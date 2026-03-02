import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch

from .util import dtype_to_str, round_up
from .io_workers import SpillHostJob, PrefetchJob


@dataclass
class NativeSubmit:
    key: str
    step_id: int
    tensor: torch.Tensor
    shape: Tuple[int, ...]
    nbytes: int
    done_evt: threading.Event
    logical_chunks: int = 1


class NativeRuntime:
    """
    Batching shim that minimizes Python control logic in hook hot paths by
    moving spill orchestration to a dedicated worker thread.
    """

    def __init__(self, cfg, storage, io, metrics):
        self.cfg = cfg
        self.storage = storage
        self.io = io
        self.metrics = metrics
        self.enabled = bool(getattr(cfg, "native_performance_mode", False))
        self.batch_submit = bool(getattr(cfg, "native_batch_submit", True))
        self.max_batch_ops = max(1, int(getattr(cfg, "native_max_batch_ops", 64)))
        self.flush_interval_s = max(0.0, float(getattr(cfg, "native_flush_interval_ms", 0.25)) / 1000.0)
        self.chunk_bytes = max(1, int(getattr(cfg, "native_chunk_bytes", 64 * 1024 * 1024)))
        self.inflight_bytes_limit = max(0, int(getattr(cfg, "native_inflight_bytes_limit", 8 * 1024**3)))
        self.step_budget_bytes = max(0, int(getattr(cfg, "native_step_budget_bytes", 16 * 1024**3)))
        self.adaptive_batching = bool(getattr(cfg, "native_adaptive_batching", True))
        self.target_batch_latency_s = max(0.0001, float(getattr(cfg, "native_target_batch_latency_ms", 1.0)) / 1000.0)

        self._q: "queue.Queue[NativeSubmit]" = queue.Queue(maxsize=max(64, self.max_batch_ops * 8))
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._fatal_lock = threading.Lock()
        self._fatal: Optional[str] = None
        self._ops = 0
        self._batches = 0
        self._logical_chunks = 0
        self._state_updates = 0
        self._inflight_bytes = 0
        self._inflight_lock = threading.Lock()
        self._prefetch_plan: List[tuple[str, object]] = []
        self._prefetch_lock = threading.Lock()
        self._batch_latency_ms_ema = 0.0
        self._step_bytes: dict[int, int] = {}
        self._step_lock = threading.Lock()
        self._budget_denials = 0

        if self.enabled:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()

    def close(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)

    def fatal(self) -> Optional[str]:
        with self._fatal_lock:
            return self._fatal

    def stats(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "ops": int(self._ops),
            "batches": int(self._batches),
            "logical_chunks": int(self._logical_chunks),
            "state_updates": int(self._state_updates),
            "inflight_bytes": int(self._inflight_bytes),
            "inflight_limit": int(self.inflight_bytes_limit),
            "fatal": self.fatal(),
            "queue_depth": int(self._q.qsize()),
            "batch_latency_ms_ema": float(self._batch_latency_ms_ema),
            "native_budget_denials": int(self._budget_denials),
        }

    def update_prefetch_schedule(self, items: List[tuple[str, object]]):
        with self._prefetch_lock:
            self._prefetch_plan = list(items)

    def _drain_prefetch_schedule(self):
        with self._prefetch_lock:
            items = list(self._prefetch_plan)
            self._prefetch_plan.clear()
        for key, meta in items:
            try:
                self.io.reserve_prefetch(key)
                self.io.submit_prefetch(PrefetchJob(key=key, meta=meta))
            except Exception:
                continue

    def submit_spill(self, key: str, step_id: int, t: torch.Tensor, done_evt: threading.Event) -> bool:
        if (not self.enabled) or self._stop.is_set():
            return False
        nbytes = int(t.numel() * t.element_size())
        if self.step_budget_bytes > 0:
            with self._step_lock:
                cur = int(self._step_bytes.get(int(step_id), 0))
                if (cur + nbytes) > self.step_budget_bytes:
                    self.metrics.spill_budget_denials += 1
                    self._budget_denials += 1
                    return False
                self._step_bytes[int(step_id)] = cur + nbytes
        with self._inflight_lock:
            if self.inflight_bytes_limit > 0 and (self._inflight_bytes + nbytes) > self.inflight_bytes_limit:
                self.metrics.inflight_budget_denials += 1
                with self._step_lock:
                    self._step_bytes[int(step_id)] = max(0, int(self._step_bytes.get(int(step_id), 0)) - int(nbytes))
                return False
            self._inflight_bytes += nbytes
        try:
            self._q.put_nowait(
                NativeSubmit(
                    key=str(key),
                    step_id=int(step_id),
                    tensor=t,
                    shape=tuple(int(x) for x in t.shape),
                    nbytes=nbytes,
                    done_evt=done_evt,
                    logical_chunks=max(1, int((nbytes + self.chunk_bytes - 1) // self.chunk_bytes)),
                )
            )
            return True
        except queue.Full:
            with self._inflight_lock:
                self._inflight_bytes = max(0, self._inflight_bytes - nbytes)
            with self._step_lock:
                self._step_bytes[int(step_id)] = max(0, int(self._step_bytes.get(int(step_id), 0)) - int(nbytes))
            return False

    def _set_fatal(self, msg: str):
        with self._fatal_lock:
            if self._fatal is None:
                self._fatal = msg

    def _flush_batch(self, batch: list[NativeSubmit]):
        if not batch:
            return
        self._batches += 1
        t0_batch = time.time()
        for sub in batch:
            try:
                t = sub.tensor
                if not t.is_contiguous():
                    t = t.contiguous()
                padded_plain = int(round_up(sub.nbytes, 4096))
                blk = self.io.acquire_pinned(padded_plain)
                if blk is None:
                    self.metrics.spill_inline_pool_exhausted += 1
                    sub.done_evt.set()
                    continue

                src_u8 = t.view(torch.uint8).reshape(-1)
                blk.u8[:sub.nbytes].copy_(src_u8, non_blocking=True)
                ready_evt = torch.cuda.Event(enable_timing=False, blocking=False)
                ready_evt.record(torch.cuda.current_stream())
                self.io.submit_spill_host(
                    SpillHostJob(
                        key=sub.key,
                        step_id=sub.step_id,
                        pinned_block=blk,
                        nbytes=sub.nbytes,
                        dtype_s=dtype_to_str(t.dtype),
                        shape=sub.shape,
                        ready_evt=ready_evt,
                        done_evt=sub.done_evt,
                    ),
                    timeout_s=float(getattr(self.cfg, "queue_put_timeout_s", 0.01)),
                )
                self._ops += 1
                self._logical_chunks += int(sub.logical_chunks)
                self._state_updates += 1
            except Exception as e:
                self._set_fatal(f"native runtime spill failed: {type(e).__name__}: {e}")
                sub.done_evt.set()
            finally:
                with self._inflight_lock:
                    self._inflight_bytes = max(0, self._inflight_bytes - int(sub.nbytes))
                with self._step_lock:
                    sid = int(sub.step_id)
                    self._step_bytes[sid] = max(0, int(self._step_bytes.get(sid, 0)) - int(sub.nbytes))
        dt = (time.time() - t0_batch) * 1000.0
        if self._batch_latency_ms_ema <= 0.0:
            self._batch_latency_ms_ema = float(dt)
        else:
            self._batch_latency_ms_ema = 0.2 * float(dt) + 0.8 * float(self._batch_latency_ms_ema)
        if self.adaptive_batching:
            if (dt / 1000.0) > self.target_batch_latency_s and self.max_batch_ops > 4:
                self.max_batch_ops = max(4, int(self.max_batch_ops * 0.8))
            elif (dt / 1000.0) < (self.target_batch_latency_s * 0.5):
                self.max_batch_ops = min(256, int(self.max_batch_ops + 2))

    def _loop(self):
        batch: list[NativeSubmit] = []
        last_flush = time.time()
        while not self._stop.is_set():
            timeout = max(0.0, self.flush_interval_s - (time.time() - last_flush))
            try:
                sub = self._q.get(timeout=timeout if self.batch_submit else 0.1)
                batch.append(sub)
                if (not self.batch_submit) or len(batch) >= self.max_batch_ops:
                    self._flush_batch(batch)
                    batch.clear()
                    self._drain_prefetch_schedule()
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch.clear()
                    self._drain_prefetch_schedule()
                    last_flush = time.time()
