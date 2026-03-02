import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .util import dtype_to_str, round_up
from .io_workers import SpillHostJob


@dataclass
class NativeSubmit:
    key: str
    step_id: int
    tensor: torch.Tensor
    shape: Tuple[int, ...]
    nbytes: int
    done_evt: threading.Event


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

        self._q: "queue.Queue[NativeSubmit]" = queue.Queue(maxsize=max(64, self.max_batch_ops * 8))
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._fatal_lock = threading.Lock()
        self._fatal: Optional[str] = None
        self._ops = 0
        self._batches = 0

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
            "fatal": self.fatal(),
            "queue_depth": int(self._q.qsize()),
        }

    def submit_spill(self, key: str, step_id: int, t: torch.Tensor, done_evt: threading.Event) -> bool:
        if (not self.enabled) or self._stop.is_set():
            return False
        nbytes = int(t.numel() * t.element_size())
        try:
            self._q.put_nowait(
                NativeSubmit(
                    key=str(key),
                    step_id=int(step_id),
                    tensor=t,
                    shape=tuple(int(x) for x in t.shape),
                    nbytes=nbytes,
                    done_evt=done_evt,
                )
            )
            return True
        except queue.Full:
            return False

    def _set_fatal(self, msg: str):
        with self._fatal_lock:
            if self._fatal is None:
                self._fatal = msg

    def _flush_batch(self, batch: list[NativeSubmit]):
        if not batch:
            return
        self._batches += 1
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
            except Exception as e:
                self._set_fatal(f"native runtime spill failed: {type(e).__name__}: {e}")
                sub.done_evt.set()

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
                    last_flush = time.time()
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = time.time()
