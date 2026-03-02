import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import torch
from .metrics import Metrics
from .staging import OverlapPipeline
from .pinned import PinnedBlockPool, PinnedBlock
from .storage import SARCStorage, TensorMeta
from .fault_injection import get_fault_injector

@dataclass
class SpillHostJob:
    key: str
    step_id: int
    pinned_block: PinnedBlock
    nbytes: int
    dtype_s: str
    shape: Tuple[int, ...]
    ready_evt: torch.cuda.Event
    done_evt: threading.Event

@dataclass
class PrefetchJob:
    key: str
    meta: TensorMeta

class IOWorkers:
    def __init__(self, storage: SARCStorage, metrics: Metrics, num_workers: int, max_queue: int,
                 pinned_pool_bytes: int, stream_priority: int, enforce_ordering: bool = False,
                 inflight_spill_bytes_limit: int = 0, inflight_prefetch_bytes_limit: int = 0,
                 fairness_per_step_bytes: int = 0, nvme_write_bw_limit_mb_s: float = 0.0,
                 nvme_read_bw_limit_mb_s: float = 0.0, queue_soft_limit: int = 0):
        self.storage = storage
        self.metrics = metrics
        self._enforce_ordering = bool(enforce_ordering)
        self._max_queue = int(max_queue)
        self._soft_queue_limit = int(max(1, queue_soft_limit if queue_soft_limit > 0 else max_queue))

        self._inflight_spill_limit = int(max(0, inflight_spill_bytes_limit))
        self._inflight_prefetch_limit = int(max(0, inflight_prefetch_bytes_limit))
        self._inflight_spill_bytes = 0
        self._inflight_prefetch_bytes = 0
        self._inflight_lock = threading.Lock()

        self._fairness_per_step_bytes = int(max(0, fairness_per_step_bytes))
        self._step_bytes: Dict[int, int] = {}
        self._step_bytes_lock = threading.Lock()

        self._write_bw_limit = float(max(0.0, nvme_write_bw_limit_mb_s))
        self._read_bw_limit = float(max(0.0, nvme_read_bw_limit_mb_s))
        self._bw_lock = threading.Lock()
        self._bw_last_read_ts = 0.0
        self._bw_last_write_ts = 0.0
        self._fault = get_fault_injector()

        alloc_fn = lambda n: storage.engine.alloc_pinned_u8(n)
        self.pool = PinnedBlockPool(alloc_fn=alloc_fn, max_bytes=pinned_pool_bytes)
        self.storage.pinned_pool = self.pool

        self.pipeline = OverlapPipeline(storage, pool=self.pool, stream_priority=stream_priority)

        self._q: "queue.Queue[Tuple[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()

        self._spill_done: Dict[str, threading.Event] = {}
        self._spill_lock = threading.Lock()

        self._fatal_lock = threading.Lock()
        self._fatal_io_error: Optional[str] = None

        self._threads = []
        worker_count = 1 if self._enforce_ordering else max(1, num_workers)
        for _ in range(worker_count):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self._threads.append(t)

    def set_soft_limits(self, queue_soft_limit: Optional[int] = None):
        if queue_soft_limit is not None:
            self._soft_queue_limit = int(max(1, min(self._max_queue, int(queue_soft_limit))))

    def shutdown(self):
        self._stop.set()
        for _ in self._threads:
            try:
                self._q.put_nowait(("STOP", None))
            except queue.Full:
                pass
        for t in self._threads:
            try:
                t.join(timeout=5.0)
            except Exception:
                pass

    def queue_depth(self) -> int:
        return self._q.qsize()

    def backpressure_state(self) -> Dict[str, int]:
        with self._inflight_lock:
            spill_b = int(self._inflight_spill_bytes)
            pref_b = int(self._inflight_prefetch_bytes)
        return {
            "queue_depth": int(self._q.qsize()),
            "queue_soft_limit": int(self._soft_queue_limit),
            "inflight_spill_bytes": spill_b,
            "inflight_prefetch_bytes": pref_b,
            "inflight_spill_limit": int(self._inflight_spill_limit),
            "inflight_prefetch_limit": int(self._inflight_prefetch_limit),
        }

    def reserve_prefetch(self, key: str) -> bool:
        return self.pipeline.reserve(key)

    def spill_done_event(self, key: str) -> threading.Event:
        with self._spill_lock:
            ev = self._spill_done.get(key)
            if ev is None:
                ev = threading.Event()
                self._spill_done[key] = ev
            return ev

    def clear_spill_done(self, key: str):
        with self._spill_lock:
            self._spill_done.pop(key, None)

    def set_fatal(self, msg: str):
        with self._fatal_lock:
            if self._fatal_io_error is None:
                self._fatal_io_error = msg

    def fatal_error(self) -> Optional[str]:
        with self._fatal_lock:
            return self._fatal_io_error

    def acquire_pinned(self, nbytes: int) -> Optional[PinnedBlock]:
        return self.pool.acquire(nbytes)

    def release_pinned(self, blk: Optional[PinnedBlock]):
        self.pool.release(blk)

    def submit_spill_host(self, job: SpillHostJob, timeout_s: Optional[float] = None) -> bool:
        if self._q.qsize() >= int(self._soft_queue_limit):
            self.metrics.spill_queue_overflow += 1
            return False
        with self._inflight_lock:
            if self._inflight_spill_limit > 0 and (self._inflight_spill_bytes + int(job.nbytes)) > self._inflight_spill_limit:
                self.metrics.inflight_budget_denials += 1
                return False
        with self._step_bytes_lock:
            cur = int(self._step_bytes.get(int(job.step_id), 0))
            if self._fairness_per_step_bytes > 0 and (cur + int(job.nbytes)) > self._fairness_per_step_bytes:
                self.metrics.fairness_denials += 1
                return False
            self._step_bytes[int(job.step_id)] = cur + int(job.nbytes)
        if not self._stop.is_set():
            try:
                if timeout_s is None:
                    self._q.put(("SPILL_HOST", job))
                else:
                    self._q.put(("SPILL_HOST", job), timeout=max(0.0, float(timeout_s)))
                with self._inflight_lock:
                    self._inflight_spill_bytes += int(job.nbytes)
                return True
            except queue.Full:
                with self._step_bytes_lock:
                    self._step_bytes[int(job.step_id)] = max(0, int(self._step_bytes.get(int(job.step_id), 0)) - int(job.nbytes))
                return False
        return False

    def submit_prefetch(self, job: PrefetchJob) -> bool:
        if self._q.qsize() >= int(self._soft_queue_limit):
            return False
        with self._inflight_lock:
            if self._inflight_prefetch_limit > 0 and (self._inflight_prefetch_bytes + int(job.meta.nbytes)) > self._inflight_prefetch_limit:
                self.metrics.inflight_budget_denials += 1
                return False
        if not self._stop.is_set():
            try:
                self._q.put(("PREFETCH", job), timeout=0.0)
                with self._inflight_lock:
                    self._inflight_prefetch_bytes += int(job.meta.nbytes)
                return True
            except queue.Full:
                return False
        return False

    def take_prefetched(self, key: str, timeout_s: float) -> Optional[torch.Tensor]:
        pr = self.pipeline.get(key)
        if not pr:
            return None
        ok = pr.ready.wait(timeout=timeout_s)
        if not ok:
            return None
        self.metrics.restore_wait_hist.add(0.0)

        if pr.cuda_event is not None:
            t0_sync = time.time()
            torch.cuda.current_stream().wait_event(pr.cuda_event)
            self.metrics.stream_sync_hist.add((time.time() - t0_sync) * 1000.0)

        self.metrics.last_used_direct = bool(self.storage.engine.last_used_direct())
        out = pr.tensor
        popped = self.pipeline.pop(key)

        if out is not None:
            out.record_stream(torch.cuda.current_stream())

        if popped and popped.enc_block is not None:
            self.pool.release(popped.enc_block)
        if popped and popped.plain_block is not None:
            self.pool.release(popped.plain_block)
        if popped and popped.work_block is not None:
            self.pool.release(popped.work_block)
        try:
            self.storage.mark_released(key)
        except Exception:
            pass
        try:
            self.storage.mark_restored(key)
        except Exception:
            pass

        return out

    def _bw_sleep(self, nbytes: int, is_write: bool):
        limit = self._write_bw_limit if is_write else self._read_bw_limit
        if limit <= 0.0:
            return
        need_s = float(nbytes) / max(1.0, limit * 1024.0 * 1024.0)
        with self._bw_lock:
            now = time.time()
            last = self._bw_last_write_ts if is_write else self._bw_last_read_ts
            elapsed = max(0.0, now - last)
            if elapsed < need_s:
                time.sleep(need_s - elapsed)
                now = time.time()
            if is_write:
                self._bw_last_write_ts = now
            else:
                self._bw_last_read_ts = now

    def _loop(self):
        while not self._stop.is_set():
            kind, payload = self._q.get()
            if kind == "STOP":
                self._q.task_done()
                return

            try:
                self._fault.delay("worker", "delay_worker_ms")
                if kind == "SPILL_HOST":
                    job: SpillHostJob = payload
                    t0_stage = time.time()
                    try:
                        if self._fault.should_enospc():
                            raise OSError("fault:ENOSPC")
                        if self._fault.should_eio():
                            raise OSError("fault:EIO")
                        job.ready_evt.synchronize()
                        self.metrics.cpu_staging_hist.add((time.time() - t0_stage) * 1000.0)
                        t0_write = time.time()
                        self.storage.put_host_bytes(
                            key=job.key,
                            plain_blk=job.pinned_block,
                            nbytes=job.nbytes,
                            dtype_s=job.dtype_s,
                            shape=job.shape,
                            step_id=job.step_id,
                            spill_done_evt=job.done_evt,
                        )
                        dt_write = (time.time() - t0_write) * 1000.0
                        self.metrics.io_write_hist.add(dt_write)
                        self._bw_sleep(job.nbytes, is_write=True)
                    except Exception as e:
                        # Fail-fast: delete uncommitted row, mark fatal, force safe mode
                        try:
                            self.storage.mark_failed(job.key)
                        except Exception:
                            pass
                        self.metrics.spilling_enabled = False
                        self.metrics.safe_mode = True
                        msg = f"SPILL failed for {job.key}: {type(e).__name__}: {e}"
                        self.metrics.disable_reason = msg
                        self.set_fatal(msg)
                    finally:
                        self.pool.release(job.pinned_block)
                        job.done_evt.set()
                        with self._inflight_lock:
                            self._inflight_spill_bytes = max(0, self._inflight_spill_bytes - int(job.nbytes))
                        with self._step_bytes_lock:
                            sid = int(job.step_id)
                            self._step_bytes[sid] = max(0, int(self._step_bytes.get(sid, 0)) - int(job.nbytes))
                        # FIX #2: prune spill_done dict entry so it doesn't grow forever
                        self.clear_spill_done(job.key)

                elif kind == "PREFETCH":
                    job: PrefetchJob = payload
                    try:
                        if self._fault.should_eio():
                            raise OSError("fault:EIO")
                        t0_read = time.time()
                        self.pipeline.prefetch(job.key, job.meta)
                        self.metrics.io_read_hist.add((time.time() - t0_read) * 1000.0)
                        self._bw_sleep(job.meta.nbytes, is_write=False)
                    except Exception as e:
                        msg = f"PREFETCH failed for {job.key}: {type(e).__name__}: {e}"
                        self.metrics.disable_reason = msg
                        self.metrics.safe_mode = True
                        self.set_fatal(msg)
                    finally:
                        with self._inflight_lock:
                            self._inflight_prefetch_bytes = max(0, self._inflight_prefetch_bytes - int(job.meta.nbytes))

            finally:
                self._q.task_done()
