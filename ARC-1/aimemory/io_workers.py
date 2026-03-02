import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import torch
from .metrics import Metrics
from .staging import OverlapPipeline
from .pinned import PinnedBlockPool, PinnedBlock
from .storage import SARCStorage, TensorMeta

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
                 pinned_pool_bytes: int, stream_priority: int):
        self.storage = storage
        self.metrics = metrics

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
        for _ in range(max(1, num_workers)):
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()
            self._threads.append(t)

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
        if not self._stop.is_set():
            try:
                if timeout_s is None:
                    self._q.put(("SPILL_HOST", job))
                else:
                    self._q.put(("SPILL_HOST", job), timeout=max(0.0, float(timeout_s)))
                return True
            except queue.Full:
                return False
        return False

    def submit_prefetch(self, job: PrefetchJob) -> bool:
        if not self._stop.is_set():
            try:
                self._q.put(("PREFETCH", job), timeout=0.0)
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

        if pr.cuda_event is not None:
            torch.cuda.current_stream().wait_event(pr.cuda_event)

        self.metrics.last_used_direct = bool(self.storage.engine.last_used_direct())
        out = pr.tensor
        popped = self.pipeline.pop(key)

        if out is not None:
            out.record_stream(torch.cuda.current_stream())

        if popped and popped.enc_block is not None:
            self.pool.release(popped.enc_block)
        if popped and popped.plain_block is not None:
            self.pool.release(popped.plain_block)

        return out

    def _loop(self):
        while not self._stop.is_set():
            kind, payload = self._q.get()
            if kind == "STOP":
                self._q.task_done()
                return

            try:
                if kind == "SPILL_HOST":
                    job: SpillHostJob = payload
                    try:
                        job.ready_evt.synchronize()
                        self.storage.put_host_bytes(
                            key=job.key,
                            plain_blk=job.pinned_block,
                            nbytes=job.nbytes,
                            dtype_s=job.dtype_s,
                            shape=job.shape,
                            step_id=job.step_id,
                            spill_done_evt=job.done_evt,
                        )
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
                        # FIX #2: prune spill_done dict entry so it doesn't grow forever
                        self.clear_spill_done(job.key)

                elif kind == "PREFETCH":
                    job: PrefetchJob = payload
                    try:
                        self.pipeline.prefetch(job.key, job.meta)
                    except Exception as e:
                        msg = f"PREFETCH failed for {job.key}: {type(e).__name__}: {e}"
                        self.metrics.disable_reason = msg
                        self.metrics.safe_mode = True
                        self.set_fatal(msg)

            finally:
                self._q.task_done()
