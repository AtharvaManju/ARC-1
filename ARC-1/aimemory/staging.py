import threading
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from .storage import TensorMeta
from .pinned import PinnedBlock
from .util import str_to_dtype

@dataclass
class PrefetchResult:
    tensor: Optional[torch.Tensor]
    ready: threading.Event
    cuda_event: Optional[torch.cuda.Event]
    enc_block: Optional[PinnedBlock]
    plain_block: Optional[PinnedBlock]
    work_block: Optional[PinnedBlock]

class OverlapPipeline:
    def __init__(self, storage, pool, stream_priority: int = -1):
        self.storage = storage
        self.pool = pool
        self.prefetch_stream = torch.cuda.Stream(priority=stream_priority)
        self._lock = threading.Lock()
        self._inflight: Dict[str, PrefetchResult] = {}

    def reserve(self, key: str) -> bool:
        with self._lock:
            if key in self._inflight:
                return False
            self._inflight[key] = PrefetchResult(None, threading.Event(), None, None, None, None)
            return True

    def get(self, key: str) -> Optional[PrefetchResult]:
        with self._lock:
            return self._inflight.get(key)

    def pop(self, key: str) -> Optional[PrefetchResult]:
        with self._lock:
            return self._inflight.pop(key, None)

    def prefetch(self, key: str, meta: TensorMeta):
        pr = self.get(key)
        if pr is None:
            return

        enc_blk = self.pool.acquire(meta.padded_bytes)
        if enc_blk is None:
            pr.ready.set()
            return

        plain_blk = None
        work_blk = None
        try:
            try:
                self.storage.mark_prefetching(key)
            except Exception:
                pass
            self.storage.read_block_to_pinned(meta, enc_blk)
            plain_blk, work_blk, owns_plain, owns_work = self.storage.decode_to_plain_block(meta, enc_blk, scratch_plain=None, scratch_work=None)
            if not owns_plain:
                plain_blk = None
            if not owns_work:
                work_blk = None
            src_u8 = plain_blk.u8[:meta.nbytes] if plain_blk is not None else enc_blk.u8[:meta.nbytes]

            dt = str_to_dtype(meta.dtype_s)
            out = torch.empty(meta.shape, device="cuda", dtype=dt)
            evt = torch.cuda.Event(enable_timing=False, blocking=False)

            src_view = src_u8.view(torch.uint8).reshape(-1)
            out_u8 = out.view(torch.uint8).reshape(-1)

            with torch.cuda.stream(self.prefetch_stream):
                out_u8.copy_(src_view, non_blocking=True)
                evt.record(self.prefetch_stream)

            pr.tensor = out
            pr.cuda_event = evt
            pr.enc_block = enc_blk
            pr.plain_block = plain_blk
            pr.work_block = work_blk
            pr.ready.set()
            try:
                self.storage.mark_resident(key)
            except Exception:
                pass

        except Exception:
            pr.ready.set()
            self.pool.release(enc_blk)
            if plain_blk is not None:
                self.pool.release(plain_blk)
            if work_blk is not None:
                self.pool.release(work_blk)
            try:
                self.storage.mark_evictable(key)
            except Exception:
                pass
            raise
