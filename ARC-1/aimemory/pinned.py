from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import bisect
import torch
from .util import round_up

@dataclass
class PinnedBlock:
    u8: torch.Tensor
    size: int

class PinnedBlockPool:
    def __init__(self, alloc_fn, max_bytes: int):
        self.alloc_fn = alloc_fn
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._free: Dict[int, List[PinnedBlock]] = {}
        self._sizes: List[int] = []  # sorted keys for O(log n) selection
        self._bytes_live = 0

    def acquire(self, min_bytes: int) -> Optional[PinnedBlock]:
        need = int(round_up(int(min_bytes)))
        with self._lock:
            # find first size >= need
            i = bisect.bisect_left(self._sizes, need)
            while i < len(self._sizes):
                k = self._sizes[i]
                lst = self._free.get(k)
                if lst:
                    blk = lst.pop()
                    if not lst:
                        # keep key (size) in _sizes; no need to remove; cheap to keep
                        pass
                    return blk
                i += 1

            if self._bytes_live + need > self.max_bytes:
                return None

            t = self.alloc_fn(need)
            b = int(t.numel())
            self._bytes_live += b
            return PinnedBlock(u8=t, size=b)

    def release(self, blk: Optional[PinnedBlock]):
        if blk is None:
            return
        sz = int(blk.size)
        with self._lock:
            if sz not in self._free:
                self._free[sz] = []
                bisect.insort(self._sizes, sz)
            self._free[sz].append(blk)
