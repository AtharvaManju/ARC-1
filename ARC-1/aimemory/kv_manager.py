import collections
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class KVBlock:
    key: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    phase: str
    last_access_ts: float
    resident: bool = True
    store_key: str = ""


class KVResidencyManager:
    """
    KV cache residency manager with prefill/decode policy controls.
    Keeps resident blocks under budget and spills cold blocks through SARCStorage.
    """

    def __init__(self, cfg, storage):
        self.cfg = cfg
        self.storage = storage
        self.enabled = bool(getattr(cfg, "kv_manager_enabled", False))
        self.budget = int(getattr(cfg, "kv_budget_bytes", 8 * 1024**3))
        self.prefill_lookahead = int(getattr(cfg, "kv_prefill_prefetch_lookahead", 8))
        self.decode_lookahead = int(getattr(cfg, "kv_decode_prefetch_lookahead", 1))
        self.latency_slo_ms = float(getattr(cfg, "kv_latency_slo_ms", 20.0))
        self.phase = "prefill"

        self._blocks: Dict[str, KVBlock] = {}
        self._resident_bytes = 0
        self._lru = collections.OrderedDict()
        self._spills = 0
        self._restores = 0

    def set_phase(self, phase: str):
        p = str(phase).lower()
        if p not in ("prefill", "decode"):
            raise ValueError("phase must be prefill|decode")
        self.phase = p

    def _touch(self, key: str):
        self._lru.pop(key, None)
        self._lru[key] = time.time()
        if key in self._blocks:
            self._blocks[key].last_access_ts = time.time()

    def _spill_one(self):
        if not self._lru:
            return False
        key, _ = next(iter(self._lru.items()))
        blk = self._blocks.get(key)
        if blk is None or (not blk.resident):
            self._lru.pop(key, None)
            return False

        # In decode phase be conservative on hot block eviction.
        if self.phase == "decode" and (time.time() - blk.last_access_ts) * 1000.0 < self.latency_slo_ms:
            self._lru.move_to_end(key)
            return False

        t = getattr(blk, "_tensor", None)
        if t is None:
            return False
        if not t.is_contiguous():
            t = t.contiguous()
        pb = self.storage.acquire_pinned(max(4096, blk.nbytes))
        src_u8 = t.view(torch.uint8).reshape(-1)
        pb.u8[:blk.nbytes].copy_(src_u8[:blk.nbytes], non_blocking=False)
        s_key = f"kv_{key}_{int(time.time()*1e6)}"
        self.storage.put_host_bytes(
            key=s_key,
            plain_blk=pb,
            nbytes=blk.nbytes,
            dtype_s="uint8",
            shape=(blk.nbytes,),
            step_id=0,
        )
        self.storage.release_pinned(pb)
        blk.resident = False
        blk.store_key = s_key
        self._resident_bytes -= int(blk.nbytes)
        delattr(blk, "_tensor")
        self._lru.pop(key, None)
        self._spills += 1
        return True

    def _ensure_budget(self):
        while self._resident_bytes > self.budget:
            if not self._spill_one():
                break

    def register(self, key: str, tensor: torch.Tensor):
        if not self.enabled:
            return
        nbytes = int(tensor.numel() * tensor.element_size())
        if key in self._blocks:
            self.release(key)
        b = KVBlock(
            key=str(key),
            shape=tuple(int(x) for x in tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=nbytes,
            phase=self.phase,
            last_access_ts=time.time(),
            resident=True,
            store_key="",
        )
        setattr(b, "_tensor", tensor)
        self._blocks[key] = b
        self._resident_bytes += nbytes
        self._touch(key)
        self._ensure_budget()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        b = self._blocks.get(key)
        if b is None:
            return None
        if b.resident:
            self._touch(key)
            return getattr(b, "_tensor", None)
        if not b.store_key:
            return None
        # Restore from spill store.
        meta = self.storage.get_meta(b.store_key)
        enc = self.storage.acquire_pinned(meta.padded_bytes)
        self.storage.read_block_to_pinned(meta, enc)
        out = torch.empty((b.nbytes,), dtype=torch.uint8)
        self.storage.restore_to_cuda_from_pinned(meta, enc, out)
        self.storage.release_pinned(enc)
        b.resident = True
        setattr(b, "_tensor", out.view(*b.shape))
        self._resident_bytes += int(b.nbytes)
        self._touch(key)
        self._restores += 1
        self._ensure_budget()
        return getattr(b, "_tensor", None)

    def release(self, key: str):
        b = self._blocks.pop(key, None)
        if b is None:
            return
        if b.resident:
            self._resident_bytes -= int(b.nbytes)
        self._lru.pop(key, None)

    def stats(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "phase": self.phase,
            "resident_blocks": int(sum(1 for b in self._blocks.values() if b.resident)),
            "spilled_blocks": int(sum(1 for b in self._blocks.values() if (not b.resident))),
            "resident_bytes": int(self._resident_bytes),
            "budget_bytes": int(self.budget),
            "spills": int(self._spills),
            "restores": int(self._restores),
            "prefetch_lookahead": int(self.prefill_lookahead if self.phase == "prefill" else self.decode_lookahead),
        }
