import collections
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch


@dataclass
class KVBlock:
    key: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    phase: str
    tenant_id: str
    request_id: str
    last_access_ts: float
    resident: bool = True
    store_key: str = ""
    clock_bit: int = 1


class KVResidencyManager:
    """
    KV cache residency manager with latency-SLO aware prefill/decode behavior,
    tenant fairness, and configurable eviction policy.
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
        self.eviction_policy = str(getattr(cfg, "kv_eviction_policy", "LRU")).upper()
        self.tenant_fair = bool(getattr(cfg, "kv_tenant_fairness", True))
        self.tenant_budget_ratio = float(getattr(cfg, "kv_tenant_budget_ratio", 0.25))
        self.request_budget_ratio = float(getattr(cfg, "kv_request_budget_ratio", 0.20))
        self.overload_drop_prefetch = bool(getattr(cfg, "kv_overload_drop_prefetch", True))

        self._blocks: Dict[str, KVBlock] = {}
        self._resident_bytes = 0
        self._tenant_bytes: Dict[str, int] = {}
        self._request_bytes: Dict[str, int] = {}
        self._lru = collections.OrderedDict()
        self._clock = collections.deque()
        self._spills = 0
        self._restores = 0
        self._token_latency_ms: List[float] = []
        self._qos_denials = 0

    def set_phase(self, phase: str):
        p = str(phase).lower()
        if p not in ("prefill", "decode"):
            raise ValueError("phase must be prefill|decode")
        self.phase = p

    def _touch(self, key: str):
        ts = time.time()
        self._lru.pop(key, None)
        self._lru[key] = ts
        b = self._blocks.get(key)
        if b is not None:
            b.last_access_ts = ts
            b.clock_bit = 1

    def _tenant_budget(self, tenant_id: str) -> int:
        if not self.tenant_fair:
            return self.budget
        return max(1, int(self.budget * self.tenant_budget_ratio))

    def _request_budget(self, request_id: str) -> int:
        if not request_id:
            return self.budget
        return max(1, int(self.budget * self.request_budget_ratio))

    def _clock_pick(self, exclude_key: Optional[str] = None) -> Optional[str]:
        if not self._clock:
            return None
        for _ in range(len(self._clock) * 2):
            k = self._clock[0]
            self._clock.rotate(-1)
            if exclude_key is not None and str(k) == str(exclude_key):
                continue
            b = self._blocks.get(k)
            if b is None or (not b.resident):
                continue
            if b.clock_bit == 0:
                return k
            b.clock_bit = 0
        if not self._clock:
            return None
        if exclude_key is None:
            return self._clock[0]
        for k in self._clock:
            if str(k) != str(exclude_key):
                return k
        return None

    def _pick_victim(self, exclude_key: Optional[str] = None) -> Optional[str]:
        if self.eviction_policy == "CLOCK":
            return self._clock_pick(exclude_key=exclude_key)
        if not self._lru:
            return None
        for k in self._lru.keys():
            if exclude_key is None or str(k) != str(exclude_key):
                return k
        return None

    def _spill_one(self, exclude_key: Optional[str] = None) -> bool:
        key = self._pick_victim(exclude_key=exclude_key)
        if key is None:
            return False
        blk = self._blocks.get(key)
        if blk is None or (not blk.resident):
            self._lru.pop(key, None)
            return False

        # Decode path is latency-sensitive; avoid spilling ultra-hot blocks.
        if self.phase == "decode" and (time.time() - blk.last_access_ts) * 1000.0 < self.latency_slo_ms:
            self._touch(key)
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
        self._tenant_bytes[blk.tenant_id] = max(0, int(self._tenant_bytes.get(blk.tenant_id, 0)) - int(blk.nbytes))
        if blk.request_id:
            self._request_bytes[blk.request_id] = max(0, int(self._request_bytes.get(blk.request_id, 0)) - int(blk.nbytes))
        try:
            delattr(blk, "_tensor")
        except Exception:
            pass
        self._lru.pop(key, None)
        self._spills += 1
        return True

    def _ensure_budget(self, exclude_key: Optional[str] = None):
        while self._resident_bytes > self.budget:
            if not self._spill_one(exclude_key=exclude_key):
                break
        if self.tenant_fair:
            # Enforce per-tenant share to avoid one request monopolizing HBM.
            for tenant_id, b in list(self._tenant_bytes.items()):
                lim = self._tenant_budget(tenant_id)
                guard = 0
                while int(b) > lim and guard < 128:
                    guard += 1
                    if not self._spill_one(exclude_key=exclude_key):
                        break
                    b = int(self._tenant_bytes.get(tenant_id, 0))

    def register(self, key: str, tensor: torch.Tensor, tenant_id: str = "default", request_id: str = ""):
        if not self.enabled:
            return False
        nbytes = int(tensor.numel() * tensor.element_size())
        rid = str(request_id or "").strip()
        if rid:
            rb = self._request_budget(rid)
            cur_req = int(self._request_bytes.get(rid, 0))
            if cur_req + nbytes > rb:
                self._qos_denials += 1
                # Under per-request overload, deny growth to protect decode tail latency.
                return False
        if key in self._blocks:
            self.release(key)
        b = KVBlock(
            key=str(key),
            shape=tuple(int(x) for x in tensor.shape),
            dtype=str(tensor.dtype),
            nbytes=nbytes,
            phase=self.phase,
            tenant_id=str(tenant_id),
            request_id=rid,
            last_access_ts=time.time(),
            resident=True,
            store_key="",
            clock_bit=1,
        )
        setattr(b, "_tensor", tensor)
        self._blocks[key] = b
        self._resident_bytes += nbytes
        self._tenant_bytes[b.tenant_id] = int(self._tenant_bytes.get(b.tenant_id, 0)) + nbytes
        if rid:
            self._request_bytes[rid] = int(self._request_bytes.get(rid, 0)) + nbytes
        self._touch(key)
        self._clock.append(key)
        self._ensure_budget()
        return True

    def get(self, key: str, token_latency_ms: Optional[float] = None) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        if token_latency_ms is not None:
            self._token_latency_ms.append(float(token_latency_ms))
            if len(self._token_latency_ms) > 8192:
                self._token_latency_ms = self._token_latency_ms[-4096:]

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
        self._tenant_bytes[b.tenant_id] = int(self._tenant_bytes.get(b.tenant_id, 0)) + int(b.nbytes)
        if b.request_id:
            self._request_bytes[b.request_id] = int(self._request_bytes.get(b.request_id, 0)) + int(b.nbytes)
        self._touch(key)
        self._restores += 1
        self._ensure_budget(exclude_key=key)
        return getattr(b, "_tensor", None)

    def on_decode_tick(self, next_keys: List[str]):
        """
        Prefetch next KV blocks keyed by token cadence.
        """
        if (not self.enabled) or self.phase != "decode":
            return
        lookahead = max(1, int(self.decode_lookahead))
        if self.overload_drop_prefetch and self._token_latency_p(99) > self.latency_slo_ms:
            lookahead = 1
        for k in list(next_keys)[:lookahead]:
            b = self._blocks.get(k)
            if b is None or b.resident:
                continue
            self.get(k)

    def release(self, key: str):
        b = self._blocks.pop(key, None)
        if b is None:
            return
        if b.resident:
            self._resident_bytes -= int(b.nbytes)
            self._tenant_bytes[b.tenant_id] = max(0, int(self._tenant_bytes.get(b.tenant_id, 0)) - int(b.nbytes))
            if b.request_id:
                self._request_bytes[b.request_id] = max(0, int(self._request_bytes.get(b.request_id, 0)) - int(b.nbytes))
        self._lru.pop(key, None)

    def _token_latency_p(self, p: float) -> float:
        if not self._token_latency_ms:
            return 0.0
        arr = np.array(self._token_latency_ms, dtype=np.float64)
        return float(np.percentile(arr, p))

    def stats(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "phase": self.phase,
            "eviction_policy": self.eviction_policy,
            "tenant_fairness": bool(self.tenant_fair),
            "resident_blocks": int(sum(1 for b in self._blocks.values() if b.resident)),
            "spilled_blocks": int(sum(1 for b in self._blocks.values() if (not b.resident))),
            "resident_bytes": int(self._resident_bytes),
            "budget_bytes": int(self.budget),
            "spills": int(self._spills),
            "restores": int(self._restores),
            "tenant_bytes": {str(k): int(v) for k, v in self._tenant_bytes.items()},
            "request_bytes": {str(k): int(v) for k, v in self._request_bytes.items()},
            "qos_denials": int(self._qos_denials),
            "overload_drop_prefetch": bool(self.overload_drop_prefetch),
            "prefetch_lookahead": int(self.prefill_lookahead if self.phase == "prefill" else self.decode_lookahead),
            "token_latency_p95_ms": self._token_latency_p(95),
            "token_latency_p99_ms": self._token_latency_p(99),
        }
