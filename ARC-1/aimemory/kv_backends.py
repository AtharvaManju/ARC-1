import importlib
from typing import Any, Optional


class BaseKVBackend:
    """
    Optional adapter interface for backend-native KV integrations.
    Return None from methods to defer to ARC-1 internal KV manager logic.
    """

    name: str = "base"

    def register(self, manager, key: str, tensor, tenant_id: str = "default", request_id: str = "") -> Optional[bool]:
        return None

    def get(self, manager, key: str, token_latency_ms: Optional[float] = None):
        return None

    def release(self, manager, key: str) -> Optional[bool]:
        return None

    def on_decode_tick(self, manager, next_keys):
        return None

    def stats(self) -> dict:
        return {"backend": self.name}


class PagedAttentionLikeBackend(BaseKVBackend):
    """
    Stub adapter point for paged-attention style integration.
    Can be replaced by vendor-specific adapters without changing manager API.
    """

    name = "paged_attention_like"

    def __init__(self):
        self.register_calls = 0
        self.get_calls = 0
        self.release_calls = 0

    def register(self, manager, key: str, tensor, tenant_id: str = "default", request_id: str = "") -> Optional[bool]:
        self.register_calls += 1
        return None

    def get(self, manager, key: str, token_latency_ms: Optional[float] = None):
        self.get_calls += 1
        return None

    def release(self, manager, key: str) -> Optional[bool]:
        self.release_calls += 1
        return None

    def stats(self) -> dict:
        return {
            "backend": self.name,
            "register_calls": int(self.register_calls),
            "get_calls": int(self.get_calls),
            "release_calls": int(self.release_calls),
        }


def load_kv_backend(spec: str) -> Optional[BaseKVBackend]:
    s = str(spec or "").strip()
    if not s:
        return None
    if s.lower() in ("paged", "paged_attention", "paged_attention_like"):
        return PagedAttentionLikeBackend()
    if ":" in s:
        mod_name, cls_name = s.split(":", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        inst = cls()
        if isinstance(inst, BaseKVBackend):
            return inst
        return inst
    return None
