import json
import os
import threading
import time


class FaultInjector:
    """
    Test-only fault injection.
    Configure with AIMEMORY_FAULTS json, for example:
      {"torn_write_every":5,"delay_spill_ms":10,"eio_every":7}
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self._lock = threading.Lock()
        self._counts = {}

    @classmethod
    def from_env(cls):
        raw = os.environ.get("AIMEMORY_FAULTS", "").strip()
        if not raw:
            return cls({})
        try:
            cfg = json.loads(raw)
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}
        return cls(cfg)

    def enabled(self) -> bool:
        return bool(self.cfg)

    def _tick(self, key: str) -> int:
        with self._lock:
            v = int(self._counts.get(key, 0)) + 1
            self._counts[key] = v
            return v

    def delay(self, key: str, ms_key: str):
        ms = float(self.cfg.get(ms_key, 0.0))
        if ms > 0:
            time.sleep(ms / 1000.0)

    def every(self, key: str, n_key: str) -> bool:
        n = int(self.cfg.get(n_key, 0))
        if n <= 0:
            return False
        c = self._tick(key)
        return (c % n) == 0

    def should_torn_write(self) -> bool:
        return self.every("torn_write", "torn_write_every")

    def should_corrupt_read(self) -> bool:
        return self.every("corrupt_read", "corrupt_read_every")

    def should_eio(self) -> bool:
        return self.every("eio", "eio_every")

    def should_enospc(self) -> bool:
        return self.every("enospc", "enospc_every")


_FAULT_INJECTOR = FaultInjector.from_env()


def get_fault_injector() -> FaultInjector:
    return _FAULT_INJECTOR
