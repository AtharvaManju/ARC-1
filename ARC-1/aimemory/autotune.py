import json
import os
import hashlib
import platform
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TuneState:
    last_adjust_step: int = -1
    profile_path: str = ""
    fingerprint: str = ""


class RuntimeAutoTuner:
    def __init__(self, cfg, metrics):
        self.cfg = cfg
        self.m = metrics
        self.enabled = bool(getattr(cfg, "autotune_enabled", True))
        self.interval = max(1, int(getattr(cfg, "autotune_adjust_interval_steps", 10)))
        self.state = TuneState()
        if not self.enabled:
            return

        pdir = str(getattr(cfg, "autotune_profile_dir", "") or os.path.join(cfg.pool_dir, "profiles"))
        os.makedirs(pdir, exist_ok=True)
        fp = self._fingerprint()
        self.state.fingerprint = fp
        self.state.profile_path = os.path.join(pdir, f"{fp}.json")
        self._load_profile()

    def _fingerprint(self) -> str:
        base = str(getattr(self.cfg, "model_profile_name", "") or "default")
        backend = str(getattr(self.m, "backend", "AUTO"))
        world = int(getattr(self.cfg, "world_size", -1))
        cuda_ver = str(getattr(torch.version, "cuda", "") or "none")
        torch_ver = str(getattr(torch, "__version__", "unknown"))
        dev_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        dev_names = []
        if torch.cuda.is_available():
            for i in range(dev_count):
                try:
                    dev_names.append(torch.cuda.get_device_name(i))
                except Exception:
                    dev_names.append("unknown")
        topo = ",".join(dev_names)
        raw = (
            f"{base}:{backend}:{world}:{cuda_ver}:{torch_ver}:"
            f"{platform.system()}:{platform.machine()}:{dev_count}:{topo}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _load_profile(self):
        p = self.state.profile_path
        if not p or not os.path.exists(p):
            return
        try:
            with open(p, "r") as f:
                prof = json.load(f)
            if "spill_min_bytes" in prof:
                self.cfg.spill_min_bytes = int(prof["spill_min_bytes"])
            if "pcc_lookahead" in prof:
                self.cfg.pcc_lookahead = int(prof["pcc_lookahead"])
            if "queue_put_timeout_s" in prof:
                self.cfg.queue_put_timeout_s = float(prof["queue_put_timeout_s"])
        except Exception:
            return

    def _save_profile(self):
        p = self.state.profile_path
        if not p:
            return
        prof = {
            "spill_min_bytes": int(self.cfg.spill_min_bytes),
            "pcc_lookahead": int(self.cfg.pcc_lookahead),
            "queue_put_timeout_s": float(self.cfg.queue_put_timeout_s),
        }
        with open(p, "w") as f:
            json.dump(prof, f, indent=2)

    def _clamp(self):
        self.cfg.spill_min_bytes = max(1 * 1024 * 1024, min(int(self.cfg.spill_min_bytes), 2 * 1024**3))
        self.cfg.pcc_lookahead = max(1, min(int(self.cfg.pcc_lookahead), 32))
        self.cfg.queue_put_timeout_s = max(0.0, min(float(self.cfg.queue_put_timeout_s), 1.0))

    def observe_step(self, step_id: int):
        if not self.enabled:
            return
        if step_id <= 0 or (step_id % self.interval != 0):
            return
        if self.state.last_adjust_step == step_id:
            return
        self.state.last_adjust_step = step_id

        changed = False
        # Backpressure/safe mode => spill less aggressively.
        if self.m.safe_mode or self.m.spill_queue_overflow > 0:
            self.cfg.spill_min_bytes = int(self.cfg.spill_min_bytes * 1.20)
            self.cfg.pcc_lookahead = max(1, int(self.cfg.pcc_lookahead) - 1)
            changed = True
        else:
            # Good prefetch + low queue => spill more aggressively.
            q = float(getattr(self.m.io_queue_depth_ema, "value", 0.0))
            if self.m.prefetch_hit_rate() >= 85.0 and q < 2.0:
                self.cfg.spill_min_bytes = int(self.cfg.spill_min_bytes * 0.92)
                self.cfg.pcc_lookahead = min(32, int(self.cfg.pcc_lookahead) + 1)
                changed = True

        if changed:
            self._clamp()
            self.m.autotune_updates += 1
            self._save_profile()
