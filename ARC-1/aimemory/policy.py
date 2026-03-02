from dataclasses import dataclass
from .metrics import Metrics

@dataclass
class PolicyState:
    step_id: int = 0
    safe_mode: bool = False
    safe_mode_until: int = 0

class AdaptivePolicy:
    def __init__(self, metrics: Metrics, overhead_sla_pct: float, safe_mode_cooldown_steps: int):
        self.m = metrics
        self.overhead_sla_pct = float(overhead_sla_pct)
        self.safe_mode_cooldown_steps = int(safe_mode_cooldown_steps)

    def update_step_baseline(self, step_ms: float, with_aimemory: bool):
        if not with_aimemory or self.m.safe_mode:
            self.m.baseline_step_ms_ema.update(step_ms)

    def enforce_sla(self, step_id: int, step_ms: float):
        if not self.m.baseline_step_ms_ema.inited:
            return
        base = self.m.baseline_step_ms_ema.value
        if base <= 0:
            return
        overhead_pct = 100.0 * (step_ms / base - 1.0)
        if overhead_pct > self.overhead_sla_pct:
            self.m.spilling_enabled = False
            self.m.safe_mode = True
            self.m.disable_reason = f"SLA: overhead {overhead_pct:.1f}% > {self.overhead_sla_pct:.1f}%"
            self.m.safe_mode_until_step = step_id + self.safe_mode_cooldown_steps

    def maybe_reenable(self, step_id: int):
        if self.m.safe_mode and step_id >= self.m.safe_mode_until_step:
            self.m.safe_mode = False
            self.m.disable_reason = ""
            self.m.spilling_enabled = True
