import collections
import os
import time
from dataclasses import dataclass

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


@dataclass
class EnvelopeStats:
    step_spill_bytes: int = 0
    window_spill_bytes: int = 0
    step_budget_denials: int = 0
    window_budget_denials: int = 0
    throttle_events: int = 0
    cliff_events: int = 0
    dynamic_prefetch_limit: int = 0
    dynamic_queue_soft_limit: int = 0
    read_write_mix_bias: float = 1.0


class PerformanceEnvelope:
    """
    Hard performance envelope controls:
    - Spill budgets (per-step + sliding window)
    - Adaptive backpressure knobs
    - Cliff detection from tail latencies and host pressure
    """

    def __init__(self, cfg, metrics):
        self.cfg = cfg
        self.m = metrics

        self.step_budget = int(getattr(cfg, "per_step_spill_budget_bytes", 16 * 1024**3))
        self.window_budget = int(getattr(cfg, "per_window_spill_budget_bytes", 128 * 1024**3))
        self.window_steps = max(1, int(getattr(cfg, "spill_budget_window_steps", 50)))
        self.fairness_step_bytes = int(getattr(cfg, "fairness_per_step_bytes", 8 * 1024**3))
        self.prefetch_min = max(1, int(getattr(cfg, "dynamic_prefetch_limit_min", 1)))
        self.prefetch_max = max(self.prefetch_min, int(getattr(cfg, "dynamic_prefetch_limit_max", 16)))

        self._step_spill = 0
        self._step_id = -1
        self._window = collections.deque(maxlen=self.window_steps)
        self._step_rank_spill = {}
        self._stats = EnvelopeStats(
            dynamic_prefetch_limit=int(getattr(cfg, "prefetch_batch_limit", 8)),
            dynamic_queue_soft_limit=max(1, int(getattr(cfg, "max_queue", 1))),
        )

    def start_step(self, step_id: int):
        self._step_id = int(step_id)
        self._step_spill = 0
        self._step_rank_spill.clear()

    def _window_bytes(self) -> int:
        return int(sum(self._window))

    def can_spill(self, nbytes: int, rank: int = 0) -> tuple[bool, str]:
        nb = int(nbytes)
        if self.step_budget > 0 and (self._step_spill + nb) > self.step_budget:
            self._stats.step_budget_denials += 1
            self.m.spill_budget_denials += 1
            return False, "step_spill_budget"
        if self.window_budget > 0 and (self._window_bytes() + self._step_spill + nb) > self.window_budget:
            self._stats.window_budget_denials += 1
            self.m.spill_budget_denials += 1
            return False, "window_spill_budget"
        cur = int(self._step_rank_spill.get(int(rank), 0))
        if self.fairness_step_bytes > 0 and (cur + nb) > self.fairness_step_bytes:
            self.m.fairness_denials += 1
            return False, "fairness_per_step_rank_budget"
        return True, ""

    def note_spill(self, nbytes: int, rank: int = 0):
        nb = int(nbytes)
        self._step_spill += nb
        self._step_rank_spill[int(rank)] = int(self._step_rank_spill.get(int(rank), 0)) + nb
        self._stats.step_spill_bytes = int(self._step_spill)
        self._stats.window_spill_bytes = int(self._window_bytes())

    def _detect_cliff(self) -> bool:
        step_p99 = float(self.m.step_hist.pct(99))
        step_p95 = float(self.m.step_hist.pct(95))
        io_w_p99 = float(self.m.io_write_hist.pct(99))
        io_r_p99 = float(self.m.io_read_hist.pct(99))
        cpu_busy = 0.0
        mem_pressure = 0.0
        if psutil is not None:
            try:
                cpu_busy = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                mem_pressure = 100.0 - float(vm.available * 100.0 / max(1.0, vm.total))
            except Exception:
                cpu_busy = 0.0
                mem_pressure = 0.0
        cliff = False
        if step_p99 > max(5.0, 1.4 * max(1.0, step_p95)):
            cliff = True
        if io_w_p99 > 50.0 or io_r_p99 > 50.0:
            cliff = True
        if cpu_busy > 92.0 or mem_pressure > 92.0:
            cliff = True
        return cliff

    def end_step(self, io=None):
        self._window.append(int(self._step_spill))
        self._stats.step_spill_bytes = int(self._step_spill)
        self._stats.window_spill_bytes = int(self._window_bytes())

        cliff = self._detect_cliff()
        if cliff:
            self._stats.cliff_events += 1
            self._stats.throttle_events += 1
            self.m.throttle_events += 1

            # Move to safer policy.
            self.cfg.prefetch_batch_limit = max(self.prefetch_min, int(self.cfg.prefetch_batch_limit * 0.7))
            self.cfg.spill_min_bytes = int(max(1 * 1024 * 1024, self.cfg.spill_min_bytes * 1.15))
            soft = max(1, int(self.cfg.max_queue * 0.5))
            self._stats.dynamic_queue_soft_limit = soft
            self._stats.dynamic_prefetch_limit = int(self.cfg.prefetch_batch_limit)
            self._stats.read_write_mix_bias = 0.7
            if io is not None and hasattr(io, "set_soft_limits"):
                io.set_soft_limits(queue_soft_limit=soft)
        else:
            # cautiously relax toward target
            self.cfg.prefetch_batch_limit = min(self.prefetch_max, int(max(self.prefetch_min, self.cfg.prefetch_batch_limit + 1)))
            soft = min(int(self.cfg.max_queue), int(max(1, self._stats.dynamic_queue_soft_limit + 1)))
            self._stats.dynamic_queue_soft_limit = soft
            self._stats.dynamic_prefetch_limit = int(self.cfg.prefetch_batch_limit)
            self._stats.read_write_mix_bias = 1.0
            if io is not None and hasattr(io, "set_soft_limits"):
                io.set_soft_limits(queue_soft_limit=soft)

        self.m.p999_step_ms = float(self.m.step_hist.pct(99.9))

    def snapshot(self) -> dict:
        return {
            "step_spill_bytes": int(self._stats.step_spill_bytes),
            "window_spill_bytes": int(self._stats.window_spill_bytes),
            "step_budget_denials": int(self._stats.step_budget_denials),
            "window_budget_denials": int(self._stats.window_budget_denials),
            "throttle_events": int(self._stats.throttle_events),
            "cliff_events": int(self._stats.cliff_events),
            "dynamic_prefetch_limit": int(self._stats.dynamic_prefetch_limit),
            "dynamic_queue_soft_limit": int(self._stats.dynamic_queue_soft_limit),
            "read_write_mix_bias": float(self._stats.read_write_mix_bias),
            "spill_budget_step": int(self.step_budget),
            "spill_budget_window": int(self.window_budget),
        }
