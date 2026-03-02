from dataclasses import dataclass

import torch
from .allocator import allocator_snapshot


@dataclass
class GovernorSnapshot:
    level: int
    free_bytes: int
    total_bytes: int
    headroom_pct: float
    adjusted: bool


class MemoryGovernor:
    """
    Converts allocator pressure into controlled degradation steps:
    level 0 = baseline policy
    level 1 = aggressive spill + lighter prefetch
    level 2 = emergency mode (max spill pressure, minimal prefetch)
    """

    def __init__(self, cfg, metrics):
        self.cfg = cfg
        self.metrics = metrics
        self.enabled = bool(getattr(cfg, "governor_enabled", True))

        self.warn_pct = max(0.0, float(getattr(cfg, "governor_warn_headroom_pct", 12.0)))
        self.emergency_pct = max(0.0, float(getattr(cfg, "governor_emergency_headroom_pct", 6.0)))
        self.cooldown_steps = max(1, int(getattr(cfg, "governor_cooldown_steps", 5)))
        self.spill_floor = max(1 * 1024 * 1024, int(getattr(cfg, "governor_spill_min_floor_bytes", 1 * 1024 * 1024)))

        self._base_spill_min = int(cfg.spill_min_bytes)
        self._base_lookahead = int(cfg.pcc_lookahead)
        self._base_prefetch_batch = int(getattr(cfg, "prefetch_batch_limit", 8))
        self._base_pack_wait = bool(getattr(cfg, "pack_wait_for_commit", False))

        self._level = 0
        self._last_adjust_step = -1

    def _read_mem(self) -> tuple[int, int]:
        if not torch.cuda.is_available():
            return 0, 0
        free_b, total_b = torch.cuda.mem_get_info()
        return int(free_b), int(total_b)

    def _pick_level(self, headroom_pct: float) -> int:
        alloc = allocator_snapshot()
        frag = float(alloc.get("fragmentation_pct", 0.0))
        if headroom_pct <= self.emergency_pct:
            return 2
        if frag >= float(getattr(self.cfg, "allocator_fragmentation_warn_pct", 35.0)):
            return max(1, int(self._level))
        if headroom_pct <= self.warn_pct:
            return 1
        return 0

    def _apply_level(self, level: int) -> bool:
        if level == self._level:
            return False
        if level == 0:
            self.cfg.spill_min_bytes = int(self._base_spill_min)
            self.cfg.pcc_lookahead = int(self._base_lookahead)
            self.cfg.prefetch_batch_limit = int(self._base_prefetch_batch)
            self.cfg.pack_wait_for_commit = bool(self._base_pack_wait)
        elif level == 1:
            self.cfg.spill_min_bytes = max(self.spill_floor, int(self._base_spill_min * 0.5))
            self.cfg.pcc_lookahead = max(1, int(self._base_lookahead * 0.5))
            self.cfg.prefetch_batch_limit = max(1, int(self._base_prefetch_batch * 0.5))
            self.cfg.pack_wait_for_commit = bool(self._base_pack_wait)
        else:
            self.cfg.spill_min_bytes = max(self.spill_floor, int(self._base_spill_min * 0.25))
            self.cfg.pcc_lookahead = 1
            self.cfg.prefetch_batch_limit = 1
            self.cfg.pack_wait_for_commit = True

        self._level = int(level)
        self.metrics.governor_level = int(level)
        self.metrics.governor_adjustments += 1
        return True

    def observe_step(self, step_id: int) -> GovernorSnapshot:
        if (not self.enabled) or (not torch.cuda.is_available()):
            return GovernorSnapshot(
                level=int(self._level),
                free_bytes=0,
                total_bytes=0,
                headroom_pct=0.0,
                adjusted=False,
            )

        free_b, total_b = self._read_mem()
        headroom = (100.0 * float(free_b) / float(total_b)) if total_b > 0 else 0.0
        self.metrics.memory_free_bytes = int(free_b)
        self.metrics.memory_total_bytes = int(total_b)
        self.metrics.memory_headroom_pct = float(headroom)

        target = self._pick_level(headroom)
        adjusted = False
        if target != self._level:
            if self._last_adjust_step < 0 or (int(step_id) - self._last_adjust_step) >= self.cooldown_steps:
                adjusted = self._apply_level(target)
                self._last_adjust_step = int(step_id)

        return GovernorSnapshot(
            level=int(self._level),
            free_bytes=int(free_b),
            total_bytes=int(total_b),
            headroom_pct=float(headroom),
            adjusted=bool(adjusted),
        )

    def on_oom_signal(self, step_id: int):
        self.metrics.oom_degrade_count += 1
        self._apply_level(2)
        self._last_adjust_step = int(step_id)
