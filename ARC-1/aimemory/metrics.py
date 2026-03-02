from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict
import numpy as np

@dataclass
class EMA:
    alpha: float
    value: float = 0.0
    inited: bool = False
    def update(self, x: float) -> float:
        if not self.inited:
            self.value = x
            self.inited = True
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

@dataclass
class Hist:
    cap: int = 4096
    xs: Deque[float] = field(default_factory=lambda: deque(maxlen=4096))
    def add(self, x: float):
        self.xs.append(float(x))
    def pct(self, p: float) -> float:
        if not self.xs:
            return 0.0
        arr = np.array(self.xs, dtype=np.float64)
        return float(np.percentile(arr, p))

@dataclass
class Metrics:
    spills: int = 0
    restores: int = 0
    spill_bytes: int = 0
    restore_bytes: int = 0

    prefetch_submitted: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    prefetch_dropped: int = 0
    spill_queue_overflow: int = 0
    spill_sync_fallback: int = 0
    spill_inline_pool_exhausted: int = 0
    autotune_updates: int = 0

    restore_stall_ms_ema: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    io_queue_depth_ema: EMA = field(default_factory=lambda: EMA(alpha=0.2))
    step_ms_ema: EMA = field(default_factory=lambda: EMA(alpha=0.1))
    baseline_step_ms_ema: EMA = field(default_factory=lambda: EMA(alpha=0.08))

    restore_hist: Hist = field(default_factory=Hist)
    step_hist: Hist = field(default_factory=Hist)

    spilling_enabled: bool = True
    safe_mode: bool = False
    disable_reason: str = ""
    safe_mode_until_step: int = 0

    last_used_direct: bool = False

    pcc_enabled: bool = True
    pcc_drift_count: int = 0

    spill_commit_wait_ms_ema: EMA = field(default_factory=lambda: EMA(alpha=0.2))

    backend: str = "AUTO"
    governor_level: int = 0
    governor_adjustments: int = 0
    oom_degrade_count: int = 0
    memory_free_bytes: int = 0
    memory_total_bytes: int = 0
    memory_headroom_pct: float = 0.0
    static_plan_hits: int = 0
    coord_policy_applied: int = 0
    kv_spills: int = 0
    kv_restores: int = 0
    throttle_events: int = 0
    spill_budget_denials: int = 0
    inflight_budget_denials: int = 0
    fairness_denials: int = 0
    quarantine_events: int = 0
    p999_step_ms: float = 0.0

    # Latency attribution histograms
    restore_wait_hist: Hist = field(default_factory=Hist)
    io_read_hist: Hist = field(default_factory=Hist)
    io_write_hist: Hist = field(default_factory=Hist)
    cpu_staging_hist: Hist = field(default_factory=Hist)
    decode_hist: Hist = field(default_factory=Hist)
    decompress_hist: Hist = field(default_factory=Hist)
    h2d_copy_hist: Hist = field(default_factory=Hist)
    stream_sync_hist: Hist = field(default_factory=Hist)
    kv_token_latency_hist: Hist = field(default_factory=Hist)

    def prefetch_hit_rate(self) -> float:
        d = self.prefetch_hits + self.prefetch_misses
        return 100.0 * (self.prefetch_hits / d) if d else 100.0

    def latency_snapshot(self) -> Dict[str, float]:
        return {
            "restore_wait_p95_ms": self.restore_wait_hist.pct(95),
            "restore_wait_p99_ms": self.restore_wait_hist.pct(99),
            "io_read_p95_ms": self.io_read_hist.pct(95),
            "io_read_p99_ms": self.io_read_hist.pct(99),
            "io_write_p95_ms": self.io_write_hist.pct(95),
            "io_write_p99_ms": self.io_write_hist.pct(99),
            "cpu_staging_p95_ms": self.cpu_staging_hist.pct(95),
            "cpu_staging_p99_ms": self.cpu_staging_hist.pct(99),
            "decode_p95_ms": self.decode_hist.pct(95),
            "decode_p99_ms": self.decode_hist.pct(99),
            "decompress_p95_ms": self.decompress_hist.pct(95),
            "decompress_p99_ms": self.decompress_hist.pct(99),
            "h2d_copy_p95_ms": self.h2d_copy_hist.pct(95),
            "h2d_copy_p99_ms": self.h2d_copy_hist.pct(99),
            "stream_sync_p95_ms": self.stream_sync_hist.pct(95),
            "stream_sync_p99_ms": self.stream_sync_hist.pct(99),
            "kv_token_latency_p95_ms": self.kv_token_latency_hist.pct(95),
            "kv_token_latency_p99_ms": self.kv_token_latency_hist.pct(99),
        }
