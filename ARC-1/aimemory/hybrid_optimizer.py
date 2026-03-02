from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HybridDecision:
    action: str  # spill | recompute | hybrid
    reason: str
    score_spill: float
    score_recompute: float


class HybridMemoryOptimizer:
    """
    Coordinates spill vs recompute preference using observed IO tails
    and compute/memory headroom. It does not implement checkpointing itself;
    instead it emits policy decisions for callers to enforce.
    """

    def __init__(self, io_tail_threshold_ms: float, compute_headroom_pct: float, recompute_bias: float = 0.5):
        self.io_tail_threshold_ms = float(max(1.0, io_tail_threshold_ms))
        self.compute_headroom_pct = float(max(0.0, compute_headroom_pct))
        self.recompute_bias = float(max(0.0, min(1.0, recompute_bias)))

    def decide(
        self,
        *,
        tensor_nbytes: int,
        io_tail_ms: float,
        memory_headroom_pct: float,
        policy: str = "balanced",
    ) -> HybridDecision:
        nmb = float(tensor_nbytes) / float(1024 * 1024)
        io_penalty = float(io_tail_ms) / max(1.0, self.io_tail_threshold_ms)
        mem_pressure = 1.0 - min(1.0, max(0.0, float(memory_headroom_pct) / max(1.0, self.compute_headroom_pct * 3.0)))
        score_spill = (nmb * (1.0 + mem_pressure)) - (io_penalty * 2.0)
        score_recompute = (io_penalty * 2.0) + (self.recompute_bias * 2.0) - (nmb * 0.2)
        p = str(policy).lower()
        if p == "throughput":
            score_recompute += 0.75
        elif p == "max_headroom":
            score_spill += 0.75
        gap = score_spill - score_recompute
        if abs(gap) < 0.5:
            return HybridDecision("hybrid", "scores_close", float(score_spill), float(score_recompute))
        if gap >= 0.5:
            return HybridDecision("spill", "spill_score_higher", float(score_spill), float(score_recompute))
        return HybridDecision("recompute", "recompute_score_higher", float(score_spill), float(score_recompute))

    def as_recommendation(
        self,
        *,
        io_tail_ms: float,
        memory_headroom_pct: float,
        policy: str,
    ) -> Dict[str, Any]:
        d = self.decide(
            tensor_nbytes=64 * 1024 * 1024,
            io_tail_ms=float(io_tail_ms),
            memory_headroom_pct=float(memory_headroom_pct),
            policy=policy,
        )
        return {
            "action": d.action,
            "reason": d.reason,
            "io_tail_threshold_ms": float(self.io_tail_threshold_ms),
            "compute_headroom_pct": float(self.compute_headroom_pct),
            "scores": {"spill": d.score_spill, "recompute": d.score_recompute},
        }
