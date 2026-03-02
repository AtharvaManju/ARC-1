import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CoordState:
    applied_step: int = -1
    last_consensus_ts: float = 0.0
    applied: int = 0


class RankCoordinator:
    """
    File-based distributed policy coordinator.
    Leader aggregates rank metrics and writes a consensus policy.
    """

    def __init__(self, root_dir: str, rank: int, world_size: int, leader_rank: int = 0):
        self.root_dir = root_dir
        self.rank = int(rank)
        self.world_size = int(max(1, world_size))
        self.leader_rank = int(leader_rank)
        self.state = CoordState()
        os.makedirs(self.root_dir, exist_ok=True)
        self.ranks_dir = os.path.join(self.root_dir, "ranks")
        os.makedirs(self.ranks_dir, exist_ok=True)
        self.consensus_path = os.path.join(self.root_dir, "consensus.json")

    def _rank_path(self, r: int) -> str:
        return os.path.join(self.ranks_dir, f"rank_{int(r)}.json")

    def publish(self, step_id: int, metrics: Dict[str, Any]):
        row = {
            "rank": self.rank,
            "step": int(step_id),
            "ts": float(time.time()),
            "spill_bytes": int(metrics.get("spill_bytes", 0)),
            "spills": int(metrics.get("spills", 0)),
            "headroom_pct": float(metrics.get("memory_headroom_pct", 0.0)),
            "safe_mode": bool(metrics.get("safe_mode", False)),
        }
        with open(self._rank_path(self.rank), "w") as f:
            json.dump(row, f, indent=2)

    def _read_rank_rows(self) -> list[dict]:
        rows = []
        for r in range(self.world_size):
            p = self._rank_path(r)
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r") as f:
                    rows.append(json.load(f))
            except Exception:
                continue
        return rows

    def leader_aggregate(self, step_id: int) -> Optional[dict]:
        if self.rank != self.leader_rank:
            return None
        rows = self._read_rank_rows()
        if not rows:
            return None
        mean_headroom = sum(float(r.get("headroom_pct", 0.0)) for r in rows) / max(1, len(rows))
        spills = [int(r.get("spills", 0)) for r in rows]
        max_spills = max(spills) if spills else 0
        min_spills = min(spills) if spills else 0
        skew = (100.0 * (max_spills - min_spills) / max(1, max_spills)) if max_spills > 0 else 0.0

        # Consensus policy: lower threshold under pressure, raise under heavy skew.
        spill_scale = 1.0
        if mean_headroom < 8.0:
            spill_scale = 0.5
        elif mean_headroom > 20.0:
            spill_scale = 1.1
        if skew > 25.0:
            spill_scale *= 1.2

        consensus = {
            "ts": float(time.time()),
            "step": int(step_id),
            "mean_headroom_pct": float(mean_headroom),
            "rank_skew_pct": float(skew),
            "spill_threshold_scale": float(spill_scale),
            "prefetch_limit_scale": float(max(0.5, min(1.2, 1.0 - (skew / 200.0)))),
        }
        with open(self.consensus_path, "w") as f:
            json.dump(consensus, f, indent=2)
        return consensus

    def poll_consensus(self, max_age_s: float = 5.0) -> Optional[dict]:
        if not os.path.exists(self.consensus_path):
            return None
        try:
            with open(self.consensus_path, "r") as f:
                c = json.load(f)
        except Exception:
            return None
        if (time.time() - float(c.get("ts", 0.0))) > float(max_age_s):
            return None
        return c

    def apply(self, cfg, consensus: dict, anti_skew: bool = True) -> bool:
        step = int(consensus.get("step", -1))
        if step <= self.state.applied_step:
            return False
        scale = float(consensus.get("spill_threshold_scale", 1.0))
        if anti_skew:
            scale = max(0.4, min(1.6, scale))
        cfg.spill_min_bytes = max(1 * 1024 * 1024, int(float(cfg.spill_min_bytes) * scale))
        pf_scale = float(consensus.get("prefetch_limit_scale", 1.0))
        cfg.prefetch_batch_limit = max(1, int(float(cfg.prefetch_batch_limit) * max(0.25, min(1.5, pf_scale))))

        self.state.applied_step = step
        self.state.last_consensus_ts = float(consensus.get("ts", time.time()))
        self.state.applied += 1
        return True
