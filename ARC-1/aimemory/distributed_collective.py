import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CollectiveState:
    applied_step: int = -1
    last_consensus_ts: float = 0.0
    applied: int = 0


def _build_consensus(rows: List[Dict[str, Any]], step_id: int, world_size: int, min_quorum_ratio: float) -> Dict[str, Any]:
    mean_headroom = sum(float(r.get("headroom_pct", 0.0)) for r in rows) / max(1, len(rows))
    min_headroom = min(float(r.get("headroom_pct", 0.0)) for r in rows)
    worst_step_p99 = max(float(r.get("step_p99_ms", 0.0)) for r in rows)
    worst_io_p99 = max(max(float(r.get("io_write_p99_ms", 0.0)), float(r.get("io_read_p99_ms", 0.0))) for r in rows)
    spills = [int(r.get("spills", 0)) for r in rows]
    max_spills = max(spills) if spills else 0
    min_spills = min(spills) if spills else 0
    skew = (100.0 * (max_spills - min_spills) / max(1, max_spills)) if max_spills > 0 else 0.0

    quorum = float(len(rows)) / float(max(1, world_size))
    quorum_ok = quorum >= float(max(0.0, min(1.0, min_quorum_ratio)))

    spill_scale = 1.0
    if min_headroom < 8.0:
        spill_scale = 0.5
    elif mean_headroom > 20.0:
        spill_scale = 1.1
    if skew > 25.0:
        spill_scale *= 1.2

    global_level = 0
    if (not quorum_ok) or min_headroom < 6.0 or worst_step_p99 > 150.0 or worst_io_p99 > 60.0:
        global_level = 2
    elif min_headroom < 12.0 or worst_step_p99 > 80.0 or worst_io_p99 > 30.0:
        global_level = 1

    mean_spills = float(sum(spills) / max(1, len(spills)))
    per_rank_scale: Dict[str, float] = {}
    for r in rows:
        rk = int(r.get("rank", 0))
        rv = float(r.get("spills", 0))
        if rv > mean_spills:
            per_rank_scale[str(rk)] = 1.15
        elif rv < mean_spills:
            per_rank_scale[str(rk)] = 0.9
        else:
            per_rank_scale[str(rk)] = 1.0

    return {
        "ts": float(time.time()),
        "step": int(step_id),
        "mean_headroom_pct": float(mean_headroom),
        "min_headroom_pct": float(min_headroom),
        "rank_skew_pct": float(skew),
        "worst_step_p99_ms": float(worst_step_p99),
        "worst_io_p99_ms": float(worst_io_p99),
        "global_level": int(global_level),
        "spill_threshold_scale": float(spill_scale),
        "prefetch_limit_scale": float(max(0.5, min(1.2, 1.0 - (skew / 200.0)))),
        "per_rank_scale": per_rank_scale,
        "quorum_ratio": float(quorum),
        "quorum_ok": bool(quorum_ok),
        "rows_seen": int(len(rows)),
        "world_size": int(world_size),
    }


class CollectiveCoordinator:
    """
    torch.distributed-backed coordination path.
    Avoids shared filesystem jitter for rank policy sync.
    """

    def __init__(self, rank: int, world_size: int, leader_rank: int = 0):
        self.rank = int(rank)
        self.world_size = int(max(1, world_size))
        self.leader_rank = int(leader_rank)
        self.state = CollectiveState()

    def _dist_ready(self) -> bool:
        try:
            import torch.distributed as dist
            return bool(dist.is_available() and dist.is_initialized())
        except Exception:
            return False

    def sync_consensus(
        self,
        step_id: int,
        metrics: Dict[str, Any],
        topology: Optional[Dict[str, Any]] = None,
        *,
        max_rank_stale_s: float = 5.0,
        min_quorum_ratio: float = 0.75,
    ) -> Optional[Dict[str, Any]]:
        if not self._dist_ready():
            return None
        import torch.distributed as dist

        lat = metrics.get("latency_attribution", {}) if isinstance(metrics.get("latency_attribution", {}), dict) else {}
        row = {
            "rank": int(self.rank),
            "step": int(step_id),
            "ts": float(time.time()),
            "spill_bytes": int(metrics.get("spill_bytes", 0)),
            "spills": int(metrics.get("spills", 0)),
            "headroom_pct": float(metrics.get("memory_headroom_pct", 0.0)),
            "safe_mode": bool(metrics.get("safe_mode", False)),
            "step_p99_ms": float(metrics.get("step_p99_ms", 0.0)),
            "io_write_p99_ms": float(lat.get("io_write_p99_ms", 0.0)),
            "io_read_p99_ms": float(lat.get("io_read_p99_ms", 0.0)),
            "topology": topology or {},
        }
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, row)

        consensus: Optional[Dict[str, Any]] = None
        if self.rank == self.leader_rank:
            now = time.time()
            rows = [r for r in gathered if isinstance(r, dict)]
            fresh = [r for r in rows if (now - float(r.get("ts", 0.0))) <= float(max_rank_stale_s)]
            rows = fresh if fresh else rows
            if rows:
                consensus = _build_consensus(rows, step_id=step_id, world_size=self.world_size, min_quorum_ratio=min_quorum_ratio)

        box: List[Optional[Dict[str, Any]]] = [consensus]
        dist.broadcast_object_list(box, src=self.leader_rank)
        return box[0]

    def apply(self, cfg, consensus: Dict[str, Any], anti_skew: bool = True, rank: Optional[int] = None) -> bool:
        if consensus is None:
            return False
        step = int(consensus.get("step", -1))
        if step <= self.state.applied_step:
            return False
        scale = float(consensus.get("spill_threshold_scale", 1.0))
        rk = self.rank if rank is None else int(rank)
        prs = consensus.get("per_rank_scale", {})
        if isinstance(prs, dict):
            scale *= float(prs.get(str(rk), 1.0))
        if anti_skew:
            scale = max(0.4, min(1.6, scale))
        cfg.spill_min_bytes = max(1 * 1024 * 1024, int(float(cfg.spill_min_bytes) * scale))
        pf_scale = float(consensus.get("prefetch_limit_scale", 1.0))
        cfg.prefetch_batch_limit = max(1, int(float(cfg.prefetch_batch_limit) * max(0.25, min(1.5, pf_scale))))
        gl = int(consensus.get("global_level", 0))
        if gl >= 2:
            cfg.pack_wait_for_commit = True
            cfg.prefetch_batch_limit = 1
        elif gl == 1:
            cfg.prefetch_batch_limit = max(1, min(cfg.prefetch_batch_limit, 2))

        self.state.applied_step = step
        self.state.last_consensus_ts = float(consensus.get("ts", time.time()))
        self.state.applied += 1
        return True
