import tempfile

from aimemory.config import AIMemoryConfig
from aimemory.distributed_coord import RankCoordinator


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_coord_") as td:
        c0 = RankCoordinator(td, rank=0, world_size=2, leader_rank=0)
        c1 = RankCoordinator(td, rank=1, world_size=2, leader_rank=0)

        c0.publish(10, {"spill_bytes": 1000, "spills": 10, "memory_headroom_pct": 5.0, "safe_mode": False})
        c1.publish(10, {"spill_bytes": 6000, "spills": 60, "memory_headroom_pct": 4.0, "safe_mode": False})
        cons = c0.leader_aggregate(10)
        assert cons is not None
        assert float(cons["rank_skew_pct"]) > 0.0

        cfg = AIMemoryConfig(spill_min_bytes=128 * 1024 * 1024, prefetch_batch_limit=8)
        applied = c1.apply(cfg, cons, anti_skew=True)
        assert applied is True
        assert int(cfg.spill_min_bytes) > 0
        assert int(cfg.prefetch_batch_limit) >= 1
    print("DISTRIBUTED_COORDINATION_OK")


if __name__ == "__main__":
    main()
