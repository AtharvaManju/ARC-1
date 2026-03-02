from aimemory.config import AIMemoryConfig
from aimemory.metrics import Metrics
from aimemory.governor import MemoryGovernor


def main():
    cfg = AIMemoryConfig(
        spill_min_bytes=128 * 1024 * 1024,
        pcc_lookahead=8,
        prefetch_batch_limit=8,
        pack_wait_for_commit=False,
    )
    m = Metrics()
    g = MemoryGovernor(cfg, m)

    assert cfg.spill_min_bytes == 128 * 1024 * 1024
    assert cfg.pcc_lookahead == 8
    assert cfg.prefetch_batch_limit == 8

    g._apply_level(1)
    assert cfg.spill_min_bytes <= 64 * 1024 * 1024
    assert cfg.pcc_lookahead <= 4
    assert cfg.prefetch_batch_limit <= 4
    assert m.governor_level == 1

    g._apply_level(2)
    assert cfg.spill_min_bytes <= 32 * 1024 * 1024
    assert cfg.pcc_lookahead == 1
    assert cfg.prefetch_batch_limit == 1
    assert cfg.pack_wait_for_commit is True
    assert m.governor_level == 2

    g.on_oom_signal(step_id=10)
    assert m.oom_degrade_count >= 1
    assert m.governor_level == 2

    g._apply_level(0)
    assert cfg.spill_min_bytes == 128 * 1024 * 1024
    assert cfg.pcc_lookahead == 8
    assert cfg.prefetch_batch_limit == 8
    assert cfg.pack_wait_for_commit is False
    assert m.governor_level == 0
    print("GOVERNOR_POLICY_OK")


if __name__ == "__main__":
    main()
