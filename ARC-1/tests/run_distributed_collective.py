from aimemory.distributed_collective import _build_consensus


def main():
    rows = [
        {"rank": 0, "headroom_pct": 10.0, "spills": 10, "step_p99_ms": 20.0, "io_write_p99_ms": 5.0, "io_read_p99_ms": 6.0},
        {"rank": 1, "headroom_pct": 8.0, "spills": 20, "step_p99_ms": 25.0, "io_write_p99_ms": 7.0, "io_read_p99_ms": 8.0},
    ]
    c = _build_consensus(rows, step_id=1, world_size=2, min_quorum_ratio=0.5)
    assert "global_level" in c and "per_rank_scale" in c
    assert float(c["quorum_ratio"]) >= 1.0
    print("DISTRIBUTED_COLLECTIVE_OK")


if __name__ == "__main__":
    main()
