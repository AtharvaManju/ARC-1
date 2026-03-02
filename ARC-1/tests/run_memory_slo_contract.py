import json
import tempfile

from aimemory.memory_slo import MemorySLOContract, MemorySLOEnforcer


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_slo_") as td:
        c = MemorySLOContract(
            never_oom=True,
            max_hbm_bytes=10_000,
            p99_overhead_ms=50.0,
            p99_overhead_pct=200.0,
            policy="balanced",
        )
        e = MemorySLOEnforcer(contract=c, out_dir=td, rank=0)
        snap = {
            "memory_total_bytes": 100_000,
            "memory_free_bytes": 95_000,
            "oom_degrade_count": 0,
            "safe_mode": False,
            "step_p99_ms": 20.0,
            "baseline_step_ms_ema": 10.0,
        }
        rep = e.emit_proof(snap)
        assert rep["ok"] is True
        bad = dict(snap)
        bad["oom_degrade_count"] = 1
        rep2 = e.emit_proof(bad)
        assert rep2["ok"] is False
        with open(e.summary_json, "r") as f:
            _ = json.load(f)
    print("MEMORY_SLO_CONTRACT_OK")


if __name__ == "__main__":
    main()
