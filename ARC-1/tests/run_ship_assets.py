import json
import os
import tempfile

from aimemory.ship_assets import build_ga_readiness, build_commercial_pack


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_ship_") as td:
        qual = os.path.join(td, "qualification.json")
        with open(qual, "w") as f:
            json.dump(
                {
                    "passed": True,
                    "gates": {"correctness": {"pass": True}, "p99_overhead": {"pass": True}},
                    "bench": {"step_ms_p95": 10.0, "step_ms_p99": 15.0, "metrics": {"prefetch_hit_rate": 90.0, "safe_mode": False, "roi": {"headroom_gain_pct": 22.0, "ooms_prevented": 3}}},
                    "headroom_gate": {"reduction_ratio": 0.25},
                },
                f,
            )
        ga = build_ga_readiness(pool_dir=td, qualification_path=qual)
        assert "stage" in ga and "checks" in ga
        out = build_commercial_pack(pool_dir=td, qualification_path=qual, out_dir=os.path.join(td, "pack"), customer="demo")
        assert os.path.exists(out["json"])
        assert os.path.exists(out["markdown"])
    print("SHIP_ASSETS_OK")


if __name__ == "__main__":
    main()
