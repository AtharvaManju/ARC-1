import json
import os
import tempfile
import torch

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController
from aimemory.control_plane import PolicyStore, build_fleet_report
from aimemory.consistency import run_consistency_check
from aimemory.storage import SARCStorage
from aimemory.static_plan import StaticPlanCompiler, model_fingerprint
from aimemory.distributed_coord import RankCoordinator


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_sim_") as td:
        store = PolicyStore(os.path.join(td, "control_plane"))
        store.push("pilot", {"spill_min_bytes": 2 * 1024 * 1024, "pcc_lookahead": 3}, reason="simulation")

        cfg = AIMemoryConfig(
            pool_dir=td,
            backend="NOOP",
            policy_name="pilot",
            control_plane_dir=os.path.join(td, "control_plane"),
            enable_jsonl_logs=False,
        )
        ctrl = AIMemoryController(cfg)
        with ctrl.step():
            pass
        snap = ctrl.metrics_snapshot()
        ctrl.shutdown()

        # Storage simulation (state-machine + compression path) on CPU.
        st = SARCStorage(
            pool_dir=td,
            rank=0,
            backend="NVME_FILE",
            compression_codec="zlib",
            compression_min_bytes=1,
            encrypt_at_rest=False,
        )
        blk = st.acquire_pinned(4096)
        blk.u8[:4096].copy_(torch.arange(4096, dtype=torch.uint8), non_blocking=False)
        st.put_host_bytes(
            key="sim_key",
            plain_blk=blk,
            nbytes=4096,
            dtype_s="uint8",
            shape=(4096,),
            step_id=1,
        )
        sp = StaticPlanCompiler(os.path.join(td, "static_plans"))
        mfp = model_fingerprint("sim", (3,), "float16")
        plan = sp.compile_from_restore_order(mfp, [1, 2, 3], lookahead=2)
        sp.save(plan)

        coord = RankCoordinator(root_dir=os.path.join(td, "coordination"), rank=0, world_size=1, leader_rank=0)
        coord.publish(1, {"spill_bytes": 1234, "spills": 2, "memory_headroom_pct": 10.0, "safe_mode": False})
        coord_cons = coord.leader_aggregate(1)
        st.close()

        cons = run_consistency_check(pool_dir=td, rank=0, repair=False)
        fleet = build_fleet_report(td)

        report = {
            "controller_snapshot": snap,
            "consistency": cons,
            "fleet": fleet,
            "policy_applied": {"spill_min_bytes": cfg.spill_min_bytes, "pcc_lookahead": cfg.pcc_lookahead},
            "manifest_replay_ok": bool(cons.get("manifest", {}).get("ok", True)),
            "static_plan_key": plan.plan_key,
            "coordination": coord_cons,
        }
        print(json.dumps(report, indent=2))
    print("PRODUCT_SIMULATION_OK")


if __name__ == "__main__":
    main()
