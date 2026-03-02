import json
import os
import tempfile
import torch

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController
from aimemory.control_plane import PolicyStore, build_fleet_report
from aimemory.consistency import run_consistency_check
from aimemory.storage import SARCStorage


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
        st.close()

        cons = run_consistency_check(pool_dir=td, rank=0, repair=False)
        fleet = build_fleet_report(td)

        report = {
            "controller_snapshot": snap,
            "consistency": cons,
            "fleet": fleet,
            "policy_applied": {"spill_min_bytes": cfg.spill_min_bytes, "pcc_lookahead": cfg.pcc_lookahead},
            "manifest_replay_ok": bool(cons.get("manifest", {}).get("ok", True)),
        }
        print(json.dumps(report, indent=2))
    print("PRODUCT_SIMULATION_OK")


if __name__ == "__main__":
    main()
