import json
import os
import tempfile

from aimemory.policy_model import MemoryPolicyModel


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_polm_") as td:
        pm = MemoryPolicyModel(td)
        feat = {
            "model_fingerprint": "m",
            "world_size": 1,
            "seq_len": 2048,
            "batch_size": 8,
            "policy": "balanced",
            "hbm_bytes": 80_000,
            "nvme_write_mb_s": 2000.0,
            "nvme_read_mb_s": 2000.0,
        }
        pol = {"spill_min_bytes": 2_097_152, "pcc_lookahead": 4, "io_workers": 2, "max_queue": 1024, "predicted_uplift_pct": 25.0}
        pm.add_sample(feat, pol, score=1.0)
        pred = pm.predict(feat)
        assert pred is not None
        os.environ["AIMEMORY_MODEL_KEY"] = "33" * 32
        pack = f"{td}/pack.json"
        pm.export_signed_pack(pack, key_uri="env://AIMEMORY_MODEL_KEY")
        pm2 = MemoryPolicyModel(f"{td}/other")
        rep = pm2.import_signed_pack(pack, key_uri="env://AIMEMORY_MODEL_KEY", require_signature=True)
        assert rep["ok"] is True
        with open(pack, "r") as f:
            _ = json.load(f)
    print("POLICY_MODEL_OK")


if __name__ == "__main__":
    main()
