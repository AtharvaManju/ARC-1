import json
import os
import tempfile

from aimemory.claims import build_claims_evidence, write_claims_evidence


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_claims_") as td:
        q = os.path.join(td, "q.json")
        with open(q, "w") as f:
            json.dump({"headroom_gate": {"reduction_ratio": 0.2}, "training_outcome_parity": {"ok": True}, "pressure_profile": {"p99_overhead_pct": 8.0}}, f)
        rep = build_claims_evidence(qualification_path=q, require_cuda_evidence=False)
        assert "status" in rep and "claims" in rep
        out = os.path.join(td, "claims.json")
        rep2 = write_claims_evidence(out_path=out, qualification_path=q, require_cuda_evidence=False)
        assert os.path.exists(out)
        assert rep2["status"] in ("READY", "INCOMPLETE", "PENDING_CUDA_EVIDENCE")
    print("CLAIMS_EVIDENCE_OK")


if __name__ == "__main__":
    main()
