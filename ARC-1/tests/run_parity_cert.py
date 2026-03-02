import json
import tempfile

from aimemory.parity_cert import certify_training_outcome_parity, certify_from_files


def main():
    base = {
        "loss_curve": [1.0, 0.8, 0.6, 0.5],
        "grad_norm_curve": [10.0, 9.0, 8.0, 7.0],
        "reproducibility_mode": True,
        "reproducibility_checksum": 123.456,
    }
    cand = {
        "loss_curve": [1.0, 0.81, 0.61, 0.49],
        "grad_norm_curve": [10.1, 9.1, 7.9, 7.1],
        "reproducibility_mode": True,
        "reproducibility_checksum": 123.456,
    }
    r = certify_training_outcome_parity(base, cand)
    assert r.ok is True
    with tempfile.TemporaryDirectory(prefix="aimemory_parity_") as td:
        b = f"{td}/b.json"
        c = f"{td}/c.json"
        with open(b, "w") as f:
            json.dump(base, f)
        with open(c, "w") as f:
            json.dump(cand, f)
        rr = certify_from_files(b, c)
        assert rr["ok"] is True
    print("PARITY_CERT_OK")


if __name__ == "__main__":
    main()
