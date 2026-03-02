import os
import tempfile

from aimemory.security_ops import generate_threat_model, security_audit, rotate_key_with_audit, write_json_report


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_security_") as td:
        tm = generate_threat_model()
        assert "assets" in tm and len(tm["assets"]) > 0
        out = os.path.join(td, "tm.json")
        write_json_report(tm, out_path=out)
        assert os.path.exists(out)

        key = os.path.join(td, "k.bin")
        with open(key, "wb") as f:
            f.write(os.urandom(32))
        os.chmod(key, 0o600)
        rep = security_audit(pool_dir=td, key_path=key)
        assert "stage" in rep

        out_key = os.path.join(td, "k2.bin")
        r = rotate_key_with_audit(key_path=key, new_key_path=out_key, audit_path=os.path.join(td, "audit.log"))
        assert r["ok"] is True and os.path.exists(out_key)
    print("SECURITY_OPS_OK")


if __name__ == "__main__":
    main()
