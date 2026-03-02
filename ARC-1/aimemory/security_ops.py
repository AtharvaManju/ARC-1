import json
import os
import stat
import time
from typing import Any, Dict

from .security import rotate_key, audit_log


def generate_threat_model(product: str = "ARC-1") -> Dict[str, Any]:
    return {
        "product": str(product),
        "generated_ts": float(time.time()),
        "assets": [
            "spill payloads",
            "metadata manifest",
            "encryption keys",
            "policy model data",
            "telemetry and logs",
        ],
        "threats": [
            "data corruption or torn writes",
            "unauthorized spill data access",
            "key compromise",
            "policy tampering",
            "cross-tenant artifact leakage",
        ],
        "controls": [
            "atomic metadata transitions with recovery",
            "checksums and optional stronger hashes",
            "encryption at rest with key-uri abstraction",
            "policy signatures and versioned rollbacks",
            "tenant namespace isolation and retention cleanup",
        ],
        "residual_risks": [
            "misconfigured filesystem permissions",
            "unvalidated external key management",
            "runtime drift outside validated hardware matrix",
        ],
    }


def security_audit(pool_dir: str, key_path: str = "", audit_path: str = "") -> Dict[str, Any]:
    pool_dir = os.path.abspath(pool_dir or ".")
    findings = []
    if os.path.exists(pool_dir):
        st = os.stat(pool_dir)
        mode = stat.S_IMODE(st.st_mode)
        if mode & 0o002:
            findings.append({"severity": "high", "issue": "pool_dir_world_writable", "mode": oct(mode)})
    else:
        findings.append({"severity": "medium", "issue": "pool_dir_missing"})

    if key_path:
        kp = os.path.abspath(key_path)
        if not os.path.exists(kp):
            findings.append({"severity": "high", "issue": "key_missing", "path": kp})
        else:
            mode = stat.S_IMODE(os.stat(kp).st_mode)
            if mode & 0o077:
                findings.append({"severity": "high", "issue": "key_permissions_too_open", "mode": oct(mode), "path": kp})

    if audit_path:
        ap = os.path.abspath(audit_path)
        if not os.path.exists(ap):
            findings.append({"severity": "low", "issue": "audit_log_missing", "path": ap})

    sev = {"high": 3, "medium": 2, "low": 1}
    max_sev = 0
    for f in findings:
        max_sev = max(max_sev, sev.get(str(f.get("severity", "")).lower(), 0))
    stage = "PASS" if max_sev == 0 else ("WARN" if max_sev <= 1 else "FAIL")
    return {
        "pool_dir": pool_dir,
        "ts": float(time.time()),
        "stage": stage,
        "findings": findings,
    }


def rotate_key_with_audit(key_path: str, new_key_path: str = "", audit_path: str = "") -> Dict[str, Any]:
    kp = os.path.abspath(key_path)
    nkp = os.path.abspath(new_key_path) if new_key_path else kp
    rotate_key(kp, new_key_path=nkp)
    if audit_path:
        audit_log(audit_path, "security_rotate_key", key_path=nkp)
    return {"ok": True, "key_path": nkp}


def write_json_report(payload: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
