import json
import os
import hashlib
import socket
from dataclasses import dataclass

@dataclass
class LicenseStatus:
    ok: bool
    reason: str = ""

def _machine_fingerprint() -> str:
    host = socket.gethostname()
    return hashlib.sha256(host.encode()).hexdigest()

def verify_license(license_path: str) -> LicenseStatus:
    if not os.path.exists(license_path):
        return LicenseStatus(False, "license file missing")
    try:
        with open(license_path, "r") as f:
            lic = json.load(f)
        expected = lic.get("fingerprint", "")
        fp = _machine_fingerprint()
        if expected != fp:
            return LicenseStatus(False, "fingerprint mismatch")
        if not lic.get("active", False):
            return LicenseStatus(False, "license inactive")
        return LicenseStatus(True, "ok")
    except Exception as e:
        return LicenseStatus(False, f"license parse error: {e}")
