import os
import platform
import subprocess
import sys
from .backend import choose_backend

def _cmd_ok(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def run_doctor(pool_dir="/mnt/nvme_pool"):
    print("[AIMemory Doctor]")
    print("  - platform:", platform.platform())
    print("  - python:", sys.version.split()[0])

    try:
        import torch
        print("  - torch:", torch.__version__)
        print("  - cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  - gpu:", torch.cuda.get_device_name(0))
            print("  - cuda version:", torch.version.cuda)
    except Exception as e:
        print("  - torch import: FAIL", e)
        return 1

    try:
        import aimemory_engine  # noqa: F401
        print("  - aimemory-engine:", "INSTALLED")
    except Exception:
        print("  - aimemory-engine:", "NOT INSTALLED (OK; will fallback)")

    print("  - nvidia-smi:", "OK" if _cmd_ok(["nvidia-smi"]) else "MISSING")

    backend = choose_backend("AUTO", pool_dir)
    print("  - chosen backend:", backend)

    if backend in ("NVME_FILE", "RAM"):
        try:
            os.makedirs(pool_dir, exist_ok=True)
            test = os.path.join(pool_dir, "aimemory_perm_test.tmp")
            with open(test, "wb") as f:
                f.write(b"ok")
            os.remove(test)
            print("  - pool dir write:", "OK")
        except Exception as e:
            print("  - pool dir write:", "FAIL", e)

    if backend == "NOOP":
        print("  - NOTE: running in NOOP mode (CPU-only or forced). Hooks are pass-through.")
    return 0
