import os
import platform
import subprocess
import sys
import shutil
from .backend import choose_backend, detect_backend_capabilities, benchmark_path
from .topology import detect_topology

def _cmd_ok(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def run_doctor(pool_dir="/mnt/nvme_pool"):
    print("[ARC-1 Doctor]")
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

    caps = detect_backend_capabilities(pool_dir)
    backend = choose_backend("AUTO", pool_dir)
    print("  - chosen backend:", backend)
    print("  - fs type:", caps.get("fs_type"))
    print("  - remote fs:", caps.get("remote"))
    print("  - tmpfs:", caps.get("tmpfs"))
    if backend in ("NVME_FILE", "TMPFS", "NETWORK_FILE"):
        pr = benchmark_path(pool_dir, probe_mb=64, probe_seconds=1.0)
        print("  - backend probe write MB/s:", f"{float(pr.get('write_mb_s', 0.0)):.1f}")
        print("  - backend probe write ms:", f"{float(pr.get('write_ms', 0.0)):.2f}")
        if float(pr.get("write_ms", 0.0)) > 1000.0:
            print("  - WARNING: storage write latency high (tail risk for p99)")

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
    try:
        du = shutil.disk_usage(pool_dir)
        print("  - disk free GB:", f"{(du.free / 1e9):.2f}")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.empty((1024,), dtype=torch.uint8, pin_memory=True)
            print("  - pinned alloc:", "OK", int(t.numel()))
    except Exception as e:
        print("  - pinned alloc:", "FAIL", e)
    topo = detect_topology(rank=0)
    print("  - numa node:", topo.get("numa_node"))
    print("  - local rank env:", topo.get("local_rank_env"))

    if backend == "NOOP":
        print("  - NOTE: running in NOOP mode (CPU-only or forced). Hooks are pass-through.")
    return 0
