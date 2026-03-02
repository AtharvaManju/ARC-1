import os
import time
import json
import psutil
import subprocess
from .backend import choose_backend, detect_backend_capabilities, benchmark_path, recommend_io_tuning

def _run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()

def throughput_test(pool_dir="/mnt/nvme_pool", size_gb=1):
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, "throughput_test.bin")
    bs = "1M"
    count = size_gb * 1024
    t0 = time.time()
    try:
        _run(["bash", "-lc", f"dd if=/dev/zero of={path} bs={bs} count={count} oflag=direct status=none"])
        dt = time.time() - t0
        gbps = (size_gb / max(1e-9, dt))
    except Exception:
        gbps = 0.0
    try:
        os.remove(path)
    except Exception:
        pass
    return gbps

def run_installer(pool_dir="/mnt/nvme_pool", out_path="./aimemory_install_report.json", apply_fixes=False):
    caps = detect_backend_capabilities(pool_dir)
    backend = choose_backend("AUTO", pool_dir)
    probe = benchmark_path(pool_dir, probe_mb=128, probe_seconds=2.0) if backend in ("NVME_FILE", "TMPFS", "NETWORK_FILE") else {"ok": True}
    tune = recommend_io_tuning(caps, probe)
    nvme = throughput_test(pool_dir) if backend == "NVME_FILE" else 0.0
    mem_gb = psutil.virtual_memory().total / 1e9

    rec = {"backend": backend}
    if backend == "NVME_FILE":
        if nvme >= 2.0:
            rec.update({"spill_min_bytes": 64 * 1024**2, "io_workers": 2})
        else:
            rec.update({"backend": "RAM", "spill_min_bytes": 256 * 1024**2, "io_workers": 1})
    elif backend == "RAM":
        rec.update({"spill_min_bytes": 256 * 1024**2, "io_workers": 1})
    else:
        rec.update({"spill_min_bytes": 10**18, "io_workers": 0})

    rec.update({
        "max_queue": int(tune.get("max_queue", 256)),
        "io_workers": int(tune.get("io_workers", rec.get("io_workers", 1))),
        "native_chunk_bytes": int(tune.get("native_chunk_bytes", 16 * 1024 * 1024)),
    })

    report = {
        "pool_dir": pool_dir,
        "chosen_backend": backend,
        "capabilities": caps,
        "probe": probe,
        "startup_tuning": tune,
        "nvme_write_gbps": nvme,
        "system_ram_gb": mem_gb,
        "recommended": rec,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("[Installer] Written:", out_path)
    print("[Installer] Recommended:", rec)
    return report
