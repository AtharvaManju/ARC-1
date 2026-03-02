import os
import time
import tempfile
import subprocess
from typing import Dict, Any

import torch


def detect_distributed():
    rank = 0
    world = 1
    is_dist = False
    is_multinode = False
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            is_dist = True
            rank = dist.get_rank()
            world = dist.get_world_size()
            nnodes = int(os.environ.get("WORLD_SIZE", str(world))) // max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
            is_multinode = nnodes > 1
    except Exception:
        pass
    return is_dist, rank, world, is_multinode


def path_is_likely_remote(pool_dir: str) -> bool:
    p = os.path.abspath(pool_dir or "")
    bad_prefixes = ("/nfs", "/net", "/afs", "/mnt/nfs", "/Volumes/", "/mnt/lustre", "/mnt/ceph")
    if p.startswith(bad_prefixes):
        return True
    low = p.lower()
    for k in ("nfs", "lustre", "ceph", "smb", "cifs", "gluster", "afs"):
        if f"/{k}" in low:
            return True
    return False


def _fs_type(pool_dir: str) -> str:
    p = os.path.abspath(pool_dir or ".")
    try:
        out = subprocess.check_output(["stat", "-f", "%T", p], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            return out.lower()
    except Exception:
        pass
    try:
        out = subprocess.check_output(["bash", "-lc", f"stat -f -c %T {p}"], stderr=subprocess.DEVNULL).decode().strip()
        if out:
            return out.lower()
    except Exception:
        pass
    return "unknown"


def benchmark_path(pool_dir: str, probe_mb: int = 128, probe_seconds: float = 2.0) -> Dict[str, Any]:
    os.makedirs(pool_dir, exist_ok=True)
    nbytes = max(1, int(probe_mb)) * 1024 * 1024
    payload = b"\0" * min(1024 * 1024, nbytes)
    test_path = ""
    w_ms = 0.0
    r_ms = 0.0
    ok = True
    err = ""
    try:
        fd, test_path = tempfile.mkstemp(prefix=".aimemory_probe_", dir=pool_dir)
        with os.fdopen(fd, "wb", buffering=0) as f:
            left = nbytes
            t0 = time.time()
            while left > 0:
                chunk = payload[: min(len(payload), left)]
                f.write(chunk)
                left -= len(chunk)
                if (time.time() - t0) > max(0.1, float(probe_seconds)):
                    break
            f.flush()
            os.fsync(f.fileno())
            w_ms = (time.time() - t0) * 1000.0
        with open(test_path, "rb", buffering=0) as f:
            t1 = time.time()
            _ = f.read(min(nbytes, 4 * 1024 * 1024))
            r_ms = (time.time() - t1) * 1000.0
    except Exception as e:
        ok = False
        err = str(e)
    finally:
        if test_path:
            try:
                os.remove(test_path)
            except Exception:
                pass
    wrote_mb = float(max(1, probe_mb))
    write_mb_s = (wrote_mb / max(1e-6, w_ms / 1000.0)) if ok else 0.0
    return {
        "ok": bool(ok),
        "error": str(err),
        "probe_mb": int(probe_mb),
        "write_ms": float(w_ms),
        "read_ms": float(r_ms),
        "write_mb_s": float(write_mb_s),
    }


def detect_backend_capabilities(pool_dir: str) -> Dict[str, Any]:
    p = os.path.abspath(pool_dir or ".")
    fs_t = _fs_type(p)
    remote = path_is_likely_remote(p)
    tmpfs = ("tmpfs" in fs_t) or p.startswith("/dev/shm")
    writable = True
    err = ""
    try:
        os.makedirs(p, exist_ok=True)
        test = os.path.join(p, ".aimemory_write_test")
        with open(test, "wb") as f:
            f.write(b"ok")
        os.remove(test)
    except Exception as e:
        writable = False
        err = str(e)
    nvme_candidate = (not remote) and (not tmpfs) and writable
    return {
        "pool_dir": p,
        "fs_type": fs_t,
        "remote": bool(remote),
        "tmpfs": bool(tmpfs),
        "writable": bool(writable),
        "error": str(err),
        "nvme_candidate": bool(nvme_candidate),
    }


def choose_backend(
    cfg_backend: str,
    pool_dir: str,
    *,
    allow_tmpfs: bool = True,
    allow_network: bool = False,
    strict_local_pool: bool = True,
) -> str:
    if cfg_backend and cfg_backend != "AUTO":
        return str(cfg_backend)
    if not torch.cuda.is_available():
        return "NOOP"

    caps = detect_backend_capabilities(pool_dir)
    if not bool(caps.get("writable", False)):
        return "RAM"
    if bool(caps.get("remote", False)):
        if allow_network and (not strict_local_pool):
            return "NETWORK_FILE"
        return "RAM"
    if bool(caps.get("tmpfs", False)):
        if allow_tmpfs:
            return "TMPFS"
        return "RAM"
    if bool(caps.get("nvme_candidate", False)):
        return "NVME_FILE"
    return "RAM"


def recommend_io_tuning(caps: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    backend = "RAM"
    if bool(caps.get("remote", False)):
        backend = "NETWORK_FILE"
    elif bool(caps.get("tmpfs", False)):
        backend = "TMPFS"
    elif bool(caps.get("nvme_candidate", False)):
        backend = "NVME_FILE"
    bw = float(probe.get("write_mb_s", 0.0))
    if backend == "NETWORK_FILE":
        return {"backend": backend, "io_workers": 1, "max_queue": 64, "native_chunk_bytes": 4 * 1024 * 1024}
    if backend == "TMPFS":
        return {"backend": backend, "io_workers": 1, "max_queue": 256, "native_chunk_bytes": 16 * 1024 * 1024}
    if backend == "NVME_FILE":
        if bw >= 2500:
            return {"backend": backend, "io_workers": 4, "max_queue": 2048, "native_chunk_bytes": 64 * 1024 * 1024}
        if bw >= 1200:
            return {"backend": backend, "io_workers": 2, "max_queue": 1024, "native_chunk_bytes": 32 * 1024 * 1024}
        return {"backend": backend, "io_workers": 1, "max_queue": 256, "native_chunk_bytes": 16 * 1024 * 1024}
    return {"backend": "RAM", "io_workers": 1, "max_queue": 256, "native_chunk_bytes": 8 * 1024 * 1024}
