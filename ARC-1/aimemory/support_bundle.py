import os
import io
import zipfile
import subprocess
import sys
import glob
from datetime import datetime

def _safe_run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except Exception as e:
        return f"[FAILED] {cmd}: {e}\n"

def build_support_bundle(pool_dir: str, out_zip: str, rank: int = 0, namespace: str = "default"):
    os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.txt", f"created={datetime.utcnow().isoformat()}Z\npool_dir={pool_dir}\nrank={rank}\n")
        py = sys.executable or "python3"
        z.writestr("doctor.txt", _safe_run([py, "-c", "from aimemory.doctor import run_doctor; run_doctor()"]))
        z.writestr("pip_freeze.txt", _safe_run([py, "-m", "pip", "freeze"]))
        z.writestr("torch_collect_env.txt", _safe_run([py, "-c", "import torch; import torch.utils.collect_env as c; print(c.get_pretty_env_info())"]))

        ns = str(namespace or "default").strip().replace("/", "_").replace("..", "_")
        if ns and ns not in ("default", "shared"):
            rank_dir = os.path.join(pool_dir, f"ns_{ns}", f"rank_{rank}")
        else:
            rank_dir = os.path.join(pool_dir, f"rank_{rank}")
        for fn in ["aimemory.log.jsonl", "metadata.db", "agent_metrics.json", "aimemory.audit.jsonl"]:
            p = os.path.join(rank_dir, fn)
            if os.path.exists(p):
                z.write(p, arcname=f"rank_{rank}/{fn}")
        for p in sorted(glob.glob(os.path.join(rank_dir, "pool_*.bin")))[-2:]:
            z.write(p, arcname=f"rank_{rank}/{os.path.basename(p)}")
        for p in sorted(glob.glob(os.path.join(rank_dir, "pool_*.manifest.json")))[-8:]:
            z.write(p, arcname=f"rank_{rank}/{os.path.basename(p)}")

        telem = "./aimemory_telemetry/telemetry.jsonl"
        if os.path.exists(telem):
            z.write(telem, arcname="telemetry.jsonl")

        if os.path.exists(rank_dir):
            pools = [os.path.basename(p) for p in sorted([p for p in os.listdir(rank_dir) if p.startswith("pool_") and p.endswith(".bin")])]
            z.writestr(f"rank_{rank}/pool_files.txt", "\n".join(pools) + "\n")

        cp_dir = os.path.join(pool_dir, "control_plane")
        if os.path.exists(cp_dir):
            for p in sorted(glob.glob(os.path.join(cp_dir, "policies", "*.json"))):
                z.write(p, arcname=f"control_plane/policies/{os.path.basename(p)}")
            for p in sorted(glob.glob(os.path.join(cp_dir, "history", "*.jsonl"))):
                z.write(p, arcname=f"control_plane/history/{os.path.basename(p)}")
            for p in sorted(glob.glob(os.path.join(cp_dir, "events", "*.jsonl"))):
                z.write(p, arcname=f"control_plane/events/{os.path.basename(p)}")

        roi_dir = os.path.join(pool_dir, "roi")
        if os.path.exists(roi_dir):
            for p in sorted(glob.glob(os.path.join(roi_dir, "*.json"))):
                z.write(p, arcname=f"roi/{os.path.basename(p)}")

    with open(out_zip, "wb") as f:
        f.write(buf.getvalue())
    return out_zip
