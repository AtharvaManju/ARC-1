import os
import time
import json
import numpy as np
import torch

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController
from aimemory.util import ensure_dir

def _step_work(x, iters=2):
    y = x
    for _ in range(iters):
        y = (y @ y).relu()
    loss = y.mean()
    loss.backward()

def run_bench(pool_dir="/mnt/nvme_pool", steps=30, warmup=10, out_dir="./aimemory_bench"):
    ensure_dir(out_dir)
    cfg = AIMemoryConfig(
        pool_dir=pool_dir,
        spill_min_bytes=8 * 1024 * 1024,
        io_workers=2,
        pinned_pool_bytes=512 * 1024**2,
        sync_each_step=True,  # bench wants full sync realism
    )
    ctrl = AIMemoryController(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for bench")

    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16, requires_grad=True)
    with ctrl.step(profiling_warmup=True):
        _step_work(x)
    ctrl.finalize_pcc_profile()

    times = []
    for i in range(warmup + steps):
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16, requires_grad=True)
        t0 = time.time()
        with ctrl.step():
            _step_work(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)

    report = {
        "steps": steps,
        "warmup": warmup,
        "step_ms_avg": sum(times) / max(1, len(times)),
        "step_ms_p95": float(np.percentile(np.array(times, dtype=np.float64), 95)) if times else 0.0,
        "step_ms_p99": float(np.percentile(np.array(times, dtype=np.float64), 99)) if times else 0.0,
        "step_ms_p999": float(np.percentile(np.array(times, dtype=np.float64), 99.9)) if times else 0.0,
        "metrics": ctrl.metrics_snapshot(),
        "config": cfg.__dict__,
    }
    out_path = os.path.join(out_dir, "bench_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("[bench] wrote", out_path)
    ctrl.shutdown()
    return report
