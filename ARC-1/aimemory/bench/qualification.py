import json
import os
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from .bench import run_bench
from .headroom_gate import run_headroom_gate
from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController


class _Tiny(nn.Module):
    def __init__(self, d: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def _convergence_delta(pool_dir: str, d: int = 2048, steps: int = 3, dtype: torch.dtype = torch.float16) -> float:
    torch.manual_seed(1234)
    xb = [torch.randn(8, d, device="cuda", dtype=dtype) for _ in range(steps)]

    def _run(use_ai: bool) -> float:
        model = _Tiny(d).to(device="cuda", dtype=dtype).train()
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        ctrl = None
        if use_ai:
            cfg = AIMemoryConfig(
                pool_dir=pool_dir,
                spill_min_bytes=2 * 1024 * 1024,
                pack_wait_for_commit=True,
                sync_each_step=True,
            )
            ctrl = AIMemoryController(cfg)
            with ctrl.step(profiling_warmup=True):
                y = model(xb[0])
                (y.float().pow(2).mean()).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            ctrl.finalize_pcc_profile()

        last = 0.0
        for i in range(1, steps):
            if ctrl:
                with ctrl.step():
                    y = model(xb[i])
                    loss = y.float().pow(2).mean()
                    loss.backward()
                    last = float(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                y = model(xb[i])
                loss = y.float().pow(2).mean()
                loss.backward()
                last = float(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)
        if ctrl:
            ctrl.shutdown()
        return last

    base = _run(False)
    ai = _run(True)
    return abs(ai - base) / max(1e-8, abs(base))


def _avg_step_ms_baseline(d: int = 2048, steps: int = 8, dtype: torch.dtype = torch.float16) -> float:
    model = _Tiny(d).to(device="cuda", dtype=dtype).train()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    vals = []
    for _ in range(steps):
        x = torch.randn(8, d, device="cuda", dtype=dtype)
        t0 = time.time()
        y = model(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        vals.append((time.time() - t0) * 1000.0)
    return sum(vals) / max(1, len(vals))


def run_qualification(
    pool_dir: str,
    out_path: str,
    threshold_multiplier: float = 3.0,
    overhead_sla_pct: float = 15.0,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for qualification")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    t0 = time.time()

    bench_dir = os.path.join(os.path.dirname(out_path) or ".", "qualification_bench")
    bench = run_bench(pool_dir=pool_dir, steps=10, warmup=4, out_dir=bench_dir)
    gate = run_headroom_gate(
        pool_dir=pool_dir,
        out_path=os.path.join(os.path.dirname(out_path) or ".", "qualification_headroom_gate.json"),
        dim=8192,
        steps=2,
        warmup=1,
        threshold_multiplier=threshold_multiplier,
        dtype_s="float16",
        max_probe=8192,
    )
    conv_delta = _convergence_delta(pool_dir=pool_dir)
    pass_headroom = bool(gate.get("passed", False))
    step_ms = float(bench.get("step_ms_avg", 0.0))
    baseline_ms = _avg_step_ms_baseline()
    overhead_pct = (100.0 * (step_ms / baseline_ms - 1.0)) if baseline_ms > 0 else 0.0
    pass_overhead = overhead_pct <= float(overhead_sla_pct)
    pass_conv = conv_delta <= 0.05

    report = {
        "headroom_gate": gate,
        "bench": bench,
        "baseline_step_ms_avg": baseline_ms,
        "aimemory_step_ms_avg": step_ms,
        "overhead_pct": overhead_pct,
        "convergence_delta_ratio": conv_delta,
        "criteria": {
            "threshold_multiplier": float(threshold_multiplier),
            "overhead_sla_pct": float(overhead_sla_pct),
            "convergence_delta_ratio_max": 0.05,
        },
        "passed": bool(pass_headroom and pass_overhead and pass_conv),
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("[qualification] wrote", out_path)
    print("[qualification]", "PASS" if report["passed"] else "FAIL")
    return report
