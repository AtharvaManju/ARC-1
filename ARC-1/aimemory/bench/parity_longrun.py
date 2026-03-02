import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController
from aimemory.parity_cert import certify_training_outcome_parity


class _Tiny(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def _run(steps: int, dim: int, device: str, dtype: torch.dtype, pool_dir: str, with_arc1: bool) -> Dict[str, List[float]]:
    torch.manual_seed(1234)
    model = _Tiny(dim).to(device=device, dtype=dtype).train()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    losses: List[float] = []
    grads: List[float] = []
    ctrl = None
    if with_arc1:
        cfg = AIMemoryConfig(
            pool_dir=pool_dir,
            backend=("AUTO" if device == "cuda" else "NOOP"),
            spill_min_bytes=1 * 1024 * 1024,
            pack_wait_for_commit=True,
            sync_each_step=(device == "cuda"),
        )
        ctrl = AIMemoryController(cfg)
        with ctrl.step(profiling_warmup=True):
            x = torch.randn(8, dim, device=device, dtype=dtype)
            y = model(x)
            y.float().pow(2).mean().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        ctrl.finalize_pcc_profile()

    try:
        for _ in range(max(4, int(steps))):
            x = torch.randn(8, dim, device=device, dtype=dtype)
            if ctrl is None:
                y = model(x)
                loss = y.float().pow(2).mean()
                loss.backward()
            else:
                with ctrl.step():
                    y = model(x)
                    loss = y.float().pow(2).mean()
                    loss.backward()
            g = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g += float(p.grad.float().norm().item())
            opt.step()
            opt.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            losses.append(float(loss.item()))
            grads.append(float(g))
    finally:
        if ctrl is not None:
            ctrl.shutdown()
    return {"loss_curve": losses, "grad_norm_curve": grads}


def run_parity_longrun(
    pool_dir: str,
    out_path: str,
    steps: int = 64,
    dim: int = 1024,
    dtype_s: str = "float16",
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(dtype_s).lower().strip() in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    elif str(dtype_s).lower().strip() in ("fp32", "float32", "float"):
        dtype = torch.float32
    else:
        dtype = torch.float16
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    base = _run(steps=steps, dim=dim, device=device, dtype=dtype, pool_dir=pool_dir, with_arc1=False)
    cand = _run(steps=steps, dim=dim, device=device, dtype=dtype, pool_dir=pool_dir, with_arc1=True)

    cert = certify_training_outcome_parity(
        {
            "loss_curve": base["loss_curve"],
            "grad_norm_curve": base["grad_norm_curve"],
            "reproducibility_mode": True,
            "reproducibility_checksum": float(sum(base["loss_curve"]) + sum(base["grad_norm_curve"])),
        },
        {
            "loss_curve": cand["loss_curve"],
            "grad_norm_curve": cand["grad_norm_curve"],
            "reproducibility_mode": True,
            "reproducibility_checksum": float(sum(cand["loss_curve"]) + sum(cand["grad_norm_curve"])),
        },
        loss_delta_pct_max=5.0,
        grad_stat_delta_pct_max=8.0,
        reproducibility_tolerance=1e-5,
    )
    report = {
        "device": device,
        "dtype": str(dtype),
        "steps": int(steps),
        "dim": int(dim),
        "parity": {
            "ok": bool(cert.ok),
            "loss_curve_max_delta_pct": float(cert.loss_curve_max_delta_pct),
            "grad_mean_delta_pct": float(cert.grad_mean_delta_pct),
            "grad_std_delta_pct": float(cert.grad_std_delta_pct),
            "reproducibility_pass": bool(cert.reproducibility_pass),
            "reasons": list(cert.reasons),
        },
        "baseline_tail_loss": float(base["loss_curve"][-1] if base["loss_curve"] else 0.0),
        "candidate_tail_loss": float(cand["loss_curve"][-1] if cand["loss_curve"] else 0.0),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
