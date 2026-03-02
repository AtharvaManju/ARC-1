import json
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from aimemory.parity_cert import certify_training_outcome_parity


class _Tiny(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.GELU(),
            nn.Linear(d, d, bias=False),
        )

    def forward(self, x):
        return self.net(x)


def _dtype_from_name(name: str) -> torch.dtype:
    n = str(name).lower().strip()
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp32", "float32", "float"):
        return torch.float32
    return torch.float16


def _effective_dtype(device: str, dt: torch.dtype) -> torch.dtype:
    if device == "cpu" and dt in (torch.float16, torch.bfloat16):
        return torch.float32
    return dt


def _run_case(dim: int, dtype_s: str, steps: int, compiled: bool, device: str) -> Dict[str, Any]:
    dt = _effective_dtype(device, _dtype_from_name(dtype_s))
    torch.manual_seed(1234)
    model = _Tiny(dim).to(device=device, dtype=dt).train()
    if compiled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as e:
            return {"ok": False, "error": f"compile_not_available:{type(e).__name__}"}
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    losses: List[float] = []
    grads: List[float] = []
    step_ms: List[float] = []
    for _ in range(max(2, int(steps))):
        x = torch.randn(8, dim, device=device, dtype=dt)
        t0 = time.time()
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
        step_ms.append((time.time() - t0) * 1000.0)
        losses.append(float(loss.item()))
        grads.append(float(g))
    return {
        "ok": True,
        "loss_curve": losses,
        "grad_norm_curve": grads,
        "step_ms_avg": float(sum(step_ms) / max(1, len(step_ms))),
        "step_ms_p99": float(sorted(step_ms)[min(len(step_ms) - 1, int(0.99 * (len(step_ms) - 1)))]),
        "dtype": str(dt),
        "device": str(device),
    }


def run_compile_matrix(
    out_path: str,
    dims: List[int],
    dtypes: List[str],
    steps: int = 4,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows: List[Dict[str, Any]] = []
    for d in dims:
        for dtype_s in dtypes:
            base = _run_case(dim=int(d), dtype_s=dtype_s, steps=steps, compiled=False, device=device)
            comp = _run_case(dim=int(d), dtype_s=dtype_s, steps=steps, compiled=True, device=device)
            if base.get("ok") and comp.get("ok"):
                cert = certify_training_outcome_parity(
                    {
                        "loss_curve": list(base["loss_curve"]),
                        "grad_norm_curve": list(base["grad_norm_curve"]),
                        "reproducibility_mode": False,
                    },
                    {
                        "loss_curve": list(comp["loss_curve"]),
                        "grad_norm_curve": list(comp["grad_norm_curve"]),
                        "reproducibility_mode": False,
                    },
                    loss_delta_pct_max=8.0,
                    grad_stat_delta_pct_max=12.0,
                )
                row = {
                    "dim": int(d),
                    "dtype": str(dtype_s),
                    "base": base,
                    "compiled": comp,
                    "parity_ok": bool(cert.ok),
                    "parity": {
                        "loss_curve_max_delta_pct": float(cert.loss_curve_max_delta_pct),
                        "grad_mean_delta_pct": float(cert.grad_mean_delta_pct),
                        "grad_std_delta_pct": float(cert.grad_std_delta_pct),
                        "reasons": list(cert.reasons),
                    },
                }
            else:
                row = {
                    "dim": int(d),
                    "dtype": str(dtype_s),
                    "base": base,
                    "compiled": comp,
                    "parity_ok": False,
                    "parity": {"reasons": ["compile_unavailable_or_failed"]},
                }
            rows.append(row)
    passed = all(bool(r.get("parity_ok", False)) for r in rows)
    report = {
        "device": device,
        "rows": rows,
        "passed": bool(passed),
        "compile_supported_rows": int(sum(1 for r in rows if bool((r.get("compiled", {}) or {}).get("ok", False)))),
        "total_rows": int(len(rows)),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
