import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController


class _Toy(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.l1 = nn.Linear(d, d, bias=False)
        self.l2 = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.l2(self.l1(x).relu())


@dataclass
class GateResult:
    baseline_max_batch: int
    aimemory_max_batch: int
    effective_headroom_multiplier: float
    threshold_multiplier: float
    passed: bool
    elapsed_s: float
    dtype: str
    dim: int
    steps: int
    warmup: int


def _oom(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda error: out of memory" in s


def _train_step(model, opt, batch: int, dim: int, dtype: torch.dtype):
    x = torch.randn(batch, dim, device="cuda", dtype=dtype, requires_grad=False)
    y = model(x)
    loss = (y.float().pow(2).mean())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def _can_run(batch: int, dim: int, steps: int, warmup: int, dtype: torch.dtype, use_aimemory: bool, pool_dir: str):
    model = _Toy(dim).cuda().train()
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    ctrl = None
    try:
        if use_aimemory:
            cfg = AIMemoryConfig(
                pool_dir=pool_dir,
                spill_min_bytes=4 * 1024 * 1024,
                io_workers=2,
                pinned_pool_bytes=512 * 1024**2,
                pcc_lookahead=4,
                queue_put_timeout_s=0.01,
                spill_queue_overflow_policy="SYNC_SPILL",
                sync_each_step=True,
            )
            ctrl = AIMemoryController(cfg)
            with ctrl.step(profiling_warmup=True):
                _train_step(model, opt, batch, dim, dtype)
            ctrl.finalize_pcc_profile()

        for _ in range(warmup + steps):
            if ctrl is None:
                _train_step(model, opt, batch, dim, dtype)
                torch.cuda.synchronize()
            else:
                with ctrl.step():
                    _train_step(model, opt, batch, dim, dtype)
        return True, (ctrl.metrics_snapshot() if ctrl else {})
    except RuntimeError as e:
        if _oom(e):
            torch.cuda.empty_cache()
            return False, {}
        raise
    finally:
        if ctrl is not None:
            ctrl.shutdown()
        del model
        torch.cuda.empty_cache()


def _search_max_batch(dim: int, steps: int, warmup: int, dtype: torch.dtype, use_aimemory: bool, pool_dir: str, max_probe: int):
    lo, hi = 1, 1
    while hi <= max_probe:
        ok, _ = _can_run(hi, dim, steps, warmup, dtype, use_aimemory, pool_dir)
        if not ok:
            break
        lo = hi
        hi = min(max_probe, hi * 2)
        if hi == lo:
            break

    if hi == lo:
        return lo

    left, right = lo, hi
    while left + 1 < right:
        mid = (left + right) // 2
        ok, _ = _can_run(mid, dim, steps, warmup, dtype, use_aimemory, pool_dir)
        if ok:
            left = mid
        else:
            right = mid
    return left


def run_headroom_gate(
    pool_dir: str,
    out_path: str,
    dim: int = 8192,
    steps: int = 2,
    warmup: int = 1,
    threshold_multiplier: float = 3.0,
    dtype_s: str = "float16",
    max_probe: int = 8192,
):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for headroom gate")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_s]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    t0 = time.time()
    baseline = _search_max_batch(dim, steps, warmup, dtype, False, pool_dir, max_probe=max_probe)
    aimem = _search_max_batch(dim, steps, warmup, dtype, True, pool_dir, max_probe=max_probe)
    mult = float(aimem) / max(1.0, float(baseline))
    elapsed = time.time() - t0

    res = GateResult(
        baseline_max_batch=int(baseline),
        aimemory_max_batch=int(aimem),
        effective_headroom_multiplier=float(mult),
        threshold_multiplier=float(threshold_multiplier),
        passed=bool(mult >= threshold_multiplier),
        elapsed_s=float(elapsed),
        dtype=dtype_s,
        dim=int(dim),
        steps=int(steps),
        warmup=int(warmup),
    )
    with open(out_path, "w") as f:
        json.dump(asdict(res), f, indent=2)

    print("[headroom-gate] wrote", out_path)
    print(
        "[headroom-gate]",
        f"baseline={baseline}",
        f"aimemory={aimem}",
        f"multiplier={mult:.3f}",
        f"threshold={threshold_multiplier:.3f}",
        ("PASS" if res.passed else "FAIL"),
    )
    return asdict(res)


def main():
    p = argparse.ArgumentParser("aimemory-headroom-gate")
    p.add_argument("--pool-dir", default="/mnt/nvme_pool")
    p.add_argument("--out", default="./aimemory_headroom_gate.json")
    p.add_argument("--dim", type=int, default=8192)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--threshold-multiplier", type=float, default=3.0)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--max-probe", type=int, default=8192)
    a = p.parse_args()
    run_headroom_gate(
        pool_dir=a.pool_dir,
        out_path=a.out,
        dim=a.dim,
        steps=a.steps,
        warmup=a.warmup,
        threshold_multiplier=a.threshold_multiplier,
        dtype_s=a.dtype,
        max_probe=a.max_probe,
    )


if __name__ == "__main__":
    main()
