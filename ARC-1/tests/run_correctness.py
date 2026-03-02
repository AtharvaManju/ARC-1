import argparse
import os
import torch
import torch.optim as optim

from tests.models import MLP, TinyAttention, ConvNet
from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

def _set(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _snap(model):
    out = {}
    for n,p in model.named_parameters():
        out[n] = None if p.grad is None else p.grad.detach().float().cpu().clone()
    return out

def _maxdiff(a,b):
    if a is None and b is None: return 0.0
    if (a is None) != (b is None): return float("inf")
    return float((a-b).abs().max().item())

def run_one(mctor, xctor, steps, use_ai, pool_dir):
    _set(0)
    m = mctor().cuda().train()
    opt = optim.AdamW(m.parameters(), lr=1e-3)
    ctrl = None
    if use_ai:
        cfg = AIMemoryConfig(
            pool_dir=pool_dir,
            spill_min_bytes=1*1024*1024,
            pinned_pool_bytes=512*1024**2,
            sync_each_step=True,  # correctness wants strict sync
        )
        ctrl = AIMemoryController(cfg)
        x = xctor()
        opt.zero_grad(set_to_none=True)
        with ctrl.step(profiling_warmup=True):
            y = m(x); (y.float().pow(2).mean()).backward()
        ctrl.finalize_pcc_profile()

    for _ in range(steps):
        x = xctor()
        opt.zero_grad(set_to_none=True)
        if ctrl:
            with ctrl.step():
                y = m(x); (y.float().pow(2).mean()).backward()
        else:
            y = m(x); (y.float().pow(2).mean()).backward()
        opt.step()

    g = _snap(m)
    if ctrl: ctrl.shutdown()
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", default="/mnt/nvme_pool")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--atol", type=float, default=5e-3)
    ap.add_argument("--rtol", type=float, default=5e-3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    os.makedirs(args.pool_dir, exist_ok=True)

    tests = [
        ("MLP-fp16", lambda: MLP(2048), lambda: torch.randn(8,2048,device="cuda",dtype=torch.float16)),
        ("TinyAttn-bf16", lambda: TinyAttention(512,8), lambda: torch.randn(4,64,512,device="cuda",dtype=torch.bfloat16)),
        ("ConvNet-fp16", lambda: ConvNet(), lambda: torch.randn(2,32,64,64,device="cuda",dtype=torch.float16)),
    ]

    any_fail = False
    for name,mctor,xctor in tests:
        b = run_one(mctor,xctor,args.steps,False,args.pool_dir)
        a = run_one(mctor,xctor,args.steps,True,args.pool_dir)
        worst=0.0; wk=None
        for k in b:
            d=_maxdiff(b[k],a[k])
            if d>worst: worst=d; wk=k
        scale=0.0
        if wk and b[wk] is not None:
            scale=float(b[wk].abs().max().item())
        thr=args.atol + args.rtol * max(1e-9, scale)
        ok = worst <= thr
        print(name, "worst", worst, "thr", thr, "=>", "OK" if ok else "FAIL")
        if not ok: any_fail=True

    if any_fail: raise SystemExit("FAILED")
    print("ALL OK")

if __name__ == "__main__":
    main()
