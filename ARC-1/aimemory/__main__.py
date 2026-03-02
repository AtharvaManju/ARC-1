import argparse
from aimemory.doctor import run_doctor
from aimemory.installer import run_installer
from aimemory.bench.bench import run_bench
from aimemory.bench.headroom_gate import run_headroom_gate
from aimemory.gc import gc_old_windows
from aimemory.support_bundle import build_support_bundle

def main():
    p = argparse.ArgumentParser(prog="aimemory")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor")
    d.add_argument("--pool-dir", default="/mnt/nvme_pool")

    i = sub.add_parser("install")
    i.add_argument("--pool-dir", default="/mnt/nvme_pool")
    i.add_argument("--out", default="./aimemory_install_report.json")

    b = sub.add_parser("bench")
    b.add_argument("--pool-dir", default="/mnt/nvme_pool")
    b.add_argument("--steps", type=int, default=30)
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--out-dir", default="./aimemory_bench")

    hg = sub.add_parser("headroom-gate")
    hg.add_argument("--pool-dir", default="/mnt/nvme_pool")
    hg.add_argument("--out", default="./aimemory_headroom_gate.json")
    hg.add_argument("--dim", type=int, default=8192)
    hg.add_argument("--steps", type=int, default=2)
    hg.add_argument("--warmup", type=int, default=1)
    hg.add_argument("--threshold-multiplier", type=float, default=3.0)
    hg.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    hg.add_argument("--max-probe", type=int, default=8192)

    g = sub.add_parser("gc")
    g.add_argument("--pool-dir", default="/mnt/nvme_pool")
    g.add_argument("--rank", type=int, default=0)
    g.add_argument("--cutoff-pool-id", type=int, required=True)
    g.add_argument("--vacuum", action="store_true")
    g.add_argument("--no-checkpoint-truncate", action="store_true")

    s = sub.add_parser("support-bundle")
    s.add_argument("--pool-dir", default="/mnt/nvme_pool")
    s.add_argument("--rank", type=int, default=0)
    s.add_argument("--out", default="./aimemory_support_bundle.zip")

    args = p.parse_args()

    if args.cmd == "doctor":
        return run_doctor(pool_dir=args.pool_dir)
    if args.cmd == "install":
        run_installer(pool_dir=args.pool_dir, out_path=args.out, apply_fixes=False)
        return 0
    if args.cmd == "bench":
        run_bench(pool_dir=args.pool_dir, steps=args.steps, warmup=args.warmup, out_dir=args.out_dir)
        return 0
    if args.cmd == "headroom-gate":
        run_headroom_gate(
            pool_dir=args.pool_dir,
            out_path=args.out,
            dim=args.dim,
            steps=args.steps,
            warmup=args.warmup,
            threshold_multiplier=args.threshold_multiplier,
            dtype_s=args.dtype,
            max_probe=args.max_probe,
        )
        return 0
    if args.cmd == "gc":
        r = gc_old_windows(
            args.pool_dir,
            args.rank,
            args.cutoff_pool_id,
            checkpoint_truncate=(not args.no_checkpoint_truncate),
            vacuum=bool(args.vacuum),
        )
        print(r)
        return 0
    if args.cmd == "support-bundle":
        out = build_support_bundle(args.pool_dir, args.out, rank=args.rank)
        print("Wrote", out)
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
