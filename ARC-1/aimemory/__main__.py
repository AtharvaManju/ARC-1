import argparse
import json
from aimemory.doctor import run_doctor
from aimemory.installer import run_installer
from aimemory.bench.bench import run_bench
from aimemory.bench.headroom_gate import run_headroom_gate
from aimemory.bench.qualification import run_qualification
from aimemory.gc import gc_old_windows
from aimemory.support_bundle import build_support_bundle
from aimemory.consistency import run_consistency_check
from aimemory.control_plane import PolicyStore, build_fleet_report
from aimemory.agent import run_agent

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

    q = sub.add_parser("qualify")
    q.add_argument("--pool-dir", default="/mnt/nvme_pool")
    q.add_argument("--out", default="./aimemory_qualification.json")
    q.add_argument("--threshold-multiplier", type=float, default=3.0)
    q.add_argument("--overhead-sla-pct", type=float, default=15.0)

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

    cc = sub.add_parser("consistency-check")
    cc.add_argument("--pool-dir", default="/mnt/nvme_pool")
    cc.add_argument("--rank", type=int, default=0)
    cc.add_argument("--repair", action="store_true")
    cc.add_argument("--out", default="")

    pp = sub.add_parser("policy-push")
    pp.add_argument("--pool-dir", default="/mnt/nvme_pool")
    pp.add_argument("--name", required=True)
    pp.add_argument("--file", required=True)
    pp.add_argument("--reason", default="manual")

    ps = sub.add_parser("policy-show")
    ps.add_argument("--pool-dir", default="/mnt/nvme_pool")
    ps.add_argument("--name", required=True)

    pr = sub.add_parser("policy-rollback")
    pr.add_argument("--pool-dir", default="/mnt/nvme_pool")
    pr.add_argument("--name", required=True)

    fr = sub.add_parser("fleet-report")
    fr.add_argument("--pool-dir", default="/mnt/nvme_pool")
    fr.add_argument("--out", default="")

    ag = sub.add_parser("agent")
    ag.add_argument("--bind", default="127.0.0.1")
    ag.add_argument("--port", type=int, default=9765)
    ag.add_argument("--metrics-path", default="")

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
    if args.cmd == "qualify":
        run_qualification(
            pool_dir=args.pool_dir,
            out_path=args.out,
            threshold_multiplier=args.threshold_multiplier,
            overhead_sla_pct=args.overhead_sla_pct,
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
    if args.cmd == "consistency-check":
        r = run_consistency_check(pool_dir=args.pool_dir, rank=args.rank, repair=bool(args.repair), out_path=args.out)
        print(r)
        return 0
    if args.cmd == "policy-push":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        with open(args.file, "r") as f:
            pol = json.load(f)
        p = store.push(args.name, pol, reason=args.reason)
        print("Wrote", p)
        return 0
    if args.cmd == "policy-show":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        pol = store.pull(args.name)
        print(json.dumps(pol or {}, indent=2))
        return 0
    if args.cmd == "policy-rollback":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        pol = store.rollback(args.name)
        print(json.dumps(pol or {}, indent=2))
        return 0
    if args.cmd == "fleet-report":
        rep = build_fleet_report(args.pool_dir)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))
        return 0
    if args.cmd == "agent":
        run_agent(bind=args.bind, port=args.port, metrics_path=args.metrics_path)
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
