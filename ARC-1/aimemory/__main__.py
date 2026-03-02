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
from aimemory.static_plan import StaticPlanCompiler, model_fingerprint_full
from aimemory.distributed_coord import RankCoordinator
from aimemory.topology import detect_topology
from aimemory.backend import detect_backend_capabilities, benchmark_path, recommend_io_tuning
from aimemory.roi import ROITracker, WorkloadIdentity
from aimemory.security import enforce_retention
from aimemory.storage import SARCStorage

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
    s.add_argument("--namespace", default="default")
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
    pp.add_argument("--stage", default="stable")
    pp.add_argument("--canary-ratio", type=float, default=1.0)
    pp.add_argument("--key-uri", default="")

    ps = sub.add_parser("policy-show")
    ps.add_argument("--pool-dir", default="/mnt/nvme_pool")
    ps.add_argument("--name", required=True)
    ps.add_argument("--require-signature", action="store_true")
    ps.add_argument("--key-uri", default="")
    ps.add_argument("--envelope", action="store_true")

    pr = sub.add_parser("policy-rollback")
    pr.add_argument("--pool-dir", default="/mnt/nvme_pool")
    pr.add_argument("--name", required=True)

    pav = sub.add_parser("policy-auto-rollback")
    pav.add_argument("--pool-dir", default="/mnt/nvme_pool")
    pav.add_argument("--name", required=True)
    pav.add_argument("--p99-ms-max", type=float, default=0.0)
    pav.add_argument("--safe-mode-max", type=int, default=0)

    fr = sub.add_parser("fleet-report")
    fr.add_argument("--pool-dir", default="/mnt/nvme_pool")
    fr.add_argument("--out", default="")

    ag = sub.add_parser("agent")
    ag.add_argument("--bind", default="127.0.0.1")
    ag.add_argument("--port", type=int, default=9765)
    ag.add_argument("--metrics-path", default="")
    ag.add_argument("--heartbeat-path", default="")
    ag.add_argument("--heartbeat-interval-s", type=float, default=5.0)

    bc = sub.add_parser("backend-capabilities")
    bc.add_argument("--pool-dir", default="/mnt/nvme_pool")
    bc.add_argument("--probe-mb", type=int, default=128)
    bc.add_argument("--probe-seconds", type=float, default=2.0)

    rb = sub.add_parser("roi-baseline")
    rb.add_argument("--pool-dir", default="/mnt/nvme_pool")
    rb.add_argument("--model", default="default")
    rb.add_argument("--batch-size", type=int, default=0)
    rb.add_argument("--seq-len", type=int, default=0)
    rb.add_argument("--precision", default="autograd")
    rb.add_argument("--world-size", type=int, default=1)
    rb.add_argument("--job", default="")
    rb.add_argument("--metrics-file", required=True)

    rr = sub.add_parser("roi-report")
    rr.add_argument("--pool-dir", default="/mnt/nvme_pool")
    rr.add_argument("--model", default="default")
    rr.add_argument("--batch-size", type=int, default=0)
    rr.add_argument("--seq-len", type=int, default=0)
    rr.add_argument("--precision", default="autograd")
    rr.add_argument("--world-size", type=int, default=1)
    rr.add_argument("--job", default="")
    rr.add_argument("--metrics-file", required=True)

    rc = sub.add_parser("retention-clean")
    rc.add_argument("--path", required=True)
    rc.add_argument("--keep-days", type=int, default=14)
    rc.add_argument("--wipe", action="store_true")
    rc.add_argument("--wipe-passes", type=int, default=1)

    sc = sub.add_parser("schema-compat")
    sc.add_argument("--pool-dir", default="/mnt/nvme_pool")
    sc.add_argument("--rank", type=int, default=0)

    spc = sub.add_parser("static-plan-compile")
    spc.add_argument("--pool-dir", default="/mnt/nvme_pool")
    spc.add_argument("--name", default="default")
    spc.add_argument("--restore-order", default="1,2,3")
    spc.add_argument("--lookahead", type=int, default=4)
    spc.add_argument("--dtype", default="float16")
    spc.add_argument("--world-size", type=int, default=1)

    sps = sub.add_parser("static-plan-show")
    sps.add_argument("--pool-dir", default="/mnt/nvme_pool")
    sps.add_argument("--plan-key", required=True)

    cs = sub.add_parser("coord-sync")
    cs.add_argument("--pool-dir", default="/mnt/nvme_pool")
    cs.add_argument("--rank", type=int, default=0)
    cs.add_argument("--world-size", type=int, default=1)
    cs.add_argument("--step", type=int, default=0)
    cs.add_argument("--spill-bytes", type=int, default=0)
    cs.add_argument("--spills", type=int, default=0)
    cs.add_argument("--headroom-pct", type=float, default=0.0)

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
        out = build_support_bundle(args.pool_dir, args.out, rank=args.rank, namespace=args.namespace)
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
        p = store.push(
            args.name,
            pol,
            reason=args.reason,
            stage=args.stage,
            canary_ratio=float(args.canary_ratio),
            key_uri=str(args.key_uri),
        )
        print("Wrote", p)
        return 0
    if args.cmd == "policy-show":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        if args.envelope:
            pol = store.pull_envelope(args.name)
        else:
            pol = store.pull(
                args.name,
                require_signature=bool(args.require_signature),
                key_uri=str(args.key_uri),
            )
        print(json.dumps(pol or {}, indent=2))
        return 0
    if args.cmd == "policy-rollback":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        pol = store.rollback(args.name)
        print(json.dumps(pol or {}, indent=2))
        return 0
    if args.cmd == "policy-auto-rollback":
        store = PolicyStore(root_dir=f"{args.pool_dir}/control_plane")
        fleet = build_fleet_report(args.pool_dir)
        pol = store.auto_rollback_on_slo(
            args.name,
            fleet,
            p99_ms_max=float(args.p99_ms_max),
            safe_mode_max=int(args.safe_mode_max),
        )
        print(json.dumps({"rolled_back": bool(pol is not None), "policy": pol or {}, "fleet": fleet}, indent=2))
        return 0
    if args.cmd == "fleet-report":
        rep = build_fleet_report(args.pool_dir)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=2)
        print(json.dumps(rep, indent=2))
        return 0
    if args.cmd == "agent":
        run_agent(
            bind=args.bind,
            port=args.port,
            metrics_path=args.metrics_path,
            heartbeat_path=args.heartbeat_path,
            heartbeat_interval_s=float(args.heartbeat_interval_s),
        )
        return 0
    if args.cmd == "backend-capabilities":
        caps = detect_backend_capabilities(args.pool_dir)
        probe = benchmark_path(args.pool_dir, probe_mb=int(args.probe_mb), probe_seconds=float(args.probe_seconds))
        tune = recommend_io_tuning(caps, probe)
        print(json.dumps({"capabilities": caps, "probe": probe, "recommended": tune}, indent=2))
        return 0
    if args.cmd == "roi-baseline":
        with open(args.metrics_file, "r") as f:
            m = json.load(f)
        wid = WorkloadIdentity(
            model=str(args.model),
            batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            precision=str(args.precision),
            world_size=int(args.world_size),
            job=str(args.job),
        )
        tr = ROITracker(f"{args.pool_dir}/roi")
        p = tr.capture_baseline(wid, m)
        print(json.dumps({"baseline_path": p}, indent=2))
        return 0
    if args.cmd == "roi-report":
        with open(args.metrics_file, "r") as f:
            m = json.load(f)
        wid = WorkloadIdentity(
            model=str(args.model),
            batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            precision=str(args.precision),
            world_size=int(args.world_size),
            job=str(args.job),
        )
        tr = ROITracker(f"{args.pool_dir}/roi")
        rep = tr.attribution(wid, m)
        print(json.dumps(rep, indent=2))
        return 0
    if args.cmd == "retention-clean":
        rep = enforce_retention(args.path, keep_days=int(args.keep_days), wipe=bool(args.wipe), wipe_passes=int(args.wipe_passes))
        print(json.dumps(rep, indent=2))
        return 0
    if args.cmd == "schema-compat":
        st = SARCStorage(pool_dir=args.pool_dir, rank=int(args.rank), backend="NVME_FILE", durable=False, encrypt_at_rest=False)
        try:
            rep = st.compatibility_report()
        finally:
            st.close()
        print(json.dumps(rep, indent=2))
        return 0
    if args.cmd == "static-plan-compile":
        root = f"{args.pool_dir}/static_plans"
        comp = StaticPlanCompiler(root)
        order = [int(x.strip()) for x in str(args.restore_order).split(",") if x.strip()]
        mfp = model_fingerprint_full(
            name=str(args.name),
            shape_sig=(len(order),),
            dtype_s=str(args.dtype),
            world_size=int(args.world_size),
            graph_key=str(args.name),
        )
        plan = comp.compile_from_restore_order(mfp, order, int(args.lookahead))
        p = comp.save(plan)
        print(json.dumps({"plan_key": plan.plan_key, "path": p}, indent=2))
        return 0
    if args.cmd == "static-plan-show":
        root = f"{args.pool_dir}/static_plans"
        comp = StaticPlanCompiler(root)
        plan = comp.load(str(args.plan_key))
        if plan is None:
            print("{}")
            return 0
        print(json.dumps({
            "plan_key": plan.plan_key,
            "created_ts": plan.created_ts,
            "model_fingerprint": plan.model_fingerprint,
            "graph_fingerprint": plan.graph_fingerprint,
            "entries": [e.__dict__ for e in plan.entries],
        }, indent=2))
        return 0
    if args.cmd == "coord-sync":
        cdir = f"{args.pool_dir}/coordination"
        coord = RankCoordinator(root_dir=cdir, rank=args.rank, world_size=max(1, args.world_size), leader_rank=0)
        coord.publish(args.step, {
            "spill_bytes": args.spill_bytes,
            "spills": args.spills,
            "memory_headroom_pct": args.headroom_pct,
            "safe_mode": False,
            "step_p99_ms": 0.0,
            "latency_attribution": {},
        }, topology=detect_topology(rank=args.rank))
        cons = None
        if int(args.rank) == 0:
            cons = coord.leader_aggregate(args.step)
        if cons is None:
            cons = coord.poll_consensus(max_age_s=5.0)
        print(json.dumps(cons or {}, indent=2))
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
