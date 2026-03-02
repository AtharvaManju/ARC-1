import json
import os
import time
from typing import Any, Dict

from .control_plane import build_fleet_report


def _load_json(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def build_ga_readiness(pool_dir: str, qualification_path: str = "") -> Dict[str, Any]:
    fleet = build_fleet_report(pool_dir)
    qual = _load_json(qualification_path)
    qg = qual.get("gates", {}) if isinstance(qual, dict) else {}
    checks = {
        "qualification_present": bool(qual),
        "qualification_pass": bool(qual.get("passed", False)) if isinstance(qual, dict) else False,
        "fleet_report_ok": bool(fleet.get("ok", False)),
        "fleet_safe_mode_ranks": int(fleet.get("safe_mode_ranks", 0)) == 0,
        "fleet_p99_within_200ms": float(fleet.get("fleet_step_p99_ms_max", 0.0)) <= 200.0,
        "rank_skew_within_25pct": float(fleet.get("rank_skew_pct", 0.0)) <= 25.0,
    }
    if isinstance(qg, dict) and qg:
        for k, v in qg.items():
            checks[f"gate_{k}"] = bool((v or {}).get("pass", False))
    all_ok = all(bool(v) for v in checks.values()) if checks else False
    stage = "GA_READY" if all_ok else ("CANARY_ONLY" if checks.get("qualification_pass", False) else "BLOCK")
    return {
        "ts": float(time.time()),
        "pool_dir": os.path.abspath(pool_dir),
        "qualification_path": os.path.abspath(qualification_path) if qualification_path else "",
        "checks": checks,
        "stage": stage,
        "fleet": fleet,
        "qualification": qual,
    }


def build_commercial_pack(pool_dir: str, qualification_path: str, out_dir: str, customer: str = "") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    ga = build_ga_readiness(pool_dir=pool_dir, qualification_path=qualification_path)
    qual = ga.get("qualification", {}) if isinstance(ga, dict) else {}
    fleet = ga.get("fleet", {}) if isinstance(ga, dict) else {}
    metrics = (qual.get("bench", {}) or {}).get("metrics", {}) if isinstance(qual, dict) else {}

    headline = {
        "customer": str(customer or "prospect"),
        "stage": str(ga.get("stage", "BLOCK")),
        "peak_hbm_reduction_estimate_pct": float((qual.get("headroom_gate", {}) or {}).get("reduction_ratio", 0.0)) * 100.0,
        "p95_step_ms": float((qual.get("bench", {}) or {}).get("step_ms_p95", 0.0)),
        "p99_step_ms": float((qual.get("bench", {}) or {}).get("step_ms_p99", 0.0)),
        "prefetch_hit_rate_pct": float((metrics or {}).get("prefetch_hit_rate", 0.0)),
        "safe_mode": bool((metrics or {}).get("safe_mode", False)),
        "rank_skew_pct": float(fleet.get("rank_skew_pct", 0.0)),
    }

    roi = (metrics or {}).get("roi", {})
    narrative = [
        f"ARC-1 rollout stage: {headline['stage']}.",
        f"Estimated peak HBM reduction: {headline['peak_hbm_reduction_estimate_pct']:.1f}%.",
        f"Observed step latency p95/p99: {headline['p95_step_ms']:.2f}ms / {headline['p99_step_ms']:.2f}ms.",
        f"Prefetch hit rate: {headline['prefetch_hit_rate_pct']:.1f}%.",
        f"Fleet rank skew: {headline['rank_skew_pct']:.1f}%.",
        f"ROI signal: headroom_gain={float((roi or {}).get('headroom_gain_pct', 0.0)):.1f}%, ooms_prevented={int((roi or {}).get('ooms_prevented', 0))}.",
    ]
    rep = {
        "headline": headline,
        "ga_readiness": ga,
        "narrative": narrative,
    }
    rep_path = os.path.join(out_dir, "arc1_commercial_report.json")
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2)
    md_path = os.path.join(out_dir, "arc1_commercial_report.md")
    with open(md_path, "w") as f:
        f.write("# ARC-1 Commercial Qualification Report\n\n")
        f.write("\n".join([f"- {line}" for line in narrative]))
        f.write("\n")
    return {"json": rep_path, "markdown": md_path, "stage": headline["stage"]}
