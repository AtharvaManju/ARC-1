import json
import os
import time
from typing import Any, Dict, List


def _load(path: str) -> Dict[str, Any]:
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


def build_claims_evidence(
    *,
    qualification_path: str = "",
    fastpath_path: str = "",
    benchmark_path: str = "",
    require_cuda_evidence: bool = True,
) -> Dict[str, Any]:
    qual = _load(qualification_path)
    fqual = _load(fastpath_path)
    bench = _load(benchmark_path)

    missing: List[str] = []
    if not qual:
        missing.append("qualification_report")
    if not fqual:
        missing.append("fastpath_report")
    if not bench:
        missing.append("benchmark_report")

    hbm_red_pct = float((qual.get("headroom_gate", {}) or {}).get("reduction_ratio", 0.0)) * 100.0
    p99_ov = float((qual.get("pressure_profile", {}) or {}).get("p99_overhead_pct", 0.0))
    parity_ok = bool((qual.get("training_outcome_parity", {}) or {}).get("ok", False))
    fast_ok = bool(fqual.get("direct_restore_success", False) or fqual.get("gds_enabled", False))

    cuda_seen = bool(
        (qual.get("bench", {}) or {}).get("metrics", {}).get("memory_total_bytes", 0)
        or fqual.get("cuda_available", False)
        or (bench.get("metrics", {}) or {}).get("memory_total_bytes", 0)
    )

    status = "READY"
    reasons = []
    if missing:
        status = "INCOMPLETE"
        reasons.append("missing_reports")
    if require_cuda_evidence and (not cuda_seen):
        status = "PENDING_CUDA_EVIDENCE"
        reasons.append("no_cuda_field_evidence")
    if not parity_ok:
        reasons.append("parity_not_confirmed")
    if p99_ov > 0.0 and p99_ov > 25.0:
        reasons.append("p99_overhead_high")

    claims = {
        "peak_hbm_reduction_pct": hbm_red_pct,
        "p99_overhead_pct": p99_ov,
        "training_parity_ok": parity_ok,
        "fastpath_evidence_ok": fast_ok,
    }
    return {
        "ts": float(time.time()),
        "status": status,
        "reasons": reasons,
        "missing": missing,
        "cuda_evidence_present": bool(cuda_seen),
        "claims": claims,
        "sources": {
            "qualification_path": os.path.abspath(qualification_path) if qualification_path else "",
            "fastpath_path": os.path.abspath(fastpath_path) if fastpath_path else "",
            "benchmark_path": os.path.abspath(benchmark_path) if benchmark_path else "",
        },
    }


def write_claims_evidence(out_path: str, **kwargs) -> Dict[str, Any]:
    rep = build_claims_evidence(**kwargs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    return rep
