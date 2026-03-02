import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np


@dataclass
class ParityCertification:
    ok: bool
    loss_curve_max_delta_pct: float
    grad_mean_delta_pct: float
    grad_std_delta_pct: float
    reproducibility_pass: bool
    reasons: List[str]


def _safe_arr(x) -> np.ndarray:
    try:
        return np.array([float(v) for v in x], dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def certify_training_outcome_parity(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    loss_delta_pct_max: float = 5.0,
    grad_stat_delta_pct_max: float = 8.0,
    reproducibility_tolerance: float = 1e-6,
) -> ParityCertification:
    b_loss = _safe_arr(baseline.get("loss_curve", []))
    c_loss = _safe_arr(candidate.get("loss_curve", []))
    n = min(int(b_loss.size), int(c_loss.size))
    reasons: List[str] = []
    if n <= 0:
        reasons.append("missing_loss_curve")
        return ParityCertification(
            ok=False,
            loss_curve_max_delta_pct=100.0,
            grad_mean_delta_pct=100.0,
            grad_std_delta_pct=100.0,
            reproducibility_pass=False,
            reasons=reasons,
        )
    b = b_loss[:n]
    c = c_loss[:n]
    loss_delta = np.abs(c - b) / np.maximum(1e-9, np.abs(b))
    max_loss_pct = float(np.max(loss_delta) * 100.0)
    b_g = _safe_arr(baseline.get("grad_norm_curve", []))
    c_g = _safe_arr(candidate.get("grad_norm_curve", []))
    gn = min(int(b_g.size), int(c_g.size))
    if gn > 0:
        bb = b_g[:gn]
        cc = c_g[:gn]
        b_mean = float(np.mean(bb))
        c_mean = float(np.mean(cc))
        b_std = float(np.std(bb))
        c_std = float(np.std(cc))
        grad_mean_delta_pct = float(abs(c_mean - b_mean) / max(1e-9, abs(b_mean)) * 100.0)
        grad_std_delta_pct = float(abs(c_std - b_std) / max(1e-9, abs(b_std)) * 100.0)
    else:
        grad_mean_delta_pct = 0.0
        grad_std_delta_pct = 0.0
    repro = bool(candidate.get("reproducibility_mode", False))
    repro_ref = float(baseline.get("reproducibility_checksum", 0.0))
    repro_cand = float(candidate.get("reproducibility_checksum", 0.0))
    repro_pass = (not repro) or (math.isclose(repro_ref, repro_cand, rel_tol=reproducibility_tolerance, abs_tol=reproducibility_tolerance))
    if max_loss_pct > float(loss_delta_pct_max):
        reasons.append("loss_curve_drift")
    if grad_mean_delta_pct > float(grad_stat_delta_pct_max):
        reasons.append("grad_mean_drift")
    if grad_std_delta_pct > float(grad_stat_delta_pct_max):
        reasons.append("grad_std_drift")
    if not repro_pass:
        reasons.append("reproducibility_checksum_mismatch")
    ok = len(reasons) == 0
    return ParityCertification(
        ok=bool(ok),
        loss_curve_max_delta_pct=float(max_loss_pct),
        grad_mean_delta_pct=float(grad_mean_delta_pct),
        grad_std_delta_pct=float(grad_std_delta_pct),
        reproducibility_pass=bool(repro_pass),
        reasons=reasons,
    )


def certify_from_files(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    with open(baseline_path, "r") as f:
        base = json.load(f)
    with open(candidate_path, "r") as f:
        cand = json.load(f)
    res = certify_training_outcome_parity(base, cand)
    return asdict(res)
