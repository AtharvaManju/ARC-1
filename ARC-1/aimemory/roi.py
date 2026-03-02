import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import numpy as np


@dataclass
class WorkloadIdentity:
    model: str = "unknown"
    batch_size: int = 0
    seq_len: int = 0
    precision: str = "unknown"
    world_size: int = 1
    job: str = ""


def identity_hash(identity: WorkloadIdentity) -> str:
    raw = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _winsorized(arr: np.ndarray, p: float = 5.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = np.percentile(arr, p)
    hi = np.percentile(arr, 100.0 - p)
    return np.clip(arr, lo, hi)


def robust_stats(xs: List[float]) -> Dict[str, float]:
    arr = np.array([float(x) for x in xs], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0}
    w = _winsorized(arr, p=5.0)
    mean = float(w.mean())
    p50 = float(np.percentile(w, 50))
    p95 = float(np.percentile(w, 95))
    p99 = float(np.percentile(w, 99))
    if w.size >= 5:
        bmeans = []
        rng = np.random.default_rng(1234)
        for _ in range(200):
            samp = rng.choice(w, size=w.size, replace=True)
            bmeans.append(float(np.mean(samp)))
        ci_lo = float(np.percentile(bmeans, 2.5))
        ci_hi = float(np.percentile(bmeans, 97.5))
    else:
        ci_lo = mean
        ci_hi = mean
    return {
        "n": int(arr.size),
        "mean": mean,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }


class ROITracker:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir or "./aimemory_roi")
        os.makedirs(self.root_dir, exist_ok=True)

    def _baseline_path(self, wid: WorkloadIdentity) -> str:
        return os.path.join(self.root_dir, f"{identity_hash(wid)}.baseline.json")

    def capture_baseline(self, wid: WorkloadIdentity, payload: Dict[str, Any]) -> str:
        row = {"ts": float(time.time()), "identity": asdict(wid), "baseline": payload}
        p = self._baseline_path(wid)
        with open(p, "w") as f:
            json.dump(row, f, indent=2)
        return p

    def load_baseline(self, wid: WorkloadIdentity) -> Optional[Dict[str, Any]]:
        p = self._baseline_path(wid)
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            return json.load(f)

    def anti_gaming(self, baseline_identity: Dict[str, Any], current_identity: Dict[str, Any]) -> Dict[str, Any]:
        b = dict(baseline_identity or {})
        c = dict(current_identity or {})
        issues = []
        for field in ("batch_size", "seq_len", "world_size"):
            bv = int(b.get(field, 0))
            cv = int(c.get(field, 0))
            if bv > 0 and cv < bv:
                issues.append(f"{field}_reduced:{bv}->{cv}")
        if str(b.get("precision", "")) and str(c.get("precision", "")) and str(c.get("precision")) != str(b.get("precision")):
            issues.append(f"precision_changed:{b.get('precision')}->{c.get('precision')}")
        return {"ok": len(issues) == 0, "issues": issues}

    def attribution(
        self,
        wid: WorkloadIdentity,
        current: Dict[str, Any],
        baseline_row: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        base = baseline_row or self.load_baseline(wid) or {}
        base_metrics = dict(base.get("baseline", {}))
        cur = dict(current or {})
        # Capacity uplift approximations.
        headroom_gain = float(cur.get("memory_headroom_pct", 0.0)) - float(base_metrics.get("memory_headroom_pct", 0.0))
        ooms_prevented = max(0, int(base_metrics.get("oom_events", 0)) - int(cur.get("oom_events", 0)))
        reruns_avoided = max(0, int(base_metrics.get("reruns", 0)) - int(cur.get("reruns", 0)))
        thpt_delta = float(cur.get("throughput", 0.0)) - float(base_metrics.get("throughput", 0.0))
        step_samples = [float(x) for x in cur.get("step_samples_ms", []) if float(x) >= 0.0]
        rs = robust_stats(step_samples)
        anti = self.anti_gaming(base.get("identity", {}), asdict(wid))
        return {
            "workload_id": identity_hash(wid),
            "headroom_gain_pct": float(headroom_gain),
            "ooms_prevented": int(ooms_prevented),
            "reruns_avoided": int(reruns_avoided),
            "throughput_delta": float(thpt_delta),
            "step_ms_stats": rs,
            "anti_gaming": anti,
        }
