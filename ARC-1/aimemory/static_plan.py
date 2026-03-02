import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class PlanEntry:
    pack_idx: int
    prefetch_lookahead: int
    priority: int


@dataclass
class StaticSpillPlan:
    plan_key: str
    created_ts: float
    model_fingerprint: str
    graph_fingerprint: str
    entries: List[PlanEntry]


def _hash_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class StaticPlanCompiler:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def plan_path(self, key: str) -> str:
        return os.path.join(self.root_dir, f"{key}.json")

    def compile_from_restore_order(
        self,
        model_fingerprint: str,
        restore_order: List[int],
        lookahead: int,
    ) -> StaticSpillPlan:
        graph_fp = _hash_obj({"restore_order": restore_order})
        plan_key = _hash_obj({"m": model_fingerprint, "g": graph_fp, "l": int(lookahead)})
        entries: List[PlanEntry] = []
        for i, pidx in enumerate(restore_order):
            entries.append(
                PlanEntry(
                    pack_idx=int(pidx),
                    prefetch_lookahead=max(1, int(lookahead)),
                    priority=int(len(restore_order) - i),
                )
            )
        return StaticSpillPlan(
            plan_key=plan_key,
            created_ts=float(time.time()),
            model_fingerprint=str(model_fingerprint),
            graph_fingerprint=str(graph_fp),
            entries=entries,
        )

    def save(self, plan: StaticSpillPlan) -> str:
        p = self.plan_path(plan.plan_key)
        with open(p, "w") as f:
            json.dump(
                {
                    "plan_key": plan.plan_key,
                    "created_ts": plan.created_ts,
                    "model_fingerprint": plan.model_fingerprint,
                    "graph_fingerprint": plan.graph_fingerprint,
                    "entries": [asdict(e) for e in plan.entries],
                },
                f,
                indent=2,
            )
        return p

    def load(self, key: str) -> StaticSpillPlan | None:
        p = self.plan_path(key)
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            raw = json.load(f)
        entries = [PlanEntry(**x) for x in raw.get("entries", [])]
        return StaticSpillPlan(
            plan_key=str(raw["plan_key"]),
            created_ts=float(raw["created_ts"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            graph_fingerprint=str(raw["graph_fingerprint"]),
            entries=entries,
        )

    def compile_capture_parity(self, baseline: Dict[str, float], candidate: Dict[str, float], tol_ratio: float = 0.05) -> dict:
        keys = sorted(set(baseline.keys()) & set(candidate.keys()))
        deltas = {}
        ok = True
        for k in keys:
            b = float(baseline[k])
            c = float(candidate[k])
            rel = abs(c - b) / max(1e-9, abs(b))
            deltas[k] = rel
            if rel > float(tol_ratio):
                ok = False
        return {"ok": bool(ok), "tol_ratio": float(tol_ratio), "deltas": deltas}


def model_fingerprint(name: str, shape_sig: Tuple[int, ...], dtype_s: str) -> str:
    return _hash_obj({"name": str(name), "shape": list(shape_sig), "dtype": str(dtype_s)})


def model_fingerprint_full(
    name: str,
    shape_sig: Tuple[int, ...],
    dtype_s: str,
    world_size: int,
    graph_key: str = "",
) -> str:
    return _hash_obj(
        {
            "name": str(name),
            "shape": list(shape_sig),
            "dtype": str(dtype_s),
            "world_size": int(world_size),
            "graph_key": str(graph_key),
        }
    )
