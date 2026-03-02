import json
import os
import time
from typing import Dict, Any, Optional


class PolicyStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.policies_dir = os.path.join(self.root_dir, "policies")
        self.history_dir = os.path.join(self.root_dir, "history")
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

    def _policy_path(self, name: str) -> str:
        return os.path.join(self.policies_dir, f"{name}.json")

    def _history_path(self, name: str) -> str:
        return os.path.join(self.history_dir, f"{name}.jsonl")

    def push(self, name: str, policy: Dict[str, Any], reason: str = "manual") -> str:
        now = time.time()
        row = {"ts": now, "reason": reason, "policy": policy}
        with open(self._history_path(name), "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(self._policy_path(name), "w") as f:
            json.dump(policy, f, indent=2)
        return self._policy_path(name)

    def pull(self, name: str) -> Optional[Dict[str, Any]]:
        p = self._policy_path(name)
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            return json.load(f)

    def rollback(self, name: str) -> Optional[Dict[str, Any]]:
        hp = self._history_path(name)
        if not os.path.exists(hp):
            return None
        with open(hp, "r") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if len(rows) < 2:
            return None
        prev = rows[-2]["policy"]
        with open(self._policy_path(name), "w") as f:
            json.dump(prev, f, indent=2)
        return prev

    def apply_to_config(self, cfg, policy: Dict[str, Any]) -> list[str]:
        applied = []
        for k, v in policy.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                applied.append(k)
        return applied


def build_fleet_report(pool_dir: str) -> Dict[str, Any]:
    ranks = []
    base = os.path.abspath(pool_dir)
    if not os.path.exists(base):
        return {"ok": False, "reason": "pool_dir_missing"}
    for name in sorted(os.listdir(base)):
        if not name.startswith("rank_"):
            continue
        p = os.path.join(base, name, "agent_metrics.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r") as f:
                row = json.load(f)
            row["rank_dir"] = name
            ranks.append(row)
        except Exception:
            continue
    total_spills = sum(int(r.get("spills", 0)) for r in ranks)
    total_restores = sum(int(r.get("restores", 0)) for r in ranks)
    safe_mode_ranks = sum(1 for r in ranks if bool(r.get("safe_mode", False)))
    total_oom_degrades = sum(int(r.get("oom_degrade_count", 0)) for r in ranks)
    total_static_plan_hits = sum(int(r.get("static_plan_hits", 0)) for r in ranks)
    total_coord_applied = sum(int(r.get("coord_policy_applied", 0)) for r in ranks)
    total_kv_spills = sum(int(r.get("kv_spills", 0)) for r in ranks)
    total_kv_restores = sum(int(r.get("kv_restores", 0)) for r in ranks)
    total_throttle = sum(int(r.get("throttle_events", 0)) for r in ranks)
    total_budget_denials = sum(int(r.get("spill_budget_denials", 0)) for r in ranks)
    total_inflight_denials = sum(int(r.get("inflight_budget_denials", 0)) for r in ranks)
    total_fairness_denials = sum(int(r.get("fairness_denials", 0)) for r in ranks)
    total_quarantine = sum(int(r.get("quarantine_events", 0)) for r in ranks)
    p95s = [float(r.get("step_p95_ms", 0.0)) for r in ranks if float(r.get("step_p95_ms", 0.0)) > 0.0]
    p99s = [float(r.get("step_p99_ms", 0.0)) for r in ranks if float(r.get("step_p99_ms", 0.0)) > 0.0]
    spills = [int(r.get("spills", 0)) for r in ranks]
    rank_skew_pct = 0.0
    if spills and max(spills) > 0:
        rank_skew_pct = 100.0 * float(max(spills) - min(spills)) / float(max(spills))
    return {
        "ok": True,
        "ranks_seen": len(ranks),
        "total_spills": total_spills,
        "total_restores": total_restores,
        "safe_mode_ranks": safe_mode_ranks,
        "total_oom_degrades": total_oom_degrades,
        "total_static_plan_hits": total_static_plan_hits,
        "total_coord_policy_applied": total_coord_applied,
        "total_kv_spills": total_kv_spills,
        "total_kv_restores": total_kv_restores,
        "total_throttle_events": total_throttle,
        "total_spill_budget_denials": total_budget_denials,
        "total_inflight_denials": total_inflight_denials,
        "total_fairness_denials": total_fairness_denials,
        "total_quarantine_events": total_quarantine,
        "fleet_step_p95_ms_max": (max(p95s) if p95s else 0.0),
        "fleet_step_p99_ms_max": (max(p99s) if p99s else 0.0),
        "rank_skew_pct": rank_skew_pct,
        "ranks": ranks,
    }
