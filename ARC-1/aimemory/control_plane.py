import hmac
import json
import hashlib
import os
import time
from typing import Dict, Any, Optional, Tuple

from .roi import robust_stats
from .security import resolve_key_from_uri


def _stable_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _policy_digest(policy: Dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(policy).encode()).hexdigest()


def _sign(policy: Dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _stable_json(policy).encode(), hashlib.sha256).hexdigest()


class PolicyStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.policies_dir = os.path.join(self.root_dir, "policies")
        self.history_dir = os.path.join(self.root_dir, "history")
        self.events_dir = os.path.join(self.root_dir, "events")
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

    def _policy_path(self, name: str) -> str:
        return os.path.join(self.policies_dir, f"{name}.json")

    def _history_path(self, name: str) -> str:
        return os.path.join(self.history_dir, f"{name}.jsonl")

    def _events_path(self, name: str) -> str:
        return os.path.join(self.events_dir, f"{name}.events.jsonl")

    def _write_event(self, name: str, event: str, **kwargs):
        row = {"ts": float(time.time()), "event": str(event)}
        row.update(kwargs)
        with open(self._events_path(name), "a") as f:
            f.write(json.dumps(row) + "\n")

    def _envelope(
        self,
        policy: Dict[str, Any],
        *,
        reason: str,
        version: int,
        stage: str,
        canary_ratio: float,
        key_uri: str = "",
    ) -> Dict[str, Any]:
        env = {
            "schema_version": 2,
            "version": int(version),
            "created_ts": float(time.time()),
            "reason": str(reason),
            "stage": str(stage),
            "canary_ratio": float(max(0.0, min(float(canary_ratio), 1.0))),
            "policy": dict(policy),
            "policy_digest": _policy_digest(policy),
            "signature": "",
            "signature_alg": "",
        }
        if key_uri:
            key = resolve_key_from_uri(key_uri)
            env["signature"] = _sign(env["policy"], key)
            env["signature_alg"] = "hmac-sha256"
        return env

    def push(
        self,
        name: str,
        policy: Dict[str, Any],
        reason: str = "manual",
        *,
        stage: str = "stable",
        canary_ratio: float = 1.0,
        key_uri: str = "",
    ) -> str:
        now = time.time()
        cur = self.pull_envelope(name)
        ver = int(cur.get("version", 0)) + 1 if cur else 1
        env = self._envelope(
            policy=policy,
            reason=reason,
            version=ver,
            stage=stage,
            canary_ratio=canary_ratio,
            key_uri=key_uri,
        )
        row = {"ts": now, "reason": reason, "envelope": env}
        with open(self._history_path(name), "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(self._policy_path(name), "w") as f:
            json.dump(env, f, indent=2)
        self._write_event(name, "push", version=int(ver), stage=str(stage), canary_ratio=float(canary_ratio))
        return self._policy_path(name)

    def pull_envelope(self, name: str) -> Optional[Dict[str, Any]]:
        p = self._policy_path(name)
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            obj = json.load(f)
        # Backward compatibility: old files may contain plain policy.
        if "policy" not in obj:
            return {
                "schema_version": 1,
                "version": 1,
                "created_ts": 0.0,
                "reason": "legacy",
                "stage": "stable",
                "canary_ratio": 1.0,
                "policy": obj,
                "policy_digest": _policy_digest(obj),
                "signature": "",
                "signature_alg": "",
            }
        return obj

    def verify_envelope(self, env: Dict[str, Any], *, require_signature: bool, key_uri: str = "") -> Tuple[bool, str]:
        pol = dict(env.get("policy", {}))
        expect_digest = _policy_digest(pol)
        if str(env.get("policy_digest", "")) != expect_digest:
            return False, "policy_digest_mismatch"
        sig = str(env.get("signature", ""))
        if require_signature and (not sig):
            return False, "missing_signature"
        if sig:
            if not key_uri:
                if require_signature:
                    return False, "signature_present_but_no_key"
                # Backward-compatible read path: allow unsigned verification mode.
                return True, ""
            key = resolve_key_from_uri(key_uri)
            calc = _sign(pol, key)
            if not hmac.compare_digest(calc, sig):
                return False, "invalid_signature"
        return True, ""

    def pull(self, name: str, *, require_signature: bool = False, key_uri: str = "") -> Optional[Dict[str, Any]]:
        env = self.pull_envelope(name)
        if env is None:
            return None
        ok, why = self.verify_envelope(env, require_signature=require_signature, key_uri=key_uri)
        if not ok:
            self._write_event(name, "pull_reject", reason=str(why), version=int(env.get("version", 0)))
            return None
        return dict(env.get("policy", {}))

    def select_for_rank(
        self,
        name: str,
        rank: int,
        *,
        require_signature: bool = False,
        key_uri: str = "",
        stage: str = "",
    ) -> Optional[Dict[str, Any]]:
        env = self.pull_envelope(name)
        if env is None:
            return None
        ok, _ = self.verify_envelope(env, require_signature=require_signature, key_uri=key_uri)
        if not ok:
            return None
        env_stage = str(env.get("stage", "stable"))
        if stage and (env_stage != stage):
            return None
        ratio = max(0.0, min(float(env.get("canary_ratio", 1.0)), 1.0))
        bucket = int(abs(int(rank)) % 1000)
        if ratio < 1.0 and bucket >= int(ratio * 1000):
            return None
        return dict(env.get("policy", {}))

    def rollback(self, name: str, *, reason: str = "manual_rollback") -> Optional[Dict[str, Any]]:
        hp = self._history_path(name)
        if not os.path.exists(hp):
            return None
        with open(hp, "r") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        if len(rows) < 2:
            return None
        prev_env = rows[-2].get("envelope", {})
        cur_env = rows[-1].get("envelope", {})
        with open(self._policy_path(name), "w") as f:
            json.dump(prev_env, f, indent=2)
        self._write_event(
            name,
            "rollback",
            reason=str(reason),
            from_version=int(cur_env.get("version", 0)),
            to_version=int(prev_env.get("version", 0)),
        )
        return dict(prev_env.get("policy", {}))

    def auto_rollback_on_slo(
        self,
        name: str,
        fleet: Dict[str, Any],
        *,
        p99_ms_max: float = 0.0,
        safe_mode_max: int = 0,
    ) -> Optional[Dict[str, Any]]:
        p99 = float(fleet.get("fleet_step_p99_ms_max", 0.0))
        safe = int(fleet.get("safe_mode_ranks", 0))
        breach = False
        reasons = []
        if p99_ms_max > 0.0 and p99 > p99_ms_max:
            breach = True
            reasons.append(f"p99>{p99_ms_max}")
        if safe_mode_max >= 0 and safe > safe_mode_max:
            breach = True
            reasons.append(f"safe_mode_ranks>{safe_mode_max}")
        if not breach:
            return None
        return self.rollback(name, reason=f"auto_rollback:{','.join(reasons)}")

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
    metric_paths = []
    for name in sorted(os.listdir(base)):
        p0 = os.path.join(base, name)
        if name.startswith("rank_"):
            metric_paths.append((name, os.path.join(p0, "agent_metrics.json")))
        elif name.startswith("ns_") and os.path.isdir(p0):
            for rname in sorted(os.listdir(p0)):
                if rname.startswith("rank_"):
                    metric_paths.append((os.path.join(name, rname), os.path.join(p0, rname, "agent_metrics.json")))
    for name, p in metric_paths:
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
    p95_rs = robust_stats(p95s)
    p99_rs = robust_stats(p99s)
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
        "fleet_step_p95_stats": p95_rs,
        "fleet_step_p99_stats": p99_rs,
        "rank_skew_pct": rank_skew_pct,
        "ranks": ranks,
    }
