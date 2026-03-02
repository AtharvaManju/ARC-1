import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class MemorySLOContract:
    never_oom: bool = False
    max_hbm_bytes: int = 0
    p99_overhead_ms: float = 0.0
    p99_overhead_pct: float = 0.0
    policy: str = "balanced"


def load_contract(path: str = "", fallback: Optional[MemorySLOContract] = None) -> MemorySLOContract:
    fb = fallback or MemorySLOContract()
    if not path:
        return fb
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return fb
    try:
        with open(p, "r") as f:
            obj = json.load(f)
        return MemorySLOContract(
            never_oom=bool(obj.get("never_oom", fb.never_oom)),
            max_hbm_bytes=int(obj.get("max_hbm_bytes", fb.max_hbm_bytes)),
            p99_overhead_ms=float(obj.get("p99_overhead_ms", fb.p99_overhead_ms)),
            p99_overhead_pct=float(obj.get("p99_overhead_pct", fb.p99_overhead_pct)),
            policy=str(obj.get("policy", fb.policy)),
        )
    except Exception:
        return fb


class MemorySLOEnforcer:
    def __init__(self, contract: MemorySLOContract, out_dir: str, rank: int):
        self.contract = contract
        self.rank = int(rank)
        self.out_dir = os.path.abspath(out_dir or ".")
        os.makedirs(self.out_dir, exist_ok=True)
        self.proof_jsonl = os.path.join(self.out_dir, f"slo_proof_rank_{self.rank}.jsonl")
        self.summary_json = os.path.join(self.out_dir, f"slo_summary_rank_{self.rank}.json")
        self.violations = 0
        self.last: Dict[str, Any] = {}

    def evaluate(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        m = dict(snapshot or {})
        used_hbm = max(0, int(m.get("memory_total_bytes", 0)) - int(m.get("memory_free_bytes", 0)))
        step_p99 = float(m.get("step_p99_ms", 0.0))
        baseline_ms = float(m.get("baseline_step_ms_ema", 0.0))
        safe_mode = bool(m.get("safe_mode", False))
        ooms = int(m.get("oom_degrade_count", 0))
        checks = []
        if self.contract.never_oom:
            checks.append(("never_oom", (ooms == 0) and (not safe_mode), {"oom_degrade_count": ooms, "safe_mode": safe_mode}))
        if int(self.contract.max_hbm_bytes) > 0:
            checks.append(("max_hbm_bytes", used_hbm <= int(self.contract.max_hbm_bytes), {"used_hbm_bytes": used_hbm}))
        if float(self.contract.p99_overhead_ms) > 0.0:
            checks.append(("p99_overhead_ms", step_p99 <= float(self.contract.p99_overhead_ms), {"step_p99_ms": step_p99}))
        if float(self.contract.p99_overhead_pct) > 0.0 and baseline_ms > 0.0:
            pct = 100.0 * (step_p99 / max(1e-9, baseline_ms) - 1.0)
            checks.append(("p99_overhead_pct", pct <= float(self.contract.p99_overhead_pct), {"step_p99_overhead_pct": pct}))
        ok = all(bool(c[1]) for c in checks) if checks else True
        failed = [c[0] for c in checks if not bool(c[1])]
        row = {
            "ts": float(time.time()),
            "rank": int(self.rank),
            "contract": asdict(self.contract),
            "ok": bool(ok),
            "failed": failed,
            "checks": [
                {"name": str(n), "pass": bool(p), "details": dict(d)} for (n, p, d) in checks
            ],
        }
        self.last = row
        if not ok:
            self.violations += 1
        return row

    def emit_proof(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        row = self.evaluate(snapshot)
        with open(self.proof_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        summary = {
            "rank": int(self.rank),
            "contract": asdict(self.contract),
            "violations": int(self.violations),
            "last": row,
        }
        with open(self.summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        return row
