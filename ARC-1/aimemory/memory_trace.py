import json
import os
import time
from collections import Counter, deque
from typing import Dict, Any, Deque, List


class MemoryTraceRecorder:
    def __init__(self, max_events: int = 20000, out_path: str = ""):
        self.events: Deque[Dict[str, Any]] = deque(maxlen=max(100, int(max_events)))
        self.out_path = str(out_path or "")

    def add(self, event: Dict[str, Any]):
        row = dict(event)
        row.setdefault("ts", float(time.time()))
        self.events.append(row)

    def trace_pack(
        self,
        *,
        key: str,
        nbytes: int,
        decision: str,
        reason: str,
        step: int,
        pack_idx: int,
    ):
        self.add(
            {
                "kind": "pack",
                "key": str(key),
                "nbytes": int(nbytes),
                "decision": str(decision),
                "reason": str(reason),
                "step": int(step),
                "pack_idx": int(pack_idx),
            }
        )

    def trace_restore(
        self,
        *,
        key: str,
        nbytes: int,
        source: str,
        stall_ms: float,
        step: int,
        pack_idx: int,
    ):
        self.add(
            {
                "kind": "restore",
                "key": str(key),
                "nbytes": int(nbytes),
                "source": str(source),
                "stall_ms": float(stall_ms),
                "step": int(step),
                "pack_idx": int(pack_idx),
            }
        )

    def summarize(self) -> Dict[str, Any]:
        arr: List[Dict[str, Any]] = list(self.events)
        if not arr:
            return {"events": 0, "recommendations": []}
        pack = [e for e in arr if e.get("kind") == "pack"]
        rst = [e for e in arr if e.get("kind") == "restore"]
        by_reason = Counter([str(e.get("reason", "")) for e in pack])
        top_peaks = sorted(pack, key=lambda x: int(x.get("nbytes", 0)), reverse=True)[:16]
        stall_ms = [float(e.get("stall_ms", 0.0)) for e in rst]
        stall_total = sum(stall_ms)
        stall_p99 = 0.0
        if stall_ms:
            s = sorted(stall_ms)
            stall_p99 = s[min(len(s) - 1, int(0.99 * (len(s) - 1)))]
        recs = []
        if by_reason.get("small_tensor_inline", 0) > max(10, len(pack) // 2):
            recs.append("Most tensors are below spill threshold; consider lowering spill_min_bytes.")
        if by_reason.get("spill_denied:step_budget", 0) + by_reason.get("spill_denied:window_budget", 0) > 0:
            recs.append("Spill budgets are frequently denying tensors; increase per-step/window spill budgets.")
        if stall_p99 > 20.0:
            recs.append("Restore stall p99 is high; increase prefetch lookahead or reduce IO queue contention.")
        if stall_total <= 1.0 and len(pack) > 0:
            recs.append("Observed low restore stalls; policy can bias toward throughput.")
        return {
            "events": int(len(arr)),
            "pack_events": int(len(pack)),
            "restore_events": int(len(rst)),
            "top_peak_tensors": top_peaks,
            "decision_reasons": dict(by_reason),
            "stall_total_ms": float(stall_total),
            "stall_p99_ms": float(stall_p99),
            "recommendations": recs,
        }

    def flush(self, path: str = "") -> str:
        out = path or self.out_path
        if not out:
            return ""
        p = os.path.abspath(out)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump({"summary": self.summarize(), "events": list(self.events)}, f, indent=2)
        return p
