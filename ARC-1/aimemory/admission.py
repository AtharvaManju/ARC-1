import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class NodeProfile:
    node_id: str
    hbm_bytes: int
    nvme_write_mb_s: float
    nvme_read_mb_s: float
    numa_node: int = -1
    local_rank: int = -1
    topology: str = ""


@dataclass
class JobRequest:
    job_id: str
    model_fingerprint: str
    requested_hbm_bytes: int
    batch_size: int
    seq_len: int
    world_size: int
    policy: str = "balanced"
    never_oom: bool = False
    max_hbm_bytes: int = 0
    p99_overhead_ms: float = 0.0
    p99_overhead_pct: float = 0.0


@dataclass
class AdmissionDecision:
    decision: str  # admit | reshape | reject
    reason: str
    predicted_uplift_pct: float
    effective_hbm_bytes: int
    suggested_batch_size: int
    placement_hint: Dict[str, Any]


def workload_identity_key(model_fingerprint: str, world_size: int, seq_len: int, batch_size: int) -> str:
    raw = f"{model_fingerprint}:{int(world_size)}:{int(seq_len)}:{int(batch_size)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class AdmissionController:
    """
    Fleet-wide effective HBM admission controller.
    Uses workload+node features to predict headroom uplift and enforce contracts.
    """

    def __init__(self, policy_predictor=None):
        self.predictor = policy_predictor

    def _predict_uplift_pct(self, req: JobRequest, node: NodeProfile) -> float:
        if self.predictor is not None:
            try:
                pred = self.predictor.predict(
                    {
                        "model_fingerprint": req.model_fingerprint,
                        "world_size": int(req.world_size),
                        "seq_len": int(req.seq_len),
                        "batch_size": int(req.batch_size),
                        "policy": str(req.policy),
                        "hbm_bytes": int(node.hbm_bytes),
                        "nvme_write_mb_s": float(node.nvme_write_mb_s),
                        "nvme_read_mb_s": float(node.nvme_read_mb_s),
                    }
                )
                if pred is not None:
                    return max(0.0, min(300.0, float(pred.get("uplift_pct", 0.0))))
            except Exception:
                pass
        # Heuristic fallback.
        bw = min(float(node.nvme_write_mb_s), float(node.nvme_read_mb_s))
        policy = str(req.policy).lower()
        base = 10.0
        if bw >= 2500:
            base = 55.0
        elif bw >= 1200:
            base = 35.0
        elif bw >= 600:
            base = 20.0
        if policy == "max_headroom":
            base *= 1.35
        elif policy == "throughput":
            base *= 0.75
        if int(req.seq_len) >= 4096:
            base *= 1.10
        return max(0.0, min(250.0, base))

    def admit(self, req: JobRequest, node: NodeProfile) -> AdmissionDecision:
        uplift = self._predict_uplift_pct(req, node)
        effective = int(float(node.hbm_bytes) * (1.0 + uplift / 100.0))
        requested = int(req.requested_hbm_bytes)
        if int(req.max_hbm_bytes) > 0:
            requested = min(requested, int(req.max_hbm_bytes))
        placement = {
            "node_id": str(node.node_id),
            "numa_node": int(node.numa_node),
            "local_rank": int(node.local_rank),
            "topology": str(node.topology),
            "recommended_backend": "NVME_FILE" if min(node.nvme_write_mb_s, node.nvme_read_mb_s) >= 500 else "RAM",
        }
        if requested <= effective:
            return AdmissionDecision(
                decision="admit",
                reason="fits_predicted_effective_hbm",
                predicted_uplift_pct=float(uplift),
                effective_hbm_bytes=int(effective),
                suggested_batch_size=int(req.batch_size),
                placement_hint=placement,
            )
        # Try reshape suggestion.
        ratio = float(effective) / float(max(1, requested))
        sug_bs = max(1, int(float(req.batch_size) * ratio))
        if sug_bs >= 1 and sug_bs < int(req.batch_size):
            if req.never_oom and sug_bs < max(1, int(req.batch_size * 0.5)):
                return AdmissionDecision(
                    decision="reject",
                    reason="never_oom_contract_unachievable_on_node",
                    predicted_uplift_pct=float(uplift),
                    effective_hbm_bytes=int(effective),
                    suggested_batch_size=int(sug_bs),
                    placement_hint=placement,
                )
            return AdmissionDecision(
                decision="reshape",
                reason="requested_hbm_exceeds_effective_hbm",
                predicted_uplift_pct=float(uplift),
                effective_hbm_bytes=int(effective),
                suggested_batch_size=int(sug_bs),
                placement_hint=placement,
            )
        return AdmissionDecision(
            decision="reject",
            reason="requested_hbm_exceeds_effective_hbm",
            predicted_uplift_pct=float(uplift),
            effective_hbm_bytes=int(effective),
            suggested_batch_size=int(max(1, sug_bs)),
            placement_hint=placement,
        )

    def admit_many(self, req: JobRequest, nodes: List[NodeProfile]) -> Dict[str, Any]:
        decisions = []
        for n in nodes:
            d = self.admit(req, n)
            decisions.append({"node": asdict(n), "decision": asdict(d)})
        decisions.sort(key=lambda x: (0 if x["decision"]["decision"] == "admit" else (1 if x["decision"]["decision"] == "reshape" else 2), -float(x["decision"]["effective_hbm_bytes"])))
        best = decisions[0] if decisions else {}
        return {"request": asdict(req), "best": best, "candidates": decisions}


def load_node_profiles(path: str) -> List[NodeProfile]:
    with open(path, "r") as f:
        arr = json.load(f)
    out = []
    for x in arr:
        out.append(
            NodeProfile(
                node_id=str(x.get("node_id", "")),
                hbm_bytes=int(x.get("hbm_bytes", 0)),
                nvme_write_mb_s=float(x.get("nvme_write_mb_s", 0.0)),
                nvme_read_mb_s=float(x.get("nvme_read_mb_s", 0.0)),
                numa_node=int(x.get("numa_node", -1)),
                local_rank=int(x.get("local_rank", -1)),
                topology=str(x.get("topology", "")),
            )
        )
    return out
