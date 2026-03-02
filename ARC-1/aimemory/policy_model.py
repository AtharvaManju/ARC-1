import json
import os
import math
import time
import hmac
import hashlib
from typing import Dict, Any, List, Optional

from .security import resolve_key_from_uri


def _stable(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class MemoryPolicyModel:
    """
    Lightweight data moat scaffold:
    stores workload/hardware -> policy samples and predicts near settings
    by nearest-neighbor distance over normalized features.
    """

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir or "./aimemory_policy_model")
        os.makedirs(self.model_dir, exist_ok=True)
        self.path = os.path.join(self.model_dir, "samples.json")
        self.samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self.samples = []
            return
        try:
            with open(self.path, "r") as f:
                obj = json.load(f)
            self.samples = list(obj.get("samples", []))
        except Exception:
            self.samples = []

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"samples": self.samples}, f, indent=2)

    def add_sample(self, features: Dict[str, Any], policy: Dict[str, Any], score: float = 0.0):
        self.samples.append(
            {
                "ts": float(time.time()),
                "features": dict(features),
                "policy": dict(policy),
                "score": float(score),
            }
        )
        if len(self.samples) > 50000:
            self.samples = self.samples[-25000:]
        self._save()

    def _dist(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ks = [
            "world_size",
            "seq_len",
            "batch_size",
            "hbm_bytes",
            "nvme_write_mb_s",
            "nvme_read_mb_s",
        ]
        d = 0.0
        for k in ks:
            av = float(a.get(k, 0.0))
            bv = float(b.get(k, 0.0))
            scale = max(1.0, abs(bv))
            d += ((av - bv) / scale) ** 2
        if str(a.get("policy", "")) != str(b.get("policy", "")):
            d += 0.25
        if str(a.get("model_fingerprint", "")) != str(b.get("model_fingerprint", "")):
            d += 0.50
        return math.sqrt(d)

    def predict(self, features: Dict[str, Any], k: int = 5) -> Optional[Dict[str, Any]]:
        if not self.samples:
            return None
        rows = []
        for s in self.samples:
            rows.append((self._dist(features, s.get("features", {})), s))
        rows.sort(key=lambda x: x[0])
        top = rows[: max(1, int(k))]
        # weighted average over numeric policy params.
        wsum = 0.0
        out: Dict[str, float] = {}
        for d, s in top:
            w = 1.0 / max(1e-6, d)
            pol = dict(s.get("policy", {}))
            for kk, vv in pol.items():
                try:
                    fv = float(vv)
                except Exception:
                    continue
                out[kk] = out.get(kk, 0.0) + w * fv
            wsum += w
        if wsum <= 0.0:
            return None
        pred = {k: (v / wsum) for k, v in out.items()}
        # keep common integral policy params stable.
        for ik in ("spill_min_bytes", "pcc_lookahead", "io_workers", "max_queue"):
            if ik in pred:
                pred[ik] = int(max(1, round(float(pred[ik]))))
        return {"policy": pred, "uplift_pct": float(pred.get("predicted_uplift_pct", 0.0))}

    def export_signed_pack(self, out_path: str, key_uri: str = "") -> str:
        pack = {
            "schema_version": 1,
            "created_ts": float(time.time()),
            "sample_count": int(len(self.samples)),
            "samples": self.samples[-10000:],
            "signature": "",
            "signature_alg": "",
            "payload_digest": "",
        }
        payload = {"schema_version": pack["schema_version"], "created_ts": pack["created_ts"], "sample_count": pack["sample_count"], "samples": pack["samples"]}
        stable = _stable(payload)
        dig = hashlib.sha256(stable.encode()).hexdigest()
        pack["payload_digest"] = dig
        if key_uri:
            key = resolve_key_from_uri(key_uri)
            pack["signature"] = hmac.new(key, stable.encode(), hashlib.sha256).hexdigest()
            pack["signature_alg"] = "hmac-sha256"
        with open(out_path, "w") as f:
            json.dump(pack, f, indent=2)
        return out_path

    def import_signed_pack(self, in_path: str, key_uri: str = "", require_signature: bool = False) -> Dict[str, Any]:
        with open(in_path, "r") as f:
            pack = json.load(f)
        payload = {"schema_version": pack.get("schema_version", 1), "created_ts": pack.get("created_ts", 0.0), "sample_count": pack.get("sample_count", 0), "samples": pack.get("samples", [])}
        stable = _stable(payload)
        got = hashlib.sha256(stable.encode()).hexdigest()
        if got != str(pack.get("payload_digest", "")):
            return {"ok": False, "reason": "payload_digest_mismatch"}
        sig = str(pack.get("signature", ""))
        if require_signature and (not sig):
            return {"ok": False, "reason": "missing_signature"}
        if sig:
            if not key_uri:
                return {"ok": False, "reason": "signature_present_no_key"}
            key = resolve_key_from_uri(key_uri)
            calc = hmac.new(key, stable.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(calc, sig):
                return {"ok": False, "reason": "invalid_signature"}
        self.samples.extend(list(payload.get("samples", [])))
        if len(self.samples) > 50000:
            self.samples = self.samples[-25000:]
        self._save()
        return {"ok": True, "imported": int(len(payload.get("samples", [])))}
