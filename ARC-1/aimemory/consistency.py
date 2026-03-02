import json
from typing import Dict, Any

from .storage import SARCStorage


def run_consistency_check(pool_dir: str, rank: int = 0, repair: bool = False, out_path: str = "") -> Dict[str, Any]:
    st = SARCStorage(
        pool_dir=pool_dir,
        rank=rank,
        backend="NVME_FILE",
        durable=False,
        encrypt_at_rest=False,
    )
    try:
        report = st.consistency_report(repair=bool(repair))
        comp = report.get("compatibility", {})
        if isinstance(comp, dict) and (not bool(comp.get("ok", True))):
            report["ok"] = False
            issues = list(report.get("issues", []))
            issues.append({"issue": "compatibility_check_failed", "details": comp})
            report["issues"] = issues
            report["count"] = int(len(issues))
    finally:
        st.close()
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report
