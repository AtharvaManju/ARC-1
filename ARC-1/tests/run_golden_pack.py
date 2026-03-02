import os
import tempfile

from aimemory.golden_pack import run_golden_qualification_pack


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_golden_") as td:
        rep = run_golden_qualification_pack(
            pool_dir=td,
            suite="golden_llm_infer_kv",
            out_dir=os.path.join(td, "pack"),
            overhead_sla_pct=15.0,
        )
        assert rep["status"] in ("PASS", "PARTIAL")
        assert os.path.exists(rep["summary_markdown"])
        assert os.path.exists(rep["artifacts_jsonl"])
        assert len(rep.get("charts", [])) >= 1
    print("GOLDEN_PACK_OK")


if __name__ == "__main__":
    main()
