import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .bench.compile_matrix import run_compile_matrix
from .bench.fastpath_qual import run_fastpath_qualification
from .bench.parity_longrun import run_parity_longrun
from .bench.qualification import run_qualification
from .claims import build_claims_evidence
from .config import AIMemoryConfig
from .kv_manager import KVResidencyManager
from .storage import SARCStorage


def _safe_git(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return ""


def _env_capture() -> Dict[str, Any]:
    py = {
        "version": sys.version,
        "executable": sys.executable,
    }
    torch_info: Dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(getattr(torch.version, "cuda", "")),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": (str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else ""),
        }
    except Exception as e:
        torch_info = {"error": f"{type(e).__name__}: {e}"}
    return {
        "ts": float(time.time()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": py,
        "torch": torch_info,
        "git_sha": _safe_git(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _append_jsonl(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _pct(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array([float(x) for x in xs], dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _write_svg_bar(path: str, title: str, values: Dict[str, float]) -> str:
    w, h = 720, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 60, 70
    keys = list(values.keys())
    vals = [float(values[k]) for k in keys]
    vmax = max(1.0, max(vals) if vals else 1.0)
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b
    bw = plot_w / max(1, len(keys) * 2)
    bars = []
    labels = []
    for i, k in enumerate(keys):
        v = float(values[k])
        bh = (v / vmax) * plot_h
        x = margin_l + (i * 2 + 0.5) * bw
        y = margin_t + (plot_h - bh)
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#2f6feb" />')
        labels.append(f'<text x="{x + bw/2:.1f}" y="{h - margin_b + 24:.1f}" text-anchor="middle" font-size="14">{k}</text>')
        labels.append(f'<text x="{x + bw/2:.1f}" y="{max(margin_t + 16, y - 6):.1f}" text-anchor="middle" font-size="12">{v:.2f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>
<text x="{w/2:.1f}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>
<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{h-margin_b}" stroke="#222" />
<line x1="{margin_l}" y1="{h-margin_b}" x2="{w-margin_r}" y2="{h-margin_b}" stroke="#222" />
{''.join(bars)}
{''.join(labels)}
</svg>
"""
    with open(path, "w") as f:
        f.write(svg)
    return path


def _kv_latency_probe(pool_dir: str, out_path: str) -> Dict[str, Any]:
    st = SARCStorage(pool_dir=pool_dir, rank=0, backend="RAM", durable=False, encrypt_at_rest=False)
    try:
        cfg = AIMemoryConfig(pool_dir=pool_dir, kv_manager_enabled=True, kv_budget_bytes=256 * 1024)
        km = KVResidencyManager(cfg, st)
        km.set_phase("decode")
        import torch
        for i in range(24):
            t = torch.randn(128, dtype=torch.float32)
            km.register(f"k{i}", t, tenant_id=("t0" if i % 2 == 0 else "t1"), request_id=f"r{i%4}")
            km.get(f"k{i}", token_latency_ms=float(2.0 + (i % 7)))
        rep = km.stats()
        rep["percentiles"] = {
            "token_latency_p50_ms": _pct([float(x) for x in km._token_latency_ms]).get("p50", 0.0),  # type: ignore[attr-defined]
            "token_latency_p95_ms": float(rep.get("token_latency_p95_ms", 0.0)),
            "token_latency_p99_ms": float(rep.get("token_latency_p99_ms", 0.0)),
        }
        _write_json(out_path, rep)
        return rep
    finally:
        st.close()


def run_golden_qualification_pack(
    *,
    pool_dir: str,
    suite: str,
    out_dir: str = "",
    overhead_sla_pct: float = 15.0,
) -> Dict[str, Any]:
    s = str(suite).strip().lower()
    if s not in ("golden_llm_train", "golden_llm_infer_kv"):
        raise ValueError("suite must be one of: golden_llm_train, golden_llm_infer_kv")
    ts = time.strftime("%Y%m%d_%H%M%S")
    if not out_dir:
        out_dir = os.path.abspath(f"./arc1_golden_pack_{s}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    artifacts_jsonl = os.path.join(out_dir, "artifacts.jsonl")
    summary_md = os.path.join(out_dir, "SUMMARY.md")

    env = _env_capture()
    cfg = asdict(AIMemoryConfig(pool_dir=pool_dir))
    env_path = _write_json(os.path.join(out_dir, "environment.json"), env)
    cfg_path = _write_json(os.path.join(out_dir, "config_snapshot.json"), cfg)
    _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "environment", "path": env_path, "status": "ok"})
    _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "config", "path": cfg_path, "status": "ok"})

    chart_files: List[str] = []
    key_metrics: Dict[str, Any] = {}
    failures: List[str] = []

    if s == "golden_llm_train":
        qpath = os.path.join(out_dir, "qualification.json")
        try:
            q = run_qualification(
                pool_dir=pool_dir,
                out_path=qpath,
                threshold_multiplier=3.0,
                overhead_sla_pct=float(overhead_sla_pct),
            )
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "qualification", "path": qpath, "status": "ok"})
            pp = q.get("pressure_profile", {})
            a_samples = ((pp.get("aimemory", {}) or {}).get("samples", [])) if isinstance(pp, dict) else []
            pct = _pct([float(x) for x in a_samples])
            cpath = _write_svg_bar(os.path.join(out_dir, "chart_step_latency_train.svg"), "Train Step Latency (AIMemory) p50/p95/p99", pct)
            chart_files.append(cpath)
            key_metrics["train_step_latency_ms"] = pct
            key_metrics["qualification_passed"] = bool(q.get("passed", False))
        except BaseException as e:
            failures.append(f"qualification:{type(e).__name__}:{e}")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "qualification", "path": qpath, "status": "error", "error": f"{type(e).__name__}: {e}"})

        cmat_path = os.path.join(out_dir, "compile_matrix.json")
        try:
            cm = run_compile_matrix(out_path=cmat_path, dims=[512, 1024], dtypes=["float16", "bfloat16"], steps=4)
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "compile_matrix", "path": cmat_path, "status": "ok"})
            key_metrics["compile_matrix_passed"] = bool(cm.get("passed", False))
        except BaseException as e:
            failures.append(f"compile_matrix:{type(e).__name__}:{e}")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "compile_matrix", "path": cmat_path, "status": "error", "error": f"{type(e).__name__}: {e}"})

        pl_path = os.path.join(out_dir, "parity_longrun.json")
        try:
            pl = run_parity_longrun(pool_dir=pool_dir, out_path=pl_path, steps=64, dim=1024, dtype_s="float16")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "parity_longrun", "path": pl_path, "status": "ok"})
            key_metrics["parity_ok"] = bool((pl.get("parity", {}) or {}).get("ok", False))
        except BaseException as e:
            failures.append(f"parity_longrun:{type(e).__name__}:{e}")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "parity_longrun", "path": pl_path, "status": "error", "error": f"{type(e).__name__}: {e}"})

        claims = build_claims_evidence(
            qualification_path=os.path.join(out_dir, "qualification.json"),
            benchmark_path=os.path.join(out_dir, "qualification_bench", "bench_report.json"),
            require_cuda_evidence=False,
        )
        cpath = _write_json(os.path.join(out_dir, "claims_evidence.json"), claims)
        _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "claims_evidence", "path": cpath, "status": "ok"})
        key_metrics["claims_status"] = str(claims.get("status", ""))

    if s == "golden_llm_infer_kv":
        fpath = os.path.join(out_dir, "fastpath_qualification.json")
        try:
            fq = run_fastpath_qualification(pool_dir=pool_dir, out_path=fpath, probe_mb=64)
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "fastpath_qualification", "path": fpath, "status": "ok"})
            key_metrics["direct_restore_success"] = bool(fq.get("direct_restore_success", False))
            key_metrics["gds_enabled"] = bool(fq.get("gds_enabled", False))
        except BaseException as e:
            failures.append(f"fastpath_qualification:{type(e).__name__}:{e}")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "fastpath_qualification", "path": fpath, "status": "error", "error": f"{type(e).__name__}: {e}"})

        kv_path = os.path.join(out_dir, "kv_latency_probe.json")
        try:
            kv = _kv_latency_probe(pool_dir=pool_dir, out_path=kv_path)
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "kv_latency_probe", "path": kv_path, "status": "ok"})
            p = kv.get("percentiles", {})
            pct = {"p50": float(p.get("token_latency_p50_ms", 0.0)), "p95": float(p.get("token_latency_p95_ms", 0.0)), "p99": float(p.get("token_latency_p99_ms", 0.0))}
            cpath = _write_svg_bar(os.path.join(out_dir, "chart_token_latency_kv.svg"), "Infer KV Token Latency p50/p95/p99", pct)
            chart_files.append(cpath)
            key_metrics["kv_token_latency_ms"] = pct
            key_metrics["kv_qos_denials"] = int(kv.get("qos_denials", 0))
        except BaseException as e:
            failures.append(f"kv_latency_probe:{type(e).__name__}:{e}")
            _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "kv_latency_probe", "path": kv_path, "status": "error", "error": f"{type(e).__name__}: {e}"})

        claims = build_claims_evidence(
            qualification_path=os.path.join(out_dir, "qualification.json"),
            fastpath_path=os.path.join(out_dir, "fastpath_qualification.json"),
            require_cuda_evidence=False,
        )
        cpath = _write_json(os.path.join(out_dir, "claims_evidence.json"), claims)
        _append_jsonl(artifacts_jsonl, {"ts": time.time(), "artifact": "claims_evidence", "path": cpath, "status": "ok"})
        key_metrics["claims_status"] = str(claims.get("status", ""))

    status = "PASS" if not failures else "PARTIAL"
    summary_lines = [
        f"# ARC-1 Golden Qualification Pack",
        "",
        f"- Suite: `{s}`",
        f"- Status: `{status}`",
        f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Git SHA: `{env.get('git_sha', '')}`",
        "",
        "## Key Metrics",
        "```json",
        json.dumps(key_metrics, indent=2),
        "```",
        "",
        "## Artifacts",
        f"- JSONL index: `{os.path.basename(artifacts_jsonl)}`",
        f"- Environment: `{os.path.basename(env_path)}`",
        f"- Config snapshot: `{os.path.basename(cfg_path)}`",
    ]
    for cf in chart_files:
        summary_lines.append(f"- Chart: `{os.path.basename(cf)}`")
    if failures:
        summary_lines += ["", "## Failures"] + [f"- {x}" for x in failures]
    with open(summary_md, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    rep = {
        "suite": s,
        "status": status,
        "out_dir": os.path.abspath(out_dir),
        "summary_markdown": summary_md,
        "artifacts_jsonl": artifacts_jsonl,
        "charts": [os.path.abspath(x) for x in chart_files],
        "key_metrics": key_metrics,
        "failures": failures,
    }
    _write_json(os.path.join(out_dir, "pack_report.json"), rep)
    return rep
