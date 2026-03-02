import json
import os
import subprocess
import tempfile


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return p.stdout.strip()


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_cli_") as td:
        out1 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "static-plan-compile",
                "--pool-dir",
                td,
                "--name",
                "m",
                "--restore-order",
                "4,2,1",
                "--lookahead",
                "3",
            ]
        )
        j1 = json.loads(out1)
        assert "plan_key" in j1
        out2 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "static-plan-show",
                "--pool-dir",
                td,
                "--plan-key",
                j1["plan_key"],
            ]
        )
        j2 = json.loads(out2)
        assert len(j2.get("entries", [])) == 3

        out3 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "coord-sync",
                "--pool-dir",
                td,
                "--rank",
                "0",
                "--world-size",
                "1",
                "--step",
                "1",
                "--spills",
                "10",
                "--headroom-pct",
                "5.0",
            ]
        )
        _ = json.loads(out3)

        out4 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "backend-capabilities",
                "--pool-dir",
                td,
                "--probe-mb",
                "8",
                "--probe-seconds",
                "0.5",
            ]
        )
        j4 = json.loads(out4)
        assert "capabilities" in j4 and "recommended" in j4

        mpath = os.path.join(td, "metrics.json")
        with open(mpath, "w") as f:
            json.dump(
                {
                    "memory_headroom_pct": 12.0,
                    "oom_events": 0,
                    "reruns": 0,
                    "throughput": 100.0,
                    "step_samples_ms": [10.0, 11.0, 9.0],
                },
                f,
            )
        out5 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "roi-baseline",
                "--pool-dir",
                td,
                "--model",
                "m",
                "--batch-size",
                "8",
                "--seq-len",
                "2048",
                "--metrics-file",
                mpath,
            ]
        )
        _ = json.loads(out5)
        out6 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "roi-report",
                "--pool-dir",
                td,
                "--model",
                "m",
                "--batch-size",
                "8",
                "--seq-len",
                "2048",
                "--metrics-file",
                mpath,
            ]
        )
        j6 = json.loads(out6)
        assert "anti_gaming" in j6 and "step_ms_stats" in j6

        snap = os.path.join(td, "snap.json")
        with open(snap, "w") as f:
            json.dump(
                {
                    "memory_total_bytes": 100000,
                    "memory_free_bytes": 95000,
                    "oom_degrade_count": 0,
                    "safe_mode": False,
                    "step_p99_ms": 10.0,
                    "baseline_step_ms_ema": 9.0,
                },
                f,
            )
        contract = os.path.join(td, "contract.json")
        with open(contract, "w") as f:
            json.dump({"never_oom": True, "max_hbm_bytes": 10000, "p99_overhead_ms": 30.0, "policy": "balanced"}, f)
        out7 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "memory-slo-check",
                "--contract",
                contract,
                "--snapshot-file",
                snap,
                "--out-dir",
                td,
            ]
        )
        j7 = json.loads(out7)
        assert "ok" in j7

        nodes = os.path.join(td, "nodes.json")
        with open(nodes, "w") as f:
            json.dump(
                [
                    {"node_id": "n1", "hbm_bytes": 80000, "nvme_write_mb_s": 2000, "nvme_read_mb_s": 2000},
                    {"node_id": "n2", "hbm_bytes": 60000, "nvme_write_mb_s": 500, "nvme_read_mb_s": 500},
                ],
                f,
            )
        job = os.path.join(td, "job.json")
        with open(job, "w") as f:
            json.dump(
                {
                    "job_id": "j",
                    "model_fingerprint": "mfp",
                    "requested_hbm_bytes": 70000,
                    "batch_size": 8,
                    "seq_len": 2048,
                    "world_size": 1,
                    "policy": "balanced",
                },
                f,
            )
        out8 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "admission-check",
                "--job-file",
                job,
                "--nodes-file",
                nodes,
            ]
        )
        j8 = json.loads(out8)
        assert "best" in j8

        base = os.path.join(td, "base.json")
        cand = os.path.join(td, "cand.json")
        with open(base, "w") as f:
            json.dump({"loss_curve": [1.0, 0.8], "grad_norm_curve": [10.0, 9.0], "reproducibility_mode": True, "reproducibility_checksum": 1.23}, f)
        with open(cand, "w") as f:
            json.dump({"loss_curve": [1.0, 0.81], "grad_norm_curve": [10.1, 9.1], "reproducibility_mode": True, "reproducibility_checksum": 1.23}, f)
        out9 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "parity-certify",
                "--baseline",
                base,
                "--candidate",
                cand,
            ]
        )
        j9 = json.loads(out9)
        assert "ok" in j9

        feat = os.path.join(td, "feat.json")
        pol = os.path.join(td, "pol.json")
        with open(feat, "w") as f:
            json.dump({"model_fingerprint": "m", "world_size": 1, "seq_len": 2048, "batch_size": 8, "policy": "balanced", "hbm_bytes": 80000, "nvme_write_mb_s": 2000, "nvme_read_mb_s": 2000}, f)
        with open(pol, "w") as f:
            json.dump({"spill_min_bytes": 1048576, "pcc_lookahead": 2}, f)
        _ = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "policy-model-add",
                "--model-dir",
                os.path.join(td, "pmodel"),
                "--features-file",
                feat,
                "--policy-file",
                pol,
            ]
        )
        out10 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "policy-model-predict",
                "--model-dir",
                os.path.join(td, "pmodel"),
                "--features-file",
                feat,
            ]
        )
        _ = json.loads(out10)

        trace = os.path.join(td, "trace.json")
        with open(trace, "w") as f:
            json.dump({"summary": {"events": 1}}, f)
        out11 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "trace-report",
                "--trace-file",
                trace,
            ]
        )
        j11 = json.loads(out11)
        assert int(j11.get("events", 0)) == 1

        out12 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "ga-readiness",
                "--pool-dir",
                td,
            ]
        )
        j12 = json.loads(out12)
        assert "stage" in j12 and "checks" in j12

        out13 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "commercial-pack",
                "--pool-dir",
                td,
                "--out-dir",
                os.path.join(td, "commercial"),
                "--customer",
                "demo",
            ]
        )
        j13 = json.loads(out13)
        assert os.path.exists(j13["json"])
        assert os.path.exists(j13["markdown"])

        out14 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "compile-matrix",
                "--out",
                os.path.join(td, "compile_matrix.json"),
                "--dims",
                "64",
                "--dtypes",
                "float32",
                "--steps",
                "2",
            ]
        )
        j14 = json.loads(out14)
        assert "rows" in j14

        out15 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "parity-longrun",
                "--pool-dir",
                td,
                "--out",
                os.path.join(td, "parity_longrun.json"),
                "--steps",
                "4",
                "--dim",
                "64",
                "--dtype",
                "float32",
            ]
        )
        j15 = json.loads(out15)
        assert "parity" in j15

        out16 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "fastpath-qualify",
                "--pool-dir",
                td,
                "--out",
                os.path.join(td, "fastpath.json"),
                "--probe-mb",
                "4",
            ]
        )
        j16 = json.loads(out16)
        assert "backend_capabilities" in j16

        out17 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "migration-report",
                "--path",
                td,
            ]
        )
        j17 = json.loads(out17)
        assert "scanned_files" in j17

        out18 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "security-threat-model",
                "--out",
                os.path.join(td, "threat_model.json"),
            ]
        )
        j18 = json.loads(out18)
        assert "assets" in j18

        out19 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "security-audit",
                "--pool-dir",
                td,
            ]
        )
        j19 = json.loads(out19)
        assert "stage" in j19

        out20 = _run(
            [
                ".venv/bin/python",
                "-m",
                "arc1",
                "claims-evidence",
                "--qualification",
                os.path.join(td, "qualification.json"),
                "--fastpath",
                os.path.join(td, "fastpath.json"),
                "--out",
                os.path.join(td, "claims.json"),
                "--no-require-cuda",
            ]
        )
        j20 = json.loads(out20)
        assert "status" in j20
    print("CLI_NEW_FEATURES_OK")


if __name__ == "__main__":
    main()
