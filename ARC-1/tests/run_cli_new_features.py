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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
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
                "aimemory",
                "trace-report",
                "--trace-file",
                trace,
            ]
        )
        j11 = json.loads(out11)
        assert int(j11.get("events", 0)) == 1
    print("CLI_NEW_FEATURES_OK")


if __name__ == "__main__":
    main()
