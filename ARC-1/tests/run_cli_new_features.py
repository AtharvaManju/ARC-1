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
    print("CLI_NEW_FEATURES_OK")


if __name__ == "__main__":
    main()
