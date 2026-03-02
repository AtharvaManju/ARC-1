import json
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
    print("CLI_NEW_FEATURES_OK")


if __name__ == "__main__":
    main()
