#!/usr/bin/env bash
set -euo pipefail

bash scripts/verify_env.sh || true
rm -rf build dist *.egg-info

python -m pip install --upgrade pip setuptools wheel
python -m pip wheel . -w dist

echo "[build_wheel] built:"
ls -lh dist
