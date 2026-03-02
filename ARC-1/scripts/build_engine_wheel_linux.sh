#!/usr/bin/env bash
set -euo pipefail

bash scripts/verify_env.sh || true
rm -rf build dist *.egg-info

python -m pip install --upgrade pip setuptools wheel
python setup_engine.py bdist_wheel

echo "[build_engine_wheel] built:"
ls -lh dist
