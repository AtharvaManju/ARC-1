#!/usr/bin/env bash
set -euo pipefail

echo "[verify_env] OS:"
uname -a || true
echo

echo "[verify_env] Python:"
python -V
python -c "import sys; print(sys.executable)"
echo

echo "[verify_env] Torch:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda version:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name(0))
PY
echo

echo "[verify_env] nvidia-smi:"
nvidia-smi || true
echo
