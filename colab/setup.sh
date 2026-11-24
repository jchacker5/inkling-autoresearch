#!/usr/bin/env bash
# Run the harness on a Linux CUDA GPU (e.g. a Colab A100) with MLX's CUDA backend.
#   !bash colab/setup.sh            (from a clone of the repo)
# A100 results are a separate series: wall-clock budgets buy far more steps than on the M4, so they go
# to results/ledger-a100.tsv (set INKLING_LEDGER) and are never compared row-for-row with M4 rows.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q "mlx[cuda12]==0.32.4" "numpy>=2.2" "pyarrow>=21" "requests>=2.32" pytest
python - <<'PY'
import mlx.core as mx
print("mlx", mx.__version__, "device:", mx.default_device())
a = mx.random.normal((4096, 4096)); mx.eval(a @ a); print("matmul ok")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python prepare.py
