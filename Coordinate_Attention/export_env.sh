#!/usr/bin/env bash
set -e

PROJ="/mnt/synology/wen.zhou/2026.05/Coordinate_Attention"
ENV_PREFIX="/home/wen.zhou/miniforge3/envs/py310_local"
CONDA_BIN="/home/wen.zhou/miniconda3/bin/conda"

mkdir -p "$PROJ"

"$CONDA_BIN" list --explicit -p "$ENV_PREFIX" > "$PROJ/conda-spec.txt"

"$ENV_PREFIX/bin/python" - <<'PY' > "$PROJ/runtime.txt"
import sys, platform
print("exe:", sys.executable)
print("version:", sys.version)
print("platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
except Exception as e:
    print("torch import failed:", repr(e))
PY

cat > "$PROJ/environment.yml" <<'YML'
name: coordinate_attention_lock
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - pillow
  - pip
YML

ls -lh "$PROJ/conda-spec.txt" "$PROJ/runtime.txt" "$PROJ/environment.yml"
