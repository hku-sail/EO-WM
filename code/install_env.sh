#!/usr/bin/env bash
# Linux / NVIDIA CUDA 11.8 / Python 3.10 environment.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"
ENV_NAME="${ENV_NAME:-eo-wm}"
command -v conda >/dev/null || { echo "Install Miniconda/Anaconda and expose conda on PATH." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# scikit-image 0.18.3 has no standard Linux cp310 wheel. Build against the
# installed NumPy with the legacy setuptools/Cython toolchain if needed.
python -m pip install numpy==1.26.4 scipy==1.13.1
if ! python -m pip install --only-binary=:all: scikit-image==0.18.3; then
  python -m pip install setuptools==59.8.0 "Cython<3" wheel pythran
  python -m pip install scikit-image==0.18.3 --no-build-isolation
  python -m pip install --upgrade setuptools
fi
python -m pip install -r requirements.txt
# CUDA extensions require a matching CUDA toolkit (nvcc) or compatible wheels.
python -m pip install flash-attn==2.6.3 --no-build-isolation --no-deps
python -m pip install xformers==0.0.27.post2 --no-deps --index-url https://download.pytorch.org/whl/cu118
# ColossalAI's upstream metadata pins a different transformers release; the
# EO-WM environment uses 4.41.2. Install its runtime dependencies
# explicitly above and keep this intentional compatibility override local.
python -m pip install colossalai==0.5.0 --no-deps
python -m pip install -e . --no-deps
python -m pip install -e ./eo-vae --no-deps
python -c 'import torch, opensora, eo_vae; print(torch.__version__); print(opensora.__file__); print(list(eo_vae.__path__))'
echo "Installed ${ENV_NAME}. Run: conda activate ${ENV_NAME}"
