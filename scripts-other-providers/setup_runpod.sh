#!/bin/bash
# Setup script for RunPod B200 instance.
#
# NOTE: RunPod's latest template has CUDA 12.8, not 13.0.
# Our kernel requires cuda_fp8.h and SM100 support.
# CUDA 12.8 supports SM100 but PyTorch version may differ.
# Use the pytorch/pytorch:2.11.0-cuda13.0 image if available,
# or install PyTorch 2.11 manually.
#
# Usage:
#   ssh <runpod-instance>
#   git clone https://github.com/LilySu/flashinfer-deepseek-sparse-attention.git
#   cd flashinfer-deepseek-sparse-attention
#   bash scripts-other-providers/setup_runpod.sh

set -e

echo "=== RunPod B200 Setup for DSA Indexer ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | sed 's/,//' || echo 'not found')"

# Check CUDA version
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | sed 's/,//')
if [[ "$CUDA_VER" != "13.0"* ]]; then
    echo ""
    echo "WARNING: CUDA $CUDA_VER detected. Contest uses CUDA 13.0."
    echo "Our CUDA kernel uses <cuda_fp8.h> which requires CUDA 12.8+."
    echo "SM100 support requires CUDA 12.8+. Should work but test carefully."
    echo ""
fi

# 1. Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch
echo "[2/5] Installing PyTorch..."
pip install --upgrade pip
# Try CUDA 13.0 wheels first, fall back to 12.8, then default
pip install torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null || \
  pip install torch --index-url https://download.pytorch.org/whl/cu128 2>/dev/null || \
  pip install torch

# 3. Install flashinfer-bench
echo "[3/5] Installing flashinfer-bench..."
pip install flashinfer-bench numpy ninja

# 4. Download dataset
echo "[4/5] Downloading contest dataset..."
if [ ! -d "mlsys26-contest" ]; then
    pip install huggingface_hub
    git lfs install
    git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
fi

# 5. Verify
echo "[5/5] Verifying setup..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
"

echo ""
echo "=== Setup complete ==="
echo "Run: source .venv/bin/activate && bash scripts-other-providers/run_nebius.sh"
echo "(run_nebius.sh works on any provider — it's provider-agnostic)"
