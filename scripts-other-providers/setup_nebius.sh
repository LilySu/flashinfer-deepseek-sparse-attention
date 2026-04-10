#!/bin/bash
# Setup script for Nebius B200 VM instance.
# Run this after SSH'ing into a Nebius B200 VM.
#
# Prerequisites: Nebius B200 instance with CUDA 13.0 and Ubuntu 24.04
#
# Usage:
#   ssh <nebius-instance>
#   git clone https://github.com/LilySu/flashinfer-deepseek-sparse-attention.git
#   cd flashinfer-deepseek-sparse-attention
#   bash scripts-other-providers/setup_nebius.sh

set -e

echo "=== Nebius B200 Setup for DSA Indexer ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | sed 's/,//' || echo 'not found')"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"

# 1. Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA 13.0 support
echo ""
echo "[2/5] Installing PyTorch..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu130 2>/dev/null || \
  pip install torch  # fallback to default

# 3. Install flashinfer-bench and dependencies
echo ""
echo "[3/5] Installing flashinfer-bench..."
pip install flashinfer-bench numpy ninja

# 4. Download contest dataset
echo ""
echo "[4/5] Downloading contest dataset..."
if [ ! -d "mlsys26-contest" ]; then
    pip install huggingface_hub
    git lfs install
    git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
fi

# 5. Verify setup
echo ""
echo "[5/5] Verifying setup..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')
print(f'CUDA version: {torch.version.cuda}')
print(f'allow_tf32: {torch.backends.cuda.matmul.allow_tf32}')
try:
    import tvm_ffi
    print(f'tvm_ffi: {tvm_ffi.__version__}')
except: print('tvm_ffi: not installed')
try:
    import flashinfer_bench
    print(f'flashinfer_bench: installed')
except: print('flashinfer_bench: not installed')
"

echo ""
echo "=== Setup complete ==="
echo "To run the benchmark:"
echo "  source .venv/bin/activate"
echo "  bash scripts-other-providers/run_nebius.sh"
