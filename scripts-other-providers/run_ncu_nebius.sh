#!/bin/bash
# Run NCU profiling on Nebius B200 VM.
# Equivalent to: modal run scripts/test_ncu_profile_modal.py
#
# Usage:
#   source .venv/bin/activate
#   bash scripts-other-providers/run_ncu_nebius.sh [basic|full|roofline]

set -e

PROFILE_SET=${1:-"basic"}
REPORT_PATH="./ncu_reports/dsa_profile_$(date +%Y%m%d_%H%M%S).ncu-rep"
mkdir -p ncu_reports

echo "=== NCU Profiling on Nebius B200 ==="
echo "Profile set: $PROFILE_SET"
echo "Output: $REPORT_PATH"

# Create profiling target
cat > /tmp/ncu_target.py << 'PYEOF'
import torch
import sys, os
sys.path.insert(0, os.getcwd())

# Import our kernel
from solution.triton.binding import kernel as dsa_kernel

PAGE_SIZE = 64; HEAD_DIM = 128; NUM_HEADS = 64; TOPK = 2048

def make_kv_cache(num_pages, device='cuda'):
    fp8_data = torch.randn(num_pages, 64, 128, device=device).clamp(-2, 2).to(torch.float8_e4m3fn)
    fp8_bytes = fp8_data.view(torch.uint8)
    scales = torch.rand(num_pages, 64, device=device, dtype=torch.float32) * 0.02 + 0.001
    scale_bytes = scales.view(torch.uint8).view(num_pages, 64, 4)
    flat = torch.zeros(num_pages, 8448, dtype=torch.uint8, device=device)
    flat[:, :8192] = fp8_bytes.reshape(num_pages, 8192)
    flat[:, 8192:] = scale_bytes.reshape(num_pages, 256)
    return flat.view(torch.int8).view(num_pages, 64, 1, 132)

B = 4; num_pages = 50
k_cache = make_kv_cache(num_pages)
q_fp8 = torch.randn(B, NUM_HEADS, HEAD_DIM, device='cuda').clamp(-2, 2).to(torch.float8_e4m3fn)
weights = torch.randn(B, NUM_HEADS, device='cuda', dtype=torch.float32)
seq_lens = torch.tensor([3200]*B, device='cuda', dtype=torch.int32)
bt = torch.zeros(B, 50, device='cuda', dtype=torch.int32)
for b in range(B):
    for pg in range(50): bt[b, pg] = (b*7+pg) % num_pages
topk = torch.full((B, TOPK), -1, device='cuda', dtype=torch.int32)

# Warmup
for _ in range(3):
    topk.fill_(-1)
    dsa_kernel(q_fp8, k_cache, weights, seq_lens, bt, topk)
torch.cuda.synchronize()

# Profiled run
print("=== PROFILED ===")
topk.fill_(-1)
dsa_kernel(q_fp8, k_cache, weights, seq_lens, bt, topk)
torch.cuda.synchronize()
print(f"Valid: {(topk[0]>=0).sum().item()}")
PYEOF

# Run NCU
ncu --set "$PROFILE_SET" \
    --clock-control none \
    --target-processes all \
    --launch-skip 20 \
    --launch-count 50 \
    --export "$REPORT_PATH" \
    python3 /tmp/ncu_target.py

echo ""
echo "Report saved: $REPORT_PATH ($(du -h "$REPORT_PATH" | cut -f1))"
echo "View: ncu-ui $REPORT_PATH"
