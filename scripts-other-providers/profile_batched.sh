#!/bin/bash
# Profile the batched pipeline at B=31 (largest workload)
set -e
cd /home/glm5/flashinfer-deepseek-sparse-attention
source flashinfer-deepseek-sparse-attention/.venv/bin/activate
mkdir -p ncu_reports

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="ncu_reports/dsa_batched_D_${TIMESTAMP}.ncu-rep"

echo "=== NCU Profiling: Batched Pipeline, B=31, 91 pages ==="

cat > /tmp/ncu_batched.py << 'PYEOF'
import torch, sys, os
sys.path.insert(0, os.getcwd())
from solution.triton.binding import kernel as dsa_kernel

PAGE_SIZE = 64; HEAD_DIM = 128; NUM_HEADS = 64; TOPK = 2048
B = 31; num_pages = 11923; pages_per_seq = 91; seq_len = pages_per_seq * PAGE_SIZE

def make_kv_cache(num_pages, device='cuda'):
    fp8_data = torch.randn(num_pages, 64, 128, device=device).clamp(-2, 2).to(torch.float8_e4m3fn)
    fp8_bytes = fp8_data.view(torch.uint8)
    scales = torch.rand(num_pages, 64, device=device, dtype=torch.float32) * 0.02 + 0.001
    scale_bytes = scales.view(torch.uint8).view(num_pages, 64, 4)
    flat = torch.zeros(num_pages, 8448, dtype=torch.uint8, device=device)
    flat[:, :8192] = fp8_bytes.reshape(num_pages, 8192)
    flat[:, 8192:] = scale_bytes.reshape(num_pages, 256)
    return flat.view(torch.int8).view(num_pages, 64, 1, 132)

k_cache = make_kv_cache(num_pages)
q_fp8 = torch.randn(B, NUM_HEADS, HEAD_DIM, device='cuda').clamp(-2, 2).to(torch.float8_e4m3fn)
weights = torch.randn(B, NUM_HEADS, device='cuda', dtype=torch.float32)
seq_lens = torch.tensor([seq_len]*B, device='cuda', dtype=torch.int32)
bt = torch.zeros(B, pages_per_seq, device='cuda', dtype=torch.int32)
for b in range(B):
    for pg in range(pages_per_seq): bt[b, pg] = (b*7+pg) % num_pages
topk = torch.full((B, TOPK), -1, device='cuda', dtype=torch.int32)

for _ in range(3):
    topk.fill_(-1)
    dsa_kernel(q_fp8, k_cache, weights, seq_lens, bt, topk)
torch.cuda.synchronize()

print("=== PROFILED ===")
topk.fill_(-1)
dsa_kernel(q_fp8, k_cache, weights, seq_lens, bt, topk)
torch.cuda.synchronize()
print(f"Valid: {(topk[0]>=0).sum().item()}")
PYEOF

# The batched pipeline has ~10 launches. Skip warmup (3 calls × ~10 = 30), capture all of profiled call.
ncu --set full \
    --clock-control none \
    --target-processes all \
    --launch-skip 30 \
    --launch-count 20 \
    --export "$REPORT" \
    python3 /tmp/ncu_batched.py 2>&1

echo "Saved: $REPORT ($(du -h "$REPORT" | cut -f1))"

# Analyze
echo ""
echo "=== ANALYSIS ==="
ncu --import "$REPORT" --csv 2>/dev/null | python3 -c "
import csv, sys
from collections import defaultdict

reader = csv.DictReader(sys.stdin)
kernels = defaultdict(dict)
for row in reader:
    kid = int(row.get('ID', '0'))
    name = row.get('Kernel Name', '')
    metric = row.get('Metric Name', '')
    val = row.get('Metric Value', '')
    block = row.get('Block Size', '')
    grid = row.get('Grid Size', '')
    unit = row.get('Metric Unit', '')
    if name and 'name' not in kernels[kid]:
        kernels[kid]['name'] = name
        kernels[kid]['block'] = block
        kernels[kid]['grid'] = grid
    if metric == 'Duration':
        kernels[kid]['dur'] = float(val.replace(',',''))
    elif metric == 'Compute (SM) Throughput':
        kernels[kid]['compute'] = val
    elif metric == 'Memory Throughput':
        kernels[kid]['memory'] = val
    elif 'Achieved Occupancy' in metric:
        kernels[kid]['occ'] = val

grand = sum(k.get('dur',0) for k in kernels.values())
print(f'{'#':>3} {'Kernel':<65} {'Grid':>15} {'Block':>10} {'Time':>7} {'Comp%':>6} {'Mem%':>7} {'Occ%':>6} {'Pct':>5}')
print('-' * 130)
for kid in sorted(kernels.keys()):
    k = kernels[kid]
    name = k.get('name','')[:65]
    dur = k.get('dur',0)
    comp = k.get('compute','-')
    mem = k.get('memory','-')
    occ = k.get('occ','-')
    grid = k.get('grid','')
    block = k.get('block','')
    pct = dur/grand*100 if grand > 0 else 0
    print(f'{kid:>3} {name:<65} {grid:>15} {block:>10} {dur:>6.1f}us {comp:>5}% {mem:>6}% {occ:>5}% {pct:>4.1f}%')
print(f'TOTAL: {grand:.1f}us ({grand/1000:.3f}ms)')
"
