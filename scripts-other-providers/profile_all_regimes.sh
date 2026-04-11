#!/bin/bash
# Profile current 2.99x solution across all 4 regimes + analyze
# Run: bash scripts-other-providers/profile_all_regimes.sh
set -e
cd /home/glm5/flashinfer-deepseek-sparse-attention
source flashinfer-deepseek-sparse-attention/.venv/bin/activate
mkdir -p ncu_reports

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Generate profiling targets for each regime
for REGIME in A B C D; do
  case $REGIME in
    A) B=1;  PAGES=10;  SEQLEN=640  ;;
    B) B=4;  PAGES=50;  SEQLEN=3200 ;;
    C) B=16; PAGES=70;  SEQLEN=4480 ;;
    D) B=31; PAGES=91;  SEQLEN=5824 ;;
  esac

  REPORT="ncu_reports/dsa_v3_${REGIME}_${TIMESTAMP}.ncu-rep"
  echo "=== Regime $REGIME: B=$B, pages=$PAGES, seq=$SEQLEN ==="

  cat > /tmp/ncu_regime_${REGIME}.py << PYEOF
import torch, sys, os
sys.path.insert(0, os.getcwd())
from solution.triton.binding import kernel as dsa_kernel

PAGE_SIZE = 64; HEAD_DIM = 128; NUM_HEADS = 64; TOPK = 2048
B = $B; num_pages = 11923; pages_per_seq = $PAGES; seq_len = $SEQLEN

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

print("=== PROFILED Regime $REGIME ===")
topk.fill_(-1)
dsa_kernel(q_fp8, k_cache, weights, seq_lens, bt, topk)
torch.cuda.synchronize()
print(f"Valid: {(topk[0]>=0).sum().item()}")
PYEOF

  # NCU: skip warmup launches, capture enough to see full pipeline
  SKIP=$((3 * B * 8 + 10))  # ~8 launches per batch * 3 warmup + setup
  COUNT=$((B * 8 + 10))     # 1 profiled iteration

  echo "  NCU: skip=$SKIP, count=$COUNT"
  ncu --set full \
      --clock-control none \
      --target-processes all \
      --launch-skip $SKIP \
      --launch-count $COUNT \
      --export "$REPORT" \
      python3 /tmp/ncu_regime_${REGIME}.py 2>&1 | tail -3

  echo "  Saved: $REPORT ($(du -h "$REPORT" | cut -f1))"
  echo ""
done

# Analyze all 4 reports
echo "=========================================="
echo "=== ANALYSIS ACROSS ALL REGIMES ==="
echo "=========================================="
for REGIME in A B C D; do
  REPORT=$(ls -t ncu_reports/dsa_v3_${REGIME}_*.ncu-rep 2>/dev/null | head -1)
  if [ -z "$REPORT" ]; then continue; fi

  case $REGIME in
    A) echo "--- Regime A: B=1, 10 pages ---" ;;
    B) echo "--- Regime B: B=4, 50 pages ---" ;;
    C) echo "--- Regime C: B=16, 70 pages ---" ;;
    D) echo "--- Regime D: B=31, 91 pages ---" ;;
  esac

  ncu --import "$REPORT" --csv --metrics gpu__time_duration.avg 2>/dev/null | python3 -c "
import csv,sys
from collections import defaultdict
reader = csv.DictReader(sys.stdin)
durations = defaultdict(list)
for row in reader:
    name = row.get('Kernel Name','')[:60]
    val = row.get('Metric Value','0')
    metric = row.get('Metric Name','')
    if metric == 'gpu__time_duration.avg':
        try: durations[name].append(float(val)/1000)
        except: pass
results = sorted([(sum(v),len(v),sum(v)/len(v),k) for k,v in durations.items()], reverse=True)
grand = sum(r[0] for r in results)
print(f'  {\"Kernel\":<55} {\"N\":>3} {\"Avg\":>7} {\"Total\":>8} {\"Pct\":>5}')
print('  ' + '-'*82)
for total,count,avg,name in results[:12]:
    short = name.replace('void at::native::','').replace('cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_','SGEMM_')
    print(f'  {short:<55} {count:>3} {avg:>6.1f}us {total:>7.1f}us {total/grand*100:>4.1f}%')
print(f'  TOTAL: {grand:.1f}us ({grand/1000:.2f}ms)')
" 2>/dev/null
  echo ""
done

echo "=== DONE ==="
