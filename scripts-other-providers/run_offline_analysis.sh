#!/bin/bash
# Run all analysis offline — NCU profiling + Triton matmul test
# Results saved to claude-outputs/ for review when back online
set -e
cd /home/glm5/flashinfer-deepseek-sparse-attention
source flashinfer-deepseek-sparse-attention/.venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="claude-outputs/${TIMESTAMP}_offline-ncu-triton-analysis.md"

echo "Starting offline analysis at $(date)"
echo "Results will be saved to: $OUTFILE"

{
echo "# Offline Analysis Results — $(date)"
echo ""

# Part 1: Triton matmul correctness
echo "## Part 1: Triton Matmul Correctness Test"
echo ""
echo '```'
python3 test_triton_matmul_correctness.py 2>&1
echo '```'
echo ""

# Part 2: NCU profiling across 4 regimes
echo "## Part 2: NCU Profiling (4 Regimes)"
echo ""
echo '```'
bash scripts-other-providers/profile_all_regimes.sh 2>&1
echo '```'
echo ""

echo "## Analysis completed at $(date)"
} > "$OUTFILE" 2>&1

echo "Done! Results saved to $OUTFILE"
