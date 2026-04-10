#!/bin/bash
# Check if flashinfer-bench evaluator has been updated (correctness check changes)
# Run: bash scripts/check_evaluator_update.sh
# Or schedule: watch -n 43200 bash scripts/check_evaluator_update.sh  (every 12 hours)

echo "=== FlashInfer Evaluator Check — $(date) ==="

# 1. Check latest flashinfer-bench version on PyPI
echo ""
echo "[1] Latest flashinfer-bench on PyPI:"
pip index versions flashinfer-bench 2>/dev/null | head -3 || \
  curl -s https://pypi.org/pypi/flashinfer-bench/json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Latest: {d['info']['version']}\")" 2>/dev/null || \
  echo "  Could not check PyPI"

# 2. Check EVALUATION.md for DSA indexer tolerance flags
echo ""
echo "[2] EVALUATION.md DSA indexer command:"
curl -s https://raw.githubusercontent.com/flashinfer-ai/flashinfer-bench-starter-kit/main/EVALUATION.md 2>/dev/null | \
  grep -A3 "dsa_topk_indexer" | head -5 || echo "  Could not fetch"

# 3. Check recent flashinfer-bench releases
echo ""
echo "[3] Recent flashinfer-bench releases:"
curl -s "https://api.github.com/repos/flashinfer-ai/flashinfer-bench/releases?per_page=3" 2>/dev/null | \
  python3 -c "import sys,json; releases=json.load(sys.stdin); [print(f\"  {r['tag_name']} — {r['published_at'][:10]} — {r['name']}\") for r in releases[:3]]" 2>/dev/null || \
  echo "  Could not fetch"

# 4. Check recent PRs mentioning evaluator/correctness
echo ""
echo "[4] Recent PRs (evaluator/correctness):"
curl -s "https://api.github.com/search/issues?q=repo:flashinfer-ai/flashinfer-bench+type:pr+correctness+OR+evaluator+OR+tolerance+OR+atol&sort=updated&order=desc&per_page=5" 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); [print(f\"  #{i['number']} {i['title'][:60]} — {i['updated_at'][:10]}\") for i in d.get('items',[])]" 2>/dev/null || \
  echo "  Could not fetch"

# 5. Check if atol/rtol appear in DSA indexer evaluation
echo ""
echo "[5] Checking for tolerance flags in DSA indexer eval:"
curl -s https://raw.githubusercontent.com/flashinfer-ai/flashinfer-bench-starter-kit/main/EVALUATION.md 2>/dev/null | \
  grep -B1 -A5 "dsa_topk_indexer" | grep -i "atol\|rtol\|tolerance\|matched" && \
  echo "  *** TOLERANCE FLAGS FOUND — RE-TEST BATCHED APPROACHES! ***" || \
  echo "  No tolerance flags for DSA indexer (still exact match)"

echo ""
echo "=== Done ==="
