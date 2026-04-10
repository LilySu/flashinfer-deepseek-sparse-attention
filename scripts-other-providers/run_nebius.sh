#!/bin/bash
# Run the DSA Indexer benchmark on a Nebius B200 VM.
# Equivalent to: modal run scripts/run_modal_simple.py
#
# Usage:
#   source .venv/bin/activate
#   bash scripts-other-providers/run_nebius.sh [--quick]

set -e

QUICK=${1:-""}
DATASET_PATH="./mlsys26-contest"

if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    echo "Run setup_nebius.sh first, or: git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest"
    exit 1
fi

echo "=== DSA Indexer Benchmark on Nebius B200 ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Pack solution
echo ""
echo "Packing solution..."
python3 scripts/pack_solution.py 2>/dev/null || python3 -c "
import json, tomllib
from pathlib import Path

with open('config.toml', 'rb') as f:
    config = tomllib.load(f)

solution_config = config['solution']
build_config = config['build']
language = build_config['language']
source_dir = Path('solution') / ('triton' if language == 'triton' else 'cuda')

sources = {}
for f in source_dir.iterdir():
    if f.is_file() and f.suffix in ('.py', '.cu', '.cuh'):
        sources[f.name] = f.read_text()

solution = {
    'name': solution_config['name'],
    'definition': solution_config['definition'],
    'author': solution_config['author'],
    'spec': {
        'language': language,
        'target_hardware': ['cuda'],
        'entry_point': build_config['entry_point'],
        'dependencies': [],
        'destination_passing_style': build_config.get('destination_passing_style', True),
    },
    'sources': [{'path': k, 'content': v} for k, v in sources.items()],
}

Path('solution.json').write_text(json.dumps(solution, indent=2))
print(f'  Name: {solution[\"name\"]}')
print(f'  Definition: {solution[\"definition\"]}')
print(f'  Sources: {[s[\"path\"] for s in solution[\"sources\"]]}')
print(f'  Saved: solution.json')
"

# Run benchmark
echo ""
echo "Running benchmark..."

if [ "$QUICK" = "--quick" ]; then
    echo "(Quick mode: 3 workloads)"
    python3 -c "
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
import json

solution = Solution.model_validate(json.loads(open('solution.json').read()))
config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
trace_set = TraceSet.from_path('$DATASET_PATH')

definition = trace_set.definitions[solution.definition]
workloads = trace_set.workloads.get(solution.definition, [])[:3]

bench_trace_set = TraceSet(
    root=trace_set.root,
    definitions={definition.name: definition},
    solutions={definition.name: [solution]},
    workloads={definition.name: workloads},
    traces={definition.name: []},
)

benchmark = Benchmark(bench_trace_set, config)
result_trace_set = benchmark.run_all(dump_traces=True)

traces = result_trace_set.traces.get(definition.name, [])
passed = failed = 0
for trace in traces:
    if trace.evaluation:
        status = trace.evaluation.status.value
        line = f'  {trace.workload.uuid[:8]}  {status}'
        if trace.evaluation.performance:
            line += f'  {trace.evaluation.performance.latency_ms:.3f}ms'
            line += f'  {trace.evaluation.performance.speedup_factor:.2f}x'
        if trace.evaluation.correctness:
            line += f'  abs_err={trace.evaluation.correctness.max_absolute_error:.2e}'
        print(line)
        if 'PASS' in status.upper() or 'CORRECT' in status.upper():
            passed += 1
        else:
            failed += 1

print(f'\nResults: {passed}/{passed+failed} passed, {failed} failed')
"
else
    echo "(Full: all workloads)"
    python3 -c "
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
import json

solution = Solution.model_validate(json.loads(open('solution.json').read()))
config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
trace_set = TraceSet.from_path('$DATASET_PATH')

definition = trace_set.definitions[solution.definition]
workloads = trace_set.workloads.get(solution.definition, [])

bench_trace_set = TraceSet(
    root=trace_set.root,
    definitions={definition.name: definition},
    solutions={definition.name: [solution]},
    workloads={definition.name: workloads},
    traces={definition.name: []},
)

benchmark = Benchmark(bench_trace_set, config)
result_trace_set = benchmark.run_all(dump_traces=True)

traces = result_trace_set.traces.get(definition.name, [])
passed = failed = 0
speedups = []
for trace in traces:
    if trace.evaluation:
        status = trace.evaluation.status.value
        line = f'  {trace.workload.uuid[:8]}  {status}'
        if trace.evaluation.performance:
            line += f'  {trace.evaluation.performance.latency_ms:.3f}ms'
            line += f'  {trace.evaluation.performance.speedup_factor:.2f}x'
            speedups.append(trace.evaluation.performance.speedup_factor)
        if trace.evaluation.correctness:
            line += f'  abs_err={trace.evaluation.correctness.max_absolute_error:.2e}'
        print(line)
        if 'PASS' in status.upper() or 'CORRECT' in status.upper():
            passed += 1
        else:
            failed += 1
            # Stop early on first failure to save time
            if failed >= 3:
                print(f'\n*** STOPPING EARLY: {failed} failures ***')
                break

print(f'\nResults: {passed}/{passed+failed} passed, {failed} failed')
if speedups:
    print(f'Average speedup: {sum(speedups)/len(speedups):.2f}x')
    print(f'Min: {min(speedups):.2f}x, Max: {max(speedups):.2f}x')
"
fi
