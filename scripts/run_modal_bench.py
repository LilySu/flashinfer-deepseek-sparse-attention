"""Run a packed DSA solution against contest workloads on Modal B200.

Matches yongwww's recommended eval env: flashinfer-bench from source (main),
CUDA 13.0 devel image (has nvcc — needed even for Python-only solutions
because some dependencies expect it).

Usage:
    KERNEL=indexer   uvx modal run scripts/run_modal_bench.py
    KERNEL=attention uvx modal run scripts/run_modal_bench.py

Pre-req: trace data must already be uploaded to the `flashinfer-trace`
modal volume (see scripts/run_modal.py header for one-time setup).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KERNEL = os.environ.get("KERNEL", "indexer")

import modal

app = modal.App(f"flashinfer-dsa-{KERNEL}-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install("torch", "numpy", "ninja")
    .run_commands(
        "git clone https://github.com/flashinfer-ai/flashinfer-bench.git /opt/flashinfer-bench",
        "cd /opt/flashinfer-bench && pip install -v -e .",
    )
    .add_local_dir(
        str(PROJECT_ROOT / KERNEL / "solution" / "python"),
        remote_path="/root/submission/solution/python",
    )
    .add_local_file(
        str(PROJECT_ROOT / KERNEL / "config.toml"),
        remote_path="/root/submission/config.toml",
    )
)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=3600,
    volumes={TRACE_SET_PATH: trace_volume},
)
def run_benchmark() -> dict:
    """Pack solution in-container, run benchmark against DSA indexer workloads."""
    import tomllib
    from pathlib import Path as P

    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet, BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    submission_root = P("/root/submission")
    with open(submission_root / "config.toml", "rb") as f:
        cfg = tomllib.load(f)

    sol_cfg = cfg["solution"]
    build_cfg = cfg["build"]

    spec = BuildSpec(
        language=build_cfg["language"],
        target_hardware=["cuda"],
        entry_point=build_cfg["entry_point"],
        destination_passing_style=build_cfg.get("destination_passing_style", True),
    )

    solution = pack_solution_from_files(
        path=str(submission_root / "solution" / "python"),
        spec=spec,
        name=sol_cfg["name"],
        definition=sol_cfg["definition"],
        author=sol_cfg["author"],
    )
    print(f"Packed solution: {solution.name} ({solution.definition})")
    print(f"  language={solution.spec.language} entry={solution.spec.entry_point}")
    print(f"  destination_passing_style={solution.spec.destination_passing_style}")
    print(f"  sources: {len(solution.sources)} files")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        raise RuntimeError(
            f"Definition '{solution.definition}' not in trace set. "
            f"Available: {sorted(trace_set.definitions.keys())}"
        )

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    print(f"\nRunning against {len(workloads)} workloads")

    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    config = BenchmarkConfig(warmup_runs=3, iterations=50, num_trials=3)
    benchmark = Benchmark(bench_ts, config)
    result_ts = benchmark.run_all(dump_traces=False)

    traces = result_ts.traces.get(definition.name, [])
    results = []
    pass_count = 0
    fail_count = 0
    for trace in traces:
        ev = trace.evaluation
        if ev is None:
            continue
        entry = {"status": ev.status.value}
        if ev.status.value == "PASSED":
            pass_count += 1
        else:
            fail_count += 1
        if ev.performance is not None:
            entry["latency_ms"] = ev.performance.latency_ms
            entry["ref_latency_ms"] = ev.performance.reference_latency_ms
            entry["speedup"] = ev.performance.speedup_factor
        if ev.correctness is not None:
            entry["max_abs_err"] = ev.correctness.max_absolute_error
            entry["max_rel_err"] = ev.correctness.max_relative_error
        results.append(entry)

    return {
        "passed": pass_count,
        "failed": fail_count,
        "total": len(traces),
        "traces": results,
    }


@app.local_entrypoint()
def main():
    result = run_benchmark.remote()
    print("\n" + "=" * 60)
    print(f"RESULT: {result['passed']}/{result['total']} PASSED, {result['failed']} FAILED")
    print("=" * 60)
    # Summary
    if result["traces"]:
        statuses = {}
        speedups = []
        for t in result["traces"]:
            statuses[t["status"]] = statuses.get(t["status"], 0) + 1
            if "speedup" in t:
                speedups.append(t["speedup"])
        for s, n in sorted(statuses.items()):
            print(f"  {s}: {n}")
        if speedups:
            import statistics
            print(f"\nSpeedup stats over {len(speedups)} workloads:")
            print(f"  mean:   {statistics.mean(speedups):.3f}x")
            print(f"  median: {statistics.median(speedups):.3f}x")
            print(f"  min:    {min(speedups):.3f}x")
            print(f"  max:    {max(speedups):.3f}x")

    # Print first 5 non-PASSED traces for diagnosis
    failed = [t for t in result["traces"] if t["status"] != "PASSED"]
    if failed:
        print("\nFirst 5 non-PASSED workloads:")
        for t in failed[:5]:
            print(f"  {t}")
