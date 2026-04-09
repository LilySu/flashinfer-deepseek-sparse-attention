"""
Simplified Modal runner that doesn't require flashinfer-bench locally.
Packs the solution manually and runs on Modal B200.
"""

import json
import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("flashinfer-bench-runner")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
TRACE_SET_PATH = "/data"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "ninja")
)


def pack_solution_manual() -> dict:
    """Pack solution into a dict without flashinfer-bench dependency."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    solution_config = config["solution"]
    build_config = config["build"]
    language = build_config["language"]

    # Read source files
    if language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    else:
        source_dir = PROJECT_ROOT / "solution" / "cuda"

    sources = {}
    for f in source_dir.iterdir():
        if f.is_file() and f.suffix in (".py", ".cu", ".cuh"):
            sources[f.name] = f.read_text()

    return {
        "name": solution_config["name"],
        "definition": solution_config["definition"],
        "author": solution_config["author"],
        "spec": {
            "language": language,
            "target_hardware": ["cuda"],
            "entry_point": build_config["entry_point"],
            "dependencies": [],
            "destination_passing_style": build_config.get("destination_passing_style", True),
        },
        "sources": [{"path": k, "content": v} for k, v in sources.items()],
    }


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark_remote(solution_dict: dict) -> dict:
    """Run benchmark on Modal B200."""
    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate(solution_dict)
    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)
    trace_set = TraceSet.from_path(TRACE_SET_PATH)

    if solution.definition not in trace_set.definitions:
        available = list(trace_set.definitions.keys())
        raise ValueError(f"Definition '{solution.definition}' not found. Available: {available}")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    # Limit workloads if quick_test mode
    if solution_dict.get("_quick_test"):
        workloads = workloads[:3]
        print(f"QUICK TEST: running {len(workloads)} workloads only")

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
    results = {}
    passed = 0
    failed = 0

    for trace in traces:
        if trace.evaluation:
            status = trace.evaluation.status.value
            entry = {"status": status, "solution": trace.solution}
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            # Capture error message if available
            if hasattr(trace.evaluation, 'error') and trace.evaluation.error:
                entry["error"] = str(trace.evaluation.error)
            if hasattr(trace.evaluation, 'message') and trace.evaluation.message:
                entry["error"] = str(trace.evaluation.message)
            if hasattr(trace.evaluation, 'stderr') and trace.evaluation.stderr:
                entry["error"] = str(trace.evaluation.stderr)
            # Try to get any string representation of the evaluation for debugging
            if status in ("RUNTIME_ERROR", "COMPILE_ERROR"):
                entry["eval_dump"] = str(trace.evaluation)[:4000]
                if hasattr(trace.evaluation, 'log') and trace.evaluation.log:
                    entry["log"] = str(trace.evaluation.log)[:4000]
            results[trace.workload.uuid] = entry
            if status == "PASSED":
                passed += 1
            else:
                failed += 1

    return {"results": results, "passed": passed, "failed": failed, "total": passed + failed}


@app.local_entrypoint()
def main():
    print("Packing solution manually...")
    solution_dict = pack_solution_manual()
    print(f"  Name: {solution_dict['name']}")
    print(f"  Definition: {solution_dict['definition']}")
    print(f"  Sources: {[s['path'] for s in solution_dict['sources']]}")

    # Also save solution.json locally
    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(json.dumps(solution_dict, indent=2))
    print(f"  Saved: {out_path}")

    # Quick test mode: set QUICK=1 to only run 3 workloads
    import os
    if os.environ.get("QUICK"):
        solution_dict["_quick_test"] = True
        print("\n[QUICK TEST MODE: 3 workloads only]")

    print("\nRunning benchmark on Modal B200...")
    result = run_benchmark_remote.remote(solution_dict)

    print(f"\n{'='*60}")
    print(f"Results: {result['passed']}/{result['total']} passed, {result['failed']} failed")
    print(f"{'='*60}")

    for uuid, r in sorted(result["results"].items(), key=lambda x: x[1]["status"]):
        status = r["status"]
        line = f"  {uuid[:8]}  {status}"
        if r.get("latency_ms") is not None:
            line += f"  {r['latency_ms']:.3f}ms"
        if r.get("speedup_factor") is not None:
            line += f"  {r['speedup_factor']:.2f}x"
        if r.get("max_abs_error") is not None:
            line += f"  abs_err={r['max_abs_error']:.2e}"
        print(line)
        if r.get("error"):
            print(f"           ERROR: {r['error'][:200]}")
        if r.get("log"):
            print(f"           LOG: {r['log']}")
        elif r.get("eval_dump"):
            print(f"           DUMP: {r['eval_dump']}")
