"""Paired A/B benchmark on Modal B200 — cancels instance-level variance.

Packs two solutions into the same TraceSet so flashinfer-bench's workload
loop runs them back-to-back on the same GPU for each workload. Reports
per-workload ratio (new/old) with geometric-mean aggregation.

Usage:
    uvx modal run scripts/run_modal_paired_bench.py

Solutions (hardcoded for attention):
    - baseline: attention/solution/python_baseline_2a/  (frozen Step 2a)
    - candidate: attention/solution/python/              (working dir; where ldmatrix work lives)
"""

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

KERNEL = "attention"  # paired bench only meaningful per-kernel

import modal

app = modal.App(f"flashinfer-dsa-{KERNEL}-paired")

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
    .add_local_dir(
        str(PROJECT_ROOT / KERNEL / "solution" / "python_baseline_2a"),
        remote_path="/root/submission/solution/python_baseline_2a",
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
def run_paired() -> dict:
    import tomllib
    from pathlib import Path as P
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet, BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    submission_root = P("/root/submission")
    with open(submission_root / "config.toml", "rb") as f:
        cfg = tomllib.load(f)

    build_cfg = cfg["build"]
    sol_cfg = cfg["solution"]
    spec = BuildSpec(
        language=build_cfg["language"],
        target_hardware=["cuda"],
        entry_point=build_cfg["entry_point"],
        destination_passing_style=build_cfg.get("destination_passing_style", True),
    )

    sol_baseline = pack_solution_from_files(
        path=str(submission_root / "solution" / "python_baseline_2a"),
        spec=spec,
        name="baseline_step2a",
        definition=sol_cfg["definition"],
        author=sol_cfg["author"],
    )
    sol_candidate = pack_solution_from_files(
        path=str(submission_root / "solution" / "python"),
        spec=spec,
        name="candidate",
        definition=sol_cfg["definition"],
        author=sol_cfg["author"],
    )
    print(f"Baseline solution:  {sol_baseline.name}  ({len(sol_baseline.sources)} srcs)")
    print(f"Candidate solution: {sol_candidate.name} ({len(sol_candidate.sources)} srcs)")

    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    definition = trace_set.definitions[sol_baseline.definition]
    workloads = trace_set.workloads.get(sol_baseline.definition, [])
    print(f"Running paired A/B on {len(workloads)} workloads")

    # Two solutions interleaved per-workload (flashinfer-bench default outer
    # loop is workload-major; for each workload both solutions run back-to-back
    # on the same GPU. Instance-level variance cancels in the per-workload ratio.
    bench_ts = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [sol_baseline, sol_candidate]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    config = BenchmarkConfig(warmup_runs=3, iterations=50, num_trials=3)
    result_ts = Benchmark(bench_ts, config).run_all(dump_traces=False)

    traces = result_ts.traces.get(definition.name, [])

    # Group traces by workload_id, then by solution_name.
    # flashinfer-bench traces carry their workload and solution identifiers.
    # flashinfer-bench iterates workload-major, solution-minor:
    # (wrk0_A, wrk0_B, wrk1_A, wrk1_B, ...). Pair by floor(i/2).
    # Dump the first trace's workload structure once for diagnostics.
    if traces:
        w0 = traces[0].workload
        print(f"[DIAG] first trace.workload type={type(w0).__name__} "
              f"repr={repr(w0)[:200]}")

    per_workload = {}  # wid → {solution_name: speedup}
    for i, trace in enumerate(traces):
        ev = trace.evaluation
        if ev is None or ev.performance is None:
            continue
        sname = trace.solution if isinstance(trace.solution, str) \
                else getattr(trace.solution, "name", str(trace.solution))
        wid = f"w{i // 2:02d}"  # pair consecutive traces
        per_workload.setdefault(wid, {})[sname] = {
            "speedup": ev.performance.speedup_factor,
            "latency_ms": ev.performance.latency_ms,
            "ref_latency_ms": ev.performance.reference_latency_ms,
            "status": ev.status.value,
        }

    return {
        "num_workloads": len(per_workload),
        "per_workload": per_workload,
    }


@app.local_entrypoint()
def main():
    import statistics
    result = run_paired.remote()

    print("\n" + "=" * 70)
    print(f"PAIRED RESULTS — {result['num_workloads']} workloads")
    print("=" * 70)

    ratios = []
    baseline_speedups = []
    candidate_speedups = []
    below_floor = []  # ratio < 0.95 workloads
    rows = []
    for wid, sols in sorted(result["per_workload"].items(), key=lambda kv: str(kv[0])):
        base = sols.get("baseline_step2a")
        cand = sols.get("candidate")
        if not base or not cand:
            print(f"  [{wid}] MISSING (base={bool(base)}, cand={bool(cand)})")
            continue
        if base["status"] != "PASSED" or cand["status"] != "PASSED":
            print(f"  [{wid}] NON-PASS (base={base['status']}, cand={cand['status']})")
            continue
        ratio = cand["speedup"] / base["speedup"]
        ratios.append(ratio)
        baseline_speedups.append(base["speedup"])
        candidate_speedups.append(cand["speedup"])
        if ratio < 0.95:
            below_floor.append((wid, ratio))
        rows.append((wid, base["speedup"], cand["speedup"], ratio))

    print(f"{'wid':<40} {'baseline':>10} {'candidate':>10} {'ratio':>8}")
    for wid, b, c, r in rows:
        print(f"{str(wid):<40} {b:>10.3f} {c:>10.3f} {r:>8.4f}")

    if ratios:
        # Geomean of ratios is the commit signal.
        geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        print("\n" + "-" * 70)
        print(f"Geomean(new/old):  {geomean:.4f}  ({(geomean-1)*100:+.2f}%)")
        print(f"Min  ratio:        {min(ratios):.4f}")
        print(f"Max  ratio:        {max(ratios):.4f}")
        print(f"Arithmetic baseline mean:  {statistics.mean(baseline_speedups):.3f}")
        print(f"Arithmetic candidate mean: {statistics.mean(candidate_speedups):.3f}")
        if below_floor:
            print(f"\n!! {len(below_floor)} workload(s) with ratio < 0.95 (regressions):")
            for wid, r in below_floor:
                print(f"     {wid}: {r:.4f}")
        else:
            print("\n✓ No workload below 0.95 floor.")
