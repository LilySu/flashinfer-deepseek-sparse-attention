# CUDA Graph Attempt — FAILED

**Date:** 2026-04-11

## What was tried

1. **Full pipeline graph capture** — TVM FFI kernels (gather_dequant, fused_index_remap) are NOT captured by CUDA graphs. They bypass PyTorch's stream management and launch on the default CUDA stream regardless of the capture stream. Graph replay produces zeros for TVM kernel outputs.

2. **Split approach** (TVM eager + PyTorch graph) — Captures only PyTorch ops (GEMM + clamp_ + mul_ + sum + topk) as a graph, runs TVM kernels eagerly outside. Correctness verified. But only 1.03-1.10x speedup — the copy/sync overhead between TVM and graph replays negates the launch overhead savings.

3. **Score masking for variable seq_lens** — Discovered that seq_lens are actually UNIFORM within each workload (all batch elements have the same seq_len). Padding/masking not needed.

## Root Cause

TVM FFI kernels use raw CUDA API (`<<<grid, block>>>`) without respecting PyTorch's current CUDA stream. They always launch on CUDA stream 0. Since `torch.cuda.graph()` captures operations on a specific stream, TVM kernel launches on stream 0 are invisible to the graph.

## Impact

- Full graph capture: **impossible** without replacing TVM FFI with PyTorch custom ops
- Split graph capture: **negligible benefit** (1.03x) due to sync/copy overhead
- CUDA graphs are a dead end for this codebase unless TVM FFI is replaced

## What would fix this

Replace TVM FFI kernels with PyTorch custom ops registered via `torch.library`:
```python
@torch.library.custom_op("dsa::gather_dequant", mutates_args={"output"})
def gather_dequant(kv_cache: Tensor, page_indices: Tensor, output: Tensor) -> None:
    # Launch CUDA kernel on current stream
    ...
```
This would make all kernels stream-aware and graph-capturable. But it requires rewriting the TVM FFI integration from scratch.
