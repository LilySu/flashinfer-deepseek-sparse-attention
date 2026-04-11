# Triton Matmul Correctness Test — FAILED

**Date:** 2026-04-11

## Result: Triton fusion path is NOT viable

Triton's `tl.dot` produces different floating-point results than cuBLAS `torch.matmul`:
- max_diff: ~0.04 per element (small FP rounding difference)
- mean_diff: ~0.007
- NOT bit-identical to cuBLAS

Impact on topk ordering:
- seq_len=64: topk matches (only 64 elements, not enough to trigger divergence)
- seq_len=640: 49/640 indices differ
- seq_len=3200: 698/2048 indices differ  
- seq_len=5824: 870/2048 indices differ

## Performance: Triton is also slower
- cuBLAS: 10.9us per call
- Triton: 16.7us per call (1.54x slower)

## Conclusion

Cannot fuse GEMM into a Triton kernel. Must keep per-batch `q_b @ K.T` as a separate cuBLAS call. The remaining optimization path is:
1. CUDA graphs (to eliminate launch overhead)
2. Further CUDA kernel fusions (non-GEMM ops)
3. Stream pipelining (if TVM FFI stream issue can be resolved)
