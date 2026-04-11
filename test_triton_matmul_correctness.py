"""Test whether Triton's tl.dot produces bit-identical results to torch.matmul.

If this passes, a Triton fused kernel (gather+dequant+GEMM+relu+weight+sum) is viable.
If it fails, we're stuck with separate cuBLAS calls and can't fuse GEMM.

Tests the exact shapes from the DSA Indexer contest:
  Q: [64, 128] (NUM_HEADS × HEAD_DIM)
  K: [seq_len, 128]
  scores = Q @ K.T → [64, seq_len]
"""

import torch
import triton
import triton.language as tl
import time

@triton.jit
def matmul_kernel(
    Q_ptr, K_ptr, OUT_ptr,
    M, N, K_dim,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Q[M, K] @ K[N, K].T → OUT[M, N]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_dim, BLOCK_K):
        k_offs = k_start + offs_k

        # Load Q block [BLOCK_M, BLOCK_K]
        q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + k_offs[None, :] * stride_qk
        q_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_dim)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Load K block [BLOCK_N, BLOCK_K] (K is stored as [N, K], we want K.T)
        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + k_offs[None, :] * stride_kk
        k_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # acc += Q @ K.T
        acc += tl.dot(q, tl.trans(k))

    # Store
    out_ptrs = OUT_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_matmul(Q, K_mat):
    """Q[M, K] @ K_mat[N, K].T → [M, N]"""
    M, K_dim = Q.shape
    N = K_mat.shape[0]
    out = torch.empty(M, N, device=Q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 128  # K_dim=128, single tile

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        Q, K_mat, out,
        M, N, K_dim,
        Q.stride(0), Q.stride(1),
        K_mat.stride(0), K_mat.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def test_correctness():
    print("=" * 60)
    print("Triton vs cuBLAS matmul correctness test")
    print("=" * 60)

    NUM_HEADS = 64
    HEAD_DIM = 128
    seq_lens = [64, 640, 2048, 3200, 4480, 5824]

    all_pass = True
    for seq_len in seq_lens:
        torch.manual_seed(42)
        Q = torch.randn(NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, HEAD_DIM, device='cuda', dtype=torch.float32)

        # cuBLAS reference
        ref = Q @ K.T
        torch.cuda.synchronize()

        # Triton
        tri = triton_matmul(Q, K)
        torch.cuda.synchronize()

        max_diff = (ref - tri).abs().max().item()
        mean_diff = (ref - tri).abs().mean().item()
        exact_match = torch.equal(ref, tri)

        # Check if topk ordering matches
        ref_sum = ref.sum(dim=0)
        tri_sum = tri.sum(dim=0)
        topk_k = min(2048, seq_len)
        _, ref_topk = torch.topk(ref_sum, topk_k)
        _, tri_topk = torch.topk(tri_sum, topk_k)
        topk_match = torch.equal(ref_topk, tri_topk)

        status = "EXACT" if exact_match else ("TOPK_OK" if topk_match else "FAIL")
        if not topk_match:
            all_pass = False

        print(f"  seq_len={seq_len:5d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
              f"exact={exact_match}, topk_match={topk_match} [{status}]")

    print()
    if all_pass:
        print("ALL TOPK MATCH — Triton fusion path is VIABLE")
    else:
        print("TOPK MISMATCH DETECTED — Triton fusion path BREAKS CORRECTNESS")
        print("Must stick with cuBLAS (per-batch Q @ K.T)")

    # Also test with relu+weight+sum pipeline
    print()
    print("=" * 60)
    print("Full pipeline test (matmul + relu + weight + sum + topk)")
    print("=" * 60)

    for seq_len in [640, 3200, 5824]:
        torch.manual_seed(42)
        Q = torch.randn(NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.float32)
        K = torch.randn(seq_len, HEAD_DIM, device='cuda', dtype=torch.float32)
        w = torch.randn(NUM_HEADS, device='cuda', dtype=torch.float32)

        # cuBLAS pipeline
        scores_ref = Q @ K.T
        scores_ref.clamp_(min=0)
        scores_ref.mul_(w[:, None])
        final_ref = scores_ref.sum(dim=0)
        topk_k = min(2048, seq_len)
        _, topk_ref = torch.topk(final_ref, topk_k)

        # Triton pipeline
        scores_tri = triton_matmul(Q, K)
        torch.cuda.synchronize()
        scores_tri.clamp_(min=0)
        scores_tri.mul_(w[:, None])
        final_tri = scores_tri.sum(dim=0)
        _, topk_tri = torch.topk(final_tri, topk_k)

        topk_match = torch.equal(topk_ref, topk_tri)
        if not topk_match:
            n_diff = (topk_ref != topk_tri).sum().item()
            print(f"  seq_len={seq_len}: TOPK MISMATCH — {n_diff}/{topk_k} indices differ")
        else:
            print(f"  seq_len={seq_len}: TOPK EXACT MATCH")

    # Timing comparison
    print()
    print("=" * 60)
    print("Timing comparison (100 iterations)")
    print("=" * 60)
    seq_len = 5824
    Q = torch.randn(NUM_HEADS, HEAD_DIM, device='cuda', dtype=torch.float32)
    K = torch.randn(seq_len, HEAD_DIM, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = Q @ K.T
        _ = triton_matmul(Q, K)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100):
        _ = Q @ K.T
    torch.cuda.synchronize()
    t_cublas = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        _ = triton_matmul(Q, K)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) * 1000

    print(f"  cuBLAS: {t_cublas:.1f}ms / 100 iters ({t_cublas/100*1000:.1f}us per call)")
    print(f"  Triton: {t_triton:.1f}ms / 100 iters ({t_triton/100*1000:.1f}us per call)")
    print(f"  Ratio:  {t_triton/t_cublas:.2f}x")


if __name__ == "__main__":
    test_correctness()
