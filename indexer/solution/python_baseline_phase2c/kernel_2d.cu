// DSA topk indexer — Phase 2d: fused scoring + threshold-skip + bounded merge.
//
// =============================================================
// STATUS: EXPLORATORY / NOT ON PRODUCTION PATH
// =============================================================
// The bounded-merge algorithm documented below is ALGORITHMICALLY
// FLAWED. Specifically:
//
//   The set-level claim "only heap[1984..2048) can change" depends on
//   heap[1984..2048) being the 64 SMALLEST entries of heap. After the
//   FIRST merge that promotes a high pending value, heap[1984..2048)
//   contains the newly promoted large values — no longer the 64
//   smallest. Subsequent bounded merges look at the wrong candidates
//   and can fail to displace entries that SHOULD be displaced.
//
// Concrete counterexample:
//   Initial heap sorted desc: [100..-1947]. threshold_top1984 = -1883.
//   Page 1 pending = [∞ × 64]: bounded-merge promotes all 64 ∞s into
//     heap[1984..2048). Now heap[1984..2048) = [∞ × 64] (not smallest).
//   Page 2 pending = [50 × 64]: threshold = min(-1883, ∞) = -1883.
//     pending_max = 50 > -1883, so merge path fires.
//     Bounded merge of (heap[1984..2048) = ∞×64) + (pending = 50×64)
//     → top 64 = ∞×64.  Heap[1984..2048) unchanged.
//   Correct top-K of union WOULD drop the lowest old heap entries
//     (-1820..-1883) and include the pending 50s. Algorithm misses this.
//
// Observed: with this kernel forced on, 71/128 workloads PASS
// correctness — exactly the seq_len < 2048 set (fill-only path, no
// merges). All seq_len ≥ 2048 workloads fail INCORRECT_NUMERICAL.
//
// =============================================================
// Why the straightforward fix doesn't save us
// =============================================================
// Correct implementation requires full 4096-pad bitonic sort per page
// (~36 cross-warp syncs, ~1.8 ms for a 2048-page row). 2c's
// scoring+logits+torch.topk path spends ~64 µs on the 64 MB DRAM
// round-trip for the same shape. So even a correct 2d is likely
// slower than 2c on the exact shapes where 2d was supposed to win.
//
// A different fusion strategy might still pay off — e.g., fusing
// scoring with CUDA radix-select rather than a sorted heap, so the
// per-page work is O(radix-pass) not O(sort 2048). Left as future
// work. For now binding.py routes to 2c on all shapes.
//
// =============================================================
// Original design below, kept for documentation.
// =============================================================
//
// One CTA per batch row. The CTA streams through every page for its row,
// scoring 64 slots per page (inner loop copied verbatim from Phase 2c), and
// maintains a top-K=2048 set in SMEM without ever writing a logits tensor
// to DRAM.
//
// ============================================================
// Bounded-merge correctness proof (SET-level, see caveat)
// ============================================================
// Claim: After merging old_heap (sorted desc, 2048 entries) with `pending`
// (64 entries), the SET of top 2048 is equal to
//     { x : x in old_heap[0..1984)  AS A SET  }   (always preserved)
//   ∪ { top-64 of ( old_heap[1984..2048)  ∪  pending )  AS A SET }
//
// Proof:
//   Let union = old_heap ∪ pending   (2048 + 64 = 2112 values).
//   top_2048(union) = union \ bottom_64(union)
//
//   bottom_64(union) ⊆ old_heap[1984..2048) ∪ pending  ...(*)
//
// Reason for (*): for every x ∈ old_heap[0..1984),  x ≥ old_heap[1983]
//   (by sorted-desc invariant).  And old_heap[1984..2048) contains 64
//   entries, each ≤ old_heap[1983].  So for any x ∈ old_heap[0..1984),
//   there are already ≥ 64 values in union that are ≤ x (namely
//   old_heap[1984..2048)).  Hence x cannot be among the bottom 64 of
//   the union.  QED.
//
// Consequence: old_heap[0..1984) is never displaced.  Its SET is
//   preserved verbatim.  Only the bottom 64 slots can change, and
//   those new bottom 64 are exactly top_64(old_heap[1984..2048) ∪ pending).
//
// ============================================================
// CAVEAT: the claim is SET-level, not POSITION-level.
// ============================================================
// Example: old_heap = [100, 99, …, −1947], pending = [∞ × 64].
//   true top 2048 of union, sorted desc:
//     [∞, ∞, …, ∞  (64 times),  100, 99, …, −1883]
//   i.e., pending occupies positions [0..64), old_heap[0..1984) shifts
//   down to positions [64..2048).
//   Our algorithm, by design, puts the 64 promoted ∞s into positions
//   [1984..2048) while leaving old_heap[0..1984) in place at [0..1984).
//   The SET is identical; positions differ.
//
// For the threshold-skip fast path we only need min(heap), which equals
//   min(threshold_top1984, heap[2047])
// where threshold_top1984 = min(heap[0..1984)) — a constant frozen at
// the moment the heap first fills to 2048 — and heap[2047] is the
// current bottom of the sorted-desc bottom-64 segment.
//
// We pay ONE full bitonic sort at end-of-kernel to produce a globally
// sorted output (positions match the torch.topk reference).
//
// ============================================================
// SMEM layout (~75 KiB per CTA, under 100 KiB target)
// ============================================================
//   q_smem[64][128]         FP32      32 KiB  (scoring state, verbatim 2c)
//   k_smem[kBytesPerPage]   uint8      ~8.3 KiB
//   w_smem[64]              FP32        256 B
//   k_bar                   uint64        8 B
//   heap_vals[2048]         FP32       8 KiB
//   heap_idx [2048]         int32      8 KiB
//   pending_vals[64]        FP32        256 B
//   pending_idx [64]        int32        256 B
//   merge_buf_vals[128]     FP32        512 B  (heap[1984..2048) + pending for merge)
//   merge_buf_idx [128]     int32        512 B
//   threshold_top1984       FP32          4 B  (min of heap[0..1984), frozen at fill)
//   heap_count              int32         4 B
//   skip_count              uint32        4 B   (diag telemetry)
//   (alignment/padding)                ~100 B

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <cstring>
#include <cstdint>

namespace {

constexpr int kNumHeads         = 64;
constexpr int kHeadDim          = 128;
constexpr int kPageSize         = 64;
constexpr int kHeadDimWithScale = 132;
constexpr int kFP8BytesPerPage  = kPageSize * kHeadDim;
constexpr int kScaleSectionOff  = kFP8BytesPerPage;
constexpr int kBytesPerPage     = kPageSize * kHeadDimWithScale;

constexpr int kThreadsPerBlock  = 128;
constexpr int kThreadsPerSlot   = 2;
constexpr int kHeadsPerGroup    = kNumHeads / kThreadsPerSlot;
constexpr int kTopK             = 2048;
constexpr int kTopK_Top1984     = kTopK - kPageSize;     // 1984
constexpr int kMergeBufLen      = kPageSize * 2;          // 128

// Toggle for debug sorting-invariant check (commented out in production).
// #define PHASE2D_ASSERT_SORTED_AFTER_MERGE 1

// ---- PTX helpers (unchanged from 2c) ----
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    uint32_t r;
    asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }"
                 : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 : : "r"(smem_addr(bar)), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                 : : "r"(smem_addr(bar)), "r"(tx));
}
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t done;
    do {
        asm volatile("{ .reg .pred p;"
                     "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                     "  selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_addr(bar)), "r"(phase));
    } while (!done);
}
__device__ __forceinline__ void cp_async_bulk_g2s(
    void* smem_dst, const void* gmem_src, uint32_t bytes, uint64_t* bar
) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        : : "r"(smem_addr(smem_dst)), "l"(gmem_src),
            "r"(bytes), "r"(smem_addr(bar))
    );
}

// ===============================================================
// Bitonic sort helpers (descending, in SMEM, 128-thread cooperative)
// ===============================================================
//
// Compare-swap pair (i, i^j) direction bit = (i & k) == 0.
// For DESCENDING overall: swap when flag==true && v[i]<v[j]  (want larger left)
//                        or flag==false && v[i]>v[j]          (want smaller left)
// Returns with SMEM writes visible to the rest of the CTA AFTER a __syncthreads.

// Sort `vals/idx` in-place, length N power of 2, descending.
template <int N>
__device__ void bitonic_sort_desc(float* vals, int32_t* idx, int tid) {
    #pragma unroll
    for (int k = 2; k <= N; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread owns N/2 / kThreadsPerBlock pairs (rounds if N<kThreadsPerBlock*2).
            for (int pair = tid; pair < (N / 2); pair += kThreadsPerBlock) {
                int i_lo = ((pair & ~(j - 1)) << 1) | (pair & (j - 1));
                int i_hi = i_lo ^ j;
                if (i_hi < N) {
                    bool flag = ((i_lo & k) == 0);
                    float v_lo = vals[i_lo], v_hi = vals[i_hi];
                    bool swap = flag ? (v_lo < v_hi) : (v_lo > v_hi);
                    if (swap) {
                        vals[i_lo] = v_hi;  vals[i_hi] = v_lo;
                        int32_t id_lo = idx[i_lo], id_hi = idx[i_hi];
                        idx [i_lo] = id_hi;  idx [i_hi] = id_lo;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Bitonic MERGE of a bitonic sequence of length N power of 2, descending.
// Prerequisite: vals[0..N) is bitonic (i.e., already the concatenation of
// one desc-sorted subsequence and one asc-sorted subsequence — which is
// what you get when you concatenate two desc-sorted halves and reverse
// the second half, OR equivalently when you place two asc-sorted halves
// back-to-back).  For our use case the two halves are already desc-sorted
// descending; we rely on the full bitonic_sort_desc above.
//
// [Not used directly — we use full bitonic_sort_desc on merge_buf since
//  N=128 is tiny and sorting vs merging is the same log²N work.]

// ---- Validation: debug-build sorted-desc assertion on [lo, hi). ----
__device__ __forceinline__ void assert_sorted_desc(
    const float* vals, int lo, int hi, int tid
) {
#ifdef PHASE2D_ASSERT_SORTED_AFTER_MERGE
    // One-thread check; cheap since hi-lo ≤ 128.
    if (tid == 0) {
        for (int i = lo + 1; i < hi; ++i) {
            if (!(vals[i - 1] >= vals[i])) {
                asm volatile("trap;");
            }
        }
    }
#else
    (void)vals; (void)lo; (void)hi; (void)tid;
#endif
}

// -----------------------------------------------------------------------------
// Phase 2d fused scoring + topk kernel
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreadsPerBlock, 1)
fused_kernel_phase2d(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ k_cache,
    const float*   __restrict__ weights,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    int32_t*       __restrict__ topk_indices,   // [B, 2048] int32 output
    uint32_t*      __restrict__ skip_counts,    // [B] uint32 telemetry (optional; nullable)
    int max_num_pages
) {
    const int batch = blockIdx.x;
    const int tid   = threadIdx.x;
    const int seq_len = seq_lens[batch];

    // --- SMEM layout ----------------------------------------------------
    __shared__ float                q_smem[kNumHeads][kHeadDim];
    __shared__ alignas(16) uint8_t  k_smem[kBytesPerPage];
    __shared__ float                w_smem[kNumHeads];
    __shared__ alignas(8) uint64_t  k_bar;

    __shared__ float    heap_vals[kTopK];
    __shared__ int32_t  heap_idx [kTopK];
    __shared__ float    pending_vals[kPageSize];
    __shared__ int32_t  pending_idx [kPageSize];
    __shared__ float    merge_buf_vals[kMergeBufLen];   // [bottom-64 | pending]
    __shared__ int32_t  merge_buf_idx [kMergeBufLen];
    __shared__ float    threshold_top1984;   // frozen once heap fills
    __shared__ int32_t  heap_count;
    __shared__ uint32_t skip_count;

    // --- Init heap, counters ------------------------------------------
    if (tid == 0) {
        threshold_top1984 = -INFINITY;
        heap_count = 0;
        skip_count = 0;
    }
    for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
        heap_vals[i] = -INFINITY;
        heap_idx [i] = -1;
    }

    // --- Empty-row fast path -------------------------------------------
    if (seq_len == 0) {
        __syncthreads();
        for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
            topk_indices[static_cast<int64_t>(batch) * kTopK + i] = -1;
        }
        if (skip_counts != nullptr && tid == 0) {
            skip_counts[batch] = 0;
        }
        return;
    }

    // --- Stage Q, W + init barrier ------------------------------------
    const int64_t q_row_off =
        static_cast<int64_t>(batch) * kNumHeads * kHeadDim;
    for (int i = tid; i < kNumHeads * kHeadDim; i += kThreadsPerBlock) {
        const int h = i / kHeadDim;
        const int d = i % kHeadDim;
        __nv_fp8_e4m3 v;
        uint8_t byte = q_fp8[q_row_off + h * kHeadDim + d];
        std::memcpy(&v, &byte, 1);
        q_smem[h][d] = static_cast<float>(v);
    }
    for (int i = tid; i < kNumHeads; i += kThreadsPerBlock) {
        w_smem[i] = weights[batch * kNumHeads + i];
    }
    if (tid == 0) {
        mbarrier_init(&k_bar, 1);
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    const int num_pages_for_seq = (seq_len + kPageSize - 1) / kPageSize;
    uint32_t bar_phase = 0;

    // ===================================================================
    // Main page loop
    // ===================================================================
    for (int page_local = 0; page_local < num_pages_for_seq; ++page_local) {
        const int page_id =
            block_table[batch * max_num_pages + page_local];

        // TMA bulk load K page.
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&k_bar, kBytesPerPage);
            const uint8_t* page_base =
                k_cache + static_cast<int64_t>(page_id) * kBytesPerPage;
            cp_async_bulk_g2s(k_smem, page_base, kBytesPerPage, &k_bar);
        }
        mbarrier_wait(&k_bar, bar_phase);
        bar_phase ^= 1;
        __syncthreads();

        // ----------- Score 64 slots (verbatim Phase 2c) ---------------
        const int page_offset = page_local * kPageSize;
        const int slot        = tid >> 1;
        const int head_group  = tid & 1;
        const int h_start     = head_group * kHeadsPerGroup;
        const int h_end       = h_start + kHeadsPerGroup;
        const int tok         = page_offset + slot;

        float partial = 0.0f;
        if (tok < seq_len) {
            float scale;
            std::memcpy(&scale, k_smem + kScaleSectionOff + slot * 4, sizeof(float));
            const uint8_t* kv_ptr = k_smem + slot * kHeadDim;
            for (int h = h_start; h < h_end; ++h) {
                float dot = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    __nv_fp8_e4m3 kv_raw;
                    uint8_t kv_byte = kv_ptr[d];
                    std::memcpy(&kv_raw, &kv_byte, 1);
                    const float k_val = static_cast<float>(kv_raw) * scale;
                    dot += q_smem[h][d] * k_val;
                }
                partial += w_smem[h] * fmaxf(dot, 0.0f);
            }
        }
        partial += __shfl_xor_sync(0xffffffffu, partial, 1);

        if (head_group == 0) {
            if (tok < seq_len) {
                pending_vals[slot] = partial;
                pending_idx [slot] = page_id * kPageSize + slot;
            } else {
                pending_vals[slot] = -INFINITY;
                pending_idx [slot] = -1;
            }
        }
        __syncthreads();

        // ----------- Heap update ----------------------------------
        // Two modes:
        //   fill  (heap_count < kTopK) : append pending to tail, sort once
        //                                 at transition to full.
        //   merge (heap_count == kTopK): threshold-skip + bounded 128-merge.
        if (heap_count < kTopK) {
            // Append pending[0..64) into heap[heap_count..heap_count+64).
            // Guarded: only threads whose target slot is < kTopK write;
            // any pending "overflow" is dropped (never happens since we
            // scale heap to exactly 2048 and page size 64, 2048/64 = 32
            // pages fit exactly before transition).
            const int write_base = heap_count;
            if (tid < kPageSize) {
                int dst = write_base + tid;
                if (dst < kTopK) {
                    heap_vals[dst] = pending_vals[tid];
                    heap_idx [dst] = pending_idx [tid];
                }
            }
            __syncthreads();

            const int new_count = heap_count + kPageSize;
            if (tid == 0) {
                heap_count = new_count > kTopK ? kTopK : new_count;
            }
            __syncthreads();

            // On transition to full, sort the whole heap desc and freeze
            // threshold_top1984 = heap[1983].
            if (new_count >= kTopK) {
                bitonic_sort_desc<kTopK>(heap_vals, heap_idx, tid);
                if (tid == 0) {
                    threshold_top1984 = heap_vals[kTopK_Top1984 - 1];  // heap[1983]
                }
                __syncthreads();
            }
        } else {
            // merge mode.  Heap is globally sorted desc; heap[1983] was
            // frozen as threshold_top1984; heap[2047] is current bottom.
            const float threshold =
                fminf(threshold_top1984, heap_vals[kTopK - 1]);

            // Pre-sort pending descending (bitonic on 64).
            bitonic_sort_desc<kPageSize>(pending_vals, pending_idx, tid);

            // Threshold-skip fast path: if pending[0] ≤ threshold,
            // no pending value can improve the heap.
            if (pending_vals[0] <= threshold) {
                if (tid == 0) ++skip_count;
                __syncthreads();
                continue;
            }

            // Partial merge: combine heap[1984..2048) (sorted desc) with
            // pending (sorted desc) into merge_buf[0..128), then
            // bitonic_sort_desc to produce top-128 desc → top 64 of that
            // gets written back to heap[1984..2048).
            if (tid < kPageSize) {
                merge_buf_vals[tid]             = heap_vals[kTopK_Top1984 + tid];
                merge_buf_idx [tid]             = heap_idx [kTopK_Top1984 + tid];
                merge_buf_vals[kPageSize + tid] = pending_vals[tid];
                merge_buf_idx [kPageSize + tid] = pending_idx [tid];
            }
            __syncthreads();

            bitonic_sort_desc<kMergeBufLen>(merge_buf_vals, merge_buf_idx, tid);

            // Top 64 of the 128-merge -> heap[1984..2048).
            if (tid < kPageSize) {
                heap_vals[kTopK_Top1984 + tid] = merge_buf_vals[tid];
                heap_idx [kTopK_Top1984 + tid] = merge_buf_idx [tid];
            }
            __syncthreads();

            assert_sorted_desc(heap_vals, kTopK_Top1984, kTopK, tid);
        }
    }

    // ===================================================================
    // Finalize: produce globally sorted-desc heap for output.
    // ===================================================================
    // If heap never filled (seq_len < kTopK), unused slots are -INFINITY
    // with idx=-1 — a single full bitonic sort pushes them to the tail
    // naturally.
    //
    // If heap filled and we did merges, heap[0..1984) is the original
    // sorted segment but heap[1984..2048) may contain values HIGHER than
    // heap[0..1984) (when a very-high pending got written into the
    // bottom segment by design).  One final full bitonic sort fixes
    // position invariants.
    bitonic_sort_desc<kTopK>(heap_vals, heap_idx, tid);

    // --- Write outputs -------------------------------------------------
    for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
        topk_indices[static_cast<int64_t>(batch) * kTopK + i] = heap_idx[i];
    }
    if (skip_counts != nullptr && tid == 0) {
        skip_counts[batch] = skip_count;
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// Host entry
// -----------------------------------------------------------------------------

void fused_phase2d(
    torch::Tensor q_u8,
    torch::Tensor k_u8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices,
    torch::Tensor skip_counts  // may be empty (numel 0) to disable telemetry
) {
    TORCH_CHECK(q_u8.is_cuda() && q_u8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(k_u8.is_cuda() && k_u8.scalar_type() == torch::kUInt8);
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32);
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32);

    const int batch_size    = q_u8.size(0);
    const int max_num_pages = block_table.size(1);
    TORCH_CHECK(topk_indices.size(0) == batch_size);
    TORCH_CHECK(topk_indices.size(1) == 2048);

    if (batch_size == 0) return;

    uint32_t* skip_ptr = nullptr;
    if (skip_counts.numel() > 0) {
        TORCH_CHECK(skip_counts.scalar_type() == torch::kUInt32 ||
                    skip_counts.scalar_type() == torch::kInt32);
        TORCH_CHECK(skip_counts.size(0) == batch_size);
        skip_ptr = reinterpret_cast<uint32_t*>(skip_counts.data_ptr());
    }

    const dim3 grid(batch_size);
    const dim3 block(kThreadsPerBlock);

    fused_kernel_phase2d<<<grid, block>>>(
        q_u8.data_ptr<uint8_t>(),
        k_u8.data_ptr<uint8_t>(),
        weights.data_ptr<float>(),
        seq_lens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        topk_indices.data_ptr<int32_t>(),
        skip_ptr,
        max_num_pages);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_kernel_phase2d launch failed: ",
                cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_phase2d", &fused_phase2d,
          "DSA topk indexer fused scoring + topk (Phase 2d step 3)");
}
