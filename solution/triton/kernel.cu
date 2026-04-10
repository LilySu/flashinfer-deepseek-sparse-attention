/*
 * DSA Indexer — CUDA kernels: vectorized dequant + Q convert.
 * Vectorized: each thread loads 16 FP8 bytes via uint4 (128-bit load).
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <cstdint>

constexpr int PAGE_SIZE = 64;
constexpr int HEAD_DIM = 128;
constexpr int NUM_HEADS = 64;
constexpr int PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4);

// ─── Vectorized FP8 dequant ───
// Each thread handles 16 consecutive FP8 dims (via uint4 = 16 bytes).
// Block: 8 threads (128 dims / 16 per thread = 8 threads per token-row)
// Grid: num_pages
// Each thread loops over 64 tokens, processing 16 dims each.
// Total: 8 threads × 64 tokens × 16 dims = 8192 elements per block = 1 page
__global__ void dequant_fp8_vec_kernel(
    const uint8_t* __restrict__ pages, float* __restrict__ output, int num_pages
) {
    int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    int tid = threadIdx.x;  // 0..7 (8 threads per block)
    int dim_start = tid * 16;  // each thread handles 16 consecutive dims

    const uint8_t* page_base = pages + (int64_t)page_idx * PAGE_BYTES;
    float* out_page = output + (int64_t)page_idx * PAGE_SIZE * HEAD_DIM;

    for (int token = 0; token < PAGE_SIZE; token++) {
        // Load 16 FP8 bytes as uint4 (128-bit load, 1 memory transaction)
        const uint4* src = (const uint4*)(page_base + token * HEAD_DIM + dim_start);
        uint4 data = *src;

        // Load scale (same for all dims in this token)
        const uint8_t* scale_ptr = page_base + PAGE_SIZE * HEAD_DIM + token * 4;
        uint32_t scale_bits = *(const uint32_t*)scale_ptr;
        float scale = __uint_as_float(scale_bits);

        // Extract 16 bytes from uint4 (4 x uint32_t = 16 bytes)
        uint8_t bytes[16];
        memcpy(bytes, &data, 16);

        // Convert each FP8 byte to FP32 and multiply by scale
        float results[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &bytes[i], 1);
            results[i] = float(fp8_val) * scale;
        }

        // Store as 4 x float4 (4 x 16-byte stores)
        float4* dst = (float4*)(out_page + token * HEAD_DIM + dim_start);
        dst[0] = make_float4(results[0], results[1], results[2], results[3]);
        dst[1] = make_float4(results[4], results[5], results[6], results[7]);
        dst[2] = make_float4(results[8], results[9], results[10], results[11]);
        dst[3] = make_float4(results[12], results[13], results[14], results[15]);
    }
}

// ─── Wide dequant: 128 threads per block (1 thread per dim, loop over tokens) ───
// 4 full warps per block → much better occupancy than 8-thread version.
// Each thread handles 1 dim across all 64 tokens.
// Reads are coalesced (adjacent threads read adjacent bytes within a token row).
// Scale load is a warp broadcast (all threads in warp read same 4 bytes).
__global__ void dequant_fp8_wide_kernel(
    const uint8_t* __restrict__ pages, float* __restrict__ output, int num_pages
) {
    int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;

    int dim = threadIdx.x;  // 0..127, one thread per dim

    const uint8_t* page_base = pages + (int64_t)page_idx * PAGE_BYTES;
    float* out_page = output + (int64_t)page_idx * PAGE_SIZE * HEAD_DIM;

    for (int token = 0; token < PAGE_SIZE; token++) {
        uint8_t fp8_byte = page_base[token * HEAD_DIM + dim];
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &fp8_byte, 1);

        // Load scale for this token (all threads in warp read same address = broadcast)
        const uint8_t* scale_ptr = page_base + PAGE_SIZE * HEAD_DIM + token * 4;
        float scale = __uint_as_float(*(const uint32_t*)scale_ptr);

        out_page[token * HEAD_DIM + dim] = float(fp8_val) * scale;
    }
}

// ─── Original dequant (fallback, proven) ───
__global__ void dequant_fp8_kernel(
    const uint8_t* __restrict__ pages, float* __restrict__ output, int num_pages
) {
    int page_idx = blockIdx.x;
    if (page_idx >= num_pages) return;
    int dim = threadIdx.x;
    const uint8_t* page_base = pages + (int64_t)page_idx * PAGE_BYTES;
    float* out_page = output + (int64_t)page_idx * PAGE_SIZE * HEAD_DIM;
    for (int token = 0; token < PAGE_SIZE; token++) {
        uint8_t fp8_byte = page_base[token * HEAD_DIM + dim];
        __nv_fp8_e4m3 fp8_val; memcpy(&fp8_val, &fp8_byte, 1);
        const uint8_t* scale_ptr = page_base + PAGE_SIZE * HEAD_DIM + token * 4;
        uint32_t scale_bits = (uint32_t)scale_ptr[0] | ((uint32_t)scale_ptr[1] << 8)
                            | ((uint32_t)scale_ptr[2] << 16) | ((uint32_t)scale_ptr[3] << 24);
        out_page[token * HEAD_DIM + dim] = float(fp8_val) * __uint_as_float(scale_bits);
    }
}

// ─── Q FP8→FP32 (proven) ───
__global__ void convert_q_fp8_to_fp32(
    const uint8_t* __restrict__ q_fp8, float* __restrict__ q_fp32, int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        __nv_fp8_e4m3 fp8_val; memcpy(&fp8_val, &q_fp8[idx], 1);
        q_fp32[idx] = float(fp8_val);
    }
}

// ─── Fused ReLU + weight multiply ───
// Replaces: scores_relu = relu(scores); weighted = scores_relu * w[:, None]
// Input: scores [NUM_HEADS, seq_len] (read-only), w [NUM_HEADS]
// Output: out [NUM_HEADS, seq_len]
__global__ void fused_relu_weight_kernel(
    const float* __restrict__ scores, const float* __restrict__ w,
    float* __restrict__ out, int num_heads, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len;
    if (idx >= total) return;
    int h = idx / seq_len;
    out[idx] = fmaxf(scores[idx], 0.0f) * w[h];
}

// ─── Fused index remapping ───
// Replaces: page_idx = topk_idx // PAGE_SIZE; offset = topk_idx % PAGE_SIZE;
//           global_page = page_indices[page_idx]; result = global_page * PAGE_SIZE + offset
// Input: topk_idx [k], page_indices [num_pages]
// Output: topk_tokens [k] as int32
__global__ void fused_index_remap_kernel(
    const int64_t* __restrict__ topk_idx, const int64_t* __restrict__ page_indices,
    int32_t* __restrict__ topk_tokens, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    int64_t idx = topk_idx[i];
    int64_t page_local = idx / PAGE_SIZE;
    int64_t offset = idx % PAGE_SIZE;
    int64_t global_page = page_indices[page_local];
    topk_tokens[i] = (int32_t)(global_page * PAGE_SIZE + offset);
}

// ─── Gather + Dequant fused ───
// Reads scattered pages from the full KV cache by page_indices, dequants in one pass.
// Eliminates the separate PyTorch gather + reshape + dequant chain.
// Grid: num_selected_pages, Block: 8 threads (same as vec kernel)
__global__ void gather_dequant_fp8_kernel(
    const uint8_t* __restrict__ kv_cache,    // full KV cache, flat [num_pages_total * PAGE_BYTES]
    const int64_t* __restrict__ page_indices, // which pages to read [num_selected]
    float* __restrict__ output,               // contiguous output [num_selected, PAGE_SIZE, HEAD_DIM]
    int num_selected
) {
    int sel_idx = blockIdx.x;
    if (sel_idx >= num_selected) return;

    int tid = threadIdx.x;  // 0..7
    int dim_start = tid * 16;

    int64_t global_page = page_indices[sel_idx];
    const uint8_t* page_base = kv_cache + global_page * PAGE_BYTES;
    float* out_page = output + (int64_t)sel_idx * PAGE_SIZE * HEAD_DIM;

    for (int token = 0; token < PAGE_SIZE; token++) {
        const uint4* src = (const uint4*)(page_base + token * HEAD_DIM + dim_start);
        uint4 data = *src;

        const uint8_t* scale_ptr = page_base + PAGE_SIZE * HEAD_DIM + token * 4;
        float scale = __uint_as_float(*(const uint32_t*)scale_ptr);

        uint8_t bytes[16];
        memcpy(bytes, &data, 16);

        float results[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            __nv_fp8_e4m3 fp8_val;
            memcpy(&fp8_val, &bytes[i], 1);
            results[i] = float(fp8_val) * scale;
        }

        float4* dst = (float4*)(out_page + token * HEAD_DIM + dim_start);
        dst[0] = make_float4(results[0], results[1], results[2], results[3]);
        dst[1] = make_float4(results[4], results[5], results[6], results[7]);
        dst[2] = make_float4(results[8], results[9], results[10], results[11]);
        dst[3] = make_float4(results[12], results[13], results[14], results[15]);
    }
}

namespace ffi = tvm::ffi;

void dequant_fp8_impl(ffi::TensorView pages_raw, ffi::TensorView output) {
    int64_t total_bytes = pages_raw.shape()[0];
    int num_pages = total_bytes / PAGE_BYTES;
    if (num_pages == 0) return;
    // Use vectorized kernel (8 threads per block, 16 dims per thread via uint4)
    // Wide kernel (128 threads) doesn't help: per-batch grids are 10-91 blocks,
    // latency-bound regardless of thread count. uint4 loads win on instruction efficiency.
    dequant_fp8_vec_kernel<<<num_pages, 8>>>(
        static_cast<const uint8_t*>(pages_raw.data_ptr()),
        static_cast<float*>(output.data_ptr()), num_pages);
}

void convert_q_impl(ffi::TensorView q_fp8, ffi::TensorView q_fp32) {
    int total = 1;
    for (int i = 0; i < q_fp8.ndim(); i++) total *= q_fp8.shape()[i];
    convert_q_fp8_to_fp32<<<(total+255)/256, 256>>>(
        static_cast<const uint8_t*>(q_fp8.data_ptr()),
        static_cast<float*>(q_fp32.data_ptr()), total);
}

void fused_relu_weight_impl(ffi::TensorView scores, ffi::TensorView w, ffi::TensorView out) {
    int num_heads = scores.shape()[0];
    int seq_len = scores.shape()[1];
    int total = num_heads * seq_len;
    fused_relu_weight_kernel<<<(total+255)/256, 256>>>(
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const float*>(w.data_ptr()),
        static_cast<float*>(out.data_ptr()),
        num_heads, seq_len);
}

void fused_index_remap_impl(ffi::TensorView topk_idx, ffi::TensorView page_indices,
                            ffi::TensorView topk_tokens) {
    int k = topk_idx.shape()[0];
    fused_index_remap_kernel<<<(k+255)/256, 256>>>(
        static_cast<const int64_t*>(topk_idx.data_ptr()),
        static_cast<const int64_t*>(page_indices.data_ptr()),
        static_cast<int32_t*>(topk_tokens.data_ptr()), k);
}

void gather_dequant_fp8_impl(ffi::TensorView kv_cache, ffi::TensorView page_indices,
                             ffi::TensorView output) {
    int num_selected = page_indices.shape()[0];
    if (num_selected == 0) return;
    gather_dequant_fp8_kernel<<<num_selected, 8>>>(
        static_cast<const uint8_t*>(kv_cache.data_ptr()),
        static_cast<const int64_t*>(page_indices.data_ptr()),
        static_cast<float*>(output.data_ptr()), num_selected);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequant_fp8, dequant_fp8_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(convert_q, convert_q_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_relu_weight, fused_relu_weight_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(fused_index_remap, fused_index_remap_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gather_dequant_fp8, gather_dequant_fp8_impl);
