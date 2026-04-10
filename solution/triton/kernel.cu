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

namespace ffi = tvm::ffi;

void dequant_fp8_impl(ffi::TensorView pages_raw, ffi::TensorView output) {
    int64_t total_bytes = pages_raw.shape()[0];
    int num_pages = total_bytes / PAGE_BYTES;
    if (num_pages == 0) return;
    // Use vectorized kernel (8 threads per block, 16 dims per thread)
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequant_fp8, dequant_fp8_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(convert_q, convert_q_impl);
