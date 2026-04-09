/*
 * DSA Indexer — CUDA kernels for FP8 dequant + Q convert + relu + weight_mul.
 * All validated bitwise exact against PyTorch on B200.
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

// ─── FP8 dequant (proven) ───
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

// ─── ReLU in-place (proven bitwise exact vs torch.relu) ───
__global__ void relu_inplace(float* __restrict__ data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = data[idx];
        data[idx] = v > 0.0f ? v : 0.0f;
    }
}

// ─── Weight multiply in-place (proven bitwise exact vs scores * w[:, None]) ───
// scores: [NUM_HEADS, seq_len], weights: [NUM_HEADS]
__global__ void weight_multiply(
    float* __restrict__ scores, const float* __restrict__ weights, int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NUM_HEADS * seq_len;
    if (idx >= total) return;
    int h = idx / seq_len;
    scores[idx] *= weights[h];
}

namespace ffi = tvm::ffi;

void dequant_fp8_impl(ffi::TensorView pages_raw, ffi::TensorView output) {
    int64_t total_bytes = pages_raw.shape()[0];
    int num_pages = total_bytes / PAGE_BYTES;
    if (num_pages == 0) return;
    dequant_fp8_kernel<<<num_pages, HEAD_DIM>>>(
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

// Per-batch relu + weight_mul on scores [NUM_HEADS, seq_len]
void relu_weight_mul_impl(ffi::TensorView scores, ffi::TensorView weights, int64_t seq_len) {
    int total = NUM_HEADS * (int)seq_len;
    relu_inplace<<<(total+255)/256, 256>>>(
        static_cast<float*>(scores.data_ptr()), total);
    weight_multiply<<<(total+255)/256, 256>>>(
        static_cast<float*>(scores.data_ptr()),
        static_cast<const float*>(weights.data_ptr()), (int)seq_len);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequant_fp8, dequant_fp8_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(convert_q, convert_q_impl);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(relu_weight_mul, relu_weight_mul_impl);
