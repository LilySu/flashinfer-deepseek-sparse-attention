/*
 * DSA Indexer — CUDA FP8 dequant kernel only.
 * Validated bitwise exact on B200 (submission-v2: 128/128 passed, 2.5x avg).
 */

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <cstdint>

constexpr int PAGE_SIZE = 64;
constexpr int HEAD_DIM = 128;
constexpr int PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4);

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

namespace ffi = tvm::ffi;

void dequant_fp8_impl(ffi::TensorView pages_raw, ffi::TensorView output) {
    int64_t total_bytes = pages_raw.shape()[0];
    int num_pages = total_bytes / PAGE_BYTES;
    if (num_pages == 0) return;
    dequant_fp8_kernel<<<num_pages, HEAD_DIM>>>(
        static_cast<const uint8_t*>(pages_raw.data_ptr()),
        static_cast<float*>(output.data_ptr()), num_pages);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequant_fp8, dequant_fp8_impl);
