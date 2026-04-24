"""Indexer FP8 MQA-logits kernel — near-verbatim port of CuTe DSL fp16_gemm_0.py.

Reference: NVIDIA/cutlass examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_0.py.
Structural differences from the reference:
  - io_dtype: Float16 → Float8E4M3FN
  - mma_inst_shape_mnk: (128,256,16) → (128,64,32) — FP8 native K=32, indexer N=64
  - mma_tiler_mnk: (128,256,64) → (128,64,128) — full HEAD_DIM=128 collapsed in K

Phase 1 MVP scope:
  - Output is the raw S = Q @ K^T tensor in FP32 (shape [128 padded heads, 64 slots, L]).
  - Python-side `binding.py` does Python pre-gather of K via block_table AND
    Python post-process: ReLU → Σ_h w[h] × S[h, slot] × scale[slot] → writes logits.
  - All warp-specialization, pipelining, and in-kernel epilogue fusion deferred to Phase 2.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float8E4M3FN
acc_dtype = cutlass.Float32

# Indexer scoring shapes (per single (batch, page) CTA):
#   Q: [NUM_HEADS=64, HEAD_DIM=128]  → MMA M=64 (padded to 128)
#   K: [PAGE_SIZE=64, HEAD_DIM=128]  → MMA N=64
#   S = Q @ K^T: [NUM_HEADS, PAGE_SIZE] in FP32 (UMMA writes 128 rows, first 64 valid).
mma_inst_shape_mnk = (128, 64, 32)
mma_tiler_mnk = (128, 64, 128)
threads_per_cta = 128

ab_stages = 1
acc_stage = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, bidz = cute.arch.block_idx()
    # Grid is (1, 1, L), so bidx==bidy==0 always; bidz selects the (batch,page).
    # Slice each 3D tensor by bidz to get a clean 2D (M,K) / (N,K) / (M,N) view.
    mA_mk = mA_mkl[None, None, bidz]
    mB_nk = mB_nkl[None, None, bidz]
    mC_mn = mC_mnl[None, None, bidz]
    mma_coord_mnk = (0, 0, None)

    # SMEM allocation
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    # TMEM allocation
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    # Prefetch tma descriptors
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # Pipeline configuration
    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, threads_per_cta,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA
    gA = cute.local_tile(mA_mk, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nk, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mn, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # subtile_cnt=1 because our N=64 is small enough that one TMEM→RMEM pass
    # covers the whole tile (128 threads × 64 fp32/thread = 8192 = M*N).  The
    # fp16_gemm_0.py reference uses subtile_cnt=4 for N=256.
    subtile_cnt = 1
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]),
                  cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)  # FP32 output

    # Mainloop
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 1):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a, tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b, tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier,
            )

            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
        acc_empty.commit()

    # Epilogue
    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        # Phase 1 MVP: pass-through (no in-kernel ReLU/weighted-sum/scale).
        # Output is FP32 to match acc_dtype; Python applies the indexer epilogue.
        tCrC.store(tCrAcc.load())
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaFP8Op(
        io_dtype, acc_dtype, mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a.element_type, ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b.element_type, ab_stages,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op_tma, a, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op_tma, b, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma,
    )

    # Grid: M_tiles × N_tiles × L. For our problem M=128 (padded), N=64, both
    # exactly fit one MMA tile, so M_tiles=N_tiles=1 and L is the full count of
    # (batch, page) pairs.
    L = c.layout.shape[2]
    grid_shape = (1, 1, L)
    kernel(
        tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor,
        c, a_smem_layout, b_smem_layout,
    ).launch(
        grid=grid_shape, block=(threads_per_cta, 1, 1),
    )


# Public entry — match the indexer kernel signature expected by binding.py.
@cute.jit
def run(
    q: cute.Tensor,        # [NUM_HEADS, HEAD_DIM, L] FP8E4M3FN; L = B*max_pages (Q replicated)
    k: cute.Tensor,        # [PAGE_SIZE, HEAD_DIM, L] FP8E4M3FN; L = B*max_pages (K gathered)
    s_out: cute.Tensor,    # [NUM_HEADS_padded=128, PAGE_SIZE, L] FP32 (raw Q @ K^T)
):
    """Single-CTA-per-(batch,page) GEMM. Outputs raw FP32 scores; Python
    fuses ReLU+weighted-sum+scale and gathers into the dense logits buffer."""
    host_function(q, k, s_out)
