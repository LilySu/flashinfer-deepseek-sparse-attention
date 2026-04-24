"""CuTe DSL port of the DSA indexer scoring kernel.

Contents:
  layouts.py — hierarchical layout scaffold (committed 2d8bb1f on cute-dsl-port)
  kernel.py  — @cute.jit host + @cute.kernel device function

Dispatch entry point is `indexer/solution/python/binding.py::kernel`; this
subpackage is loaded only when INDEXER_BACKEND=cute_dsl. Default dispatch
remains Phase 2c to preserve submission-v12 behavior.
"""
