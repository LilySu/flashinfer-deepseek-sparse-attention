# Offline Analysis Results — Fri Apr 10 23:16:43 UTC 2026

## Part 1: Triton Matmul Correctness Test

```
/tmp/tmpvb1i58qn/cuda_utils.c:7:10: fatal error: Python.h: No such file or directory
    7 | #include <Python.h>
      |          ^~~~~~~~~~
compilation terminated.
============================================================
Triton vs cuBLAS matmul correctness test
============================================================
Traceback (most recent call last):
  File "/home/glm5/flashinfer-deepseek-sparse-attention/test_triton_matmul_correctness.py", line 197, in <module>
    test_correctness()
  File "/home/glm5/flashinfer-deepseek-sparse-attention/test_triton_matmul_correctness.py", line 100, in test_correctness
    tri = triton_matmul(Q, K)
          ^^^^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/test_triton_matmul_correctness.py", line 69, in triton_matmul
    matmul_kernel[grid](
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 370, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/jit.py", line 700, in run
    device = driver.active.get_current_device()
             ^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 28, in active
    self._active = self.default
                   ^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 22, in default
    self._default = _create_driver()
                    ^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/driver.py", line 10, in _create_driver
    return active_drivers[0]()
           ^^^^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 720, in __init__
    self.utils = CudaUtils()  # TODO: make static
                 ^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/backends/nvidia/driver.py", line 62, in __init__
    mod = compile_module_from_src(
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 93, in compile_module_from_src
    so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/runtime/build.py", line 48, in _build
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
  File "/usr/lib/python3.12/subprocess.py", line 413, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/tmpvb1i58qn/cuda_utils.c', '-O3', '-shared', '-fPIC', '-Wno-psabi', '-o', '/tmp/tmpvb1i58qn/cuda_utils.cpython-312-x86_64-linux-gnu.so', '-l:libcuda.so.1', '-L/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/backends/nvidia/lib', '-L/lib/x86_64-linux-gnu', '-I/home/glm5/flashinfer-deepseek-sparse-attention/flashinfer-deepseek-sparse-attention/.venv/lib/python3.12/site-packages/triton/backends/nvidia/include', '-I/tmp/tmpvb1i58qn', '-I/usr/include/python3.12']' returned non-zero exit status 1.
