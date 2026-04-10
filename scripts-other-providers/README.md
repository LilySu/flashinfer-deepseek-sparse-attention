# Running DSA Indexer on Alternative GPU Providers

When Modal is down, use Nebius or RunPod B200 instances.

## Environment Requirements

The contest evaluation runs on:
- **GPU:** NVIDIA B200 (SM100, Compute Capability 10.0)
- **CUDA:** 13.0
- **PyTorch:** 2.11.0+cu130
- **Python:** 3.12
- **flashinfer-bench:** 0.1.2+
- **tvm-ffi:** 0.1.10+

### Key: `torch.backends.cuda.matmul.allow_tf32 = False` is the default on PyTorch 2.11 B200.

## Provider Comparison

| | Modal (contest) | Nebius | RunPod |
|--|----------------|--------|--------|
| B200 available | Yes | Yes (HGX B200) | Yes |
| CUDA 13.0 | Yes | Yes | No (12.8 latest) |
| PyTorch 2.11 | Yes | Manual install | No (2.8 latest) |
| SM100 support | Yes | Yes | Needs CUDA 13.0 |
| Price | Contest credits | ~$5/hr | $4.99/hr |

**Nebius is the best alternative** — has CUDA 13.0 and B200, closest to contest environment.
RunPod has B200 but older CUDA (12.8) which may not have full SM100 support.
