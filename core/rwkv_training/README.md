# RWKV Training Model Integration

This directory contains the training-capable RWKV model extracted from the official [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository.

## Why This is Necessary

The `rwkv` package available via `pip install rwkv` is an **inference-only** library that:
- ❌ Does NOT support gradient computation (no `backward()` pass)
- ❌ Does NOT include CUDA kernels for training (`wkv_cuda_backward`)
- ❌ Cannot be used for model training or fine-tuning

The official RWKV-LM repository provides the full training implementation with:
- ✅ CUDA kernels with backward pass support  
- ✅ Proper gradient computation through WKV operators
- ✅ Full training capabilities

## Files Included

### From RWKV-LM/RWKV-v5/src/:
- `model.py` - Full RWKV model with training support and custom CUDA kernels
- `__init__.py` - Package initialization

### From RWKV-LM/RWKV-v5/cuda/:
- `wkv5_cuda.cu` - CUDA kernel for WKV5 forward and backward
- `wkv5_op.cpp` - C++ operator wrapper
- `wkv6_cuda.cu` - CUDA kernel for WKV6 forward and backward
- `wkv6_op.cpp` - C++ operator wrapper
- Additional CUDA files for different RWKV versions

## Usage

The `core/architecture.py` file has been updated to use this training-capable model instead of the inference-only pip package.

```python
# OLD (WRONG - Inference only):
from rwkv.model import RWKV

# NEW (CORRECT - Training capable):
from core.rwkv_training.model import RWKV
```

## Requirements

To compile the CUDA kernels, you need:
1. CUDA Toolkit matching your PyTorch version
2. C++ compiler (GCC on Linux, MSVC on Windows)
3. PyTorch with CUDA support

The kernels are compiled via JIT (Just-In-Time) when first imported, using:
```python
from torch.utils.cpp_extension import load
```

## Environment Variables

The RWKV training model requires certain environment variables:
- `RWKV_JIT_ON` - Enable JIT compilation (set to "1")
- `RWKV_HEAD_SIZE_A` - Head size for attention (typically 64)
- `RWKV_MY_TESTING` - Version identifier (e.g., "x060" for v6)
- `RWKV_TRAIN_TYPE` - Training mode (e.g., "states")
- `RWKV_CTXLEN` - Context length for compilation optimization

## License

The RWKV model code is from the official RWKV-LM repository:
https://github.com/BlinkDL/RWKV-LM

Licensed under Apache License 2.0.

## References

- RWKV Paper: https://arxiv.org/abs/2305.13048
- RWKV-LM Repository: https://github.com/BlinkDL/RWKV-LM
- RWKV Chat/Inference Package: https://github.com/BlinkDL/ChatRWKV
