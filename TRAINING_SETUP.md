# RWKV-Music Training Setup Guide

## Critical Issue Fixed: Inference vs Training Models

The original codebase incorrectly used the **inference-only** `rwkv` pip package, which does NOT support training because:

1. ❌ No gradient computation (no `backward()` implementation)
2. ❌ No CUDA kernels for `wkv_cuda_backward` 
3. ❌ Training would crash on first backward pass

## Solution: Training-Capable RWKV v8 "Heron" Model

This repository now includes the proper **training-capable** RWKV v8 "Heron" model from the official [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository with full backward pass support.

### What Changed

**BEFORE (Broken):**
```python
# requirements.txt
rwkv>=0.8.0  # ❌ Inference-only, no training support

# architecture.py  
from rwkv.model import RWKV  # ❌ Will crash during training
```

**AFTER (Fixed - Using v8):**
```python
# requirements.txt
# rwkv removed - using RWKV v8 training model from core/rwkv_training/
pytorch-lightning>=2.0.0  # ✅ Required for training

# architecture.py
from core.rwkv_training.rwkv_v8_model import RWKV_x070  # ✅ v8 "Heron" with full training support
```

## RWKV v8 "Heron" Architecture

RWKV v8 is the **latest stable architecture** featuring:
- ✅ Advanced WKV7s (state-based) CUDA kernels
- ✅ Improved attention and FFN mechanisms
- ✅ Better long-context handling
- ✅ Enhanced numerical stability

## Prerequisites for Training

### 1. CUDA Toolkit
The RWKV training model uses custom CUDA kernels that are JIT-compiled on first use.

**Required:**
- CUDA Toolkit 11.8+ or 12.x (must match PyTorch CUDA version)
- C++ compiler: GCC 7+ (Linux) or MSVC 2019+ (Windows)

**Installation:**
- Linux: `sudo apt install nvidia-cuda-toolkit build-essential`
- Windows: Install Visual Studio 2019+ with "Desktop development with C++" workload

### 2. PyTorch with CUDA
```bash
# Check your PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Install matching CUDA toolkit
# Example for CUDA 11.8:
conda install -c nvidia cuda-toolkit=11.8
```

### 3. Environment Variables

The training model requires these environment variables (automatically set by `env_hijack.py`):

```bash
export RWKV_JIT_ON="1"              # Enable JIT compilation
export RWKV_HEAD_SIZE_A="64"        # Attention head size  
export RWKV_MY_TESTING="x060"       # RWKV version (v6)
export RWKV_TRAIN_TYPE="states"     # Training mode
export RWKV_CUDA_ON="1"             # Enable CUDA kernels
```

## Training Model Files

Located in `core/rwkv_training/`:

```
core/rwkv_training/
├── README.md           # Documentation
├── model.py            # Training model with backward support
├── __init__.py         # Package init
└── cuda/               # CUDA kernels
    ├── wkv5_cuda.cu    # WKV5 CUDA implementation
    ├── wkv5_op.cpp     # WKV5 operator wrapper
    ├── wkv6_cuda.cu    # WKV6 CUDA implementation  
    ├── wkv6_op.cpp     # WKV6 operator wrapper
    └── ...             # Other CUDA files
```

## Verification

To verify your setup is correct:

```python
import torch
from core.rwkv_training.model import RWKV

# This should work without errors
print("✓ Training model imported successfully")

# Check CUDA compilation (will compile on first run)
# This may take a few minutes
```

## Common Issues

### Issue 1: "No module named 'core.rwkv_training.model'"

**Solution:** Ensure you're running from the project root directory:
```bash
cd /path/to/RWKV-Music
python train_parallel.py ...
```

### Issue 2: CUDA compilation fails

**Solution:** 
1. Check CUDA toolkit matches PyTorch: `nvcc --version` vs `torch.version.cuda`
2. On Windows: Ensure Visual Studio C++ tools are installed
3. Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability

### Issue 3: "RuntimeError: Cannot train with inference-only RWKV model"

**Solution:** The code detected you're using the wrong model. Make sure:
1. `core/rwkv_training/` directory exists with model files
2. Import path is correct: `from core.rwkv_training.model import RWKV`

## Testing Training

Quick test to ensure backward pass works:

```python
import torch
from core.architecture import PianoMuseRWKV

# Create dummy model (will fail if using inference-only package)
model = PianoMuseRWKV("path/to/pretrained.pth", strategy='cuda bf16')
model.train()

# Test forward and backward
input_ids = torch.randint(0, 1000, (2, 100)).cuda()
ctx_lengths = torch.tensor([50, 50])

logits = model(input_ids, ctx_lengths=ctx_lengths)
loss = logits.sum()
loss.backward()  # ✅ Should work with training model

print("✓ Backward pass successful!")
```

## References

- **RWKV-LM Repository:** https://github.com/BlinkDL/RWKV-LM
- **RWKV Paper:** https://arxiv.org/abs/2305.13048  
- **Training Examples:** See `RWKV-LM/RWKV-v5/` in the official repo

## Credits

The training model is extracted from the official RWKV-LM repository by BlinkDL.
Licensed under Apache License 2.0.
