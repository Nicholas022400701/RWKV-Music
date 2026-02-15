# RWKV v8 "Heron" with ROSA Training Model Integration

This directory contains the training-capable RWKV v8 "Heron" model with **ROSA (Rapid Online Suffix Automaton)** extracted from the official [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository.

## RWKV v8 "Heron" with ROSA Architecture

RWKV v8 "Heron" is the latest RWKV architecture featuring:
- ✅ **ROSA (Rapid Online Suffix Automaton)** - Revolutionary pattern matching mechanism
- ✅ WKV7s CUDA kernels (state-based) with full backward pass support
- ✅ Improved attention mechanism with better long-context handling
- ✅ Enhanced numerical stability and training efficiency
- ✅ Optimized for modern GPUs (RTX 4090, etc.)

### What is ROSA?

**ROSA (Rapid Online Suffix Automaton)** is a groundbreaking innovation in RWKV v8 that:
- 🚀 **Pattern Recognition:** Automatically discovers and exploits repeating patterns in sequences
- 🚀 **Online Learning:** Updates suffix automaton structure in real-time during training
- 🚀 **Memory Efficiency:** Compresses sequence representations using pattern matching
- 🚀 **Copy & Count Abilities:** Achieves strong copying and counting capabilities with minimal parameters

ROSA enables models to learn complex tasks (arithmetic, reversal, copying) with orders of magnitude fewer parameters than traditional transformers!

## Why This is Necessary

The `rwkv` package available via `pip install rwkv` is an **inference-only** library that:
- ❌ Does NOT support gradient computation (no `backward()` pass)
- ❌ Does NOT include CUDA kernels for training
- ❌ Cannot be used for model training or fine-tuning

The official RWKV-LM repository provides the full training implementation with:
- ✅ CUDA kernels with backward pass support  
- ✅ Proper gradient computation through WKV operators
- ✅ Full training capabilities

## Files Included

### Model Files:
- `rwkv_v8_model.py` - RWKV v8 "Heron" model implementation
- `rosa_layer.py` - **ROSA (Rapid Online Suffix Automaton) layer implementation**
- `rosa_train.py` - **ROSA training script with 1-bit quantization**
- `rosa_lm.py` - **Pure ROSA language model (no RWKV, only ROSA + FFN)**
- `__init__.py` - Package initialization

### CUDA Kernels (from RWKV-LM/RWKV-v7/cuda/):
- `wkv7s.cu` - CUDA kernel for WKV7s (state-based) with forward and backward
- `wkv7s_op.cpp` - C++ operator wrapper

Note: v8 uses wkv7s kernels ("7s" = v7 with state management), which are optimized for RNN-mode inference and training.

## Key Features of RWKV v8 with ROSA

### ROSA (Rapid Online Suffix Automaton)

**What ROSA Does:**
ROSA is a suffix automaton that tracks repeating patterns in sequences and learns embeddings for them:

1. **Pattern Discovery:** As the model processes tokens, ROSA builds a suffix automaton that identifies all repeating subsequences
2. **Smart Embeddings:** Instead of traditional embeddings, ROSA uses pattern-based lookups:
   - If a pattern has been seen before → use learned embedding for that pattern
   - If pattern is new → use default embedding
3. **1-bit Quantization:** ROSA can work with 1-bit (binary) representations, drastically reducing memory

**Real-World Performance:**
- **40K parameters** can reverse 60-digit numbers with 99.8% accuracy (v7 needs millions of params!)
- **1M parameters** solve 40-digit arithmetic with 99% accuracy
- **Pure ROSA model (L12)** trained only on 1.5B tokens achieves copy & count abilities

### Architecture Improvements
- **Enhanced Time-Mixing:** Better attention mechanism with improved state propagation
- **Optimized Channel-Mixing:** More efficient FFN with `enn.weight` for token-specific adaptations
- **State Management:** Superior RNN-mode performance with efficient state updates
- **Numerical Stability:** Improved training stability through better weight initialization
- **ROSA Integration:** Suffix automaton for pattern-based sequence understanding

### Performance
- Faster inference in RNN mode (O(1) per token)
- More efficient training with better gradient flow
- Better long-context capability than v7
- Optimized for both GPT-mode (parallel) and RNN-mode (sequential)

## Usage

### Standard RWKV v8 Model

The model supports both RNN-mode (sequential, O(1) memory) and GPT-mode (parallel training):

```python
from core.rwkv_training.rwkv_v8_model import RWKV_x070

# Initialize model
model = RWKV_x070(args)

# RNN-mode (one token at a time)
output, state = model.forward_one(token_id, state)

# GPT-mode (sequence at once, for training)
output, state = model.forward_seq(token_ids, state, full_output=True)
```

### ROSA Layer Integration

Add ROSA pattern matching to your model:

```python
from core.rwkv_training.rosa_layer import ROSA_1bit_LAYER

# Add ROSA layer to your architecture
rosa_layer = ROSA_1bit_LAYER(n_embd, tau=1e-3)

# Forward pass
x = your_embeddings  # [B, T, C]
rosa_output = rosa_layer(x)  # Pattern-based representation

# ROSA automatically:
# 1. Builds suffix automaton from input sequence
# 2. Identifies repeating patterns
# 3. Returns learned embeddings for seen patterns
```

### Pure ROSA Language Model

Train a model using **only ROSA + FFN** (no RWKV attention):

```python
from core.rwkv_training.rosa_lm import ROSA_LM

# This is a lightweight model that achieves surprising capabilities
# with minimal parameters using pure pattern matching
model = ROSA_LM(n_layer=12, n_embd=768)
```

### Music Generation with ROSA

For music generation, ROSA is particularly powerful because:
- **Melodic Patterns:** Music contains many repeating motifs, perfect for ROSA
- **Chord Progressions:** Common chord sequences are automatically recognized
- **Rhythm Patterns:** Repeating rhythmic structures are efficiently encoded
- **Memory Efficiency:** 1-bit ROSA reduces memory, allowing longer sequences

## Requirements

To compile the CUDA kernels, you need:
1. **CUDA Toolkit** matching your PyTorch version
2. **C++ compiler** (GCC on Linux, MSVC on Windows)
3. **PyTorch** with CUDA support
4. **Head size must be 64** (hardcoded in CUDA kernel for optimal performance)

The kernels are compiled via JIT (Just-In-Time) when first imported:
```python
from torch.utils.cpp_extension import load
```

## Environment Variables (Required)

The RWKV v8 model requires these environment variables:
- `RWKV_JIT_ON="1"` - Enable JIT compilation
- `RWKV_HEAD_SIZE="64"` - Head size for attention (must be 64)
- `RWKV_MY_TESTING="x070"` - Version identifier (v8 uses x070 kernels)
- `RWKV_CUDA_ON="1"` - Enable CUDA kernels

These are automatically set by `core/env_hijack.py`.

## Model Architecture Details

### Time-Mixing (Attention)
v8 uses an improved time-mixing mechanism with:
- Learnable interpolation parameters (x_r, x_w, x_k, x_v, x_a, x_g)
- Multi-component mixing (w0, w1, w2, a0, a1, a2)
- Enhanced state management for better long-term dependencies

### Channel-Mixing (FFN)
v8 introduces token-specific enhancements:
- `enn.weight` for per-token adaptations
- Improved key-value transformations
- Better gradient flow through the network

## Performance Notes

### Memory Requirements
- State-based processing (wkv7s) for efficient RNN mode
- Supports both parallel (GPT-mode) and sequential (RNN-mode) processing
- Memory efficient for long sequences in RNN mode

### Speed
- **RNN mode:** O(1) time and memory per token (constant regardless of context length)
- **GPT mode:** O(T) parallel processing for training
- Optimized for bfloat16 on modern GPUs

### Training
- Supports both modes: use GPT-mode for parallel training, RNN-mode for inference
- Better gradient stability than previous versions
- Faster convergence with improved architecture

## Migration from v7

Key differences when migrating from v7:
1. **CUDA kernel:** v8 uses `wkv7s` (state-based) instead of `wkv7_clampw`
2. **Model structure:** Enhanced time-mixing and channel-mixing components
3. **State management:** Improved state propagation mechanism
4. **Token-specific FFN:** Added `enn.weight` for per-token adaptations

## License

The RWKV model code is from the official RWKV-LM repository:
https://github.com/BlinkDL/RWKV-LM

Licensed under Apache License 2.0.

## References

- **RWKV v8 Demo:** https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v8_rc00_demo.py
- **RWKV v8/ROSA:** https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8
- **RWKV Paper:** https://arxiv.org/abs/2305.13048
- **RWKV-LM Repository:** https://github.com/BlinkDL/RWKV-LM
