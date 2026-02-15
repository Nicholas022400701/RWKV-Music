"""
O(1) Memory Inference Engine for Piano Completion.
Switches to RNN mode for constant memory generation regardless of sequence length.
"""

import os
import torch
import argparse
from pathlib import Path

# Must hijack environment BEFORE importing RWKV
from core.env_hijack import hijack_windows_cuda_env, verify_cuda_setup
hijack_windows_cuda_env()

from core.tokenization import PianoTokenizer


@torch.no_grad()
def generate_inspiration(
    model,
    context_tokens: list,
    max_new_tokens: int = 256,
    temperature: float = 0.85,
    top_p: float = 0.90,
    top_k: int = 0
) -> list:
    """
    Generate musical completion given context using RNN mode.
    
    The beauty of RWKV: During training, use parallel O(T) mode for efficiency.
    During inference, switch to RNN mode for O(1) memory per step.
    
    All musical history, melodic motifs, harmonic progressions are losslessly
    compressed into an O(1) state matrix through exponential decay.
    
    Args:
        model: RWKV model instance
        context_tokens: List of context token IDs
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold (lower = more conservative)
        top_k: Top-k sampling (0 = disabled)
    
    Returns:
        List of generated token IDs
    """
    # Initialize state (None for first token)
    # Musical memory represented as a constant-size state matrix
    state = None
    
    print(f"[Generation] Processing {len(context_tokens)} context tokens...")
    
    # Phase 1: Prefill context (consume context to build state)
    for i, token in enumerate(context_tokens[:-1]):
        if (i + 1) % 100 == 0:
            print(f"[Generation] Processed {i+1}/{len(context_tokens)} context tokens")
        _, state = model.forward([token], state)
    
    # Get output from last context token
    out, state = model.forward([context_tokens[-1]], state)
    
    print(f"[Generation] Context processed. Starting generation...")
    
    # Phase 2: Autoregressive generation with constant memory
    generated = []
    current_token = sample_token(out, temperature, top_p, top_k)
    
    for step in range(max_new_tokens):
        generated.append(current_token)
        
        if (step + 1) % 50 == 0:
            print(f"[Generation] Generated {step+1}/{max_new_tokens} tokens")
        
        # Forward one step: O(1) memory, O(1) time
        # State update: State_t = State_{t-1} * exp(-w) + K * V
        # This is the mathematical equivalence that makes RWKV efficient
        out, state = model.forward([current_token], state)
        
        # Sample next token with temperature and nucleus sampling
        current_token = sample_token(out, temperature, top_p, top_k)
    
    print(f"[Generation] Complete! Generated {len(generated)} tokens")
    return generated


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 0
) -> int:
    """
    Sample next token from logits using temperature and nucleus sampling.
    
    Nucleus sampling truncates unreasonable off-key noise while preserving
    creative diversity.
    
    Args:
        logits: Logits from model [vocab_size]
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling (0 = disabled)
    
    Returns:
        Sampled token ID
    """
    # Apply temperature scaling
    logits = logits / temperature
    
    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Top-k sampling (if enabled)
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        probs_filtered = torch.zeros_like(probs)
        probs_filtered[top_k_indices] = top_k_probs
        probs = probs_filtered / probs_filtered.sum()
    
    # Nucleus (top-p) sampling
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find tokens to remove (cumulative probability > top_p)
        remove_mask = cumsum_probs > top_p
        
        # Keep at least one token
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        
        # Zero out removed tokens
        probs[sorted_indices[remove_mask]] = 0.0
        probs = probs / probs.sum()
    
    # Sample from filtered distribution
    token = torch.multinomial(probs, 1).item()
    return token


def main(args):
    """Main inference function."""
    
    print("=" * 70)
    print("RWKV Piano Music Completion - Inference Engine")
    print("=" * 70)
    
    # Verify CUDA setup
    verify_cuda_setup()
    
    # Initialize tokenizer
    print("\n" + "=" * 70)
    print("Initializing Tokenizer")
    print("=" * 70)
    tokenizer = PianoTokenizer()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Load model
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    try:
        from rwkv.model import RWKV
    except ImportError:
        print("[ERROR] RWKV library not installed!")
        print("Install from: https://github.com/BlinkDL/RWKV-LM")
        return
    
    print(f"Loading model from: {args.model_path}")
    model = RWKV(model=args.model_path, strategy='cuda fp16')
    print("[Model] Loaded successfully in RNN inference mode")
    
    # Load and tokenize context MIDI
    print("\n" + "=" * 70)
    print("Processing Context")
    print("=" * 70)
    print(f"Context MIDI: {args.context_midi}")
    
    context_tokens = tokenizer.tokenize_midi(args.context_midi)
    print(f"Context length: {len(context_tokens)} tokens")
    
    if len(context_tokens) == 0:
        print("[ERROR] Context MIDI file produced no tokens!")
        return
    
    # Truncate context if too long
    if args.max_context_len and len(context_tokens) > args.max_context_len:
        print(f"[WARNING] Context too long, truncating to {args.max_context_len} tokens")
        context_tokens = context_tokens[-args.max_context_len:]
    
    # Generate completion
    print("\n" + "=" * 70)
    print("Generating Completion")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Top-k: {args.top_k}")
    
    generated_tokens = generate_inspiration(
        model,
        context_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Combine context and generated tokens
    full_sequence = context_tokens + generated_tokens
    
    # Save to MIDI
    print("\n" + "=" * 70)
    print("Saving Output")
    print("=" * 70)
    
    output_path = Path(args.output_dir) / f"completion_{Path(args.context_midi).stem}.mid"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tokenizer.detokenize(full_sequence, str(output_path))
        print(f"Saved to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save MIDI: {e}")
        
        # Save tokens as fallback
        tokens_path = output_path.with_suffix('.txt')
        with open(tokens_path, 'w') as f:
            f.write(' '.join(map(str, full_sequence)))
        print(f"Saved tokens to: {tokens_path}")
    
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total tokens: {len(full_sequence)}")
    print(f"Context: {len(context_tokens)} tokens")
    print(f"Generated: {len(generated_tokens)} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RWKV Piano Completion Inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained RWKV model")
    
    # Input arguments
    parser.add_argument("--context_midi", type=str, required=True,
                       help="Path to context MIDI file")
    parser.add_argument("--max_context_len", type=int, default=2048,
                       help="Maximum context length (tokens)")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.85,
                       help="Sampling temperature (0.1-2.0, higher=more random)")
    parser.add_argument("--top_p", type=float, default=0.90,
                       help="Nucleus sampling threshold (0.0-1.0)")
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k sampling (0=disabled)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save generated MIDI")
    
    args = parser.parse_args()
    
    main(args)
