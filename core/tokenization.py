"""
MIDI Tokenization using REMI (Revamped MIDI-derived events) representation.
Provides structured token conversion for piano music completion.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    from miditok import REMI, TokenizerConfig
    from miditok.constants import TEMPO, TIME_SIGNATURE
except ImportError:
    print("[WARNING] miditok not installed. Install with: pip install miditok")
    REMI = None


class PianoTokenizer:
    """
    MIDI tokenizer specifically designed for piano music completion tasks.
    Uses REMI representation which explicitly encodes bars for easy segmentation.
    """
    
    def __init__(self, vocab_size: int = 65536, max_bar_embedding: int = 300):
        """
        Initialize piano tokenizer with REMI configuration.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_bar_embedding: Maximum number of bars for positional encoding
        """
        if REMI is None:
            raise ImportError("miditok is required. Install with: pip install miditok")
        
        # Configure REMI tokenizer for piano music
        config = TokenizerConfig(
            num_velocities=32,  # Velocity quantization levels
            use_chords=False,   # Don't use chord detection (keep raw notes)
            use_rests=True,     # Explicitly encode rests
            use_tempos=True,    # Include tempo changes
            use_time_signatures=True,  # Include time signature changes
            use_programs=False,  # Piano only, no need for program changes
            beat_res={(0, 4): 8, (4, 12): 4},  # Beat resolution: 8th notes for 0-4 beats, 16th for 4-12
            num_tempos=32,      # Tempo quantization levels
            tempo_range=(40, 250),  # BPM range
        )
        
        self.tokenizer = REMI(config)
        self.vocab_size = vocab_size
        self.max_bar_embedding = max_bar_embedding
        
    def tokenize_midi(self, midi_path: str) -> List[int]:
        """
        Convert MIDI file to token sequence.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.midi_to_tokens(midi_path)
        
        # REMI returns list of tokens per track, flatten for single track (piano)
        if isinstance(tokens, list) and len(tokens) > 0:
            if isinstance(tokens[0], list):
                # Multiple tracks, take the first one or merge
                token_ids = [self.tokenizer.token_to_id(token) for token in tokens[0]]
            else:
                token_ids = [self.tokenizer.token_to_id(token) for token in tokens]
        else:
            token_ids = []
            
        return token_ids
    
    def detokenize(self, token_ids: List[int], output_path: str):
        """
        Convert token sequence back to MIDI file.
        
        Args:
            token_ids: List of token IDs
            output_path: Path to save MIDI file
        """
        tokens = [self.tokenizer.id_to_token(tid) for tid in token_ids]
        midi = self.tokenizer.tokens_to_midi([tokens])
        midi.dump(output_path)
        
    def find_bar_indices(self, token_ids: List[int]) -> List[int]:
        """
        Find indices of all Bar tokens in the sequence.
        These serve as anchors for sliding window segmentation.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of indices where Bar tokens occur
        """
        bar_indices = []
        for i, token_id in enumerate(token_ids):
            token_str = self.tokenizer.id_to_token(token_id)
            if token_str.startswith("Bar"):
                bar_indices.append(i)
        return bar_indices
    
    def extract_metadata_tokens(self, token_ids: List[int], up_to_index: int) -> Dict[str, Optional[int]]:
        """
        Extract the most recent tempo and time signature tokens up to a given index.
        This ensures context segments have complete musical metadata.
        
        Args:
            token_ids: Full token sequence
            up_to_index: Index to search backwards from
            
        Returns:
            Dictionary with 'tempo' and 'time_signature' token IDs (or None if not found)
        """
        metadata = {'tempo': None, 'time_signature': None}
        
        # Search backwards for most recent metadata tokens
        for i in range(up_to_index - 1, -1, -1):
            token_str = self.tokenizer.id_to_token(token_ids[i])
            
            if metadata['tempo'] is None and token_str.startswith("Tempo"):
                metadata['tempo'] = token_ids[i]
            
            if metadata['time_signature'] is None and token_str.startswith("TimeSig"):
                metadata['time_signature'] = token_ids[i]
            
            # Stop if we found both
            if metadata['tempo'] is not None and metadata['time_signature'] is not None:
                break
                
        return metadata
    
    def is_structural_token(self, token_id: int) -> bool:
        """
        Check if a token represents an atomic musical boundary.
        Used for safe truncation to avoid breaking NoteOn/Pitch/Velocity groups.
        
        Args:
            token_id: Token ID to check
            
        Returns:
            True if token is a structural boundary (Bar, Pitch, NoteOn, Tempo, TimeSig)
        """
        token_str = self.tokenizer.id_to_token(token_id)
        return token_str.startswith(("Bar", "Pitch", "NoteOn", "Tempo", "TimeSig"))
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer.vocab)


def create_context_completion_pairs(
    token_ids: List[int],
    tokenizer: PianoTokenizer,
    n_context_bars: int = 4,
    n_completion_bars: int = 2,
    step: int = 1
) -> List[Dict[str, List[int]]]:
    """
    Create context-completion pairs using bar-based sliding window algorithm.
    
    This is the core data preparation algorithm that segments a full piece
    into training examples of [N bars context] -> [M bars completion].
    
    Args:
        token_ids: Full token sequence from a MIDI file
        tokenizer: PianoTokenizer instance
        n_context_bars: Number of bars for context
        n_completion_bars: Number of bars for completion target
        step: Stride of sliding window (in bars)
        
    Returns:
        List of dictionaries with 'context' and 'completion' token lists
    """
    # Find all bar boundaries
    bar_indices = tokenizer.find_bar_indices(token_ids)
    
    # Check if piece is long enough
    total_bars_needed = n_context_bars + n_completion_bars
    if len(bar_indices) < total_bars_needed:
        # Piece too short, discard
        return []
    
    data_pairs = []
    
    # Slide window across the piece
    for i in range(0, len(bar_indices) - total_bars_needed + 1, step):
        # Define boundaries
        context_start_idx = bar_indices[i]
        completion_start_idx = bar_indices[i + n_context_bars]
        
        # Completion end: either next window boundary or end of piece
        if (i + total_bars_needed) < len(bar_indices):
            completion_end_idx = bar_indices[i + total_bars_needed]
        else:
            completion_end_idx = len(token_ids)
        
        # Extract context and completion segments
        context_ids = token_ids[context_start_idx:completion_start_idx]
        completion_ids = token_ids[completion_start_idx:completion_end_idx]
        
        # Prepend essential metadata to context if it's from middle of piece
        if i > 0:
            metadata = tokenizer.extract_metadata_tokens(token_ids, context_start_idx)
            prepend_tokens = []
            
            if metadata['tempo'] is not None:
                prepend_tokens.append(metadata['tempo'])
            if metadata['time_signature'] is not None:
                prepend_tokens.append(metadata['time_signature'])
            
            if prepend_tokens:
                context_ids = prepend_tokens + context_ids
        
        data_pairs.append({
            'context': context_ids,
            'completion': completion_ids
        })
    
    return data_pairs


def process_midi_directory(
    midi_dir: str,
    tokenizer: PianoTokenizer,
    n_context_bars: int = 4,
    n_completion_bars: int = 2,
    step: int = 1
) -> List[Dict[str, List[int]]]:
    """
    Process all MIDI files in a directory to create training dataset.
    
    Args:
        midi_dir: Directory containing MIDI files
        tokenizer: PianoTokenizer instance
        n_context_bars: Number of bars for context
        n_completion_bars: Number of bars for completion
        step: Stride of sliding window
        
    Returns:
        List of all context-completion pairs from all MIDI files
    """
    all_pairs = []
    midi_files = list(Path(midi_dir).glob("**/*.mid")) + list(Path(midi_dir).glob("**/*.midi"))
    
    print(f"[Tokenization] Found {len(midi_files)} MIDI files")
    
    for idx, midi_file in enumerate(midi_files):
        if (idx + 1) % 100 == 0:
            print(f"[Tokenization] Processed {idx + 1}/{len(midi_files)} files, "
                  f"generated {len(all_pairs)} pairs so far")
        
        try:
            token_ids = tokenizer.tokenize_midi(str(midi_file))
            pairs = create_context_completion_pairs(
                token_ids, tokenizer, n_context_bars, n_completion_bars, step
            )
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"[WARNING] Failed to process {midi_file}: {e}")
            continue
    
    print(f"[Tokenization] Complete! Generated {len(all_pairs)} training pairs from {len(midi_files)} files")
    return all_pairs
