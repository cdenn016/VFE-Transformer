# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 13:48:03 2025

@author: chris and christine
"""

#!/usr/bin/env python
"""
Example: How to integrate WikiText-103 loader into visualization script.

This shows you exactly what "integrate your WikiText-103 loader" means.
"""

import torch
from transformer.data import create_dataloaders

def get_validation_sequence_from_wikitext(
    dataset='wikitext-103',
    max_seq_len=128,
    batch_idx=0,
    sample_idx=0,
):
    """
    Load a REAL validation sequence from WikiText-103.

    Args:
        dataset: 'wikitext-2' or 'wikitext-103'
        max_seq_len: Sequence length
        batch_idx: Which batch to sample from (default: first batch)
        sample_idx: Which sequence in the batch (default: first sequence)

    Returns:
        token_ids: (1, N) tensor - the actual sequence
        description: str - what this sequence is
        tokenizer: the tokenizer (for decoding)
    """
    print(f"Loading real validation data from {dataset}...")

    # Use YOUR existing data loader!
    train_loader, val_loader, vocab_size = create_dataloaders(
        max_seq_len=max_seq_len,
        batch_size=8,
        vocab_size=50257,  # GPT-2 vocab size
        dataset=dataset,
    )

    # Get the validation dataset to access tokenizer
    val_dataset = val_loader.dataset

    # Get a specific batch
    for i, (input_ids, target_ids) in enumerate(val_loader):
        if i == batch_idx:
            # Extract specific sequence
            token_ids = input_ids[sample_idx:sample_idx+1]  # (1, N)

            # Try to get tokenizer for decoding
            tokenizer = None
            if hasattr(val_dataset, 'tokenizer'):
                tokenizer = val_dataset.tokenizer
            elif hasattr(val_dataset, 'enc'):  # tiktoken
                tokenizer = val_dataset.enc

            # Create description
            description = f"VALIDATION DATA: {dataset.upper()}, batch {batch_idx}, sample {sample_idx}"

            return token_ids, description, tokenizer

    raise ValueError(f"Batch {batch_idx} not found in validation loader")


# =============================================================================
# Example usage: Replace the TODO in visualize_attention_with_context.py
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("EXAMPLE: Loading Real Validation Data")
    print("=" * 80)

    # This is what you'd put in the visualization script
    token_ids, description, tokenizer = get_validation_sequence_from_wikitext(
        dataset='wikitext-2',  # Start with smaller dataset for speed
        max_seq_len=128,
        batch_idx=0,
        sample_idx=0,
    )

    print(f"\n[LOADED] {description}")
    print(f"  Shape: {token_ids.shape}")
    print(f"  Token IDs (first 20): {token_ids[0, :20].tolist()}")

    # Decode the sequence
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, 'decode'):
                # HuggingFace tokenizer
                decoded_text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
                token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]
            elif hasattr(tokenizer, 'decode_tokens_bytes'):
                # tiktoken
                decoded_text = tokenizer.decode(token_ids[0].tolist())
                # For tiktoken, decode each token separately
                token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]
            else:
                decoded_text = "[tokenizer decode not available]"
                token_strs = [f"tok{i}" for i in token_ids[0].tolist()]

            print(f"\n[DECODED TEXT]")
            print(f"  {decoded_text[:300]}{'...' if len(decoded_text) > 300 else ''}")
            print(f"\n[TOKENS]")
            print(f"  {' | '.join(token_strs[:20])}{'...' if len(token_strs) > 20 else ''}")

        except (KeyError, IndexError, TypeError, UnicodeDecodeError) as e:
            print(f"[WARN] Could not decode: {e}")
    else:
        print(f"[WARN] No tokenizer available for decoding")

    print("\n" + "=" * 80)
    print("✓ This is REAL validation data, not random/fake text!")
    print("  Now you can visualize attention on actual WikiText sequences.")
    print("=" * 80)