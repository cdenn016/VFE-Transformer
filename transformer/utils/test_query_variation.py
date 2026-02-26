#!/usr/bin/env python3
"""
Quick test: Query-side variation for λ_β = 1.0 model

Tests if different tokens attend differently despite uniform key-side patterns.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from transformer.core.model import GaugeTransformerLM


def analyze_query_variation(beta, input_ids=None, tokenizer=None):
    """
    Analyze if different queries (tokens) attend differently.

    Args:
        beta: (B, H, N, N) attention weights
        input_ids: (B, N) token IDs (optional, for display)
        tokenizer: For decoding (optional)
    """
    B, H, N, _ = beta.shape
    beta_np = beta[0].cpu().numpy()

    print("=" * 70)
    print("QUERY-SIDE VARIATION ANALYSIS")
    print("=" * 70)

    # For each head, compute pairwise L2 distances between attention ROWS
    # (Each row is how one query attends to all keys)

    for h in range(H):
        print(f"\nHead {h}:")

        # Compute all pairwise row distances
        dists = []
        for i in range(min(N, 20)):  # First 20 tokens
            for j in range(i + 1, min(N, 20)):
                # L2 distance between row i and row j
                dist = np.linalg.norm(beta_np[h, i, :] - beta_np[h, j, :])
                dists.append(dist)

        if len(dists) > 0:
            mean_dist = np.mean(dists)
            max_dist = np.max(dists)
            min_dist = np.min(dists)

            print(f"  Query variation (L2 between attention patterns):")
            print(f"    Mean: {mean_dist:.6f}")
            print(f"    Range: [{min_dist:.6f}, {max_dist:.6f}]")

            if mean_dist < 0.001:
                print(f"    → TRULY UNIFORM! All queries attend identically!")
            elif mean_dist < 0.01:
                print(f"    → Very low variation (queries almost identical)")
            elif mean_dist < 0.05:
                print(f"    → Low variation (weak differentiation)")
            else:
                print(f"    → Good variation (queries differ)")

        # Show actual attention patterns for first few tokens
        if input_ids is not None and tokenizer is not None and h == 0:
            print(f"\n  Sample attention patterns (first 5 tokens):")
            for i in range(min(5, N)):
                try:
                    token = tokenizer.decode([input_ids[0, i].item()])
                    attn_vals = beta_np[h, i, :10]  # First 10 positions
                    print(f"    Token {i} ({repr(token)}): {attn_vals}")
                except (KeyError, IndexError, TypeError, UnicodeDecodeError):
                    attn_vals = beta_np[h, i, :10]
                    print(f"    Token {i}: {attn_vals}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    all_dists = []
    for h in range(H):
        for i in range(min(N, 20)):
            for j in range(i + 1, min(N, 20)):
                dist = np.linalg.norm(beta_np[h, i, :] - beta_np[h, j, :])
                all_dists.append(dist)

    if len(all_dists) > 0:
        overall_mean = np.mean(all_dists)
        print(f"Average query variation across all heads: {overall_mean:.6f}")

        if overall_mean < 0.001:
            print("\n❌ CRITICAL: Truly uniform attention!")
            print("   All queries attend identically to all keys.")
            print("   Model cannot differentiate context based on query token.")
            print("   → How does this work?! Mystery!")
        elif overall_mean < 0.01:
            print("\n⚠️  Very weak query variation.")
            print("   Different tokens attend almost identically.")
            print("   Differentiation must come from elsewhere (φ, FFN, etc.)")
        elif overall_mean < 0.05:
            print("\n⚠️  Weak query variation.")
            print("   Some differentiation exists but is subtle.")
        else:
            print("\n✓ Significant query variation!")
            print("   Different tokens attend to different context patterns.")


def test_belief_similarity(model, input_ids):
    """
    Check if embeddings are truly compressed (all similar).
    """
    print("\n" + "=" * 70)
    print("BELIEF SPACE ANALYSIS")
    print("=" * 70)

    with torch.no_grad():
        # Get initial embeddings
        mu, sigma, phi = model.token_embed(input_ids)

        # Compute pairwise distances in μ space
        mu_batch = mu[0]  # (N, K)
        N, K = mu_batch.shape

        dists = []
        for i in range(min(N, 20)):
            for j in range(i + 1, min(N, 20)):
                dist = torch.norm(mu_batch[i] - mu_batch[j]).item()
                dists.append(dist)

        mean_dist = np.mean(dists)
        print(f"Average pairwise distance in μ space: {mean_dist:.4f}")

        if mean_dist < 0.5:
            print("  → ❌ Embeddings COLLAPSED! All tokens very similar.")
        elif mean_dist < 1.0:
            print("  → ⚠️  Embeddings compressed but distinct.")
        else:
            print("  → ✓ Embeddings well-separated.")

        # Show sample embeddings
        print(f"\nFirst 3 token embeddings (μ only, first 5 dims):")
        for i in range(min(3, N)):
            print(f"  Token {i}: {mu_batch[i, :5].cpu().numpy()}")


def main(checkpoint_path: str = None):
    """
    Run query variation analysis on a trained model.

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses command line arg.
    """
    import argparse

    if checkpoint_path is None:
        parser = argparse.ArgumentParser(description='Analyze query-side variation in attention')
        parser.add_argument('checkpoint', type=str, nargs='?', default=None,
                          help='Path to model checkpoint (best_model.pt)')
        args = parser.parse_args()
        checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        print("ERROR: No checkpoint path provided.")
        print("Usage: python test_query_variation.py <path/to/best_model.pt>")
        return

    print("Loading model...")

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("\nPlease update the checkpoint_path in this script to point to your λ_β=1.0 model.")
        print("Exiting...")
        return

    # Try to load experiment_config.json first (more reliable)
    checkpoint_dir = Path(checkpoint_path).parent
    config_json_path = checkpoint_dir / "experiment_config.json"

    config = None
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        with open(config_json_path, 'r') as f:
            json_data = json.load(f)

        # Check if config is nested under a 'config' key
        if 'config' in json_data and isinstance(json_data['config'], dict):
            config = json_data['config']
            print(f"✓ Loaded config from experiment_config.json (nested under 'config' key)")
        else:
            config = json_data
            print(f"✓ Loaded config from experiment_config.json")
    else:
        print(f"Warning: {config_json_path} not found, trying to extract from checkpoint...")

        # Try to extract config from checkpoint pickle
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                config = ckpt_config
            elif hasattr(ckpt_config, '__dict__'):
                config = vars(ckpt_config)
            print(f"✓ Extracted config from checkpoint")
        else:
            print("❌ No config found! Using fallback defaults")
            config = {
                'vocab_size': 50257,
                'embed_dim': 28,
                'n_layers': 6,
                'irrep_spec': [('ℓ0', 5, 1), ('ℓ1', 3, 3), ('ℓ2', 1, 5)],
                'hidden_dim': 112,
                'max_seq_len': 128,
                'kappa_beta': 1.0,
                'dropout': 0.1,
                'pos_encoding_mode': 'learned',
                'evolve_sigma': True,
                'evolve_phi': False,
                'tie_embeddings': True,
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }

    # Add missing keys with sensible defaults if needed
    # Legacy compat: old configs stored kappa_beta_base instead of kappa_beta
    if 'kappa_beta' not in config:
        config['kappa_beta'] = config.pop('kappa_beta_base', 1.0)
    config.pop('kappa_beta_auto_scale', None)
    config.pop('kappa_beta_base', None)
    config.pop('kappa_beta_k_ref', None)

    if 'use_diagonal_covariance' not in config and 'diagonal_covariance' in config:
        config['use_diagonal_covariance'] = config['diagonal_covariance']

    # Show the actual config we're using
    print(f"\nConfig:")
    for key in ['vocab_size', 'embed_dim', 'n_layers', 'irrep_spec', 'kappa_beta', 'ffn_mode']:
        if key in config:
            print(f"  {key}: {config[key]}")

    model = GaugeTransformerLM(config)

    # Load model weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"✓ Loaded checkpoint from {checkpoint_path}")

    model.eval()

    # Create test input
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        text = "The cat sat on the mat while the dog chased the ball"
        input_ids = tokenizer.encode(text, return_tensors='pt')
        print(f"\nTest text: {text}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    except (ImportError, OSError, ValueError):
        print("Warning: Could not load tokenizer, using random input")
        input_ids = torch.randint(0, config.vocab_size, (1, 20))
        tokenizer = None

    # Get attention
    print("\nComputing attention...")
    with torch.no_grad():
        _, attn_info = model.forward_with_attention(input_ids)
        beta = attn_info['beta']  # (B, H, N, N)

    print(f"Attention shape: {beta.shape}")

    # Analyze query variation
    analyze_query_variation(beta, input_ids, tokenizer)

    # Analyze belief space
    test_belief_similarity(model, input_ids)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()