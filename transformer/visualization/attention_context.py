#!/usr/bin/env python
"""
Visualize attention patterns WITH FULL CONTEXT.

This script shows:
1. WHAT sequence is being visualized (decode tokens to readable text)
2. WHERE it came from (training data? validation? specific example?)
3. Per-head attention patterns (not averaged)
4. Diagnostic information about uniformity

Usage:
    # Random sequence (for debugging)
    python visualize_attention_with_context.py --mode random

    # From validation data (real examples)
    python visualize_attention_with_context.py --mode validation --data-path path/to/data

    # Specific text
    python visualize_attention_with_context.py --mode text --text "The quick brown fox jumps"
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from transformer.core.model import GaugeTransformerLM


def create_tokenizer(vocab_size=50257):
    """
    Create a simple tokenizer for visualization.

    In practice, use the actual tokenizer from your training setup!
    """
    try:
        # Try to use actual BPE tokenizer if available
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print(f"[Tokenizer] Using GPT2Tokenizer (vocab size: {tokenizer.vocab_size})")
        return tokenizer
    except ImportError:
        # Fallback: simple character-level tokenizer
        print(f"[Tokenizer] Using simple character tokenizer (vocab size: {vocab_size})")

        class SimpleTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size

            def encode(self, text):
                # Simple: use ASCII values modulo vocab_size
                return [ord(c) % self.vocab_size for c in text[:128]]

            def decode(self, token_ids, skip_special_tokens=True):
                # Simple decode
                return ''.join([chr(min(max(t, 32), 126)) for t in token_ids])

        return SimpleTokenizer(vocab_size)


def get_sequence(mode, tokenizer, args):
    """
    Get a sequence to visualize based on mode.

    Returns:
        token_ids: (1, N) tensor
        description: str describing what this sequence is
        token_strs: List[str] of decoded tokens
    """
    if mode == 'random':
        # Random sequence for debugging
        N = args.seq_len
        token_ids = torch.randint(0, min(100, tokenizer.vocab_size), (1, N))

        description = f"RANDOM SEQUENCE (length {N})"
        token_strs = [f"tok{i}" for i in token_ids[0].tolist()]

        print(f"\n[SEQUENCE] {description}")
        print(f"  Token IDs: {token_ids[0].tolist()[:20]}{'...' if N > 20 else ''}")

    elif mode == 'text':
        # Specific text provided by user
        text = args.text
        token_ids = tokenizer.encode(text)
        token_ids = torch.tensor([token_ids])

        N = token_ids.shape[1]
        description = f"USER TEXT: \"{text}\""

        # Decode each token individually to show subwords
        try:
            token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]
        except (KeyError, IndexError, RuntimeError):
            token_strs = [f"tok{i}" for i in token_ids[0].tolist()]

        print(f"\n[SEQUENCE] {description}")
        print(f"  Length: {N} tokens")
        print(f"  Tokens: {' | '.join(token_strs[:15])}{'...' if N > 15 else ''}")

    elif mode == 'validation':
        # Load REAL validation data from WikiText
        try:
            from transformer.data import create_dataloaders

            dataset_name = args.dataset if hasattr(args, 'dataset') else 'wikitext-2'
            print(f"Loading REAL validation data from {dataset_name}...")

            # Use existing data loader
            train_loader, val_loader, vocab_size = create_dataloaders(
                max_seq_len=args.seq_len if hasattr(args, 'seq_len') else 128,
                batch_size=8,
                vocab_size=50257,
                dataset=dataset_name,
            )

            # Get validation dataset
            val_dataset = val_loader.dataset

            # Get first batch, first sequence
            for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
                if batch_idx == 0:
                    token_ids = input_ids[0:1]  # (1, N)
                    break

            # Get tokenizer from dataset
            if hasattr(val_dataset, 'tokenizer'):
                tokenizer = val_dataset.tokenizer
            elif hasattr(val_dataset, 'enc'):  # tiktoken
                tokenizer = val_dataset.enc

            description = f"REAL VALIDATION DATA: {dataset_name.upper()}"

            # Decode tokens
            try:
                if hasattr(tokenizer, 'decode'):
                    decoded_text = tokenizer.decode(token_ids[0].tolist(), skip_special_tokens=True)
                    token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]
                else:
                    decoded_text = tokenizer.decode(token_ids[0].tolist())
                    token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]

                print(f"\n[SEQUENCE] {description}")
                print(f"  Decoded text: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")
            except (KeyError, IndexError, RuntimeError) as e:
                print(f"[WARN] Could not decode: {e}")
                token_strs = [f"tok{i}" for i in token_ids[0].tolist()]

        except (ImportError, RuntimeError, OSError, ValueError) as e:
            # Fallback to example text if data loading fails
            print(f"[ERROR] Could not load validation data: {e}")
            print(f"  Falling back to example text...")

            text = "The transformer uses attention mechanisms to process sequences."
            token_ids = tokenizer.encode(text)
            token_ids = torch.tensor([token_ids])
            description = f"EXAMPLE TEXT (fallback): \"{text}\""

            try:
                token_strs = [tokenizer.decode([t]) for t in token_ids[0].tolist()]
            except (KeyError, IndexError, RuntimeError):
                token_strs = [f"tok{i}" for i in token_ids[0].tolist()]

        N = token_ids.shape[1]
        print(f"  Length: {N} tokens")
        print(f"  Tokens: {' | '.join(token_strs[:15])}{'...' if N > 15 else ''}")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return token_ids, description, token_strs


def visualize_attention_with_labels(model, token_ids, description, token_strs, save_path='attention_analysis.png'):
    """
    Visualize attention with full context: what sequence, what tokens, which heads.
    """
    model.eval()

    with torch.no_grad():
        logits, attn_info = model.forward_with_attention(token_ids)

    beta = attn_info['beta']  # (B, n_heads, N, N)
    kl_matrix = attn_info.get('kl_matrix')  # (B, N, N)

    B, n_heads, N, N = beta.shape

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, n_heads + 1, figure=fig, hspace=0.4, wspace=0.3)

    # Title with sequence information
    fig.suptitle(f'Attention Analysis: {description}', fontsize=14, fontweight='bold', y=0.98)

    # =========================================================================
    # Row 0: Display the actual sequence (scrollable text)
    # =========================================================================
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')

    # Format tokens for display
    if N <= 30:
        tokens_display = ' | '.join([f"{i}:{t}" for i, t in enumerate(token_strs)])
    else:
        tokens_display = ' | '.join([f"{i}:{t}" for i, t in enumerate(token_strs[:25])]) + f" ... +{N-25} more"

    ax_text.text(0.5, 0.5, f"Sequence Tokens:\n{tokens_display}",
                ha='center', va='center', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # =========================================================================
    # Row 1: Individual attention heads
    # =========================================================================
    for head_idx in range(n_heads):
        ax = fig.add_subplot(gs[1, head_idx])

        beta_head = beta[0, head_idx].numpy()  # (N, N)

        # Log scale for better visualization
        beta_plot = np.log10(np.maximum(beta_head, 1e-6))

        im = ax.imshow(beta_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
        ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        # Add token labels if sequence is short enough
        if N <= 20:
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels([f"{i}\n{t[:5]}" for i, t in enumerate(token_strs)],
                              rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels([f"{i}:{t[:8]}" for i, t in enumerate(token_strs)],
                              fontsize=7)

        # Compute row uniformity
        beta_safe = beta_head.copy()
        np.fill_diagonal(beta_safe, np.nan)
        row_std = np.nanstd(beta_safe, axis=1).mean()

        color = 'green' if row_std > 0.1 else 'orange' if row_std > 0.05 else 'red'
        ax.text(0.02, 0.98, f'Avg row std: {row_std:.3f}',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

    # Colorbar
    cbar_ax = fig.add_subplot(gs[1, -1])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('log₁₀(β_ij)', rotation=270, labelpad=15)

    # =========================================================================
    # Row 2: Averaged attention + statistics
    # =========================================================================
    ax_avg = fig.add_subplot(gs[2, :n_heads//2])
    beta_avg = beta[0].mean(dim=0).numpy()
    beta_avg_plot = np.log10(np.maximum(beta_avg, 1e-6))

    im_avg = ax_avg.imshow(beta_avg_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
    ax_avg.set_title('HEAD-AVERAGED Attention (⚠️ loses per-head patterns)',
                     fontsize=11, fontweight='bold')
    ax_avg.set_xlabel('Key Position')
    ax_avg.set_ylabel('Query Position')

    # Token labels if short
    if N <= 20:
        ax_avg.set_xticks(range(N))
        ax_avg.set_yticks(range(N))
        ax_avg.set_xticklabels([f"{i}\n{t[:5]}" for i, t in enumerate(token_strs)],
                              rotation=45, ha='right', fontsize=7)
        ax_avg.set_yticklabels([f"{i}:{t[:8]}" for i, t in enumerate(token_strs)],
                              fontsize=7)

    plt.colorbar(im_avg, ax=ax_avg, label='log₁₀(β_ij)')

    # Row uniformity statistics
    ax_stats = fig.add_subplot(gs[2, n_heads//2+1:])
    row_stds = []
    for i in range(N):
        row = beta_avg[i, :i+1]  # Causal
        std = np.std(row) if len(row) > 1 else 0
        row_stds.append(std)

    ax_stats.bar(range(N), row_stds, color='steelblue', alpha=0.7)
    ax_stats.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Uniform threshold')
    ax_stats.set_xlabel('Query Position')
    ax_stats.set_ylabel('Row Std Dev')
    ax_stats.set_title('Row Uniformity (Averaged Attention)')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)

    # =========================================================================
    # Row 3: KL divergence analysis
    # =========================================================================
    if kl_matrix is not None:
        ax_kl = fig.add_subplot(gs[3, :n_heads//2])
        kl_np = kl_matrix[0].numpy()

        im_kl = ax_kl.imshow(kl_np, cmap='Reds', aspect='auto')
        ax_kl.set_title('KL Divergence: KL(q_i || Ω_ij[q_j])', fontsize=11, fontweight='bold')
        ax_kl.set_xlabel('Key Position (j)')
        ax_kl.set_ylabel('Query Position (i)')

        if N <= 20:
            ax_kl.set_xticks(range(N))
            ax_kl.set_yticks(range(N))
            ax_kl.set_xticklabels([f"{i}\n{t[:5]}" for i, t in enumerate(token_strs)],
                                 rotation=45, ha='right', fontsize=7)
            ax_kl.set_yticklabels([f"{i}:{t[:8]}" for i, t in enumerate(token_strs)],
                                 fontsize=7)

        plt.colorbar(im_kl, ax=ax_kl, label='KL divergence')

        # KL statistics
        ax_kl_stats = fig.add_subplot(gs[3, n_heads//2+1:])
        kl_safe = kl_np.copy()
        np.fill_diagonal(kl_safe, np.nan)
        kl_row_stds = np.nanstd(kl_safe, axis=1)

        ax_kl_stats.bar(range(N), kl_row_stds, color='coral', alpha=0.7)
        ax_kl_stats.set_xlabel('Query Position')
        ax_kl_stats.set_ylabel('KL Row Std Dev')
        ax_kl_stats.set_title('KL Divergence Variation')
        ax_kl_stats.grid(True, alpha=0.3)

        mean_kl_std = kl_row_stds.mean()
        if mean_kl_std < 0.5:
            ax_kl_stats.text(0.5, 0.95,
                           '⚠️ WARNING: KL divergences are nearly uniform!\nThis causes uniform attention.',
                           transform=ax_kl_stats.transAxes,
                           fontsize=9, color='red', weight='bold',
                           va='top', ha='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n[SAVED] {save_path}")

    # =========================================================================
    # Print diagnostic summary
    # =========================================================================
    print("\n" + "="*80)
    print("ATTENTION DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Sequence: {description}")
    print(f"Length: {N} tokens")
    print(f"Attention heads: {n_heads}")
    print("-"*80)

    for head_idx in range(n_heads):
        beta_head = beta[0, head_idx].numpy()
        beta_safe = beta_head.copy()
        np.fill_diagonal(beta_safe, np.nan)
        row_std = np.nanstd(beta_safe, axis=1).mean()

        status = "✓ SHARP" if row_std > 0.1 else "⚠ MEDIUM" if row_std > 0.05 else "❌ UNIFORM"
        print(f"  Head {head_idx}: row_std = {row_std:.4f}  [{status}]")

    beta_avg_safe = beta_avg.copy()
    np.fill_diagonal(beta_avg_safe, np.nan)
    avg_row_std = np.nanstd(beta_avg_safe, axis=1).mean()
    status_avg = "✓ SHARP" if avg_row_std > 0.1 else "⚠ MEDIUM" if avg_row_std > 0.05 else "❌ UNIFORM"
    print(f"\n  AVERAGED: row_std = {avg_row_std:.4f}  [{status_avg}]")

    if kl_matrix is not None:
        kl_std = kl_row_stds.mean()
        print(f"\n  KL Matrix: row_std = {kl_std:.4f}")
        if kl_std < 0.5:
            print(f"    ⚠️  Root cause: KL divergences too uniform → uniform attention")

    print("="*80)

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize attention with full context')
    parser.add_argument('--mode', choices=['random', 'text', 'validation'], default='text',
                       help='Source of sequence to visualize')
    parser.add_argument('--text', type=str,
                       default='The quick brown fox jumps over the lazy dog',
                       help='Text to visualize (for --mode text)')
    parser.add_argument('--dataset', type=str, choices=['wikitext-2', 'wikitext-103'],
                       default='wikitext-2',
                       help='Which WikiText dataset to use (for --mode validation)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='DEPRECATED: Dataset auto-downloads now')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--save-path', type=str, default='attention_with_context.png',
                       help='Where to save the figure')

    args = parser.parse_args()

    print("="*80)
    print("ATTENTION VISUALIZATION WITH FULL CONTEXT")
    print("="*80)

    # Create model
    config = {
        'vocab_size': 50257,  # GPT-2 vocab size
        'embed_dim': 63,
        'n_layers': 1,
        'hidden_dim': 128,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'mask_self_attention': True,
        'evolve_sigma': True,
        # Irrep specification: (name, num_irreps, dimension)
        # Total: 32*1 + 8*3 + 2*4 = 32 + 24 + 8 = 64
        'irrep_spec': [('l0', 34, 1), ('l1', 8, 3), ('l2', 1, 5)],
    }

    print(f"\n[MODEL CONFIG]")
    print(f"  embed_dim: {config['embed_dim']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  kappa_beta: {config['kappa_beta']}")
    print(f"  mask_self_attention: {config['mask_self_attention']}")

    model = GaugeTransformerLM(config)
    model.eval()

    # Get tokenizer and sequence
    tokenizer = create_tokenizer(config['vocab_size'])
    token_ids, description, token_strs = get_sequence(args.mode, tokenizer, args)

    # Visualize
    visualize_attention_with_labels(model, token_ids, description, token_strs, args.save_path)

    print(f"\n✓ Done! Open {args.save_path} to see the results.")


if __name__ == '__main__':
    main()