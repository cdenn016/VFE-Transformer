#!/usr/bin/env python
"""
Visualize attention patterns for individual heads (not averaged).
Shows how different heads learn different patterns.

Usage: python visualize_attention_heads.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from transformer.core.model import GaugeTransformerLM

def visualize_per_head_attention():
    """Visualize attention for each head separately."""

    # Simple model
    config = {
        'vocab_size': 100,
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

    model = GaugeTransformerLM(config)
    model.eval()

    # Test sequence
    token_ids = torch.randint(0, 100, (1, 16))  # (B=1, N=16)

    # Forward pass
    with torch.no_grad():
        logits, attn_info = model.forward_with_attention(token_ids)

    beta = attn_info['beta']  # (B, n_heads, N, N)
    kl_matrix = attn_info.get('kl_matrix')  # (B, N, N) - already head-averaged

    B, n_heads, N, N = beta.shape
    print(f"Attention shape: {beta.shape}")
    print(f"  B={B} (batch), H={n_heads} (heads), N={N} (sequence)")

    # Create subplot grid for all heads
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, n_heads + 1, figure=fig, hspace=0.3, wspace=0.3)

    # Plot individual heads
    for head_idx in range(n_heads):
        ax = fig.add_subplot(gs[0, head_idx])

        beta_head = beta[0, head_idx].numpy()  # (N, N)

        # Use log scale to see medium-weight connections
        beta_plot = np.log10(np.maximum(beta_head, 1e-6))

        im = ax.imshow(beta_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
        ax.set_title(f'Head {head_idx}', fontsize=10)
        ax.set_xlabel('Key (j)')
        ax.set_ylabel('Query (i)')

        # Compute statistics
        beta_safe = beta_head.copy()
        np.fill_diagonal(beta_safe, np.nan)  # Exclude diagonal
        std = np.nanstd(beta_safe, axis=1).mean()  # Avg row std

        ax.text(0.02, 0.98, f'Row std: {std:.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Add colorbar
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('log₁₀(β_ij)', rotation=270, labelpad=15)

    # Plot head-averaged attention (what you're currently seeing)
    ax_avg = fig.add_subplot(gs[1, :n_heads//2])
    beta_avg = beta[0].mean(dim=0).numpy()  # Average over heads
    beta_avg_plot = np.log10(np.maximum(beta_avg, 1e-6))
    im_avg = ax_avg.imshow(beta_avg_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
    ax_avg.set_title('AVERAGED OVER HEADS (what you see now)', fontsize=12, fontweight='bold')
    ax_avg.set_xlabel('Key (j)')
    ax_avg.set_ylabel('Query (i)')
    plt.colorbar(im_avg, ax=ax_avg, label='log₁₀(β_ij)')

    # Plot row statistics for averaged attention
    ax_stats = fig.add_subplot(gs[1, n_heads//2+1:])
    row_stds = []
    for i in range(N):
        row = beta_avg[i, :i+1]  # Causal: only valid positions
        if len(row) > 1:
            std = np.std(row)
            row_stds.append(std)
        else:
            row_stds.append(0)

    ax_stats.bar(range(len(row_stds)), row_stds, color='steelblue', alpha=0.7)
    ax_stats.axhline(y=0.05, color='red', linestyle='--', label='Uniform threshold (0.05)')
    ax_stats.set_xlabel('Query Position')
    ax_stats.set_ylabel('Row Std Dev')
    ax_stats.set_title('Row Uniformity (Averaged Attention)')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)

    # Plot KL divergence matrix
    if kl_matrix is not None:
        ax_kl = fig.add_subplot(gs[2, :n_heads//2])
        kl_np = kl_matrix[0].numpy()  # (N, N)
        im_kl = ax_kl.imshow(kl_np, cmap='Reds', aspect='auto')
        ax_kl.set_title('KL Divergence Matrix: KL(q_i || Ω_ij[q_j])', fontsize=12)
        ax_kl.set_xlabel('Key (j)')
        ax_kl.set_ylabel('Query (i)')
        plt.colorbar(im_kl, ax=ax_kl, label='KL divergence')

        # KL statistics
        ax_kl_stats = fig.add_subplot(gs[2, n_heads//2+1:])
        kl_np_safe = kl_np.copy()
        np.fill_diagonal(kl_np_safe, np.nan)
        kl_row_stds = np.nanstd(kl_np_safe, axis=1)

        ax_kl_stats.bar(range(N), kl_row_stds, color='coral', alpha=0.7)
        ax_kl_stats.set_xlabel('Query Position')
        ax_kl_stats.set_ylabel('KL Row Std Dev')
        ax_kl_stats.set_title('KL Divergence Variation per Query')
        ax_kl_stats.grid(True, alpha=0.3)

        # Check if KL is uniform
        mean_kl_std = kl_row_stds.mean()
        if mean_kl_std < 0.5:
            ax_kl_stats.text(0.5, 0.95,
                            'WARNING: KL divergences are nearly uniform!\nThis causes uniform attention.',
                            transform=ax_kl_stats.transAxes,
                            fontsize=10, color='red',
                            verticalalignment='top',
                            horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.suptitle('Per-Head Attention Analysis', fontsize=16, fontweight='bold')

    plt.savefig('attention_heads_separate.png', dpi=200, bbox_inches='tight')
    print(f"\nSaved: attention_heads_separate.png")

    # Print diagnostic summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    for head_idx in range(n_heads):
        beta_head = beta[0, head_idx].numpy()
        beta_safe = beta_head.copy()
        np.fill_diagonal(beta_safe, np.nan)
        row_std = np.nanstd(beta_safe, axis=1).mean()

        status = "UNIFORM ❌" if row_std < 0.05 else "SHARP ✓"
        print(f"Head {head_idx}: Avg row std = {row_std:.4f}  [{status}]")

    beta_avg = beta[0].mean(dim=0).numpy()
    beta_avg_safe = beta_avg.copy()
    np.fill_diagonal(beta_avg_safe, np.nan)
    avg_row_std = np.nanstd(beta_avg_safe, axis=1).mean()
    status_avg = "UNIFORM ❌" if avg_row_std < 0.05 else "SHARP ✓"
    print(f"\nAveraged: Avg row std = {avg_row_std:.4f}  [{status_avg}]")

    if kl_matrix is not None:
        kl_np = kl_matrix[0].numpy()
        kl_np_safe = kl_np.copy()
        np.fill_diagonal(kl_np_safe, np.nan)
        kl_std = np.nanstd(kl_np_safe, axis=1).mean()
        print(f"\nKL Matrix: Avg row std = {kl_std:.4f}")
        if kl_std < 0.5:
            print("  ⚠️  KL divergences are too uniform → uniform attention!")

    print("="*80)

    plt.show()


if __name__ == '__main__':
    print("Visualizing per-head attention patterns...")
    visualize_per_head_attention()