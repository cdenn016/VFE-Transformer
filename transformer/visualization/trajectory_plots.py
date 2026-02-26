# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 21:05:21 2025

@author: chris and christine
"""

"""
Trajectory Visualization for Transformer
==========================================

Plotting utilities for visualizing:
1. Belief evolution through transformer layers
2. Phase space trajectories
3. Attention patterns
4. Training convergence

Requires matplotlib. Falls back gracefully if not available.

Author: Chris & Claude
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")

# Import trajectory classes
from transformer.analysis.trajectory import (
    ForwardTrajectory,
    LayerTrajectory,
    TrajectoryRecorder,
)


# =============================================================================
# Color Schemes
# =============================================================================

# Professional color palette
COLORS = {
    'energy_H': '#2E86AB',      # Blue for total energy
    'energy_T': '#E94F37',      # Red for kinetic
    'energy_V': '#44AF69',      # Green for potential
    'mu': '#7B68EE',            # Purple for belief means
    'sigma': '#FF8C00',         # Orange for covariances
    'phi': '#20B2AA',           # Teal for gauge frames
    'attention': '#DC143C',     # Crimson for attention
    'layer': '#4169E1',         # Royal blue for layers
}

# Custom colormap for attention matrices
def get_attention_cmap():
    """Blue-white-red colormap for attention."""
    if not MATPLOTLIB_AVAILABLE:
        return None
    colors = ['#2166AC', '#F7F7F7', '#B2182B']
    return LinearSegmentedColormap.from_list('attention', colors)


# =============================================================================
# Belief Evolution Plots
# =============================================================================

def plot_mu_evolution(
    trajectory: ForwardTrajectory,
    batch_idx: int = 0,
    token_idx: int = -1,
    n_components: int = 8,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot belief mean μ evolution through transformer layers.

    Shows how μ components evolve from embedding through each layer.

    Args:
        trajectory: ForwardTrajectory with layer data
        batch_idx: Which batch element to plot
        token_idx: Which token to plot (-1 = last token)
        n_components: Number of μ components to show
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    if token_idx == -1:
        token_idx = trajectory.seq_len - 1

    # Extract μ at each layer
    mu_trace = [trajectory.mu_embed[batch_idx, token_idx]]
    layer_names = ['Embed']

    for lt in trajectory.layer_trajectories:
        mu_trace.append(lt.mu_out[batch_idx, token_idx])
        layer_names.append(f'L{lt.layer_idx}')

    mu_trace = np.array(mu_trace)  # (n_layers+1, K)
    n_layers, K = mu_trace.shape

    # Limit components
    n_show = min(n_components, K)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Line plot of μ components over layers
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, n_show))

    for i in range(n_show):
        ax.plot(range(n_layers), mu_trace[:, i], 'o-',
                color=colors[i], label=f'μ[{i}]', alpha=0.7)

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_names)
    ax.set_xlabel('Layer')
    ax.set_ylabel('μ value')
    ax.set_title(f'Belief Evolution (Token {token_idx})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Heatmap of all components
    ax = axes[1]
    im = ax.imshow(mu_trace.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(mu_trace).max(), vmax=np.abs(mu_trace).max())
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('μ component')
    ax.set_title(f'μ Heatmap (Token {token_idx})')
    plt.colorbar(im, ax=ax, label='μ value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_mu_norm_evolution(
    trajectory: ForwardTrajectory,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot ||μ|| evolution for all tokens through layers.

    Args:
        trajectory: ForwardTrajectory with layer data
        batch_idx: Which batch element to plot
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    N = trajectory.seq_len

    # Collect ||μ|| for each token at each layer
    mu_norms = []
    layer_names = ['Embed']

    # Embedding
    mu_embed = trajectory.mu_embed[batch_idx]  # (N, K)
    mu_norms.append([np.linalg.norm(mu_embed[i]) for i in range(N)])

    for lt in trajectory.layer_trajectories:
        mu_out = lt.mu_out[batch_idx]  # (N, K)
        mu_norms.append([np.linalg.norm(mu_out[i]) for i in range(N)])
        layer_names.append(f'L{lt.layer_idx}')

    mu_norms = np.array(mu_norms)  # (n_layers+1, N)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(mu_norms.T, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Token index')
    ax.set_title('||μ|| Evolution Through Layers')
    plt.colorbar(im, ax=ax, label='||μ||')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Attention Pattern Plots
# =============================================================================

def plot_attention_pattern(
    trajectory: ForwardTrajectory,
    layer_idx: int = -1,
    head_idx: int = 0,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (6, 5),
    save_path: Optional[str] = None,
    mask_diagonal: bool = True,
    log_scale: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot attention matrix β for a specific layer and head.

    Args:
        trajectory: ForwardTrajectory with attention data
        layer_idx: Which layer (-1 = last)
        head_idx: Which attention head
        batch_idx: Which batch element
        figsize: Figure size
        save_path: If provided, save figure to this path
        mask_diagonal: If True, mask out diagonal (self-attention) for clarity
        log_scale: If True, use log scale to enhance small values

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    if layer_idx == -1:
        layer_idx = len(trajectory.layer_trajectories) - 1

    lt = trajectory.layer_trajectories[layer_idx]

    if lt.beta is None:
        print("No attention data recorded. Enable record_attention=True.")
        return None

    # Handle different beta shapes
    beta = lt.beta[batch_idx]  # (n_heads, N, N) or (N, N)

    if beta.ndim == 3:
        beta = beta[head_idx]  # (N, N)

    # Convert to numpy for manipulation
    beta = np.array(beta)

    # Mask diagonal (self-attention often dominates)
    if mask_diagonal:
        beta_plot = beta.copy()
        np.fill_diagonal(beta_plot, np.nan)
    else:
        beta_plot = beta

    # Apply log scale to enhance small values
    if log_scale:
        eps = 1e-6
        beta_plot = np.log10(np.maximum(beta_plot, eps))
        cbar_label = 'log₁₀(β_ij)'
        cmap = 'viridis'
    else:
        cbar_label = 'β_ij'
        cmap = get_attention_cmap()

    fig, ax = plt.subplots(figsize=figsize)

    # Use focused colorbar range for log scale to see medium-weight connections
    if log_scale:
        im = ax.imshow(beta_plot, cmap=cmap, aspect='auto', vmin=-3, vmax=0)
    else:
        im = ax.imshow(beta_plot, cmap=cmap, aspect='auto')
    ax.set_xlabel('Key (j)')
    ax.set_ylabel('Query (i)')
    title = f'Attention β (Layer {layer_idx}, Head {head_idx})'
    if mask_diagonal:
        title += ' [diag masked]'
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_summary(
    trajectory: ForwardTrajectory,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot attention summary across all layers.

    Shows attention entropy and sparsity per layer.

    Args:
        trajectory: ForwardTrajectory with attention data
        batch_idx: Which batch element
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    # Compute attention entropy per layer
    entropies = []
    sparsities = []
    layer_names = []

    for lt in trajectory.layer_trajectories:
        if lt.beta is None:
            continue

        beta = lt.beta[batch_idx]
        if beta.ndim == 3:
            beta = beta.mean(axis=0)  # Average over heads

        # Entropy: -Σ β log β
        entropy = -(beta * np.log(beta + 1e-10)).sum(axis=-1).mean()
        entropies.append(entropy)

        # Sparsity: fraction of attention below 0.01
        sparsity = (beta < 0.01).mean()
        sparsities.append(sparsity)

        layer_names.append(f'L{lt.layer_idx}')

    if not entropies:
        print("No attention data recorded.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Entropy
    ax = axes[0]
    x = np.arange(len(entropies))
    ax.bar(x, entropies, color=COLORS['attention'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Entropy')
    ax.set_title('Attention Entropy per Layer')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Sparsity
    ax = axes[1]
    ax.bar(x, sparsities, color=COLORS['layer'], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Fraction β < 0.01')
    ax.set_title('Attention Sparsity per Layer')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Training Diagnostics Plots
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot training curves from history dict.

    Expected keys: 'loss', 'ce_loss', 'free_energy', etc.

    Args:
        history: Dict mapping metric name to list of values
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    # Determine number of subplots
    n_metrics = len(history)
    if n_metrics == 0:
        return None

    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for idx, (name, values) in enumerate(history.items()):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        steps = np.arange(len(values))
        ax.plot(steps, values, linewidth=1.5, color=COLORS.get('layer', '#4169E1'))
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(history), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Phase Space Visualization (Advanced)
# =============================================================================

def plot_phase_space_2d(
    trajectory: ForwardTrajectory,
    batch_idx: int = 0,
    token_idx: int = -1,
    component_x: int = 0,
    component_y: int = 1,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot 2D phase space trajectory (μ_x vs μ_y) through layers.

    Args:
        trajectory: ForwardTrajectory with layer data
        batch_idx: Which batch element
        token_idx: Which token (-1 = last)
        component_x: μ component for x-axis
        component_y: μ component for y-axis
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    if token_idx == -1:
        token_idx = trajectory.seq_len - 1

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory.layer_trajectories)))

    # Start with embedding
    mu_x = [trajectory.mu_embed[batch_idx, token_idx, component_x]]
    mu_y = [trajectory.mu_embed[batch_idx, token_idx, component_y]]

    ax.scatter(mu_x, mu_y, c='black', s=100, marker='*', zorder=10, label='Embed')

    for layer_idx, lt in enumerate(trajectory.layer_trajectories):
        # Layer output
        out_x = lt.mu_out[batch_idx, token_idx, component_x]
        out_y = lt.mu_out[batch_idx, token_idx, component_y]
        ax.scatter(out_x, out_y, c=[colors[layer_idx]], s=50, marker='o',
                   label=f'L{lt.layer_idx}')

        mu_x.append(out_x)
        mu_y.append(out_y)

    # Connect layer outputs
    ax.plot(mu_x, mu_y, 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel(f'μ[{component_x}]')
    ax.set_ylabel(f'μ[{component_y}]')
    ax.set_title(f'Phase Space Trajectory (Token {token_idx})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Comprehensive Dashboard
# =============================================================================

def plot_trajectory_dashboard(
    trajectory: ForwardTrajectory,
    batch_idx: int = 0,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Create comprehensive dashboard of trajectory visualizations.

    Includes:
    - Energy conservation
    - Belief evolution
    - Attention patterns
    - Phase space

    Args:
        trajectory: ForwardTrajectory with all data
        batch_idx: Which batch element
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # =========================================================================
    # Row 1: Summary
    # =========================================================================

    # 1.1-1.2: Reserved for future use
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')

    # 1.3: Summary stats
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    stats_text = f"""
    FFN Mode: {trajectory.ffn_mode}
    Batch Size: {trajectory.batch_size}
    Seq Length: {trajectory.seq_len}
    Layers: {len(trajectory.layer_trajectories)}
    Output Entropy: {trajectory.logits_entropy:.4f if trajectory.logits_entropy else 'N/A'}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', va='center')
    ax.set_title('Summary')

    # =========================================================================
    # Row 2: Belief Evolution
    # =========================================================================

    # 2.1: μ evolution (last token)
    ax = fig.add_subplot(gs[1, 0])
    token_idx = trajectory.seq_len - 1
    mu_trace = [trajectory.mu_embed[batch_idx, token_idx]]
    for lt in trajectory.layer_trajectories:
        mu_trace.append(lt.mu_out[batch_idx, token_idx])
    mu_trace = np.array(mu_trace)

    n_show = min(8, mu_trace.shape[1])
    colors_mu = plt.cm.viridis(np.linspace(0, 1, n_show))
    for i in range(n_show):
        ax.plot(mu_trace[:, i], 'o-', color=colors_mu[i], alpha=0.7, markersize=4)

    ax.set_xticks(range(len(mu_trace)))
    ax.set_xticklabels(['Emb'] + [f'L{i}' for i in range(len(trajectory.layer_trajectories))])
    ax.set_ylabel('μ')
    ax.set_title(f'μ Evolution (Token {token_idx})')
    ax.grid(True, alpha=0.3)

    # 2.2: ||μ|| heatmap
    ax = fig.add_subplot(gs[1, 1])
    mu_norms = [np.linalg.norm(trajectory.mu_embed[batch_idx], axis=-1)]
    for lt in trajectory.layer_trajectories:
        mu_norms.append(np.linalg.norm(lt.mu_out[batch_idx], axis=-1))
    mu_norms = np.array(mu_norms)

    im = ax.imshow(mu_norms.T, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(mu_norms)))
    ax.set_xticklabels(['Emb'] + [f'L{i}' for i in range(len(trajectory.layer_trajectories))])
    ax.set_ylabel('Token')
    ax.set_title('||μ|| Heatmap')
    plt.colorbar(im, ax=ax)

    # 2.3: φ evolution (last token)
    ax = fig.add_subplot(gs[1, 2])
    phi_trace = [trajectory.phi_embed[batch_idx, token_idx]]
    for lt in trajectory.layer_trajectories:
        phi_trace.append(lt.phi_out[batch_idx, token_idx])
    phi_trace = np.array(phi_trace)

    for i, label in enumerate(['φ_x', 'φ_y', 'φ_z']):
        ax.plot(phi_trace[:, i], 'o-', label=label, markersize=4)

    ax.set_xticks(range(len(phi_trace)))
    ax.set_xticklabels(['Emb'] + [f'L{i}' for i in range(len(trajectory.layer_trajectories))])
    ax.set_ylabel('φ')
    ax.set_title(f'Gauge Frame Evolution (Token {token_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Row 3: Attention & Phase Space
    # =========================================================================

    # 3.1: Attention (last layer) - with diagonal masked and log scale
    ax = fig.add_subplot(gs[2, 0])
    last_layer = trajectory.layer_trajectories[-1] if trajectory.layer_trajectories else None
    if last_layer and last_layer.beta is not None:
        beta = last_layer.beta[batch_idx]
        if beta.ndim == 3:
            beta = beta[0]  # First head
        beta = np.array(beta)
        # Mask diagonal and use log scale for better visualization
        beta_plot = beta.copy()
        np.fill_diagonal(beta_plot, np.nan)
        beta_plot = np.log10(np.maximum(beta_plot, 1e-6))
        # vmin=-3, vmax=0 to see medium-weight connections (0.001 to 1.0)
        im = ax.imshow(beta_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        ax.set_title(f'Attention (L{last_layer.layer_idx}) [log, diag masked]')
        plt.colorbar(im, ax=ax, label='log₁₀(β)')
    else:
        ax.text(0.5, 0.5, 'No attention data', ha='center', va='center')
        ax.set_title('Attention (no data)')

    # 3.2: Phase space 2D
    ax = fig.add_subplot(gs[2, 1])
    mu_x = [trajectory.mu_embed[batch_idx, token_idx, 0]]
    mu_y = [trajectory.mu_embed[batch_idx, token_idx, 1]]

    ax.scatter(mu_x, mu_y, c='black', s=100, marker='*', zorder=10)

    colors_layer = plt.cm.plasma(np.linspace(0, 1, len(trajectory.layer_trajectories)))
    for layer_idx, lt in enumerate(trajectory.layer_trajectories):
        out_x = lt.mu_out[batch_idx, token_idx, 0]
        out_y = lt.mu_out[batch_idx, token_idx, 1]
        ax.scatter(out_x, out_y, c=[colors_layer[layer_idx]], s=50)
        mu_x.append(out_x)
        mu_y.append(out_y)

    ax.plot(mu_x, mu_y, 'k--', alpha=0.3)
    ax.set_xlabel('μ[0]')
    ax.set_ylabel('μ[1]')
    ax.set_title(f'Phase Space (Token {token_idx})')
    ax.grid(True, alpha=0.3)

    # 3.3: KL matrix (last layer)
    ax = fig.add_subplot(gs[2, 2])
    if last_layer and last_layer.kl_matrix is not None:
        kl = last_layer.kl_matrix[batch_idx]
        im = ax.imshow(kl, cmap='Reds')
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        ax.set_title(f'KL(q_i||q_j) (L{last_layer.layer_idx})')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'No KL data', ha='center', va='center')
        ax.set_title('KL Matrix (no data)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install with: pip install matplotlib")
        exit(1)

    print("=" * 70)
    print("TRAJECTORY PLOTTING TEST")
    print("=" * 70)

    # Create synthetic trajectory data
    from transformer.analysis.trajectory import (
        TrajectoryRecorder, ForwardTrajectory, LayerTrajectory
    )
    import torch

    B, N, K = 2, 8, 16

    # Create recorder and simulate data
    recorder = TrajectoryRecorder(enabled=True, record_attention=True)

    mu = torch.randn(B, N, K)
    Sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    phi = torch.randn(B, N, 3) * 0.1

    recorder.start_forward(B, N, ffn_mode='VFE_dynamic')
    recorder.record_embeddings(mu, Sigma, phi)

    # Simulate layers
    for layer_idx in range(4):
        recorder.start_layer(layer_idx)
        recorder.record_layer_input(mu, Sigma, phi)

        # Simulate attention
        beta = torch.softmax(torch.randn(B, 4, N, N), dim=-1)
        kl_matrix = torch.rand(B, N, N) * 5
        recorder.record_attention(beta, kl_matrix)

        mu = mu + torch.randn_like(mu) * 0.2
        recorder.record_layer_output(mu, Sigma, phi)
        recorder.end_layer()

    logits = torch.randn(B, N, 100)
    trajectory = recorder.end_forward(mu, logits)

    print(f"\n[1] Testing μ evolution plot...")
    fig = plot_mu_evolution(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[2] Testing ||μ|| evolution plot...")
    fig = plot_mu_norm_evolution(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[3] Testing attention pattern plot...")
    fig = plot_attention_pattern(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[4] Testing attention summary plot...")
    fig = plot_attention_summary(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[5] Testing phase space plot...")
    fig = plot_phase_space_2d(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[6] Testing comprehensive dashboard...")
    fig = plot_trajectory_dashboard(trajectory)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print(f"\n[7] Testing training curves plot...")
    history = {
        'loss': list(np.exp(-np.linspace(0, 2, 100)) + np.random.randn(100) * 0.1),
        'ce_loss': list(np.exp(-np.linspace(0, 2, 100)) + np.random.randn(100) * 0.05),
    }
    fig = plot_training_curves(history)
    if fig:
        plt.close(fig)
        print(f"    OK")

    print("\n" + "=" * 70)
    print("ALL PLOTTING TESTS PASSED")
    print("=" * 70)