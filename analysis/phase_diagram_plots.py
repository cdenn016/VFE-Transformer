# -*- coding: utf-8 -*-
"""
Phase Diagram Visualization for Kappa Phase Transitions
========================================================

Visualization tools for studying polarization phase transitions:
1. Phase diagram in (κ, d) space
2. Order parameter vs temperature
3. Attention matrix heatmaps
4. Critical scaling plots

Author: VFE Transformer Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions will not work.")


# =============================================================================
# Phase Diagram Plots
# =============================================================================

def plot_phase_diagram(
    kappa_range: Tuple[float, float] = (0.1, 10.0),
    n_points: int = 100,
    epsilon: float = 0.01,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot the phase diagram in (κ, d) space.

    Shows the critical curve d_c(κ) = √(2κ|ln(ε)|) separating:
    - Mixed phase (d < d_c): cross-group attention significant
    - Polarized phase (d > d_c): groups decoupled
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    kappa = np.linspace(kappa_range[0], kappa_range[1], n_points)
    d_c = np.sqrt(2 * kappa * abs(np.log(epsilon)))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot critical curve
    ax.plot(kappa, d_c, 'k-', linewidth=2.5, label=r'Critical: $d_c = \sqrt{2\kappa|\ln\epsilon|}$')

    # Shade regions
    d_max = d_c.max() * 1.3
    ax.fill_between(kappa, d_c, d_max, alpha=0.3, color='blue', label='Polarized Phase')
    ax.fill_between(kappa, 0, d_c, alpha=0.3, color='red', label='Mixed Phase')

    # Add annotations
    ax.annotate('POLARIZED\n(groups decoupled)',
                xy=(kappa_range[1]*0.7, d_max*0.8),
                fontsize=14, ha='center', color='darkblue')
    ax.annotate('MIXED\n(cross-group coupling)',
                xy=(kappa_range[1]*0.7, d_max*0.2),
                fontsize=14, ha='center', color='darkred')

    # Labels and formatting
    ax.set_xlabel(r'Temperature $\kappa$', fontsize=14)
    ax.set_ylabel(r'Mahalanobis Distance $d_M$', fontsize=14)
    ax.set_title(r'Phase Diagram: Polarization Stability', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.set_xlim(kappa_range)
    ax.set_ylim(0, d_max)
    ax.grid(True, alpha=0.3)

    # Add formula annotation
    ax.text(0.98, 0.02,
            r'$\epsilon = ' + f'{epsilon}$',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_order_parameter_vs_kappa(
    separation: float = 2.0,
    kappa_range: Tuple[float, float] = (0.1, 5.0),
    n_points: int = 100,
    epsilon: float = 0.01,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot order parameter η vs temperature κ at fixed separation.

    Order parameter: η = 1 - 2*β_cross/β_within
    - η → 1: complete polarization
    - η → 0: critical point
    - η < 0: mixed phase (cross-group dominates)
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    kappa = np.linspace(kappa_range[0], kappa_range[1], n_points)
    kappa_c = separation**2 / (2 * abs(np.log(epsilon)))

    # Simplified order parameter model
    # β_cross/β_within ≈ exp(-d²/(2κ)) for equal group sizes
    ratio = np.exp(-separation**2 / (2 * kappa))
    order_param = 1 - 2 * ratio

    fig, ax = plt.subplots(figsize=figsize)

    # Plot order parameter
    ax.plot(kappa, order_param, 'b-', linewidth=2.5, label=r'$\eta = 1 - 2\beta_{cross}/\beta_{within}$')

    # Mark critical point
    ax.axvline(kappa_c, color='r', linestyle='--', linewidth=2, label=f'$\\kappa_c = {kappa_c:.3f}$')
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    # Shade phases
    ax.fill_between(kappa, order_param, 1, where=(kappa < kappa_c),
                    alpha=0.2, color='blue', label='Polarized')
    ax.fill_between(kappa, order_param, -1, where=(kappa > kappa_c),
                    alpha=0.2, color='red', label='Mixed')

    # Mean-field scaling prediction near critical point
    kappa_near_c = kappa[kappa < kappa_c * 0.95]
    if len(kappa_near_c) > 0:
        scaling = np.sqrt(np.maximum(kappa_c - kappa_near_c, 0))
        scaling = scaling / scaling.max() * order_param[kappa < kappa_c * 0.95].max()
        ax.plot(kappa_near_c, scaling, 'g--', linewidth=1.5, alpha=0.7,
                label=r'Mean-field: $\eta \sim (\kappa_c - \kappa)^{1/2}$')

    ax.set_xlabel(r'Temperature $\kappa$', fontsize=14)
    ax.set_ylabel(r'Order Parameter $\eta$', fontsize=14)
    ax.set_title(f'Order Parameter vs Temperature (d = {separation})', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(kappa_range)
    ax.set_ylim(-0.5, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_attention_heatmap(
    beta_matrix: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Attention Matrix",
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None
):
    """
    Plot attention matrix as heatmap, optionally reordered by group.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    n_agents = beta_matrix.shape[0]

    # Reorder by group if labels provided
    if labels is not None:
        order = np.argsort(labels)
        beta_ordered = beta_matrix[order][:, order]
        labels_ordered = labels[order]
    else:
        beta_ordered = beta_matrix
        labels_ordered = None

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(beta_ordered, cmap='viridis', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Attention $\beta_{ij}$', fontsize=12)

    # Add group boundaries if labels provided
    if labels_ordered is not None:
        unique_labels = np.unique(labels_ordered)
        boundaries = []
        for label in unique_labels[:-1]:
            boundary = np.where(labels_ordered == label)[0][-1] + 0.5
            boundaries.append(boundary)

        for b in boundaries:
            ax.axhline(b, color='white', linewidth=2)
            ax.axvline(b, color='white', linewidth=2)

    ax.set_xlabel('Agent j', fontsize=12)
    ax.set_ylabel('Agent i', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_critical_scaling(
    separations: np.ndarray,
    kappa_c_measured: np.ndarray,
    epsilon: float = 0.01,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Plot measured κ_c vs separation to verify d² scaling.

    Theory: κ_c = d² / (2|ln(ε)|)
    Log-log plot should show slope = 2.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    kappa_c_theory = separations**2 / (2 * abs(np.log(epsilon)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Linear plot
    ax1 = axes[0]
    ax1.plot(separations, kappa_c_measured, 'bo-', markersize=8, linewidth=2, label='Measured')
    ax1.plot(separations, kappa_c_theory, 'r--', linewidth=2, label='Theory')
    ax1.set_xlabel('Separation d', fontsize=12)
    ax1.set_ylabel(r'Critical $\kappa_c$', fontsize=12)
    ax1.set_title('Critical Temperature vs Separation', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Log-log plot
    ax2 = axes[1]
    ax2.loglog(separations, kappa_c_measured, 'bo-', markersize=8, linewidth=2, label='Measured')
    ax2.loglog(separations, kappa_c_theory, 'r--', linewidth=2, label=r'Theory: $\kappa_c \propto d^2$')

    # Fit slope
    log_d = np.log(separations)
    log_kappa = np.log(kappa_c_measured)
    slope, intercept = np.polyfit(log_d, log_kappa, 1)
    ax2.text(0.05, 0.95, f'Slope = {slope:.2f}\n(Theory: 2.0)',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Separation d', fontsize=12)
    ax2.set_ylabel(r'Critical $\kappa_c$', fontsize=12)
    ax2.set_title('Log-Log Scaling', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


def plot_attention_vs_distance(
    separations: np.ndarray,
    beta_within: np.ndarray,
    beta_cross: np.ndarray,
    kappa: float,
    epsilon: float = 0.01,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot within-group and cross-group attention vs separation distance.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    d_c = np.sqrt(2 * kappa * abs(np.log(epsilon)))

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogy(separations, beta_within, 'b-o', markersize=8, linewidth=2,
                label=r'$\beta_{within}$')
    ax.semilogy(separations, beta_cross, 'r-s', markersize=8, linewidth=2,
                label=r'$\beta_{cross}$')

    # Mark critical distance
    ax.axvline(d_c, color='green', linestyle='--', linewidth=2,
               label=f'$d_c = {d_c:.2f}$')

    # Mark epsilon threshold
    ax.axhline(epsilon, color='gray', linestyle=':', linewidth=1.5,
               label=f'$\\epsilon = {epsilon}$')

    ax.set_xlabel('Separation Distance d', fontsize=14)
    ax.set_ylabel('Average Attention', fontsize=14)
    ax.set_title(f'Attention vs Distance ($\\kappa = {kappa}$)', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Shade polarized region
    ax.axvspan(d_c, separations.max() * 1.1, alpha=0.1, color='blue')
    ax.text(d_c + 0.1, ax.get_ylim()[1] * 0.5, 'Polarized',
            fontsize=12, color='darkblue', rotation=90, va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


# =============================================================================
# Combined Summary Plot
# =============================================================================

def plot_phase_transition_summary(
    separation: float = 2.0,
    kappa_range: Tuple[float, float] = (0.1, 5.0),
    epsilon: float = 0.01,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
):
    """
    Create a 2x2 summary of phase transition analysis.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for plotting")
        return None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    kappa = np.linspace(kappa_range[0], kappa_range[1], 100)
    kappa_c = separation**2 / (2 * abs(np.log(epsilon)))
    d_c_curve = np.sqrt(2 * kappa * abs(np.log(epsilon)))

    # Panel 1: Phase diagram
    ax1 = axes[0, 0]
    ax1.plot(kappa, d_c_curve, 'k-', linewidth=2.5)
    ax1.fill_between(kappa, d_c_curve, d_c_curve.max() * 1.3, alpha=0.3, color='blue')
    ax1.fill_between(kappa, 0, d_c_curve, alpha=0.3, color='red')
    ax1.axhline(separation, color='green', linestyle='--', linewidth=2, label=f'd = {separation}')
    ax1.set_xlabel(r'$\kappa$', fontsize=12)
    ax1.set_ylabel(r'$d_M$', fontsize=12)
    ax1.set_title('Phase Diagram', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim(kappa_range)
    ax1.set_ylim(0, d_c_curve.max() * 1.3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Order parameter
    ax2 = axes[0, 1]
    ratio = np.exp(-separation**2 / (2 * kappa))
    order_param = 1 - 2 * ratio
    ax2.plot(kappa, order_param, 'b-', linewidth=2.5)
    ax2.axvline(kappa_c, color='r', linestyle='--', linewidth=2, label=f'$\\kappa_c$')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(r'$\kappa$', fontsize=12)
    ax2.set_ylabel(r'Order Parameter $\eta$', fontsize=12)
    ax2.set_title('Order Parameter', fontsize=14)
    ax2.legend()
    ax2.set_xlim(kappa_range)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cross-group attention
    ax3 = axes[1, 0]
    beta_cross = np.exp(-separation**2 / (2 * kappa))
    ax3.semilogy(kappa, beta_cross, 'r-', linewidth=2.5)
    ax3.axvline(kappa_c, color='green', linestyle='--', linewidth=2)
    ax3.axhline(epsilon, color='gray', linestyle=':', linewidth=1.5, label=f'$\\epsilon$')
    ax3.set_xlabel(r'$\kappa$', fontsize=12)
    ax3.set_ylabel(r'$\beta_{cross}$', fontsize=12)
    ax3.set_title('Cross-Group Attention', fontsize=14)
    ax3.legend()
    ax3.set_xlim(kappa_range)
    ax3.grid(True, alpha=0.3, which='both')

    # Panel 4: Text summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Phase Transition Summary
    ========================

    Separation: d = {separation}
    Threshold: ε = {epsilon}

    Critical temperature:
    κ_c = d²/(2|ln(ε)|) = {kappa_c:.4f}

    Critical distance (at κ=1):
    d_c = √(2κ|ln(ε)|) = {np.sqrt(2 * abs(np.log(epsilon))):.4f}

    Polarization Stability:
    • κ < κ_c: Polarized (β_cross < ε)
    • κ > κ_c: Mixed (β_cross > ε)

    Scaling: κ_c ∝ d²
    Exponent: β = 1/2 (mean-field)
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('Kappa Phase Transition Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, axes


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PHASE DIAGRAM VISUALIZATION")
    print("=" * 70)

    if HAS_MATPLOTLIB:
        # Generate all plots
        print("\n[1] Phase Diagram...")
        plot_phase_diagram(save_path=None)

        print("\n[2] Order Parameter...")
        plot_order_parameter_vs_kappa(separation=2.0, save_path=None)

        print("\n[3] Summary Plot...")
        plot_phase_transition_summary(separation=2.0, save_path=None)

        plt.show()
        print("\nPlots displayed.")
    else:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
