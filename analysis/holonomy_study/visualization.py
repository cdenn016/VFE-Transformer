"""
Holonomy Study Visualization
==============================

Plots for the flat bundle conjecture experiment:
    1. Holonomy distributions by condition (ironic / literal / control)
    2. Layer-resolved curvature profiles
    3. Per-pair comparison (same sentence, different context)
    4. Attention asymmetry comparison (Method 0)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List

from .experiment import ExperimentResult
from .holonomy import HolonomyResult


COLORS = {
    'ironic': '#d62728',    # red
    'literal': '#2ca02c',   # green
    'control': '#1f77b4',   # blue
}


def plot_holonomy_distributions(
    result: ExperimentResult,
    output_path: Optional[str] = None,
    title: str = 'Holonomy Distribution by Condition',
):
    """
    Violin + strip plot of per-sentence mean kappa by condition.

    The core visualization: if the conjecture holds, the ironic
    distribution should be shifted right relative to literal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels_order = ['literal', 'ironic', 'control']
    kappas = {}
    for label in labels_order:
        if label in result.results_by_label and len(result.results_by_label[label]) > 0:
            kappas[label] = np.array([
                hr.kappa_mean for hr in result.results_by_label[label]
            ])

    if not kappas:
        print("No holonomy data to plot")
        return

    # --- Left: violin + strip ---
    ax = axes[0]
    positions = []
    data = []
    colors = []
    tick_labels = []

    for idx, label in enumerate(labels_order):
        if label not in kappas:
            continue
        positions.append(idx)
        data.append(kappas[label])
        colors.append(COLORS[label])
        tick_labels.append(f"{label}\n(n={len(kappas[label])})")

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.3)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('gray')

    # Strip plot overlay
    for i, (pos, d) in enumerate(zip(positions, data)):
        jitter = np.random.RandomState(42).uniform(-0.1, 0.1, size=len(d))
        ax.scatter(pos + jitter, d, c=colors[i], alpha=0.5, s=15, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(r'Mean holonomy $\kappa$')
    ax.set_title(title)

    # Add significance annotation
    if result.ironic_vs_literal and result.ironic_vs_literal['p_value'] < 0.05:
        p = result.ironic_vs_literal['p_value']
        d = result.ironic_vs_literal['cohens_d']
        ax.annotate(
            f'p = {p:.4f}\nd = {d:.3f}',
            xy=(0.5, 0.95), xycoords='axes fraction',
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    # --- Right: histogram overlay ---
    ax = axes[1]
    all_vals = np.concatenate(list(kappas.values()))
    bins = np.linspace(np.min(all_vals), np.max(all_vals), 30)

    for label in labels_order:
        if label in kappas:
            ax.hist(kappas[label], bins=bins, alpha=0.4,
                    color=COLORS[label], label=label, density=True)

    ax.set_xlabel(r'Mean holonomy $\kappa$')
    ax.set_ylabel('Density')
    ax.set_title('Holonomy Distributions')
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_layer_profile(
    layer_resolved: dict,
    output_path: Optional[str] = None,
):
    """
    Plot holonomy kappa as a function of network depth.

    Shows how curvature builds up through layers.
    Expect: ironic sentences show curvature emerging at later (semantic) layers.
    """
    if not layer_resolved:
        print("No layer-resolved data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by label
    ironic_profiles = {}
    literal_profiles = {}

    for key, profile in layer_resolved.items():
        label = key.split('_')[0]
        if label == 'ironic':
            ironic_profiles[key] = profile
        elif label == 'literal':
            literal_profiles[key] = profile

    # Plot individual traces with low alpha, then means
    for profiles, color, label in [
        (ironic_profiles, COLORS['ironic'], 'ironic'),
        (literal_profiles, COLORS['literal'], 'literal'),
    ]:
        if not profiles:
            continue

        all_depths = set()
        for p in profiles.values():
            all_depths.update(p.keys())
        depths = sorted(all_depths)

        # Individual traces
        for key, profile in profiles.items():
            xs = sorted(profile.keys())
            ys = [profile[d] for d in xs]
            ax.plot(xs, ys, color=color, alpha=0.15, linewidth=1)

        # Mean trace
        mean_ys = []
        for d in depths:
            vals = [p[d] for p in profiles.values() if d in p]
            mean_ys.append(np.mean(vals))
        ax.plot(depths, mean_ys, color=color, linewidth=2.5, label=f'{label} (mean)')

    ax.set_xlabel('Network Depth (layers)')
    ax.set_ylabel(r'Mean holonomy $\kappa$')
    ax.set_title('Curvature Emergence by Depth')
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_paired_comparison(
    result: ExperimentResult,
    output_path: Optional[str] = None,
):
    """
    For paired examples (same target sentence, different context),
    plot ironic vs literal kappa as paired points connected by lines.

    Each line connects the same sentence in ironic vs literal context.
    If the conjecture holds, most lines should slope upward (ironic > literal).
    """
    # Match pairs by pair_id
    ironic_by_pair = {}
    literal_by_pair = {}

    for hr in result.results_by_label.get('ironic', []):
        pid = hr.metadata.get('pair_id')
        if pid is not None:
            ironic_by_pair[pid] = hr.kappa_mean

    for hr in result.results_by_label.get('literal', []):
        pid = hr.metadata.get('pair_id')
        if pid is not None:
            literal_by_pair[pid] = hr.kappa_mean

    # Find shared pair IDs
    shared = sorted(set(ironic_by_pair.keys()) & set(literal_by_pair.keys()))
    if not shared:
        print("No paired examples found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ironic_vals = [ironic_by_pair[pid] for pid in shared]
    literal_vals = [literal_by_pair[pid] for pid in shared]

    # Paired lines
    for iv, lv in zip(ironic_vals, literal_vals):
        color = COLORS['ironic'] if iv > lv else COLORS['literal']
        ax.plot([0, 1], [lv, iv], color=color, alpha=0.4, linewidth=1)

    # Points
    ax.scatter([0] * len(literal_vals), literal_vals,
               c=COLORS['literal'], s=40, zorder=3, label='literal')
    ax.scatter([1] * len(ironic_vals), ironic_vals,
               c=COLORS['ironic'], s=40, zorder=3, label='ironic')

    # Summary
    n_higher = sum(1 for i, l in zip(ironic_vals, literal_vals) if i > l)
    total = len(shared)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Literal context', 'Ironic context'])
    ax.set_ylabel(r'Mean holonomy $\kappa$')
    ax.set_title(f'Paired Comparison: Same Sentence, Different Context\n'
                 f'Ironic > Literal in {n_higher}/{total} pairs '
                 f'({100*n_higher/total:.0f}%)')
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_asymmetry_comparison(
    asymmetry_by_label: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
):
    """
    Plot Method 0 attention asymmetry distributions.
    Quick sanity check before running full transport analysis.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels_order = ['literal', 'ironic', 'control']
    for label in labels_order:
        if label in asymmetry_by_label:
            data = asymmetry_by_label[label]
            ax.hist(data, bins=20, alpha=0.4, color=COLORS[label],
                    label=f'{label} (n={len(data)})', density=True)

    ax.set_xlabel('Mean attention path defect')
    ax.set_ylabel('Density')
    ax.set_title('Method 0: Attention Path Defect')
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def generate_all_plots(
    result: ExperimentResult,
    output_dir: str = 'results/holonomy_study',
):
    """Generate all visualization plots from experiment results."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if result.results_by_label:
        plot_holonomy_distributions(
            result, output_path=str(out / 'holonomy_distributions.png')
        )
        plot_paired_comparison(
            result, output_path=str(out / 'paired_comparison.png')
        )

    if result.asymmetry_by_label:
        plot_asymmetry_comparison(
            result.asymmetry_by_label,
            output_path=str(out / 'asymmetry_comparison.png'),
        )

    if result.layer_resolved:
        plot_layer_profile(
            result.layer_resolved,
            output_path=str(out / 'layer_profile.png'),
        )

    print(f"All plots saved to {out}/")
