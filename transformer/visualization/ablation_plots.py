"""
Ablation Study Figures: Gauge VFE vs Standard Transformer
==========================================================

Simple functions to generate comparison figures. No CLI needed.

Usage:
    from transformer.plot_ablation import plot_ablation

    # Just call with two directories
    plot_ablation(
        vfe_dir='checkpoints_publication/vfe_run',
        std_dir='checkpoints_publication/standard_baseline',
    )

Author: Ablation study for gauge transformer paper
Date: December 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


# =============================================================================
# Publication Style
# =============================================================================

PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}

# Color scheme for models
COLORS = {
    'vfe': '#1f77b4',      # Blue for gauge VFE
    'standard': '#ff7f0e', # Orange for standard
    'vfe_val': '#2ca02c',  # Green for VFE validation
    'std_val': '#d62728',  # Red for standard validation
}


@dataclass
class TrainingRun:
    """Data from a training run."""
    name: str
    steps: List[int]
    train_ppl: List[float]
    val_ppl: List[float]
    val_steps: List[int]
    wall_time: List[float]  # Cumulative wall-clock time
    tokens_seen: List[int]  # Cumulative tokens processed
    final_ppl: float
    total_time: float
    total_tokens: int
    params: int
    config: Dict


def load_training_history(run_dir: Path, name: str = "model") -> Optional[TrainingRun]:
    """Load training history from a checkpoint directory."""
    run_dir = Path(run_dir)

    # Try to load from CSV first (more detailed)
    csv_path = run_dir / "training_history.csv"
    json_path = run_dir / "training_history.json"
    metrics_csv = run_dir / "metrics.csv"

    steps = []
    train_ppl = []
    val_ppl = []
    val_steps = []
    wall_time = []
    tokens_seen = []

    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            cumulative_time = 0.0
            cumulative_tokens = 0

            for row in reader:
                step = int(row.get('step', 0))
                steps.append(step)

                # PPL
                ppl = float(row.get('train_ppl', 0))
                train_ppl.append(ppl if ppl < 1e6 else float('nan'))

                # Validation PPL
                v_ppl = row.get('val_ppl', '')
                if v_ppl and v_ppl != 'None' and v_ppl != '':
                    try:
                        val_ppl.append(float(v_ppl))
                        val_steps.append(step)
                    except ValueError:
                        pass

                # Time
                step_time = float(row.get('step_time', 0))
                cumulative_time += step_time
                wall_time.append(cumulative_time)

                # Tokens (estimate if not available)
                tps = float(row.get('tokens_per_sec', 0))
                if tps > 0 and step_time > 0:
                    cumulative_tokens += int(tps * step_time)
                tokens_seen.append(cumulative_tokens)

    elif metrics_csv.exists():
        # Standard baseline format
        with open(metrics_csv, 'r') as f:
            reader = csv.DictReader(f)
            cumulative_time = 0.0
            cumulative_tokens = 0

            for row in reader:
                step = int(row.get('step', 0))
                steps.append(step)

                # PPL
                ppl = float(row.get('train_ppl', 0))
                train_ppl.append(ppl if ppl < 1e6 else float('nan'))

                # Validation PPL
                v_ppl = row.get('val_ppl', '')
                if v_ppl and v_ppl != 'None' and v_ppl != '':
                    try:
                        val_ppl.append(float(v_ppl))
                        val_steps.append(step)
                    except ValueError:
                        pass

                # Time
                step_time = float(row.get('step_time', 0))
                cumulative_time += step_time
                wall_time.append(cumulative_time)

                # Tokens
                tps = float(row.get('tokens_per_sec', 0))
                if tps > 0 and step_time > 0:
                    cumulative_tokens += int(tps * step_time)
                tokens_seen.append(cumulative_tokens)

    elif json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)

        history = data.get('history', [])
        cumulative_time = 0.0
        cumulative_tokens = 0

        for entry in history:
            step = entry.get('step', 0)
            steps.append(step)

            ppl = entry.get('train_ppl', 0)
            train_ppl.append(ppl if ppl < 1e6 else float('nan'))

            v_ppl = entry.get('val_ppl')
            if v_ppl is not None:
                val_ppl.append(v_ppl)
                val_steps.append(step)

            step_time = entry.get('step_time', 0)
            cumulative_time += step_time
            wall_time.append(cumulative_time)

            tps = entry.get('tokens_per_sec', 0)
            if tps > 0 and step_time > 0:
                cumulative_tokens += int(tps * step_time)
            tokens_seen.append(cumulative_tokens)

    else:
        print(f"[WARN] No training history found in {run_dir}")
        return None

    if not steps:
        print(f"[WARN] Empty training history in {run_dir}")
        return None

    # Load config if available
    config = {}
    config_paths = [run_dir / "config.json", run_dir / "training_log.json"]
    for cp in config_paths:
        if cp.exists():
            with open(cp, 'r') as f:
                data = json.load(f)
                config = data.get('config', data)
                break

    # Calculate totals
    final_ppl = val_ppl[-1] if val_ppl else train_ppl[-1]
    total_time = wall_time[-1] if wall_time else 0
    total_tokens = tokens_seen[-1] if tokens_seen else 0
    params = config.get('total_params', 0)

    return TrainingRun(
        name=name,
        steps=steps,
        train_ppl=train_ppl,
        val_ppl=val_ppl,
        val_steps=val_steps,
        wall_time=wall_time,
        tokens_seen=tokens_seen,
        final_ppl=final_ppl,
        total_time=total_time,
        total_tokens=total_tokens,
        params=params,
        config=config,
    )


def plot_step_comparison(
    vfe_run: TrainingRun,
    std_run: TrainingRun,
    save_dir: Path,
    title: str = "Step-by-Step Comparison",
) -> plt.Figure:
    """
    Plot PPL vs training steps for both models.

    This is a step-matched comparison (same steps, different compute).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Training PPL
    ax = axes[0]
    ax.semilogy(vfe_run.steps, vfe_run.train_ppl,
                color=COLORS['vfe'], label=f'Gauge VFE ({vfe_run.name})',
                alpha=0.8, linewidth=1.5)
    ax.semilogy(std_run.steps, std_run.train_ppl,
                color=COLORS['standard'], label=f'Standard ({std_run.name})',
                alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('(a) Training Perplexity vs Steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) Validation PPL
    ax = axes[1]
    if vfe_run.val_steps and vfe_run.val_ppl:
        ax.semilogy(vfe_run.val_steps, vfe_run.val_ppl,
                    'o-', color=COLORS['vfe'], label=f'Gauge VFE',
                    markersize=4, linewidth=1.5)
    if std_run.val_steps and std_run.val_ppl:
        ax.semilogy(std_run.val_steps, std_run.val_ppl,
                    's-', color=COLORS['standard'], label=f'Standard',
                    markersize=4, linewidth=1.5)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Perplexity (log scale)')
    ax.set_title('(b) Validation Perplexity vs Steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation box with final metrics
    textstr = (
        f"Final Val PPL:\n"
        f"  VFE: {vfe_run.final_ppl:.1f}\n"
        f"  Std: {std_run.final_ppl:.1f}\n"
        f"Steps: {vfe_run.steps[-1]:,}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1].text(0.98, 0.98, textstr, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "ablation_step_comparison.pdf")
    fig.savefig(save_dir / "ablation_step_comparison.png", dpi=300)
    print(f"Saved step comparison to {save_dir}/ablation_step_comparison.png")

    return fig


def plot_compute_comparison(
    vfe_run: TrainingRun,
    std_run: TrainingRun,
    save_dir: Path,
    title: str = "Compute-Matched Comparison",
) -> plt.Figure:
    """
    Plot PPL vs wall-clock time (fair compute comparison).

    This shows efficiency: which model achieves lower PPL per unit compute.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to minutes for readability
    vfe_time_min = [t / 60 for t in vfe_run.wall_time]
    std_time_min = [t / 60 for t in std_run.wall_time]

    # (a) Training PPL vs Time
    ax = axes[0]
    ax.semilogy(vfe_time_min, vfe_run.train_ppl,
                color=COLORS['vfe'], label=f'Gauge VFE',
                alpha=0.8, linewidth=1.5)
    ax.semilogy(std_time_min, std_run.train_ppl,
                color=COLORS['standard'], label=f'Standard',
                alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('(a) Training PPL vs Compute Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) Validation PPL vs Time
    ax = axes[1]

    # Map val_steps to wall_time
    def steps_to_time(steps_list, all_steps, all_times):
        """Map step numbers to wall times."""
        times = []
        for s in steps_list:
            # Find closest step
            idx = min(range(len(all_steps)), key=lambda i: abs(all_steps[i] - s))
            times.append(all_times[idx] / 60)  # Convert to minutes
        return times

    if vfe_run.val_steps and vfe_run.val_ppl:
        vfe_val_times = steps_to_time(vfe_run.val_steps, vfe_run.steps, vfe_run.wall_time)
        ax.semilogy(vfe_val_times, vfe_run.val_ppl,
                    'o-', color=COLORS['vfe'], label=f'Gauge VFE',
                    markersize=4, linewidth=1.5)

    if std_run.val_steps and std_run.val_ppl:
        std_val_times = steps_to_time(std_run.val_steps, std_run.steps, std_run.wall_time)
        ax.semilogy(std_val_times, std_run.val_ppl,
                    's-', color=COLORS['standard'], label=f'Standard',
                    markersize=4, linewidth=1.5)

    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Validation Perplexity (log scale)')
    ax.set_title('(b) Validation PPL vs Compute Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation box with compute stats
    speedup = std_run.total_time / vfe_run.total_time if vfe_run.total_time > 0 else 0
    textstr = (
        f"Total Time:\n"
        f"  VFE: {vfe_run.total_time/60:.1f} min\n"
        f"  Std: {std_run.total_time/60:.1f} min\n"
        f"Speedup: {speedup:.1f}x"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1].text(0.98, 0.98, textstr, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "ablation_compute_comparison.pdf")
    fig.savefig(save_dir / "ablation_compute_comparison.png", dpi=300)
    print(f"Saved compute comparison to {save_dir}/ablation_compute_comparison.png")

    return fig


def plot_combined_ablation(
    vfe_run: TrainingRun,
    std_run: TrainingRun,
    save_dir: Path,
) -> plt.Figure:
    """
    Combined 2x2 ablation figure for publication.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Convert to minutes
    vfe_time_min = [t / 60 for t in vfe_run.wall_time]
    std_time_min = [t / 60 for t in std_run.wall_time]

    # (a) Training PPL vs Steps
    ax = axes[0, 0]
    ax.semilogy(vfe_run.steps, vfe_run.train_ppl,
                color=COLORS['vfe'], label='Gauge VFE', alpha=0.8, linewidth=1.5)
    ax.semilogy(std_run.steps, std_run.train_ppl,
                color=COLORS['standard'], label='Standard', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Perplexity')
    ax.set_title('(a) Training PPL vs Steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (b) Validation PPL vs Steps
    ax = axes[0, 1]
    if vfe_run.val_steps:
        ax.semilogy(vfe_run.val_steps, vfe_run.val_ppl,
                    'o-', color=COLORS['vfe'], label='Gauge VFE', markersize=4)
    if std_run.val_steps:
        ax.semilogy(std_run.val_steps, std_run.val_ppl,
                    's-', color=COLORS['standard'], label='Standard', markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title('(b) Validation PPL vs Steps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (c) Training PPL vs Time
    ax = axes[1, 0]
    ax.semilogy(vfe_time_min, vfe_run.train_ppl,
                color=COLORS['vfe'], label='Gauge VFE', alpha=0.8, linewidth=1.5)
    ax.semilogy(std_time_min, std_run.train_ppl,
                color=COLORS['standard'], label='Standard', alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Wall-Clock Time (minutes)')
    ax.set_ylabel('Perplexity')
    ax.set_title('(c) Training PPL vs Compute Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # (d) Efficiency: Final PPL vs Total Compute
    ax = axes[1, 1]

    # Bar chart of final PPL with time annotations
    models = ['Gauge VFE', 'Standard']
    final_ppls = [vfe_run.final_ppl, std_run.final_ppl]
    times = [vfe_run.total_time / 60, std_run.total_time / 60]
    colors = [COLORS['vfe'], COLORS['standard']]

    bars = ax.bar(models, final_ppls, color=colors, edgecolor='black', linewidth=1)

    # Add PPL values on bars
    for bar, ppl, t in zip(bars, final_ppls, times):
        height = bar.get_height()
        ax.annotate(f'PPL: {ppl:.1f}\n({t:.1f} min)',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Final Validation Perplexity')
    ax.set_title('(d) Final Performance Summary')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ablation Study: Gauge VFE vs Standard Transformer',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "ablation_combined.pdf")
    fig.savefig(save_dir / "ablation_combined.png", dpi=300)
    print(f"Saved combined ablation to {save_dir}/ablation_combined.png")

    return fig


def plot_ablation(
    vfe_dir: str,
    std_dir: str,
    output_dir: str = 'outputs/ablation',
    vfe_name: str = 'Gauge VFE',
    std_name: str = 'Standard',
):
    """
    Generate all ablation figures with one call.

    Args:
        vfe_dir: Path to VFE training results
        std_dir: Path to standard transformer results
        output_dir: Where to save figures
        vfe_name: Label for VFE model
        std_name: Label for standard model

    Example:
        plot_ablation(
            'checkpoints_publication/vfe_so2',
            'checkpoints_publication/standard_baseline',
        )
    """
    plt.rcParams.update(PUBLICATION_STYLE)

    vfe_run = load_training_history(Path(vfe_dir), name=vfe_name)
    std_run = load_training_history(Path(std_dir), name=std_name)

    if not vfe_run or not std_run:
        print("ERROR: Could not load training histories")
        return None

    print(f"VFE: {len(vfe_run.steps)} steps, final PPL={vfe_run.final_ppl:.1f}")
    print(f"Std: {len(std_run.steps)} steps, final PPL={std_run.final_ppl:.1f}")

    output_dir = Path(output_dir)
    plot_step_comparison(vfe_run, std_run, output_dir)
    plot_compute_comparison(vfe_run, std_run, output_dir)
    fig = plot_combined_ablation(vfe_run, std_run, output_dir)

    return fig


# Quick test
if __name__ == '__main__':
    
    plot_ablation(
    vfe_dir='checkpoints_publication/your_vfe_run',
    std_dir='checkpoints_publication/standard_baseline_stable',
)