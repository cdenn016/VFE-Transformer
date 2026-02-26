#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Gauge VFE Manuscript
=============================================================

Generates clean figures with properly formatted x-axis labels (no overlap).

Usage:
    python scripts/generate_publication_figures.py --metrics_file <path> --output_dir <path>

Example:
    python scripts/generate_publication_figures.py \
        --metrics_file 230_K=100_N=128_so20_200k_3batch_ffn_VFE_dynamic/metrics.csv \
        --output_dir figures/

Author: Generated for LLM_VFE_manuscript
Date: January 2026
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# Publication-quality settings
rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_metrics_csv(csv_path: Path) -> dict:
    """Load training metrics from CSV file."""
    metrics = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'step' in row:
                metrics['steps'].append(int(row['step']))

            for key, value in row.items():
                if key == 'step':
                    continue
                try:
                    if value and value not in ['', 'None', 'nan']:
                        metrics[key].append(float(value))
                    else:
                        metrics[key].append(None)
                except (ValueError, TypeError):
                    metrics[key].append(None)

    return dict(metrics)


def format_steps_k(x, pos):
    """Format step numbers as 'Xk' (e.g., 50000 -> '50k')."""
    if x >= 1000:
        return f'{int(x/1000)}k'
    return str(int(x))


def create_training_curves_figure(metrics: dict, output_dir: Path, title_suffix: str = ""):
    """Create training curves figure with proper formatting."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    steps = np.array(metrics.get('steps', []))
    train_loss = np.array(metrics.get('train_loss_total', metrics.get('train_loss', [])))
    val_loss = np.array(metrics.get('val_loss', []))
    val_ppl = np.array(metrics.get('val_ppl', []))

    # Filter out None values for validation data
    val_mask = np.array([v is not None for v in val_loss])
    val_steps = steps[val_mask] if any(val_mask) else []
    val_loss_clean = np.array([v for v in val_loss if v is not None])

    val_ppl_mask = np.array([v is not None and v > 0 for v in val_ppl])
    val_ppl_steps = steps[val_ppl_mask] if any(val_ppl_mask) else []
    val_ppl_clean = np.array([v for v in val_ppl if v is not None and v > 0])

    # Panel (a): Loss curves
    ax = axes[0]
    ax.plot(steps, train_loss, label='Train', alpha=0.7, linewidth=1.5, color='#1f77b4')
    if len(val_loss_clean) > 0:
        ax.plot(val_steps, val_loss_clean, 'o-', label='Validation',
                linewidth=2, markersize=4, color='#ff7f0e', markevery=max(1, len(val_steps)//10))

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training and Validation Loss')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # Panel (b): Perplexity
    ax = axes[1]
    if len(val_ppl_clean) > 0:
        ax.semilogy(val_ppl_steps, val_ppl_clean, 'o-',
                   linewidth=2, markersize=4, color='#2ca02c', markevery=max(1, len(val_ppl_steps)//10))

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title('(b) Validation Perplexity')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['pdf', 'png']:
        fig_path = output_dir / f'training_curves.{fmt}'
        plt.savefig(fig_path, format=fmt)
        print(f"Saved: {fig_path}")

    plt.close()


def create_train_val_gap_figure(metrics: dict, output_dir: Path):
    """Create train-validation gap figure."""

    fig, ax = plt.subplots(figsize=(6, 4))

    steps = np.array(metrics.get('steps', []))
    train_loss = np.array(metrics.get('train_loss_total', metrics.get('train_loss', [])))
    val_loss = np.array(metrics.get('val_loss', []))

    # Get validation points
    val_mask = np.array([v is not None for v in val_loss])
    val_steps = steps[val_mask] if any(val_mask) else []
    val_loss_clean = np.array([v for v in val_loss if v is not None])
    train_at_val = train_loss[val_mask] if any(val_mask) else []

    if len(val_loss_clean) > 0 and len(train_at_val) > 0:
        gap = val_loss_clean - train_at_val

        ax.plot(val_steps, gap, 'o-', linewidth=2, markersize=5, color='#d62728')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Validation - Train Loss')
        ax.set_title('Train-Validation Gap')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['pdf', 'png']:
        fig_path = output_dir / f'train_val_gap.{fmt}'
        plt.savefig(fig_path, format=fmt)
        print(f"Saved: {fig_path}")

    plt.close()


def create_combined_figure(metrics: dict, output_dir: Path, model_name: str = "Gauge VFE"):
    """Create a combined 2x2 figure for publication."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    steps = np.array(metrics.get('steps', []))
    train_loss = np.array(metrics.get('train_loss_total', metrics.get('train_loss', [])))
    val_loss = np.array(metrics.get('val_loss', []))
    val_ppl = np.array(metrics.get('val_ppl', []))
    train_ppl = np.array(metrics.get('train_ppl', []))

    # Filter validation data
    val_mask = np.array([v is not None for v in val_loss])
    val_steps = steps[val_mask] if any(val_mask) else []
    val_loss_clean = np.array([v for v in val_loss if v is not None])

    val_ppl_mask = np.array([v is not None and v > 0 for v in val_ppl])
    val_ppl_steps = steps[val_ppl_mask] if any(val_ppl_mask) else []
    val_ppl_clean = np.array([v for v in val_ppl if v is not None and v > 0])

    # (a) Training loss
    ax = axes[0, 0]
    ax.plot(steps, train_loss, linewidth=1.5, color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'(a) {model_name} Training Loss')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # (b) Validation loss
    ax = axes[0, 1]
    if len(val_loss_clean) > 0:
        ax.plot(val_steps, val_loss_clean, 'o-', linewidth=2, markersize=5, color='#ff7f0e')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title(f'(b) {model_name} Validation Loss')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # (c) Validation perplexity (log scale)
    ax = axes[1, 0]
    if len(val_ppl_clean) > 0:
        ax.semilogy(val_ppl_steps, val_ppl_clean, 'o-', linewidth=2, markersize=5, color='#2ca02c')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_title(f'(c) {model_name} Validation Perplexity')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # (d) Train-val gap
    ax = axes[1, 1]
    if len(val_loss_clean) > 0:
        train_at_val = train_loss[val_mask]
        gap = val_loss_clean - train_at_val
        ax.plot(val_steps, gap, 'o-', linewidth=2, markersize=5, color='#d62728')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Val - Train Loss')
    ax.set_title(f'(d) {model_name} Generalization Gap')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_steps_k))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ['pdf', 'png']:
        fig_path = output_dir / f'combined_training_figure.{fmt}'
        plt.savefig(fig_path, format=fmt)
        print(f"Saved: {fig_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='Path to metrics.csv file')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    parser.add_argument('--model_name', type=str, default='Gauge VFE (SO(20))',
                       help='Model name for figure titles')
    args = parser.parse_args()

    metrics_path = Path(args.metrics_file)
    output_dir = Path(args.output_dir)

    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return

    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics_csv(metrics_path)
    print(f"Loaded {len(metrics.get('steps', []))} steps")

    print(f"\nGenerating figures in: {output_dir}")

    # Generate all figures
    create_training_curves_figure(metrics, output_dir)
    create_train_val_gap_figure(metrics, output_dir)
    create_combined_figure(metrics, output_dir, args.model_name)

    print("\nDone!")


if __name__ == '__main__':
    main()
