#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze RG Flow in VFE Gauge Transformers
==========================================

This script performs comprehensive Renormalization Group (RG) flow analysis
on trained VFE gauge transformer models.

The analysis reveals:
1. Meta-agent emergence through modularity in attention
2. Information compression via effective rank reduction
3. Belief coherence within emergent clusters
4. Multi-scale self-similarity (scale invariance)
5. Fixed points and phase transitions

Usage:
    # Analyze a trained checkpoint
    python scripts/analyze_rg_flow.py --checkpoint path/to/model.pt --output rg_analysis/

    # Quick analysis (fewer batches)
    python scripts/analyze_rg_flow.py --checkpoint path/to/model.pt --n_batches 5

    # With custom dataset
    python scripts/analyze_rg_flow.py --checkpoint path/to/model.pt --dataset wikitext-2

Author: VFE Transformer Team
Date: January 2026
"""

import sys
import os

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Import analysis modules
from transformer.analysis.rg_flow_analysis import (
    RGFlowTracker,
    run_rg_analysis,
    plot_rg_flow,
    plot_rg_phase_space,
    plot_attention_block_structure,
    plot_multiscale_comparison,
    RGFlowTrajectory,
)
from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    compute_modularity,
    compute_effective_rank,
    RGFlowSummary,
    analyze_rg_flow as analyze_rg_flow_metrics,
)

# Import model and data
from transformer.core.model import GaugeTransformerLM
from transformer.data import create_dataloaders


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """
    Load model from checkpoint.

    Returns:
        model: GaugeTransformerLM instance
        config: Model configuration dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Try to infer config from state dict
        raise ValueError("Checkpoint must contain 'config' or 'model_config'")

    # Create model
    model = GaugeTransformerLM(config)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


def analyze_single_batch(
    model,
    batch,
    device: str = 'cuda',
    verbose: bool = True,
) -> dict:
    """
    Analyze RG structure for a single batch.

    Returns detailed diagnostics for visualization and interpretation.
    """
    # Get input tokens
    if isinstance(batch, dict):
        input_ids = batch['input_ids'].to(device)
    elif isinstance(batch, (tuple, list)):
        input_ids = batch[0].to(device)
    else:
        input_ids = batch.to(device)

    with torch.no_grad():
        # Forward with attention tracking
        logits, attn_info = model.forward_with_attention(input_ids)

        beta = attn_info['beta']  # (B, H, N, N) or (B, N, N)
        mu = attn_info['mu']      # (B, N, K)
        sigma = attn_info.get('sigma')

        if sigma is None:
            sigma = torch.ones_like(mu)

        # Average over heads if multi-head
        if beta.dim() == 4:
            n_heads = beta.shape[1]
            beta_avg = beta.mean(dim=1)
            if verbose:
                print(f"  Multi-head attention: {n_heads} heads")
        else:
            beta_avg = beta

        B, N, K = mu.shape
        if verbose:
            print(f"  Batch size: {B}, Sequence length: {N}, Embedding dim: {K}")

        # Compute RG diagnostics
        diagnostics = compute_rg_diagnostics(
            mu=mu, sigma=sigma, beta=beta_avg,
            step=0, auto_cluster=True
        )

        # Compute per-head diagnostics if multi-head
        per_head_mods = []
        per_head_ranks = []
        if beta.dim() == 4:
            for h in range(n_heads):
                beta_h = beta[:, h]
                mod_h = compute_modularity(beta_h)
                rank_h = compute_effective_rank(beta_h)
                per_head_mods.append(mod_h)
                per_head_ranks.append(rank_h)

        # Multi-scale analysis
        tracker = RGFlowTracker(track_multiscale=True)
        multiscale = tracker.analyze_multiscale(beta_avg, mu, sigma)

        # Build result
        result = {
            'diagnostics': diagnostics,
            'multiscale': multiscale,
            'beta': beta_avg.cpu(),
            'mu': mu.cpu(),
            'sigma': sigma.cpu() if isinstance(sigma, torch.Tensor) else sigma,
            'cluster_labels': diagnostics.cluster_labels.cpu() if diagnostics.cluster_labels is not None else None,
            'per_head': {
                'modularity': per_head_mods,
                'effective_rank': per_head_ranks,
            } if per_head_mods else None,
        }

        if verbose:
            print(f"  Modularity: {diagnostics.modularity:.4f}")
            print(f"  Effective rank: {diagnostics.effective_rank:.2f}")
            print(f"  N clusters: {diagnostics.n_clusters}")
            print(f"  KL within: {diagnostics.kl_within_mean:.4f}")
            print(f"  KL between: {diagnostics.kl_between_mean:.4f}")
            print(f"  Scales detected: {multiscale.n_scales}")

    return result


def run_full_analysis(
    model,
    dataloader,
    output_dir: str,
    n_batches: int = 20,
    device: str = 'cuda',
):
    """
    Run comprehensive RG flow analysis.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("RG FLOW ANALYSIS")
    print("="*70)
    print(f"Output directory: {output_path}")
    print(f"Number of batches: {n_batches}")
    print(f"Device: {device}")

    # Run analysis
    results = run_rg_analysis(
        model=model,
        dataloader=dataloader,
        device=device,
        n_batches=n_batches,
        track_per_batch=True,
        output_dir=str(output_path),
    )

    # Additional per-head analysis for first batch
    print("\n[Per-Head Analysis (first batch)]")
    first_batch = next(iter(dataloader))
    detailed = analyze_single_batch(model, first_batch, device=device, verbose=True)

    if detailed['per_head'] is not None:
        print("\n  Per-head modularity:")
        for h, mod in enumerate(detailed['per_head']['modularity']):
            print(f"    Head {h}: Q={mod:.4f}")

    # Generate detailed plots
    print("\n[Generating detailed visualizations...]")

    # Plot attention block structure for first batch
    plot_attention_block_structure(
        beta=detailed['beta'],
        cluster_labels=detailed['cluster_labels'],
        save_path=str(output_path / 'attention_blocks.png'),
        title='Attention Block Structure (First Batch)',
    )

    # If we have trajectory data, analyze fixed points
    if results['trajectories']:
        print("\n[Fixed Point Analysis]")
        tracker = RGFlowTracker()

        # Build combined trajectory
        all_mods = [d.modularity for d in results['diagnostics']]
        all_ranks = [d.effective_rank for d in results['diagnostics']]
        all_kl_w = [d.kl_within_mean for d in results['diagnostics']]
        all_kl_b = [d.kl_between_mean for d in results['diagnostics']]
        all_ent = [d.beta_entropy for d in results['diagnostics']]
        all_n_c = [d.n_clusters for d in results['diagnostics']]

        combined_traj = RGFlowTrajectory(
            steps=list(range(len(all_mods))),
            modularity=all_mods,
            effective_rank=all_ranks,
            n_clusters=all_n_c,
            kl_within=all_kl_w,
            kl_between=all_kl_b,
            entropy=all_ent,
        )

        # Detect fixed points
        fps = tracker.detect_fixed_points(combined_traj)
        print(f"  Candidate fixed points found: {len(fps)}")

        # Compute scaling exponents
        scaling = tracker.compute_scaling_exponents(combined_traj)
        print(f"  Modularity exponent: {scaling.modularity_exponent:.3f} (R²={scaling.fit_quality['modularity']:.2f})")
        print(f"  Rank exponent: {scaling.rank_exponent:.3f} (R²={scaling.fit_quality['rank']:.2f})")
        print(f"  Correlation length: {scaling.correlation_length:.2f}")

        # Save scaling analysis
        with open(output_path / 'scaling_analysis.json', 'w') as f:
            json.dump(scaling.to_dict(), f, indent=2)

        # Save fixed points
        if fps:
            fp_data = [fp.to_dict() for fp in fps]
            with open(output_path / 'fixed_points.json', 'w') as f:
                json.dump(fp_data, f, indent=2)

    # Generate summary report
    report = generate_report(results, output_path)
    with open(output_path / 'rg_analysis_report.md', 'w') as f:
        f.write(report)

    print(f"\n[Analysis complete! Results saved to {output_path}]")

    return results


def generate_report(results: dict, output_path: Path) -> str:
    """
    Generate a markdown report of the RG analysis.
    """
    summary = results['summary']
    rg_sigs = summary['rg_signatures']

    report = f"""# RG Flow Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Mean | Std |
|--------|------|-----|
| Modularity | {summary['modularity']['mean']:.4f} | {summary['modularity']['std']:.4f} |
| Effective Rank | {summary['effective_rank']['mean']:.2f} | {summary['effective_rank']['std']:.2f} |
| N Clusters | {summary['n_clusters']['mean']:.1f} | {summary['n_clusters']['std']:.1f} |
| KL Within | {summary['kl_within']['mean']:.4f} | {summary['kl_within']['std']:.4f} |
| KL Between | {summary['kl_between']['mean']:.4f} | {summary['kl_between']['std']:.4f} |
| Scale Invariance | {summary['scale_invariance']['mean']:.4f} | {summary['scale_invariance']['std']:.4f} |

**KL Ratio (within/between):** {summary['kl_ratio']:.4f}

## RG Behavior Signatures

The following signatures indicate healthy RG flow behavior:

| Signature | Status | Interpretation |
|-----------|--------|----------------|
| High Modularity | {'YES' if rg_sigs['high_modularity'] else 'NO'} | Block structure in attention (meta-agents forming) |
| Low KL Ratio | {'YES' if rg_sigs['low_kl_ratio'] else 'NO'} | Clusters tighter than separations |
| Rank Reduction | {'YES' if rg_sigs['moderate_rank_reduction'] else 'NO'} | Information compression occurring |
| Scale Invariance | {'YES' if rg_sigs['scale_invariant'] else 'NO'} | Self-similar structure across scales |

## Interpretation

"""
    # Add interpretation based on signatures
    all_good = all(rg_sigs.values())
    some_good = any(rg_sigs.values())

    if all_good:
        report += """
The model exhibits **strong RG behavior** with all signatures present:
- Meta-agents are emerging as coherent clusters of tokens
- Information is being compressed while maintaining distinctness
- The dynamics show self-similarity across coarse-graining scales
- This suggests the VFE is approaching optimal representations

This is consistent with the theoretical prediction that VFE descent implements
an RG transformation from fine-grained token representations to coarse-grained
semantic summaries.
"""
    elif some_good:
        report += """
The model shows **partial RG behavior**:
"""
        if rg_sigs['high_modularity']:
            report += "- Meta-agents ARE forming (high modularity)\n"
        else:
            report += "- Meta-agents NOT clearly forming (low modularity)\n"

        if rg_sigs['low_kl_ratio']:
            report += "- Clusters ARE coherent (low KL ratio)\n"
        else:
            report += "- Clusters NOT well-separated (high KL ratio)\n"

        if rg_sigs['scale_invariant']:
            report += "- Structure IS self-similar across scales\n"
        else:
            report += "- Structure NOT self-similar across scales\n"

        report += """
This partial RG behavior may indicate:
- The model is still learning and hasn't converged
- The VFE parameters (kappa, alpha) need adjustment
- The task doesn't require hierarchical representation
"""
    else:
        report += """
The model shows **weak RG behavior** - none of the expected signatures are present.
This may indicate:
- Random or uniform attention (no structure learning)
- Insufficient training
- Architecture issues preventing meta-agent formation

Consider:
- Training longer
- Adjusting kappa_beta (attention temperature)
- Checking that gauge frames are evolving
"""

    report += f"""

## Files Generated

- `rg_analysis_summary.json` - Raw statistics
- `rg_flow.png` - RG observables over batches
- `rg_phase_space.png` - Phase space trajectory
- `attention_blocks.png` - Block structure visualization
- `multiscale.png` - Multi-scale comparison
- `scaling_analysis.json` - Critical exponents
- `fixed_points.json` - Detected fixed points

## References

The RG interpretation of VFE dynamics follows from the observation that
meta-agents (clusters of tokens with coherent beliefs) satisfy the same
definition as individual agents. This self-similarity is the defining
property of renormalization group theories.

See: Wilson, K. G. (1974). The renormalization group and the epsilon expansion.
"""

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RG flow in VFE gauge transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/analyze_rg_flow.py --checkpoint checkpoints/best_model.pt

  # With more batches
  python scripts/analyze_rg_flow.py --checkpoint checkpoints/best_model.pt --n_batches 50

  # Custom output directory
  python scripts/analyze_rg_flow.py --checkpoint checkpoints/best_model.pt --output my_analysis/
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='rg_analysis',
        help='Output directory for results (default: rg_analysis)',
    )
    parser.add_argument(
        '--n_batches', '-n',
        type=int,
        default=20,
        help='Number of batches to analyze (default: 20)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='wikitext-103',
        choices=['wikitext-2', 'wikitext-103'],
        help='Dataset to use (default: wikitext-103)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size (default: 4)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)',
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=128,
        help='Sequence length (default: 128)',
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("="*70)
    print("RG FLOW ANALYZER FOR VFE GAUGE TRANSFORMERS")
    print("="*70)

    # Load model
    print(f"\n[Loading model from {args.checkpoint}...]")
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)
    print(f"  Model loaded successfully")
    print(f"  Config: embed_dim={config.get('embed_dim')}, n_layers={config.get('n_layers')}")

    # Create dataloader
    print(f"\n[Creating dataloader for {args.dataset}...]")

    # Get vocab size and max_seq_len from config
    vocab_size = config.get('vocab_size', 50257)
    max_seq_len = config.get('max_seq_len', args.seq_len)

    train_loader, val_loader = create_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
        num_workers=0,  # Avoid multiprocessing issues
    )
    print(f"  Dataloader created (batch_size={args.batch_size}, seq_len={max_seq_len})")

    # Run analysis
    results = run_full_analysis(
        model=model,
        dataloader=val_loader,
        output_dir=args.output,
        n_batches=args.n_batches,
        device=args.device,
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
