# -*- coding: utf-8 -*-
"""
RG Flow Analysis: Dynamic vs Static Modes
==========================================

This module clarifies the two different manifestations of RG structure
in VFE transformers:

1. DYNAMIC RG (multiple VFE steps per forward pass):
   - Clusters form within a single forward pass
   - Track RG metrics across VFE iterations
   - Requires: n_vfe_steps > 1

2. STATIC RG (single VFE step, standard training):
   - RG structure is learned over training
   - Embedded in the priors/embeddings
   - Analyze attention patterns of trained model

Both are valid manifestations of RG - the question is WHERE the
coarse-graining happens: within inference or within learning.

Author: VFE Transformer Team
Date: January 2026
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from transformer.analysis.rg_metrics import (
    compute_modularity,
    compute_effective_rank,
    compute_beta_entropy,
    detect_clusters_spectral,
)


@dataclass
class DynamicRGTrajectory:
    """
    RG trajectory within a single forward pass (multiple VFE steps).

    This captures the DYNAMIC emergence of clusters as beliefs converge.
    """
    vfe_steps: List[int]
    modularity: List[float]
    effective_rank: List[float]
    n_clusters: List[int]
    kl_within: List[float]
    total_vfe: List[float]  # Free energy decreasing

    def is_converging(self) -> bool:
        """Check if RG metrics show convergence."""
        if len(self.modularity) < 3:
            return False
        # Modularity should increase or plateau
        mod_trend = np.diff(self.modularity[-5:]).mean() if len(self.modularity) >= 5 else 0
        return mod_trend >= -0.01  # Not decreasing


@dataclass
class StaticRGSnapshot:
    """
    RG structure at a single training step (for single-VFE-step models).

    This captures the LEARNED structure in trained embeddings.
    """
    training_step: int
    modularity: float
    effective_rank: float
    n_clusters: int
    cluster_sizes: List[int]
    attention_entropy: float

    # Compare to random baseline
    modularity_vs_random: float  # How much better than random?


def analyze_dynamic_rg_from_beta_history(
    beta_history: List[torch.Tensor],
) -> DynamicRGTrajectory:
    """
    Analyze RG flow from a list of attention matrices (beta_history).

    This is the cleanest way to track dynamic RG - use the beta_history
    returned by VariationalFFN when return_beta_history=True.

    Args:
        beta_history: List of (B, N, N) attention matrices from VFE iterations

    Returns:
        DynamicRGTrajectory with metrics at each VFE step
    """
    trajectory = DynamicRGTrajectory(
        vfe_steps=[], modularity=[], effective_rank=[],
        n_clusters=[], kl_within=[], total_vfe=[]
    )

    for step, beta in enumerate(beta_history):
        if beta.dim() == 4:
            beta = beta.mean(dim=1)  # Average heads if multi-head

        mod = compute_modularity(beta)
        rank = compute_effective_rank(beta)
        clusters = detect_clusters_spectral(beta)
        n_clust = int(clusters.max().item()) + 1

        trajectory.vfe_steps.append(step)
        trajectory.modularity.append(mod)
        trajectory.effective_rank.append(rank)
        trajectory.n_clusters.append(n_clust)
        trajectory.kl_within.append(0.0)  # Would need μ, Σ for this
        trajectory.total_vfe.append(0.0)

    return trajectory


def analyze_dynamic_rg(
    model,
    input_ids: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    device: str = 'cuda',
) -> DynamicRGTrajectory:
    """
    Track RG metrics across VFE iterations within a forward pass.

    Works with GaugeTransformerLM that has VariationalFFN layers.
    The key is that VariationalFFN can return beta_history when configured.

    Args:
        model: GaugeTransformerLM or similar with VariationalFFN
        input_ids: (B, N) input tokens
        targets: (B, N) target tokens (optional)
        device: Device

    Returns:
        DynamicRGTrajectory with metrics at each VFE step
    """
    model.eval()
    model.to(device)
    input_ids = input_ids.to(device)
    if targets is not None:
        targets = targets.to(device)

    trajectory = DynamicRGTrajectory(
        vfe_steps=[], modularity=[], effective_rank=[],
        n_clusters=[], kl_within=[], total_vfe=[]
    )

    # Try to get beta_history from the model
    # Different model architectures expose this differently

    with torch.no_grad():
        # Method 1: Check if model has transformer stack with FFN that returns history
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                if hasattr(block, 'ffn') and hasattr(block.ffn, 'n_iterations'):
                    n_iters = block.ffn.n_iterations
                    if n_iters > 1:
                        print(f"Found VariationalFFN with n_iterations={n_iters}")
                    break

        # Method 2: For PureFEPTransformer
        if hasattr(model, 'layers') and hasattr(model.layers[0], 'vfe_step'):
            layer = model.layers[0]
            mu_q, sigma_q, phi = model.embed(input_ids)
            mu_p = mu_q.clone()
            sigma_p = sigma_q.clone()

            B, N, K = mu_q.shape
            mask = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
            mask = mask.unsqueeze(0).expand(B, -1, -1)

            # Get n_iterations from config or default
            n_vfe_steps = getattr(model.config, 'n_vfe_steps', 10)

            for step in range(n_vfe_steps):
                mu_q, sigma_q, phi, metrics = layer.vfe_step(
                    mu_q, sigma_q, mu_p, sigma_p, phi,
                    targets=targets, mask=mask,
                    compute_metrics=True
                )

                beta = metrics.get('beta')
                if beta is None:
                    continue

                mod = compute_modularity(beta)
                rank = compute_effective_rank(beta)
                clusters = detect_clusters_spectral(beta)
                n_clust = int(clusters.max().item()) + 1

                trajectory.vfe_steps.append(step)
                trajectory.modularity.append(mod)
                trajectory.effective_rank.append(rank)
                trajectory.n_clusters.append(n_clust)
                trajectory.total_vfe.append(metrics.get('total_vfe', 0.0))

            return trajectory

        # Method 3: Standard forward, just get final attention
        # (only one point, not a trajectory)
        logits, attn_info = model.forward_with_attention(input_ids, targets)
        beta = attn_info['beta']
        if beta.dim() == 4:
            beta = beta.mean(dim=1)

        # For single-step models, we just have one point
        mod = compute_modularity(beta)
        rank = compute_effective_rank(beta)
        clusters = detect_clusters_spectral(beta)
        n_clust = int(clusters.max().item()) + 1

        trajectory.vfe_steps.append(0)
        trajectory.modularity.append(mod)
        trajectory.effective_rank.append(rank)
        trajectory.n_clusters.append(n_clust)

        print("Note: Model uses single VFE step. For dynamic RG, increase ffn_n_iterations.")

    return trajectory


def analyze_static_rg(
    model,
    dataloader,
    n_batches: int = 10,
    device: str = 'cuda',
) -> StaticRGSnapshot:
    """
    Analyze learned RG structure in a trained model (single VFE step).

    This examines the attention patterns that emerge from learned embeddings.
    Even with single VFE step, the TRAINED model encodes RG structure.

    Args:
        model: GaugeTransformerLM or similar
        dataloader: Data to analyze
        n_batches: Number of batches to average over
        device: Device

    Returns:
        StaticRGSnapshot with averaged metrics
    """
    model.eval()
    model.to(device)

    all_mod = []
    all_rank = []
    all_entropy = []
    all_clusters = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            else:
                input_ids = batch[0].to(device)

            # Forward with attention
            _, attn_info = model.forward_with_attention(input_ids)

            beta = attn_info['beta']
            if beta.dim() == 4:
                beta = beta.mean(dim=1)  # Average heads

            mod = compute_modularity(beta)
            rank = compute_effective_rank(beta)
            entropy = compute_beta_entropy(beta)
            clusters = detect_clusters_spectral(beta)
            n_clust = int(clusters.max().item()) + 1

            all_mod.append(mod)
            all_rank.append(rank)
            all_entropy.append(entropy)
            all_clusters.append(n_clust)

    # Compute random baseline modularity
    N = beta.shape[-1]
    random_beta = torch.softmax(torch.randn(1, N, N), dim=-1)
    random_mod = compute_modularity(random_beta)

    avg_mod = float(np.mean(all_mod))

    # Get cluster sizes from last batch
    unique, counts = torch.unique(clusters, return_counts=True)
    cluster_sizes = counts.tolist()

    return StaticRGSnapshot(
        training_step=-1,  # Unknown
        modularity=avg_mod,
        effective_rank=float(np.mean(all_rank)),
        n_clusters=int(np.mean(all_clusters)),
        cluster_sizes=cluster_sizes,
        attention_entropy=float(np.mean(all_entropy)),
        modularity_vs_random=avg_mod - random_mod,
    )


def compare_rg_modes() -> str:
    """
    Return explanation of dynamic vs static RG.
    """
    return """
╔═══════════════════════════════════════════════════════════════════════╗
║                    RG FLOW IN VFE TRANSFORMERS                        ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  DYNAMIC RG (multiple VFE steps, e.g., Pure FEP with n_steps=20)     ║
║  ─────────────────────────────────────────────────────────────────    ║
║                                                                       ║
║    Forward pass for ONE sequence:                                     ║
║                                                                       ║
║    Step 0    Step 5    Step 10   Step 15   Step 20                   ║
║    ┌───┐    ┌───┐     ┌───┐     ┌───┐     ┌───┐                      ║
║    │···│ →  │·:·│  →  │:::│  →  │█·█│  →  │███│                      ║
║    │···│    │·:·│     │:::│     │·█·│     │███│                      ║
║    │···│    │·:·│     │:::│     │█·█│     │███│                      ║
║    └───┘    └───┘     └───┘     └───┘     └───┘                      ║
║    random   weak      moderate  strong    converged                   ║
║             clusters  clusters  blocks    meta-agents                 ║
║                                                                       ║
║    Modularity: ────────────────────────────►  INCREASES              ║
║    Eff. Rank:  ◄────────────────────────────  DECREASES              ║
║    KL within:  ◄────────────────────────────  DECREASES              ║
║                                                                       ║
║    ✓ See RG flow WITHIN a single forward pass                        ║
║    ✓ Beliefs dynamically converge to form meta-agents                ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  STATIC RG (single VFE step, standard VFE_dynamic training)          ║
║  ─────────────────────────────────────────────────────────            ║
║                                                                       ║
║    Training over MANY steps:                                          ║
║                                                                       ║
║    Step 0     Step 1K    Step 10K   Step 50K   Step 100K             ║
║    ┌───┐     ┌───┐      ┌───┐      ┌───┐      ┌───┐                  ║
║    │···│  →  │···│   →  │·:·│   →  │:::│   →  │███│                  ║
║    │···│     │···│      │·:·│      │:::│      │███│                  ║
║    │···│     │···│      │·:·│      │:::│      │███│                  ║
║    └───┘     └───┘      └───┘      └───┘      └───┘                  ║
║    init      learning   structure  semantic   converged               ║
║    random    begins     emerging   clusters   embeddings              ║
║                                                                       ║
║    RG structure is LEARNED and STORED in embeddings (priors)         ║
║                                                                       ║
║    ✓ Trained model has RG structure in attention patterns            ║
║    ✗ Cannot see dynamic RG within single forward pass                ║
║    ✓ Analyze by looking at trained model's attention matrices        ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  BOTH ARE VALID RG!                                                   ║
║                                                                       ║
║  The coarse-graining happens at different timescales:                ║
║    • Dynamic: τ_inference (milliseconds, within forward pass)        ║
║    • Static:  τ_learning  (hours/days, across training)              ║
║                                                                       ║
║  The theoretical RG structure (meta-agents, scale invariance) can    ║
║  emerge in either case - just manifests differently.                 ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# Visualization
# =============================================================================

def plot_dynamic_vs_static(
    dynamic_traj: Optional[DynamicRGTrajectory],
    static_snap: Optional[StaticRGSnapshot],
    save_path: Optional[str] = None,
):
    """
    Plot comparison of dynamic and static RG analysis.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dynamic RG (left)
    ax = axes[0]
    if dynamic_traj and len(dynamic_traj.vfe_steps) > 0:
        ax.plot(dynamic_traj.vfe_steps, dynamic_traj.modularity, 'b-o', label='Modularity')
        ax.plot(dynamic_traj.vfe_steps,
                np.array(dynamic_traj.effective_rank) / max(dynamic_traj.effective_rank),
                'r-s', label='Eff. Rank (normalized)')
        ax.set_xlabel('VFE Step')
        ax.set_ylabel('Value')
        ax.set_title('DYNAMIC RG\n(within forward pass)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No dynamic data\n(requires n_vfe_steps > 1)',
                ha='center', va='center', fontsize=12)
        ax.set_title('DYNAMIC RG\n(not available)')

    # Static RG (right)
    ax = axes[1]
    if static_snap:
        metrics = ['Modularity', 'Mod vs Random', 'Eff. Rank\n(÷10)', 'N Clusters']
        values = [
            static_snap.modularity,
            static_snap.modularity_vs_random,
            static_snap.effective_rank / 10,
            static_snap.n_clusters / 10,
        ]
        colors = ['blue', 'green', 'red', 'purple']
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('STATIC RG\n(learned in trained model)')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No static data',
                ha='center', va='center', fontsize=12)
        ax.set_title('STATIC RG')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print(compare_rg_modes())

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
    Q: Does RG only emerge with multiple VFE steps?

    A: NO! RG structure can emerge in TWO ways:

       1. DYNAMIC (n_vfe_steps > 1):
          Clusters form within a single forward pass
          → Track modularity ACROSS VFE iterations

       2. STATIC (n_vfe_steps = 1):
          Structure learned over training
          → Analyze attention patterns of TRAINED model

       Both show the same theoretical RG signatures
       (modularity, rank reduction, KL coherence),
       just at different timescales!
    """)
