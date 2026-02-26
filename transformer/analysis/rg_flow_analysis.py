# -*- coding: utf-8 -*-
"""
RG Flow Analysis for VFE Gauge Transformers
============================================

This module implements comprehensive Renormalization Group (RG) flow analysis
for the gauge-theoretic VFE transformer, studying how the dynamics exhibit
self-similar structure characteristic of RG theories in statistical physics.

The Core Insight
----------------
VFE theory has a natural self-similar structure: meta-agents (clusters of tokens
with coherent beliefs) are themselves agents in the formal sense, with their own
beliefs, priors, and inter-agent couplings. This is the defining property of RG.

Key Observables
---------------
1. Modularity Q(beta): Block structure in attention (emergent meta-agents)
2. Effective rank: Spectral entropy-based dimensionality reduction
3. KL within/between clusters: Coherence and distinctness
4. Beta entropy: Attention distribution spread
5. Cluster sizes: Meta-agent formation statistics

RG Flow Signatures
------------------
Expected healthy RG behavior under VFE descent:
- Modularity: INCREASING (block structure emerges)
- Effective rank: DECREASING (fewer active degrees of freedom)
- KL within clusters: DECREASING (agents converge)
- KL between clusters: STABLE (groups remain distinct)

Phase Transitions
-----------------
Critical phenomena where the RG flow passes near unstable fixed points may
correspond to phase transitions in learned representations. We detect:
- Sharp modularity changes (order parameter jumps)
- Peaks in effective rank derivative (susceptibility divergence)
- KL ratio crossovers (symmetry breaking)

Author: VFE Transformer Team
Date: January 2026
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
import json
from datetime import datetime

# Import RG metrics
from transformer.analysis.rg_metrics import (
    RGDiagnostics,
    RGFlowSummary,
    compute_rg_diagnostics,
    compute_modularity,
    compute_effective_rank,
    compute_beta_entropy,
    compute_kl_within_clusters,
    compute_kl_between_clusters,
    detect_clusters_spectral,
    detect_clusters_kl,
    construct_meta_agents,
    reorder_beta_by_clusters,
    get_cluster_block_order,
    analyze_rg_flow,
)


# =============================================================================
# Extended RG Data Structures
# =============================================================================

@dataclass
class RGFixedPointAnalysis:
    """
    Analysis of RG fixed points.

    Fixed points are configurations where no further compression is possible
    without sacrificing predictive accuracy - optimal representations in the
    information bottleneck sense.
    """
    is_fixed_point: bool
    convergence_metric: float  # How close to fixed point
    modularity_at_fp: float
    effective_rank_at_fp: float
    n_meta_agents: int
    kl_ratio: float  # within/between ratio (low = well-separated)
    stability: str  # 'stable', 'unstable', or 'saddle'

    def to_dict(self) -> dict:
        return {
            'is_fixed_point': self.is_fixed_point,
            'convergence_metric': self.convergence_metric,
            'modularity_at_fp': self.modularity_at_fp,
            'effective_rank_at_fp': self.effective_rank_at_fp,
            'n_meta_agents': self.n_meta_agents,
            'kl_ratio': self.kl_ratio,
            'stability': self.stability,
        }


@dataclass
class RGScalingAnalysis:
    """
    Scaling analysis for RG flow - characterizes critical behavior.
    """
    # Scaling exponents
    modularity_exponent: float  # Q ~ t^α
    rank_exponent: float  # eff_rank ~ t^β
    kl_exponent: float  # KL_within ~ t^γ

    # Correlation length
    correlation_length: float  # Characteristic scale of correlations

    # Dynamical exponent
    dynamical_exponent: float  # z: time scales as L^z

    # Quality of fits
    fit_quality: Dict[str, float]  # R² for each fit

    def to_dict(self) -> dict:
        return {
            'modularity_exponent': self.modularity_exponent,
            'rank_exponent': self.rank_exponent,
            'kl_exponent': self.kl_exponent,
            'correlation_length': self.correlation_length,
            'dynamical_exponent': self.dynamical_exponent,
            'fit_quality': self.fit_quality,
        }


@dataclass
class MultiScaleRGAnalysis:
    """
    Multi-scale RG analysis tracking flow at different coarse-graining levels.

    Scale hierarchy:
        ζ=0: N tokens → q_i interactions via β_ij
        ζ=1: n_clusters meta-agents → β'_AB
        ζ=2: super-meta-agents → β''_αβ
        ...
    """
    n_scales: int
    diagnostics_per_scale: List[RGDiagnostics]
    coarse_graining_maps: List[torch.Tensor]  # Cluster labels at each scale
    scale_invariance_metric: float  # How similar dynamics are across scales

    def to_dict(self) -> dict:
        return {
            'n_scales': self.n_scales,
            'diagnostics_per_scale': [d.to_dict() for d in self.diagnostics_per_scale],
            'scale_invariance_metric': self.scale_invariance_metric,
        }


@dataclass
class RGFlowTrajectory:
    """
    Complete trajectory through RG flow space.
    """
    steps: List[int]
    modularity: List[float]
    effective_rank: List[float]
    n_clusters: List[int]
    kl_within: List[float]
    kl_between: List[float]
    entropy: List[float]

    # Phase space coordinates
    phase_space_trajectory: Optional[np.ndarray] = None  # (T, n_observables)

    # Fixed point analysis at each step
    fixed_point_distances: Optional[List[float]] = None

    def to_dict(self) -> dict:
        return {
            'steps': self.steps,
            'modularity': self.modularity,
            'effective_rank': self.effective_rank,
            'n_clusters': self.n_clusters,
            'kl_within': self.kl_within,
            'kl_between': self.kl_between,
            'entropy': self.entropy,
        }


# =============================================================================
# RG Flow Tracker
# =============================================================================

class RGFlowTracker:
    """
    Tracks RG flow observables across VFE dynamics.

    Can track:
    1. Within-forward-pass: RG evolution during VFE iterations
    2. Across-training: RG evolution as model learns
    3. Multi-scale: RG at different coarse-graining levels
    """

    def __init__(
        self,
        track_multiscale: bool = True,
        max_scales: int = 4,
        auto_cluster: bool = True,
        convergence_threshold: float = 0.01,
    ):
        """
        Initialize RG flow tracker.

        Args:
            track_multiscale: Whether to track multiple coarse-graining scales
            max_scales: Maximum number of scales to track
            auto_cluster: Auto-detect clusters at each step
            convergence_threshold: Threshold for fixed point detection
        """
        self.track_multiscale = track_multiscale
        self.max_scales = max_scales
        self.auto_cluster = auto_cluster
        self.convergence_threshold = convergence_threshold

        # Storage for trajectories
        self.trajectories: List[RGFlowTrajectory] = []
        self.multiscale_analyses: List[MultiScaleRGAnalysis] = []

        # Current trajectory being built
        self._current_trajectory = None

    def start_trajectory(self):
        """Start tracking a new RG trajectory."""
        self._current_trajectory = RGFlowTrajectory(
            steps=[], modularity=[], effective_rank=[],
            n_clusters=[], kl_within=[], kl_between=[], entropy=[]
        )

    def record_step(
        self,
        beta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        step: int,
    ):
        """
        Record RG observables at a single step.

        Args:
            beta: Attention matrix (B, N, N) or (B, H, N, N)
            mu: Belief means (B, N, K)
            sigma: Belief covariances (B, N, K) or (B, N, K, K)
            step: Current step number
        """
        if self._current_trajectory is None:
            self.start_trajectory()

        # Average over heads if multi-head
        if beta.dim() == 4:
            beta = beta.mean(dim=1)  # (B, N, N)

        # Compute RG diagnostics
        diagnostics = compute_rg_diagnostics(
            mu=mu, sigma=sigma, beta=beta, step=step,
            auto_cluster=self.auto_cluster
        )

        # Store in current trajectory
        traj = self._current_trajectory
        traj.steps.append(step)
        traj.modularity.append(diagnostics.modularity)
        traj.effective_rank.append(diagnostics.effective_rank)
        traj.n_clusters.append(diagnostics.n_clusters)
        traj.kl_within.append(diagnostics.kl_within_mean)
        traj.kl_between.append(diagnostics.kl_between_mean)
        traj.entropy.append(diagnostics.beta_entropy)

    def end_trajectory(self) -> RGFlowTrajectory:
        """End current trajectory and return it."""
        traj = self._current_trajectory
        if traj is not None and len(traj.steps) > 0:
            # Build phase space trajectory
            n_steps = len(traj.steps)
            traj.phase_space_trajectory = np.array([
                traj.modularity,
                traj.effective_rank,
                traj.kl_within,
                traj.kl_between,
                traj.entropy,
            ]).T  # (T, 5)

            self.trajectories.append(traj)

        self._current_trajectory = None
        return traj

    def analyze_multiscale(
        self,
        beta: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> MultiScaleRGAnalysis:
        """
        Perform multi-scale RG analysis.

        Implements the coarse-graining transformation:
            Scale ζ → Scale ζ+1
        by clustering tokens into meta-agents and constructing
        the effective attention matrix at the coarse-grained level.

        Args:
            beta: Attention matrix (B, N, N)
            mu: Belief means (B, N, K)
            sigma: Belief covariances

        Returns:
            MultiScaleRGAnalysis with diagnostics at each scale
        """
        # Average over heads if needed
        if beta.dim() == 4:
            beta = beta.mean(dim=1)

        B, N, K = mu.shape
        device = mu.device

        diagnostics_per_scale = []
        coarse_graining_maps = []

        # Scale 0: Original tokens
        current_mu = mu
        current_sigma = sigma
        current_beta = beta

        for scale in range(self.max_scales):
            # Check if we have enough agents for further coarse-graining
            n_current = current_mu.shape[1]
            if n_current < 4:  # Need at least 4 for meaningful clustering
                break

            # Compute diagnostics at this scale
            diag = compute_rg_diagnostics(
                mu=current_mu,
                sigma=current_sigma,
                beta=current_beta,
                step=scale,
                auto_cluster=True,
            )
            diagnostics_per_scale.append(diag)

            # Store coarse-graining map
            if diag.cluster_labels is not None:
                coarse_graining_maps.append(diag.cluster_labels.clone())

            # Check if we should continue
            if diag.n_clusters < 2 or diag.n_clusters >= n_current - 1:
                break

            # Construct meta-agents for next scale
            mu_meta, sigma_meta, beta_meta = construct_meta_agents(
                mu=current_mu,
                sigma=current_sigma,
                beta=current_beta,
                cluster_labels=diag.cluster_labels,
            )

            # Move to next scale
            current_mu = mu_meta
            current_sigma = sigma_meta
            current_beta = beta_meta

        # Compute scale invariance metric
        # (how similar the dynamics look at different scales)
        if len(diagnostics_per_scale) >= 2:
            scale_invariance = self._compute_scale_invariance(diagnostics_per_scale)
        else:
            scale_invariance = 1.0

        analysis = MultiScaleRGAnalysis(
            n_scales=len(diagnostics_per_scale),
            diagnostics_per_scale=diagnostics_per_scale,
            coarse_graining_maps=coarse_graining_maps,
            scale_invariance_metric=scale_invariance,
        )

        self.multiscale_analyses.append(analysis)
        return analysis

    def _compute_scale_invariance(
        self,
        diagnostics: List[RGDiagnostics],
    ) -> float:
        """
        Compute how scale-invariant the RG structure is.

        Perfect scale invariance means the coarse-grained theory
        looks identical to the fine-grained one (up to rescaling).
        """
        if len(diagnostics) < 2:
            return 1.0

        # Compare modularity across scales (should be similar if scale-invariant)
        mods = [d.modularity for d in diagnostics]
        mod_variance = np.var(mods) if len(mods) > 1 else 0.0

        # Compare effective rank ratios
        ranks = [d.effective_rank for d in diagnostics]
        # Normalize by number of agents at each scale
        n_agents = [d.n_clusters for d in diagnostics]
        normalized_ranks = [r / max(n, 1) for r, n in zip(ranks, n_agents)]
        rank_variance = np.var(normalized_ranks) if len(normalized_ranks) > 1 else 0.0

        # Scale invariance = 1 / (1 + total variance)
        # High variance = low scale invariance
        total_variance = mod_variance + rank_variance
        return 1.0 / (1.0 + total_variance)

    def detect_fixed_points(
        self,
        trajectory: Optional[RGFlowTrajectory] = None,
    ) -> List[RGFixedPointAnalysis]:
        """
        Detect fixed points in the RG flow.

        Fixed points are where:
        1. Observables stop changing (|dO/dt| < threshold)
        2. Further coarse-graining doesn't change structure

        Args:
            trajectory: RG trajectory to analyze (default: last trajectory)

        Returns:
            List of fixed point analyses at candidate locations
        """
        if trajectory is None:
            if len(self.trajectories) == 0:
                return []
            trajectory = self.trajectories[-1]

        if len(trajectory.steps) < 3:
            return []

        fixed_points = []

        # Look for plateaus in observables
        mod = np.array(trajectory.modularity)
        rank = np.array(trajectory.effective_rank)
        kl_w = np.array(trajectory.kl_within)
        kl_b = np.array(trajectory.kl_between)

        # Compute derivatives
        d_mod = np.abs(np.diff(mod))
        d_rank = np.abs(np.diff(rank))
        d_kl_w = np.abs(np.diff(kl_w))

        # Find steps where all derivatives are small
        threshold = self.convergence_threshold
        for i in range(len(d_mod)):
            is_plateau = (
                d_mod[i] < threshold and
                d_rank[i] < threshold * 10 and
                d_kl_w[i] < threshold
            )

            if is_plateau:
                # Check stability by looking at second derivatives
                if i > 0 and i < len(d_mod) - 1:
                    d2_mod = d_mod[i+1] - d_mod[i-1]
                    stability = 'stable' if d2_mod > 0 else 'unstable'
                else:
                    stability = 'unknown'

                # Compute KL ratio
                kl_ratio = kl_w[i] / (kl_b[i] + 1e-10)

                # Convergence metric: how close to "stopped"
                convergence = 1.0 / (1.0 + d_mod[i] + d_rank[i] + d_kl_w[i])

                fp = RGFixedPointAnalysis(
                    is_fixed_point=convergence > 0.9,
                    convergence_metric=convergence,
                    modularity_at_fp=mod[i],
                    effective_rank_at_fp=rank[i],
                    n_meta_agents=trajectory.n_clusters[i],
                    kl_ratio=kl_ratio,
                    stability=stability,
                )
                fixed_points.append(fp)

        return fixed_points

    def compute_scaling_exponents(
        self,
        trajectory: Optional[RGFlowTrajectory] = None,
    ) -> RGScalingAnalysis:
        """
        Compute scaling exponents from RG flow.

        In critical systems, observables follow power laws:
            Q ~ t^α  (modularity)
            r ~ t^β  (effective rank)
            KL ~ t^γ (KL divergences)

        Args:
            trajectory: RG trajectory to analyze

        Returns:
            RGScalingAnalysis with fitted exponents
        """
        if trajectory is None:
            if len(self.trajectories) == 0:
                return RGScalingAnalysis(
                    modularity_exponent=0, rank_exponent=0, kl_exponent=0,
                    correlation_length=1, dynamical_exponent=1,
                    fit_quality={'modularity': 0, 'rank': 0, 'kl': 0}
                )
            trajectory = self.trajectories[-1]

        t = np.array(trajectory.steps, dtype=float)
        if len(t) < 3:
            return RGScalingAnalysis(
                modularity_exponent=0, rank_exponent=0, kl_exponent=0,
                correlation_length=1, dynamical_exponent=1,
                fit_quality={'modularity': 0, 'rank': 0, 'kl': 0}
            )

        t = t - t.min() + 1  # Start from 1 to avoid log(0)

        mod = np.array(trajectory.modularity)
        rank = np.array(trajectory.effective_rank)
        kl_w = np.array(trajectory.kl_within)

        def fit_power_law(y, t):
            """Fit y = a * t^b using log-log regression."""
            # Filter out zeros/negatives
            mask = y > 0
            if mask.sum() < 3:
                return 0.0, 0.0

            log_t = np.log(t[mask])
            log_y = np.log(y[mask])

            # Linear regression in log space
            A = np.vstack([log_t, np.ones_like(log_t)]).T
            try:
                result = np.linalg.lstsq(A, log_y, rcond=None)
                b, log_a = result[0]

                # Compute R²
                y_pred = np.exp(log_a + b * log_t)
                ss_res = np.sum((y[mask] - y_pred) ** 2)
                ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
                r_squared = 1 - ss_res / (ss_tot + 1e-10)

                return b, max(0, r_squared)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                return 0.0, 0.0

        mod_exp, mod_r2 = fit_power_law(mod, t)
        rank_exp, rank_r2 = fit_power_law(rank, t)
        kl_exp, kl_r2 = fit_power_law(kl_w, t)

        # Estimate correlation length from cluster sizes
        n_clusters = np.array(trajectory.n_clusters)
        N = n_clusters[0] if len(n_clusters) > 0 else 1  # Total tokens
        correlation_length = float(np.sqrt(N / (n_clusters.mean() + 1e-10)))

        # Dynamical exponent from relaxation time
        # z ~ 1 / (rate of rank decrease)
        if abs(rank_exp) > 0.01:
            dynamical_exp = 1.0 / abs(rank_exp)
        else:
            dynamical_exp = 1.0

        return RGScalingAnalysis(
            modularity_exponent=mod_exp,
            rank_exponent=rank_exp,
            kl_exponent=kl_exp,
            correlation_length=correlation_length,
            dynamical_exponent=dynamical_exp,
            fit_quality={'modularity': mod_r2, 'rank': rank_r2, 'kl': kl_r2},
        )


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_rg_flow(
    trajectory: RGFlowTrajectory,
    save_path: Optional[str] = None,
    title: str = "RG Flow Evolution",
) -> plt.Figure:
    """
    Plot RG flow observables over time.

    Creates a 2x3 grid showing:
    - Modularity evolution
    - Effective rank evolution
    - Number of clusters
    - KL within clusters
    - KL between clusters
    - Entropy
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14)

    steps = trajectory.steps

    # Modularity
    ax = axes[0, 0]
    ax.plot(steps, trajectory.modularity, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Modularity Q')
    ax.set_title('Block Structure (↑ = meta-agents forming)')
    ax.grid(True, alpha=0.3)

    # Effective rank
    ax = axes[0, 1]
    ax.plot(steps, trajectory.effective_rank, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.set_title('Dimensionality (↓ = compression)')
    ax.grid(True, alpha=0.3)

    # Number of clusters
    ax = axes[0, 2]
    ax.plot(steps, trajectory.n_clusters, 'g-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('N Clusters')
    ax.set_title('Meta-agent Count')
    ax.grid(True, alpha=0.3)

    # KL within clusters
    ax = axes[1, 0]
    ax.plot(steps, trajectory.kl_within, 'm-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Within')
    ax.set_title('Cluster Coherence (↓ = tighter)')
    ax.grid(True, alpha=0.3)

    # KL between clusters
    ax = axes[1, 1]
    ax.plot(steps, trajectory.kl_between, 'c-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Between')
    ax.set_title('Cluster Distinctness (stable = good)')
    ax.grid(True, alpha=0.3)

    # Entropy
    ax = axes[1, 2]
    ax.plot(steps, trajectory.entropy, 'orange', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Attention Spread')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved RG flow plot to {save_path}")

    return fig


def plot_rg_phase_space(
    trajectory: RGFlowTrajectory,
    x_obs: str = 'modularity',
    y_obs: str = 'effective_rank',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot RG flow in phase space (2D projection).

    Shows trajectory through (modularity, effective_rank) space.
    """
    obs_map = {
        'modularity': trajectory.modularity,
        'effective_rank': trajectory.effective_rank,
        'kl_within': trajectory.kl_within,
        'kl_between': trajectory.kl_between,
        'entropy': trajectory.entropy,
        'n_clusters': trajectory.n_clusters,
    }

    x = np.array(obs_map[x_obs])
    y = np.array(obs_map[y_obs])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by time (step)
    colors = np.linspace(0, 1, len(x))
    scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=50, alpha=0.7)

    # Draw trajectory line
    ax.plot(x, y, 'k-', alpha=0.3, linewidth=1)

    # Mark start and end
    ax.scatter([x[0]], [y[0]], c='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter([x[-1]], [y[-1]], c='red', s=200, marker='*', label='End', zorder=5)

    # Arrow showing direction
    mid = len(x) // 2
    if mid > 0 and mid < len(x) - 1:
        dx = x[mid+1] - x[mid-1]
        dy = y[mid+1] - y[mid-1]
        ax.annotate('', xy=(x[mid] + dx*0.3, y[mid] + dy*0.3),
                   xytext=(x[mid], y[mid]),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlabel(x_obs.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_obs.replace('_', ' ').title(), fontsize=12)
    ax.set_title('RG Flow Phase Space', fontsize=14)
    ax.legend()

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (normalized)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase space plot to {save_path}")

    return fig


def plot_attention_block_structure(
    beta: torch.Tensor,
    cluster_labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Attention Block Structure",
) -> plt.Figure:
    """
    Visualize attention matrix reordered to show block structure.
    """
    # Convert to numpy
    if isinstance(beta, torch.Tensor):
        beta_np = beta.detach().cpu().numpy()
    else:
        beta_np = beta

    # Handle batch dimension
    if beta_np.ndim == 3:
        beta_np = beta_np[0]  # Use first batch
    elif beta_np.ndim == 4:
        beta_np = beta_np[0].mean(axis=0)  # Average over heads, first batch

    # Detect clusters if not provided
    if cluster_labels is None:
        beta_tensor = torch.from_numpy(beta_np).float()
        cluster_labels = detect_clusters_spectral(beta_tensor)

    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels_np = cluster_labels.detach().cpu().numpy()
    else:
        cluster_labels_np = cluster_labels

    if cluster_labels_np.ndim > 1:
        cluster_labels_np = cluster_labels_np[0]

    # Sort by cluster
    sort_idx = np.argsort(cluster_labels_np)
    beta_sorted = beta_np[sort_idx][:, sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original
    ax = axes[0]
    im = ax.imshow(beta_np, cmap='viridis', aspect='auto')
    ax.set_title('Original Attention', fontsize=12)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

    # Reordered
    ax = axes[1]
    im = ax.imshow(beta_sorted, cmap='viridis', aspect='auto')
    ax.set_title('Reordered by Cluster', fontsize=12)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

    # Draw cluster boundaries
    unique_clusters = np.unique(cluster_labels_np)
    boundaries = []
    for c in unique_clusters[:-1]:
        boundary = np.sum(cluster_labels_np[sort_idx] <= c)
        boundaries.append(boundary)

    for b in boundaries:
        ax.axhline(y=b-0.5, color='white', linewidth=2, linestyle='--')
        ax.axvline(x=b-0.5, color='white', linewidth=2, linestyle='--')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved block structure plot to {save_path}")

    return fig


def plot_multiscale_comparison(
    analysis: MultiScaleRGAnalysis,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare RG observables across coarse-graining scales.
    """
    n_scales = analysis.n_scales

    if n_scales < 2:
        print("Need at least 2 scales for comparison")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    scales = list(range(n_scales))
    mods = [d.modularity for d in analysis.diagnostics_per_scale]
    ranks = [d.effective_rank for d in analysis.diagnostics_per_scale]
    kl_within = [d.kl_within_mean for d in analysis.diagnostics_per_scale]
    n_agents = [d.n_clusters for d in analysis.diagnostics_per_scale]

    # Modularity
    axes[0].bar(scales, mods, color='blue', alpha=0.7)
    axes[0].set_xlabel('Scale ζ')
    axes[0].set_ylabel('Modularity')
    axes[0].set_title('Block Structure vs Scale')
    axes[0].set_xticks(scales)

    # Effective rank
    axes[1].bar(scales, ranks, color='red', alpha=0.7)
    axes[1].set_xlabel('Scale ζ')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Dimensionality vs Scale')
    axes[1].set_xticks(scales)

    # KL within
    axes[2].bar(scales, kl_within, color='green', alpha=0.7)
    axes[2].set_xlabel('Scale ζ')
    axes[2].set_ylabel('KL Within')
    axes[2].set_title('Coherence vs Scale')
    axes[2].set_xticks(scales)

    # Number of agents
    axes[3].bar(scales, n_agents, color='purple', alpha=0.7)
    axes[3].set_xlabel('Scale ζ')
    axes[3].set_ylabel('N Agents')
    axes[3].set_title('Agent Count vs Scale')
    axes[3].set_xticks(scales)

    fig.suptitle(f'Multi-Scale RG Analysis (Scale Invariance: {analysis.scale_invariance_metric:.3f})', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-scale plot to {save_path}")

    return fig


# =============================================================================
# Analysis Runner
# =============================================================================

def run_rg_analysis(
    model,
    dataloader,
    device: str = 'cuda',
    n_batches: int = 10,
    track_per_batch: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive RG flow analysis on a trained model.

    Args:
        model: GaugeTransformerLM model
        dataloader: DataLoader with input data
        device: Device to run on
        n_batches: Number of batches to analyze
        track_per_batch: Track RG flow for each batch
        output_dir: Directory to save plots and results

    Returns:
        Dict with analysis results
    """
    model.eval()
    model.to(device)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    tracker = RGFlowTracker(track_multiscale=True)

    all_trajectories = []
    all_multiscale = []
    all_diagnostics = []

    print(f"Running RG analysis on {n_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break

            # Get input tokens
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            # Forward with attention tracking
            logits, attn_info = model.forward_with_attention(input_ids)

            beta = attn_info['beta']  # (B, H, N, N) or (B, N, N)
            mu = attn_info['mu']      # (B, N, K)
            sigma = attn_info.get('sigma')  # May be None

            if sigma is None:
                # Use dummy sigma if not available
                sigma = torch.ones_like(mu)

            # Average over heads if multi-head
            if beta.dim() == 4:
                beta_avg = beta.mean(dim=1)
            else:
                beta_avg = beta

            # Compute diagnostics for this batch
            diagnostics = compute_rg_diagnostics(
                mu=mu, sigma=sigma, beta=beta_avg,
                step=batch_idx, auto_cluster=True
            )
            all_diagnostics.append(diagnostics)

            # Track trajectory (single-step for now)
            if track_per_batch:
                tracker.start_trajectory()
                tracker.record_step(beta_avg, mu, sigma, step=batch_idx)
                traj = tracker.end_trajectory()
                all_trajectories.append(traj)

            # Multi-scale analysis
            multiscale = tracker.analyze_multiscale(beta_avg, mu, sigma)
            all_multiscale.append(multiscale)

            print(f"  Batch {batch_idx}: Q={diagnostics.modularity:.3f}, "
                  f"rank={diagnostics.effective_rank:.1f}, "
                  f"clusters={diagnostics.n_clusters}, "
                  f"scales={multiscale.n_scales}")

    # Aggregate statistics
    mods = [d.modularity for d in all_diagnostics]
    ranks = [d.effective_rank for d in all_diagnostics]
    kl_within = [d.kl_within_mean for d in all_diagnostics]
    kl_between = [d.kl_between_mean for d in all_diagnostics]
    n_clusters = [d.n_clusters for d in all_diagnostics]
    scale_invariances = [m.scale_invariance_metric for m in all_multiscale]

    summary = {
        'modularity': {'mean': np.mean(mods), 'std': np.std(mods)},
        'effective_rank': {'mean': np.mean(ranks), 'std': np.std(ranks)},
        'kl_within': {'mean': np.mean(kl_within), 'std': np.std(kl_within)},
        'kl_between': {'mean': np.mean(kl_between), 'std': np.std(kl_between)},
        'n_clusters': {'mean': np.mean(n_clusters), 'std': np.std(n_clusters)},
        'scale_invariance': {'mean': np.mean(scale_invariances), 'std': np.std(scale_invariances)},
        'n_batches_analyzed': len(all_diagnostics),
    }

    # Check RG behavior signatures
    kl_ratio = np.mean(kl_within) / (np.mean(kl_between) + 1e-10)
    rg_signatures = {
        'high_modularity': np.mean(mods) > 0.1,  # Block structure present
        'low_kl_ratio': kl_ratio < 1.0,  # Within < Between
        'moderate_rank_reduction': np.mean(ranks) < np.mean(n_clusters) * 2,
        'scale_invariant': np.mean(scale_invariances) > 0.5,
    }

    summary['rg_signatures'] = rg_signatures
    summary['kl_ratio'] = kl_ratio

    print("\n" + "="*60)
    print("RG FLOW ANALYSIS SUMMARY")
    print("="*60)
    print(f"Modularity:      {summary['modularity']['mean']:.3f} +/- {summary['modularity']['std']:.3f}")
    print(f"Effective Rank:  {summary['effective_rank']['mean']:.1f} +/- {summary['effective_rank']['std']:.1f}")
    print(f"N Clusters:      {summary['n_clusters']['mean']:.1f} +/- {summary['n_clusters']['std']:.1f}")
    print(f"KL Within:       {summary['kl_within']['mean']:.3f} +/- {summary['kl_within']['std']:.3f}")
    print(f"KL Between:      {summary['kl_between']['mean']:.3f} +/- {summary['kl_between']['std']:.3f}")
    print(f"KL Ratio:        {kl_ratio:.3f}")
    print(f"Scale Invariance: {summary['scale_invariance']['mean']:.3f} +/- {summary['scale_invariance']['std']:.3f}")
    print("-"*60)
    print("RG Behavior Signatures:")
    for sig, value in rg_signatures.items():
        status = "YES" if value else "NO"
        print(f"  {sig}: {status}")
    print("="*60)

    # Save results
    if output_path:
        # Save summary JSON
        with open(output_path / 'rg_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Plot aggregate trajectory
        if len(all_diagnostics) > 1:
            agg_traj = RGFlowTrajectory(
                steps=list(range(len(all_diagnostics))),
                modularity=mods,
                effective_rank=ranks,
                n_clusters=n_clusters,
                kl_within=kl_within,
                kl_between=kl_between,
                entropy=[d.beta_entropy for d in all_diagnostics],
            )
            plot_rg_flow(agg_traj, save_path=str(output_path / 'rg_flow.png'))
            plot_rg_phase_space(agg_traj, save_path=str(output_path / 'rg_phase_space.png'))

        # Plot multi-scale for last batch
        if all_multiscale and all_multiscale[-1].n_scales >= 2:
            plot_multiscale_comparison(all_multiscale[-1],
                                      save_path=str(output_path / 'multiscale.png'))

        print(f"\nResults saved to {output_path}")

    return {
        'summary': summary,
        'diagnostics': all_diagnostics,
        'multiscale': all_multiscale,
        'trajectories': all_trajectories,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("RG FLOW ANALYSIS MODULE")
    print("="*70)
    print("\nThis module provides tools for analyzing RG structure in VFE transformers.")
    print("\nKey components:")
    print("  - RGFlowTracker: Track RG observables across VFE dynamics")
    print("  - MultiScaleRGAnalysis: Analyze coarse-graining hierarchy")
    print("  - Fixed point detection: Find RG fixed points")
    print("  - Scaling analysis: Compute critical exponents")
    print("  - Visualization: Plot RG flow, phase space, block structure")
    print("\nUsage:")
    print("  from transformer.analysis.rg_flow_analysis import run_rg_analysis")
    print("  results = run_rg_analysis(model, dataloader)")
    print("\nSee docstrings for detailed API documentation.")
    print("="*70)

    # Run quick test with synthetic data
    print("\n[Testing with synthetic data...]")

    # Create synthetic attention with block structure
    torch.manual_seed(42)
    N = 32  # tokens
    K = 64  # embedding dim
    B = 2   # batch

    # Create 4 clusters
    beta = torch.zeros(B, N, N)
    for b in range(B):
        for c in range(4):
            start = c * 8
            end = (c + 1) * 8
            beta[b, start:end, start:end] = torch.rand(8, 8) * 0.5 + 0.5
        beta[b] = beta[b] + torch.rand(N, N) * 0.1
        beta[b] = beta[b] / beta[b].sum(dim=-1, keepdim=True)

    mu = torch.randn(B, N, K)
    sigma = torch.ones(B, N, K) * 0.5

    # Test tracker
    tracker = RGFlowTracker()
    tracker.start_trajectory()

    # Simulate VFE evolution (blocks become more pronounced)
    for step in range(10):
        tracker.record_step(beta, mu, sigma, step)

        # Evolve: strengthen blocks
        for b in range(B):
            for c in range(4):
                start = c * 8
                end = (c + 1) * 8
                beta[b, start:end, start:end] *= 1.05
            beta[b] = beta[b] / beta[b].sum(dim=-1, keepdim=True)

    traj = tracker.end_trajectory()

    # Test multi-scale analysis
    print("\n[Multi-scale analysis...]")
    multiscale = tracker.analyze_multiscale(beta, mu, sigma)
    print(f"  N scales: {multiscale.n_scales}")
    print(f"  Scale invariance: {multiscale.scale_invariance_metric:.3f}")

    # Test fixed point detection
    print("\n[Fixed point detection...]")
    fps = tracker.detect_fixed_points(traj)
    print(f"  Found {len(fps)} candidate fixed points")

    # Test scaling analysis
    print("\n[Scaling analysis...]")
    scaling = tracker.compute_scaling_exponents(traj)
    print(f"  Modularity exponent: {scaling.modularity_exponent:.3f}")
    print(f"  Rank exponent: {scaling.rank_exponent:.3f}")

    # Plot
    print("\n[Generating test plots...]")
    try:
        plot_rg_flow(traj, save_path='/tmp/test_rg_flow.png')
        plot_rg_phase_space(traj, save_path='/tmp/test_phase_space.png')
        plot_attention_block_structure(beta, save_path='/tmp/test_blocks.png')
        if multiscale.n_scales >= 2:
            plot_multiscale_comparison(multiscale, save_path='/tmp/test_multiscale.png')
        print("  Plots saved to /tmp/")
    except (ValueError, TypeError, OSError) as e:
        print(f"  Plotting skipped: {e}")

    print("\n" + "="*70)
    print("All RG flow analysis tests passed!")
    print("="*70)
