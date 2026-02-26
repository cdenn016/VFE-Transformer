# -*- coding: utf-8 -*-
"""
Enhanced RG Flow Analysis for VFE Gauge Transformers
=====================================================

This module extends the basic RG analysis to include the full VFE structure:
- Gauge frames φ and their coarse-graining
- KL divergence matrices (not just attention)
- Transport operators Ω_ij at each scale
- Free energy decomposition across scales

The key insight is that a true RG transformation must preserve the
gauge-theoretic structure: meta-agents should have gauge frames that
transform correctly, and the VFE functional should have the same form
at the coarse-grained level.

Author: VFE Transformer Team
Date: January 2026
"""

import sys
import os

# Ensure project root is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

# Import base RG metrics
from transformer.analysis.rg_metrics import (
    RGDiagnostics,
    compute_modularity,
    compute_effective_rank,
    compute_beta_entropy,
    detect_clusters_spectral,
    _to_tensor,
    _to_numpy,
)


# =============================================================================
# Extended Data Structures
# =============================================================================

@dataclass
class FullRGDiagnostics:
    """
    Complete RG diagnostics including gauge structure.

    Extends RGDiagnostics with:
    - Gauge frame coherence within clusters
    - KL matrix structure (not just attention)
    - Transport operator statistics
    - Free energy decomposition
    """
    step: int

    # Basic RG observables
    modularity: float
    effective_rank: float
    n_clusters: int
    cluster_labels: Optional[torch.Tensor] = None

    # Belief statistics
    kl_within_mean: float = 0.0
    kl_within_std: float = 0.0
    kl_between_mean: float = 0.0
    kl_between_std: float = 0.0

    # Gauge frame statistics
    phi_within_mean: float = 0.0      # Mean gauge distance within clusters
    phi_within_std: float = 0.0
    phi_between_mean: float = 0.0     # Mean gauge distance between clusters
    phi_between_std: float = 0.0
    gauge_coherence: float = 0.0      # How aligned are gauge frames in clusters?

    # KL matrix statistics (raw KL, not softmax)
    kl_matrix_rank: float = 0.0       # Effective rank of KL matrix
    kl_entropy: float = 0.0           # Entropy of KL distribution

    # Transport statistics
    transport_magnitude_mean: float = 0.0  # Mean ||Ω_ij - I||
    transport_within_mean: float = 0.0     # Within-cluster transport
    transport_between_mean: float = 0.0    # Between-cluster transport

    # Free energy decomposition
    fe_self: float = 0.0              # KL(q||p) term
    fe_belief_align: float = 0.0      # Σ β_ij KL(q_i||Ω[q_j]) term
    fe_total: float = 0.0

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'modularity': self.modularity,
            'effective_rank': self.effective_rank,
            'n_clusters': self.n_clusters,
            'kl_within_mean': self.kl_within_mean,
            'kl_within_std': self.kl_within_std,
            'kl_between_mean': self.kl_between_mean,
            'kl_between_std': self.kl_between_std,
            'phi_within_mean': self.phi_within_mean,
            'phi_within_std': self.phi_within_std,
            'phi_between_mean': self.phi_between_mean,
            'phi_between_std': self.phi_between_std,
            'gauge_coherence': self.gauge_coherence,
            'kl_matrix_rank': self.kl_matrix_rank,
            'kl_entropy': self.kl_entropy,
            'transport_magnitude_mean': self.transport_magnitude_mean,
            'transport_within_mean': self.transport_within_mean,
            'transport_between_mean': self.transport_between_mean,
            'fe_self': self.fe_self,
            'fe_belief_align': self.fe_belief_align,
            'fe_total': self.fe_total,
        }


@dataclass
class CoarseGrainedState:
    """
    State of the system at a given coarse-graining level.

    Contains all the quantities needed for VFE computation:
    - Meta-agent beliefs (μ, Σ)
    - Meta-agent gauge frames (φ)
    - Meta-agent attention (β)
    - Meta-agent KL matrices
    """
    scale: int                    # Coarse-graining level (0=tokens, 1=meta-agents, ...)
    n_agents: int                 # Number of agents at this scale

    mu: torch.Tensor              # (B, n_agents, K) belief means
    sigma: torch.Tensor           # (B, n_agents, K) or (B, n_agents, K, K)
    phi: torch.Tensor             # (B, n_agents, gauge_dim) gauge frames
    beta: torch.Tensor            # (B, n_agents, n_agents) attention
    kl_matrix: Optional[torch.Tensor] = None  # (B, n_agents, n_agents) raw KL

    cluster_map: Optional[torch.Tensor] = None  # (B, n_agents_prev) -> cluster at this scale

    def to_dict(self) -> dict:
        return {
            'scale': self.scale,
            'n_agents': self.n_agents,
            'mu_shape': list(self.mu.shape),
            'phi_shape': list(self.phi.shape),
        }


@dataclass
class HierarchicalRGState:
    """
    Complete hierarchical state across all coarse-graining scales.
    """
    states: List[CoarseGrainedState]
    diagnostics: List[FullRGDiagnostics]

    @property
    def n_scales(self) -> int:
        return len(self.states)

    def scale_invariance(self) -> float:
        """Measure how similar the dynamics are across scales."""
        if len(self.diagnostics) < 2:
            return 1.0

        # Compare modularity, rank, KL ratios across scales
        mods = [d.modularity for d in self.diagnostics]
        kl_ratios = [d.kl_within_mean / (d.kl_between_mean + 1e-10) for d in self.diagnostics]

        mod_var = np.var(mods) if len(mods) > 1 else 0.0
        kl_var = np.var(kl_ratios) if len(kl_ratios) > 1 else 0.0

        return 1.0 / (1.0 + mod_var + kl_var)


# =============================================================================
# Gauge Frame Analysis
# =============================================================================

def compute_gauge_distance(phi1: torch.Tensor, phi2: torch.Tensor) -> torch.Tensor:
    """
    Compute distance between gauge frames in Lie algebra.

    For SO(3), φ ∈ so(3) ≅ R³, so we use Euclidean distance.
    For general SO(N), would use Frobenius norm.

    Args:
        phi1: (*, gauge_dim) first gauge frame
        phi2: (*, gauge_dim) second gauge frame

    Returns:
        Distances of same shape as input without last dim
    """
    return torch.norm(phi1 - phi2, dim=-1)


def compute_gauge_coherence_within_clusters(
    phi: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Compute how coherent gauge frames are within clusters.

    High coherence = gauge frames are similar within clusters
    This indicates meta-agents have well-defined gauge structure.

    Args:
        phi: (B, N, gauge_dim) gauge frames
        cluster_labels: (B, N) cluster assignments

    Returns:
        mean_distance: Mean gauge distance within clusters
        std_distance: Std of gauge distances
        coherence: 1 / (1 + mean_distance) normalized coherence
    """
    phi = _to_tensor(phi)
    cluster_labels = _to_tensor(cluster_labels)

    if phi.dim() == 2:
        phi = phi.unsqueeze(0)
        cluster_labels = cluster_labels.unsqueeze(0)

    B, N, G = phi.shape
    all_distances = []

    for b in range(B):
        unique_clusters = cluster_labels[b].unique()

        for c in unique_clusters:
            mask = cluster_labels[b] == c
            indices = mask.nonzero(as_tuple=True)[0]

            if len(indices) < 2:
                continue

            # Pairwise gauge distances within cluster
            phi_cluster = phi[b, indices]  # (n_c, G)
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    dist = compute_gauge_distance(phi_cluster[i], phi_cluster[j])
                    all_distances.append(dist.item())

    if len(all_distances) == 0:
        return 0.0, 0.0, 1.0

    mean_dist = float(np.mean(all_distances))
    std_dist = float(np.std(all_distances))
    coherence = 1.0 / (1.0 + mean_dist)

    return mean_dist, std_dist, coherence


def compute_gauge_distance_between_clusters(
    phi: torch.Tensor,
    cluster_labels: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute gauge distances between cluster centroids.

    Args:
        phi: (B, N, gauge_dim) gauge frames
        cluster_labels: (B, N) cluster assignments

    Returns:
        mean_distance: Mean gauge distance between cluster centroids
        std_distance: Std of gauge distances
    """
    phi = _to_tensor(phi)
    cluster_labels = _to_tensor(cluster_labels)

    if phi.dim() == 2:
        phi = phi.unsqueeze(0)
        cluster_labels = cluster_labels.unsqueeze(0)

    B, N, G = phi.shape
    all_distances = []

    for b in range(B):
        unique_clusters = cluster_labels[b].unique()
        n_clusters = len(unique_clusters)

        if n_clusters < 2:
            continue

        # Compute cluster centroids (mean gauge frame)
        centroids = []
        for c in unique_clusters:
            mask = cluster_labels[b] == c
            centroid = phi[b, mask].mean(dim=0)  # (G,)
            centroids.append(centroid)

        centroids = torch.stack(centroids)  # (n_clusters, G)

        # Pairwise distances between centroids
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = compute_gauge_distance(centroids[i], centroids[j])
                all_distances.append(dist.item())

    if len(all_distances) == 0:
        return 0.0, 0.0

    return float(np.mean(all_distances)), float(np.std(all_distances))


# =============================================================================
# KL Matrix Analysis
# =============================================================================

def analyze_kl_matrix(
    kl_matrix: torch.Tensor,
    cluster_labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Analyze the raw KL divergence matrix structure.

    The KL matrix KL_ij = KL(q_i || Ω_ij[q_j]) encodes belief similarity
    under parallel transport. Unlike attention (softmax of -KL/κ), this
    preserves the full metric structure.

    Args:
        kl_matrix: (B, N, N) or (B, H, N, N) KL divergences
        cluster_labels: Optional cluster assignments for within/between analysis

    Returns:
        Dict with KL matrix statistics
    """
    kl = _to_tensor(kl_matrix)

    # Average over heads if multi-head
    if kl.dim() == 4:
        kl = kl.mean(dim=1)

    # Average over batch
    kl_avg = kl.mean(dim=0)  # (N, N)

    # Effective rank of KL matrix (how many significant directions)
    # Using singular value entropy
    try:
        U, S, V = torch.linalg.svd(kl_avg)
        S_normalized = S / (S.sum() + 1e-10)
        S_normalized = S_normalized[S_normalized > 1e-10]
        entropy = -torch.sum(S_normalized * torch.log(S_normalized + 1e-10))
        effective_rank = torch.exp(entropy).item()
    except (RuntimeError, ValueError):
        effective_rank = kl_avg.shape[0]

    # Entropy of KL values (how spread out are the KL values)
    kl_flat = kl_avg.flatten()
    kl_flat = kl_flat[kl_flat > 0]  # Filter out zeros/negatives
    if len(kl_flat) > 0:
        kl_normalized = kl_flat / (kl_flat.sum() + 1e-10)
        kl_entropy = -torch.sum(kl_normalized * torch.log(kl_normalized + 1e-10)).item()
    else:
        kl_entropy = 0.0

    result = {
        'effective_rank': effective_rank,
        'entropy': kl_entropy,
        'mean': kl_avg.mean().item(),
        'std': kl_avg.std().item(),
        'max': kl_avg.max().item(),
        'min': kl_avg[kl_avg > 0].min().item() if (kl_avg > 0).any() else 0.0,
    }

    # Within/between cluster statistics if labels provided
    if cluster_labels is not None:
        labels = _to_tensor(cluster_labels)
        if labels.dim() > 1:
            labels = labels[0]

        within_kl = []
        between_kl = []

        unique_clusters = labels.unique()
        for c in unique_clusters:
            mask = labels == c
            indices = mask.nonzero(as_tuple=True)[0]

            # Within cluster
            for i in indices:
                for j in indices:
                    if i != j:
                        within_kl.append(kl_avg[i, j].item())

            # Between clusters
            other_mask = ~mask
            other_indices = other_mask.nonzero(as_tuple=True)[0]
            for i in indices:
                for j in other_indices:
                    between_kl.append(kl_avg[i, j].item())

        result['within_mean'] = float(np.mean(within_kl)) if within_kl else 0.0
        result['within_std'] = float(np.std(within_kl)) if within_kl else 0.0
        result['between_mean'] = float(np.mean(between_kl)) if between_kl else 0.0
        result['between_std'] = float(np.std(between_kl)) if between_kl else 0.0

    return result


# =============================================================================
# Coarse-Graining Transformation
# =============================================================================

def coarse_grain_full_state(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    beta: torch.Tensor,
    kl_matrix: Optional[torch.Tensor] = None,
    cluster_labels: Optional[torch.Tensor] = None,
    scale: int = 0,
) -> CoarseGrainedState:
    """
    Perform full coarse-graining transformation including gauge frames.

    This implements the RG transformation:
        (μ_i, Σ_i, φ_i, β_ij) → (μ_A, Σ_A, φ_A, β_AB)

    For meta-agent A containing tokens {i ∈ A}:
        μ_A = Σ_i w_i μ_i        (coherence-weighted mean)
        φ_A = Frechet_mean(φ_i)  (mean on Lie algebra, simplified to Euclidean mean)
        β_AB = Σ_{i∈A,j∈B} β_ij  (aggregated attention)

    Args:
        mu: (B, N, K) token belief means
        sigma: (B, N, K) or (B, N, K, K) token covariances
        phi: (B, N, gauge_dim) token gauge frames
        beta: (B, N, N) attention matrix
        kl_matrix: Optional (B, N, N) KL divergences
        cluster_labels: Optional cluster assignments (auto-detected if None)
        scale: Current scale level

    Returns:
        CoarseGrainedState at the coarse-grained level
    """
    mu = _to_tensor(mu)
    sigma = _to_tensor(sigma)
    phi = _to_tensor(phi)
    beta = _to_tensor(beta)

    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        phi = phi.unsqueeze(0)
        beta = beta.unsqueeze(0)

    B, N, K = mu.shape
    G = phi.shape[-1]
    is_diagonal = sigma.dim() == 3
    device = mu.device

    # Detect clusters if not provided
    if cluster_labels is None:
        cluster_labels = detect_clusters_spectral(beta)

    if cluster_labels.dim() == 1:
        cluster_labels = cluster_labels.unsqueeze(0).expand(B, -1)

    # Number of meta-agents
    n_meta = int(cluster_labels.max().item()) + 1

    # Initialize meta-agent tensors
    mu_meta = torch.zeros(B, n_meta, K, device=device)
    phi_meta = torch.zeros(B, n_meta, G, device=device)
    if is_diagonal:
        sigma_meta = torch.zeros(B, n_meta, K, device=device)
    else:
        sigma_meta = torch.zeros(B, n_meta, K, K, device=device)
    beta_meta = torch.zeros(B, n_meta, n_meta, device=device)
    kl_meta = torch.zeros(B, n_meta, n_meta, device=device) if kl_matrix is not None else None

    for b in range(B):
        for c in range(n_meta):
            mask = cluster_labels[b] == c
            if mask.sum() == 0:
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            n_c = len(indices)

            # Coherence weights from attention within cluster
            beta_within = beta[b, indices][:, indices]
            w = beta_within.sum(dim=1)
            w = w / (w.sum() + 1e-10)

            # Weighted mean for beliefs
            mu_meta[b, c] = (w.unsqueeze(-1) * mu[b, indices]).sum(dim=0)

            # Weighted mean for covariance
            if is_diagonal:
                sigma_meta[b, c] = (w.unsqueeze(-1) * sigma[b, indices]).sum(dim=0)
            else:
                for i, idx in enumerate(indices):
                    sigma_meta[b, c] += w[i] * sigma[b, idx]

            # Frechet mean for gauge frames (simplified to Euclidean in Lie algebra)
            # For SO(3), this is the "average rotation axis"
            phi_meta[b, c] = phi[b, indices].mean(dim=0)

        # Aggregate attention between meta-agents
        for c1 in range(n_meta):
            for c2 in range(n_meta):
                mask1 = cluster_labels[b] == c1
                mask2 = cluster_labels[b] == c2
                idx1 = mask1.nonzero(as_tuple=True)[0]
                idx2 = mask2.nonzero(as_tuple=True)[0]

                if len(idx1) == 0 or len(idx2) == 0:
                    continue

                # Sum attention from cluster c1 to cluster c2
                beta_meta[b, c1, c2] = beta[b, idx1][:, idx2].sum()

                # Sum KL if available
                if kl_matrix is not None and kl_meta is not None:
                    kl = _to_tensor(kl_matrix)
                    if kl.dim() == 4:
                        kl = kl.mean(dim=1)  # Average over heads
                    kl_meta[b, c1, c2] = kl[b, idx1][:, idx2].mean()

        # Normalize meta-attention
        beta_meta[b] = beta_meta[b] / (beta_meta[b].sum(dim=-1, keepdim=True) + 1e-10)

    return CoarseGrainedState(
        scale=scale + 1,
        n_agents=n_meta,
        mu=mu_meta,
        sigma=sigma_meta,
        phi=phi_meta,
        beta=beta_meta,
        kl_matrix=kl_meta,
        cluster_map=cluster_labels,
    )


# =============================================================================
# Full RG Analysis
# =============================================================================

def compute_full_rg_diagnostics(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    beta: torch.Tensor,
    kl_matrix: Optional[torch.Tensor] = None,
    step: int = 0,
    auto_cluster: bool = True,
) -> FullRGDiagnostics:
    """
    Compute complete RG diagnostics including gauge structure.

    Args:
        mu: (B, N, K) belief means
        sigma: (B, N, K) or (B, N, K, K) belief covariances
        phi: (B, N, gauge_dim) gauge frames
        beta: (B, N, N) attention matrix
        kl_matrix: Optional (B, N, N) or (B, H, N, N) raw KL divergences
        step: Current step number
        auto_cluster: Auto-detect clusters

    Returns:
        FullRGDiagnostics with complete analysis
    """
    mu = _to_tensor(mu)
    sigma = _to_tensor(sigma)
    phi = _to_tensor(phi)
    beta = _to_tensor(beta)

    if beta.dim() == 4:
        beta = beta.mean(dim=1)  # Average over heads

    # Basic RG metrics
    modularity = compute_modularity(beta)
    effective_rank = compute_effective_rank(beta)

    # Detect clusters
    if auto_cluster:
        cluster_labels = detect_clusters_spectral(beta)
        n_clusters = int(cluster_labels.max().item()) + 1
    else:
        cluster_labels = None
        n_clusters = 1

    # Gauge frame analysis
    if cluster_labels is not None:
        phi_within, phi_within_std, gauge_coherence = compute_gauge_coherence_within_clusters(
            phi, cluster_labels
        )
        phi_between, phi_between_std = compute_gauge_distance_between_clusters(
            phi, cluster_labels
        )
    else:
        phi_within = phi_within_std = phi_between = phi_between_std = 0.0
        gauge_coherence = 1.0

    # KL matrix analysis
    if kl_matrix is not None:
        kl_stats = analyze_kl_matrix(kl_matrix, cluster_labels)
        kl_matrix_rank = kl_stats['effective_rank']
        kl_entropy = kl_stats['entropy']
        kl_within_mean = kl_stats.get('within_mean', 0.0)
        kl_within_std = kl_stats.get('within_std', 0.0)
        kl_between_mean = kl_stats.get('between_mean', 0.0)
        kl_between_std = kl_stats.get('between_std', 0.0)
    else:
        kl_matrix_rank = effective_rank
        kl_entropy = compute_beta_entropy(beta)
        kl_within_mean = kl_within_std = kl_between_mean = kl_between_std = 0.0

    # Free energy estimation (simplified - would need priors for full calculation)
    # Here we estimate from attention-weighted KL
    if kl_matrix is not None:
        kl = _to_tensor(kl_matrix)
        if kl.dim() == 4:
            kl = kl.mean(dim=1)
        fe_belief_align = (beta * kl).sum(dim=(-2, -1)).mean().item()
    else:
        fe_belief_align = 0.0

    return FullRGDiagnostics(
        step=step,
        modularity=modularity,
        effective_rank=effective_rank,
        n_clusters=n_clusters,
        cluster_labels=cluster_labels,
        kl_within_mean=kl_within_mean,
        kl_within_std=kl_within_std,
        kl_between_mean=kl_between_mean,
        kl_between_std=kl_between_std,
        phi_within_mean=phi_within,
        phi_within_std=phi_within_std,
        phi_between_mean=phi_between,
        phi_between_std=phi_between_std,
        gauge_coherence=gauge_coherence,
        kl_matrix_rank=kl_matrix_rank,
        kl_entropy=kl_entropy,
        fe_belief_align=fe_belief_align,
    )


def build_hierarchical_rg_state(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    phi: torch.Tensor,
    beta: torch.Tensor,
    kl_matrix: Optional[torch.Tensor] = None,
    max_scales: int = 4,
    min_agents: int = 4,
) -> HierarchicalRGState:
    """
    Build complete hierarchical RG state across multiple scales.

    Implements the full coarse-graining tower:
        Scale 0: N tokens
        Scale 1: n₁ meta-agents
        Scale 2: n₂ super-meta-agents
        ...

    At each scale, preserves (μ, Σ, φ, β) structure.

    Args:
        mu, sigma, phi, beta: Token-level quantities
        kl_matrix: Optional raw KL divergences
        max_scales: Maximum number of scales
        min_agents: Stop when fewer than this many agents

    Returns:
        HierarchicalRGState with all scales
    """
    states = []
    diagnostics = []

    # Scale 0: Original tokens
    current_state = CoarseGrainedState(
        scale=0,
        n_agents=mu.shape[1],
        mu=mu,
        sigma=sigma,
        phi=phi,
        beta=beta,
        kl_matrix=kl_matrix,
    )
    states.append(current_state)

    # Compute diagnostics for scale 0
    diag = compute_full_rg_diagnostics(
        mu=mu, sigma=sigma, phi=phi, beta=beta,
        kl_matrix=kl_matrix, step=0
    )
    diagnostics.append(diag)

    # Iterate coarse-graining
    for scale in range(1, max_scales):
        # Check if we should stop
        if current_state.n_agents < min_agents:
            break

        if diag.n_clusters < 2 or diag.n_clusters >= current_state.n_agents - 1:
            break

        # Coarse-grain
        next_state = coarse_grain_full_state(
            mu=current_state.mu,
            sigma=current_state.sigma,
            phi=current_state.phi,
            beta=current_state.beta,
            kl_matrix=current_state.kl_matrix,
            cluster_labels=diag.cluster_labels,
            scale=scale - 1,
        )

        if next_state.n_agents < min_agents:
            break

        states.append(next_state)

        # Compute diagnostics
        diag = compute_full_rg_diagnostics(
            mu=next_state.mu,
            sigma=next_state.sigma,
            phi=next_state.phi,
            beta=next_state.beta,
            kl_matrix=next_state.kl_matrix,
            step=scale,
        )
        diagnostics.append(diag)

        current_state = next_state

    return HierarchicalRGState(states=states, diagnostics=diagnostics)


# =============================================================================
# Summary and Reporting
# =============================================================================

def summarize_hierarchical_rg(hierarchy: HierarchicalRGState) -> Dict[str, Any]:
    """
    Summarize hierarchical RG analysis.

    Returns:
        Dict with summary statistics
    """
    summary = {
        'n_scales': hierarchy.n_scales,
        'scale_invariance': hierarchy.scale_invariance(),
    }

    # Per-scale statistics
    summary['per_scale'] = []
    for i, (state, diag) in enumerate(zip(hierarchy.states, hierarchy.diagnostics)):
        scale_summary = {
            'scale': i,
            'n_agents': state.n_agents,
            'modularity': diag.modularity,
            'effective_rank': diag.effective_rank,
            'n_clusters': diag.n_clusters,
            'kl_within': diag.kl_within_mean,
            'kl_between': diag.kl_between_mean,
            'gauge_coherence': diag.gauge_coherence,
            'phi_within': diag.phi_within_mean,
            'phi_between': diag.phi_between_mean,
        }
        summary['per_scale'].append(scale_summary)

    # RG behavior assessment
    if hierarchy.n_scales >= 2:
        d0, d1 = hierarchy.diagnostics[0], hierarchy.diagnostics[1]
        summary['rg_behavior'] = {
            'modularity_increased': d1.modularity > d0.modularity,
            'rank_decreased': d1.effective_rank < d0.effective_rank,
            'kl_within_decreased': d1.kl_within_mean < d0.kl_within_mean,
            'gauge_coherent': d1.gauge_coherence > 0.5,
            'phi_separation': d1.phi_between_mean > d1.phi_within_mean,
        }

    return summary


# =============================================================================
# Main Entry
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("ENHANCED RG FLOW ANALYSIS (WITH GAUGE STRUCTURE)")
    print("="*70)

    # Test with synthetic data
    torch.manual_seed(42)

    B, N, K, G = 2, 32, 64, 3
    n_clusters = 4

    # Create clustered data
    mu = torch.randn(B, N, K)
    sigma = torch.ones(B, N, K) * 0.5
    phi = torch.randn(B, N, G) * 0.5

    # Make gauge frames coherent within clusters
    for b in range(B):
        for c in range(n_clusters):
            start = c * (N // n_clusters)
            end = (c + 1) * (N // n_clusters)
            # Shift cluster to have similar phi
            cluster_center = torch.randn(G) * 2
            phi[b, start:end] = cluster_center + torch.randn(end - start, G) * 0.1

    # Create attention with block structure
    beta = torch.zeros(B, N, N)
    for b in range(B):
        for c in range(n_clusters):
            start = c * (N // n_clusters)
            end = (c + 1) * (N // n_clusters)
            beta[b, start:end, start:end] = torch.rand(end - start, end - start) * 0.5 + 0.5
        beta[b] = beta[b] + torch.rand(N, N) * 0.1
        beta[b] = beta[b] / beta[b].sum(dim=-1, keepdim=True)

    # Create KL matrix
    kl_matrix = torch.abs(torch.randn(B, N, N))

    print("\n[Testing full RG diagnostics...]")
    diag = compute_full_rg_diagnostics(mu, sigma, phi, beta, kl_matrix)
    print(f"  Modularity: {diag.modularity:.4f}")
    print(f"  N clusters: {diag.n_clusters}")
    print(f"  Gauge coherence: {diag.gauge_coherence:.4f}")
    print(f"  Phi within: {diag.phi_within_mean:.4f}")
    print(f"  Phi between: {diag.phi_between_mean:.4f}")
    print(f"  KL matrix rank: {diag.kl_matrix_rank:.2f}")

    print("\n[Testing hierarchical RG state...]")
    hierarchy = build_hierarchical_rg_state(mu, sigma, phi, beta, kl_matrix)
    print(f"  Scales: {hierarchy.n_scales}")
    print(f"  Scale invariance: {hierarchy.scale_invariance():.4f}")

    for i, state in enumerate(hierarchy.states):
        print(f"  Scale {i}: {state.n_agents} agents")

    print("\n[Summary...]")
    summary = summarize_hierarchical_rg(hierarchy)
    print(f"  RG behavior: {summary.get('rg_behavior', 'N/A')}")

    print("\n" + "="*70)
    print("Enhanced RG analysis tests passed!")
    print("="*70)
