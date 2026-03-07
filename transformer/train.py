"""
Training Loop for Gauge-Theoretic Transformer
==============================================

Implements COMPLETE free energy minimization with all gauge-theoretic terms.

Full Free Energy:
    F = (1) Σ_i KL(q_i || p_i)                    [Belief prior - alpha]
      + (3) Σ_{i,j} β_{ij} · KL(q_i || Ω_{ij} q_j) [Belief alignment - beta]
      + (4) Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j) [Model alignment - gamma]
      - (5) E[log p(o|x)]                         [Observation likelihood]

where:
    - q_i = N(μ_i, Σ_i): Agent beliefs (evolved through transformer)
    - p_i = N(μ_embed[i], Σ_embed[i]): Embedding priors (initial)
    - β_{ij} = softmax_j(-KL(q_i||Ω_{ij}q_j)/κ): Belief coupling weights
    - γ_{ij} = softmax_j(-KL(p_i||Ω_{ij}p_j)/κ'): Prior coupling weights
    - Ω_{ij}: Parallel transport operator

Note: (2) Model prior Σ_i KL(s_i || r_i) = 0 since s_i = r_i

Author: Implementation from validated suite + complete gamma term
Date: November 2025
"""

# Suppress Triton and CuPy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import time
import json
import numpy as np
from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)

# Import attention computation for gamma term
from transformer.core.attention import compute_attention_weights
from transformer.core.sanitization import san
from transformer.training.config import TrainingConfig



def compute_rg_metrics_from_attention(
    attn_info: Dict,
    step: int,
    auto_cluster: bool = True,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute RG metrics from attention info returned by forward_with_attention.

    This analyzes the emergent renormalization group structure in the
    attention-belief dynamics, detecting meta-agent emergence.

    Args:
        attn_info: Dict with 'beta', 'mu', 'sigma' from forward_with_attention
        step: Current training step
        auto_cluster: Auto-detect clusters via spectral clustering
        n_clusters: Fixed number of clusters (None = auto)

    Returns:
        Dict with RG metrics for logging:
            - rg/modularity: Block structure in attention (higher = more meta-agents)
            - rg/effective_rank: Effective dimensionality (lower = concentrated)
            - rg/n_clusters: Number of detected meta-agents
            - rg/kl_within_mean: KL divergence within clusters (lower = tighter)
            - rg/kl_between_mean: KL divergence between clusters (stable = distinct)
            - rg/beta_entropy: Attention distribution entropy
    """
    beta = attn_info.get('beta')  # (B, n_heads, N, N) or (B, N, N)
    mu = attn_info.get('mu')      # (B, N, K)
    sigma = attn_info.get('sigma')  # (B, N, K) or (B, N, K, K)

    if beta is None or mu is None:
        return {}

    # Average over heads if multi-head attention
    if beta.dim() == 4:
        beta_avg = beta.mean(dim=1)  # (B, N, N)
    else:
        beta_avg = beta

    # Handle sigma - default to ones if None
    if sigma is None:
        sigma = torch.ones_like(mu)

    # Compute RG diagnostics
    try:
        diagnostics = compute_rg_diagnostics(
            mu=mu,
            sigma=sigma,
            beta=beta_avg,
            step=step,
            auto_cluster=auto_cluster,
            n_clusters=n_clusters,
        )

        # Convert to metrics dict
        rg_metrics = {
            'rg/modularity': diagnostics.modularity,
            'rg/effective_rank': diagnostics.effective_rank,
            'rg/n_clusters': diagnostics.n_clusters,
            'rg/kl_within_mean': diagnostics.kl_within_mean,
            'rg/kl_within_std': diagnostics.kl_within_std,
            'rg/kl_between_mean': diagnostics.kl_between_mean,
            'rg/kl_between_std': diagnostics.kl_between_std,
            'rg/beta_entropy': diagnostics.beta_entropy,
        }

        # Add meta-agent sizes if available
        if diagnostics.meta_agent_sizes:
            rg_metrics['rg/meta_agent_sizes'] = diagnostics.meta_agent_sizes

        return rg_metrics

    except (ValueError, RuntimeError, FloatingPointError) as e:
        # Return empty metrics on error (don't crash training)
        print(f"[WARNING] RG metrics computation failed: {e}")
        return {}


def compute_dynamic_rg_metrics(
    rg_info: Dict,
    step: int,
) -> Dict[str, Any]:
    """
    Compute RG flow metrics from beta_history (dynamic RG within forward pass).

    This tracks how attention structure evolves across VFE iterations,
    revealing dynamic cluster formation.

    Args:
        rg_info: Dict from forward_with_rg_tracking() containing 'beta_history'
        step: Current training step

    Returns:
        Dict with dynamic RG metrics:
            - rg/dynamic/n_iterations: Number of VFE steps
            - rg/dynamic/modularity_init: Modularity at first VFE step
            - rg/dynamic/modularity_final: Modularity at last VFE step
            - rg/dynamic/modularity_change: Final - Init (positive = emergence)
            - rg/dynamic/rank_init: Effective rank at first step
            - rg/dynamic/rank_final: Effective rank at last step
            - rg/dynamic/rank_change: Final - Init (negative = compression)
    """
    beta_history = rg_info.get('beta_history')

    if beta_history is None or len(beta_history) == 0:
        return {'rg/dynamic/n_iterations': 0}

    n_iterations = len(beta_history)

    # Import RG metrics
    from transformer.analysis.rg_metrics import compute_modularity, compute_effective_rank

    # Compute metrics at first and last step
    beta_init = beta_history[0]
    beta_final = beta_history[-1]

    if beta_init.dim() == 4:
        beta_init = beta_init.mean(dim=1)
        beta_final = beta_final.mean(dim=1)

    mod_init = compute_modularity(beta_init)
    mod_final = compute_modularity(beta_final)
    rank_init = compute_effective_rank(beta_init)
    rank_final = compute_effective_rank(beta_final)

    metrics = {
        'rg/dynamic/n_iterations': n_iterations,
        'rg/dynamic/modularity_init': mod_init,
        'rg/dynamic/modularity_final': mod_final,
        'rg/dynamic/modularity_change': mod_final - mod_init,
        'rg/dynamic/rank_init': rank_init,
        'rg/dynamic/rank_final': rank_final,
        'rg/dynamic/rank_change': rank_final - rank_init,
    }

    # If enough iterations, compute mid-point too
    if n_iterations >= 3:
        mid_idx = n_iterations // 2
        beta_mid = beta_history[mid_idx]
        if beta_mid.dim() == 4:
            beta_mid = beta_mid.mean(dim=1)
        metrics['rg/dynamic/modularity_mid'] = compute_modularity(beta_mid)
        metrics['rg/dynamic/rank_mid'] = compute_effective_rank(beta_mid)

    return metrics


# =============================================================================
# Gaussian KL Divergence (Proper Implementation)
# =============================================================================

def gaussian_kl_divergence(
    mu_q: torch.Tensor,      # (B, N, K)
    sigma_q: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    mu_p: torch.Tensor,      # (B, N, K)
    sigma_p: torch.Tensor,   # (B, N, K, K) full, (B, N, K) diagonal, or None (uses identity)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute KL(N(μ_q, Σ_q) || N(μ_p, Σ_p)) for Gaussian distributions.

    Handles both full covariance matrices (B, N, K, K) and diagonal covariances (B, N, K).
    Diagonal covariances are detected automatically based on tensor dimensions.

    For full covariances:
        KL = 0.5 * [tr(Σ_p⁻¹ Σ_q) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + log(|Σ_p|/|Σ_q|)]

    For diagonal covariances (O(K) instead of O(K³)):
        KL = 0.5 * [Σ_k(σ_q[k]/σ_p[k]) + Σ_k((μ_p[k]-μ_q[k])²/σ_p[k]) - K + Σ_k(log(σ_p[k])-log(σ_q[k]))]

    Args:
        mu_q: Posterior means (B, N, K)
        sigma_q: Posterior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        mu_p: Prior means (B, N, K)
        sigma_p: Prior covariances - (B, N, K, K) full or (B, N, K) diagonal or None
        eps: Numerical stability constant

    Returns:
        kl: KL divergence per agent, shape (B, N)
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect if covariances are diagonal (3D) or full (4D)
    sigma_q_is_diagonal = sigma_q is not None and sigma_q.dim() == 3
    sigma_p_is_diagonal = sigma_p is not None and sigma_p.dim() == 3

    # If either is diagonal, use diagonal path for both (more efficient)
    use_diagonal = sigma_q_is_diagonal or sigma_p_is_diagonal

    if use_diagonal:
        # =================================================================
        # DIAGONAL PATH: O(K) per agent instead of O(K³)
        # =================================================================
        # Convert to diagonal variances if needed
        if sigma_q is None:
            sigma_q_diag = torch.ones(*mu_q.shape, device=device, dtype=dtype)
        elif sigma_q.dim() == 3:
            sigma_q_diag = sigma_q  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_q_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1)  # (B, N, K)

        if sigma_p is None:
            sigma_p_diag = torch.ones(*mu_p.shape, device=device, dtype=dtype)
        elif sigma_p.dim() == 3:
            sigma_p_diag = sigma_p  # Already diagonal (B, N, K)
        else:
            # Extract diagonal from full matrix
            sigma_p_diag = torch.diagonal(sigma_p, dim1=-2, dim2=-1)  # (B, N, K)

        # Ensure positivity
        sigma_q_diag = sigma_q_diag.clamp(min=eps)
        sigma_p_diag = sigma_p_diag.clamp(min=eps)

        # Trace term: Σ_k (σ_q[k] / σ_p[k])
        trace_term = (sigma_q_diag / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Mahalanobis term: Σ_k ((μ_p[k] - μ_q[k])² / σ_p[k])
        delta_mu = mu_p - mu_q  # (B, N, K)
        mahal_term = ((delta_mu ** 2) / sigma_p_diag).sum(dim=-1)  # (B, N)

        # Log determinant term: Σ_k (log(σ_p[k]) - log(σ_q[k]))
        logdet_term = (torch.log(sigma_p_diag) - torch.log(sigma_q_diag)).sum(dim=-1)  # (B, N)

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    else:
        # =================================================================
        # FULL COVARIANCE PATH: O(K³) via Cholesky
        # =================================================================
        # Default to identity covariances if not provided
        if sigma_q is None:
            sigma_q = torch.eye(K, device=device, dtype=dtype).expand(*mu_q.shape[:-1], K, K)
        if sigma_p is None:
            sigma_p = torch.eye(K, device=device, dtype=dtype).expand(*mu_p.shape[:-1], K, K)

        # Regularize for numerical stability (1e-4 floor for robustness)
        reg = max(eps, 1e-4)
        I_K = torch.eye(K, device=device, dtype=dtype)
        sigma_q_reg = sigma_q + reg * I_K
        sigma_p_reg = sigma_p + reg * I_K

        try:
            # Compute Σ_p⁻¹ via Cholesky
            L_p = torch.linalg.cholesky(sigma_p_reg)

            # Trace term: tr(Σ_p⁻¹ Σ_q)
            Y = torch.linalg.solve_triangular(L_p, sigma_q_reg, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N)

            # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
            delta_mu = mu_p - mu_q  # (B, N, K)
            v = torch.linalg.solve_triangular(L_p, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
            mahal_term = torch.sum(v ** 2, dim=-1)  # (B, N)

            # Log determinant terms
            logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)
            L_q = torch.linalg.cholesky(sigma_q_reg)
            logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)
            logdet_term = logdet_p - logdet_q  # (B, N)
        except RuntimeError:
            # Eigenvalue fallback for non-SPD matrices
            san.record('cholesky_fallback')
            eigvals_p = torch.linalg.eigvalsh(sigma_p_reg).clamp(min=1e-6)
            eigvals_q = torch.linalg.eigvalsh(sigma_q_reg).clamp(min=1e-6)
            logdet_p = torch.sum(torch.log(eigvals_p), dim=-1)
            logdet_q = torch.sum(torch.log(eigvals_q), dim=-1)
            logdet_term = logdet_p - logdet_q

            sigma_p_inv = torch.linalg.pinv(sigma_p_reg)
            trace_term = torch.einsum('...kk->...', sigma_p_inv @ sigma_q_reg)
            delta_mu = mu_p - mu_q
            mahal_term = torch.einsum('...k,...k->...',
                delta_mu, torch.einsum('...kl,...l->...k', sigma_p_inv, delta_mu))

        # KL divergence
        kl = 0.5 * (trace_term + mahal_term - K + logdet_term)

    return torch.clamp(kl, min=0.0, max=100.0)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Free Energy Loss Computation (ATTENTION-WEIGHTED)
# =============================================================================

def compute_free_energy_loss(
    model,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.0,           # Self-consistency weight
    lambda_beta: float = 1.0,     # Belief alignment weight
    lambda_gamma: float = 0.0,    # Model alignment weight
    kappa_gamma: float = 1.0,     # Temperature for γ_ij coupling weights
    pad_token_id: int = -100,     # Token ID to ignore in loss (padding)
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute COMPLETE free energy loss with all gauge-theoretic terms.

    Full Free Energy:
        F = (1) α · Σ_i KL(q_i || p_i)                    [Belief prior]
          + (3) λ_β · Σ_{i,j} β_ij · KL(q_i || Ω_{ij}q_j) [Belief alignment]
          + (4) λ_γ · Σ_{i,j} γ_ij · KL(p_i || Ω_{ij}p_j) [Model alignment]
          - (5) E[log p(o|x)]                             [Observation likelihood]

    where:
        - q_i = N(μ_i, Σ_i): Agent beliefs (evolved through transformer)
        - p_i = N(μ_embed, Σ_embed): Embedding priors (initial, from token_embed)
        - β_ij = softmax_j(-KL(q_i||Ω_{ij}q_j)/κ_β): Belief coupling weights
        - γ_ij = softmax_j(-KL(p_i||Ω_{ij}p_j)/κ_γ): Prior coupling weights
        - Ω_{ij}: Parallel transport operator (gauge connection)
        - PyTorch autodiff handles ∂β_ij/∂μ_i and ∂γ_ij/∂μ_embed automatically!

    This is the CORRECT formulation from active inference + gauge theory:
        - Belief alignment (β): Encourages consistency between evolved beliefs
        - Model alignment (γ): Encourages consistency between embedding priors

    Args:
        model: GaugeTransformerLM with forward_with_attention() method
        token_ids: (B, N) input token IDs
        targets: (B, N) target token IDs
        alpha: Weight for belief prior KL(q||p) (default: 0.0)
        lambda_beta: Weight for belief alignment term (default: 1.0)
        lambda_gamma: Weight for model alignment term (default: 0.0)
        kappa_gamma: Temperature for γ_ij coupling weights (default: 1.0)
        pad_token_id: Token ID for padding (ignored in loss). Default -100.

    Returns:
        total_loss: Scalar loss for backprop
        metrics: Dict with loss components

    Example:
        >>> # Standard training (gamma disabled)
        >>> loss, metrics = compute_free_energy_loss(
        ...     model, inputs, targets,
        ...     alpha=0.001, lambda_beta=0.1, lambda_gamma=0.0
        ... )

        >>> # With model alignment (regularize embedding space)
        >>> loss, metrics = compute_free_energy_loss(
        ...     model, inputs, targets,
        ...     alpha=0.001, lambda_beta=0.1, lambda_gamma=0.01
        ... )
    """
    # =================================================================
    # Forward pass with attention weights and KL matrices
    # =================================================================
    # NOTE: Do NOT pass targets here! Passing targets allows VFE FFN to
    # "cheat" by using targets to adjust beliefs before CE is computed,
    # causing CE to collapse to 0. Targets should only be used for loss
    # computation AFTER the forward pass.
    logits, attn_info = model.forward_with_attention(token_ids, targets=None)

    beta = attn_info['beta']    # (B, n_heads, N, N)
    kl = attn_info['kl']        # (B, n_heads, N, N)
    mu_q = attn_info['mu']      # (B, N, K) - evolved beliefs
    sigma_q = attn_info['sigma']  # (B, N, K, K) or None

    # Extract priors for gamma term
    mu_p = attn_info['mu_prior']      # (B, N, K) - embedding priors
    sigma_p = attn_info['sigma_prior']  # (B, N, K, K)
    phi_p = attn_info['phi_prior']      # (B, N, 3)
    generators = model.generators      # (3, K, K)

    # =================================================================
    # 1. Observation Likelihood: -E[log p(o|x)] = Cross-Entropy
    # =================================================================
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # (B*N, V)
        targets.reshape(-1),                   # (B*N,)
        reduction='mean',
        ignore_index=pad_token_id,  # Ignore padding tokens in loss
    )

    # =================================================================
    # 2. Attention-Weighted Free Energy: Σ_ij β_ij · KL(q_i||Ω_ij[q_j])
    # =================================================================
    # This is the CRITICAL term from the validated suite!
    # Line 189 from gradients.free_energy_clean.py: weighted_field = beta_ij * kl_field

    if lambda_beta > 0.0:
        # Pointwise multiplication: β_ij * KL_ij for each head
        weighted_kl = beta * kl  # (B, n_heads, N, N)

        # Sum over all pairs (i,j) and average over heads and batch
        # Note: Averaging over batch and heads, summing over agent pairs
        belief_align_loss = weighted_kl.sum(dim=(-2, -1)).mean()  # Mean over (batch, heads)

        # Normalize by √K to stabilize loss scale for large latent dimensions.
        # KL between K-dim Gaussians scales as O(K); √K normalization
        # prevents belief alignment loss from dominating CE.
        K = mu_q.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        belief_align_loss = lambda_beta * belief_align_loss / dim_scale
    else:
        belief_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 3. Self-Consistency: α·KL(q||p) - Beliefs should stay close to priors
    # =================================================================
    # This is the key VFE term that pulls evolved beliefs back toward
    # their embedding priors. Without this, beliefs can drift arbitrarily.
    #
    # KL(q||p) = KL(N(μ_q, Σ_q) || N(μ_p, Σ_p))
    #
    # This provides gradients to embeddings even when belief evolution is detached!
    # =================================================================
    if alpha > 0.0:
        K = mu_q.shape[-1]
        # Proper KL divergence between evolved beliefs and embedding priors.
        # Gradients flow through both mu and sigma to train embeddings.
        # The diagonal path (common case) uses no Cholesky — just σ_q/σ_p ratios.
        # The full-covariance path uses Cholesky with eigenvalue fallback for robustness.
        kl_per_agent = gaussian_kl_divergence(
            mu_q=mu_q,        # Evolved beliefs
            sigma_q=sigma_q,  # Allow gradient flow to sigma embeddings
            mu_p=mu_p,        # Embedding priors
            sigma_p=sigma_p,  # Allow gradient flow to sigma embeddings
        )  # (B, N)

        # Normalize by √K: KL between K-dim Gaussians scales as O(K);
        # √K normalization prevents self-consistency from dominating CE.
        dim_scale = math.sqrt(max(K, 1))

        # Average over batch and agents, normalized by √K
        self_consistency_loss = alpha * kl_per_agent.mean() / dim_scale
    else:
        self_consistency_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # 4. Model Alignment: λ_γ·Σ_{i,j} γ_ij · KL(p_i || Ω_{ij} p_j)
    # =================================================================
    # This term encourages consistency between embedding priors p_i across agents.
    #
    # Formula:
    #   L_model = λ_γ · Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j)
    #
    # where:
    #   - p_i = N(μ_embed[i], Σ_embed[i]): Initial embedding prior
    #   - γ_{ij} = softmax_j(-KL(p_i || Ω_{ij} p_j) / κ_γ): Prior coupling weights
    #   - Ω_{ij} = exp(φ_i) · exp(-φ_j): Parallel transport operator
    #
    # This is symmetric to belief alignment (β term), but operates on
    # the embedding space rather than the evolved belief space.
    #
    # Use cases:
    #   - Regularize embedding space to be gauge-consistent
    #   - Prevent embeddings from having arbitrary gauge choices
    #   - Encourage smooth gauge structure in token space
    # =================================================================
    if lambda_gamma > 0.0:
        batch_size, num_agents, K = mu_p.shape
        device = mu_p.device

        # Causal mask (same as for beliefs)
        mask = torch.tril(torch.ones(num_agents, num_agents, device=device))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute γ_{ij} coupling weights and KL(p_i || Ω_{ij} p_j)
        # Using same attention mechanism as β_{ij}, but on priors
        # Detect if sigma_p is diagonal (3D) or full (4D)
        diagonal_cov = sigma_p is not None and sigma_p.dim() == 3
        gamma, kl_prior = compute_attention_weights(
            mu_p,
            sigma_p,
            phi_p,
            generators,
            kappa_gamma,
            epsilon=1e-8,
            mask=mask,
            use_numba=False,  # Use PyTorch for gradient tracking
            return_kl=True,
            diagonal_covariance=diagonal_cov,
        )
        # gamma: (B, N, N)
        # kl_prior: (B, N, N)

        # Weighted model alignment: Σ_{i,j} γ_{ij} · KL(p_i || Ω_{ij} p_j)
        weighted_kl_prior = gamma * kl_prior  # (B, N, N)

        # Sum over all agent pairs and average over batch
        # Normalize by √K for dimension-stable loss scale
        K = mu_p.shape[-1]
        dim_scale = math.sqrt(max(K, 1))
        model_align_loss = lambda_gamma * weighted_kl_prior.sum(dim=(-2, -1)).mean() / dim_scale
    else:
        model_align_loss = torch.tensor(0.0, device=ce_loss.device)

    # =================================================================
    # Total Free Energy (ALL FOUR TERMS)
    # =================================================================
    total_loss = ce_loss + belief_align_loss + self_consistency_loss + model_align_loss

    # Compute attention metrics outside the computation graph
    with torch.no_grad():
        beta_avg = beta.mean(dim=1)  # (B, N, N) - average over heads
        beta_safe = beta_avg.clamp(min=1e-10)
        attn_entropy = -(beta_safe * beta_safe.log()).sum(dim=-1).mean()
        attn_concentration = beta_avg.max(dim=-1)[0].mean()

    # Metrics
    metrics = {
        'loss/total': total_loss.item(),
        'loss/ce': ce_loss.item(),
        'loss/belief_align': belief_align_loss.item(),
        'loss/self_consistency': self_consistency_loss.item() if alpha > 0 else 0.0,
        'loss/model_align': model_align_loss.item() if lambda_gamma > 0 else 0.0,
        'attention/beta_mean': beta.mean().item(),
        'attention/kl_mean': kl.mean().item(),
        'attention/entropy': attn_entropy.item(),
        'attention/concentration': attn_concentration.item(),
    }

    # Bayesian alpha diagnostics
    with torch.no_grad():
        for block in model.transformer.blocks:
            vffn = getattr(block.ffn, 'variational_ffn', None)
            if vffn is not None and vffn.learnable_alpha and mu_q is not None and mu_p is not None:
                import torch.nn.functional as _F
                alpha_vals = vffn.get_bayesian_alpha(mu_q, mu_p, sigma_p)
                a0 = _F.softplus(vffn.raw_a0)
                b0 = _F.softplus(vffn.raw_b0)
                delta = mu_q - mu_p
                if sigma_p.dim() == 3:
                    mahal_sq = (delta ** 2 / sigma_p.clamp(min=1e-6)).sum(dim=-1)
                else:
                    K = mu_q.shape[-1]
                    sp_inv = torch.linalg.inv(sigma_p + 1e-6 * torch.eye(K, device=mu_q.device))
                    mahal_sq = torch.einsum('bni,bnij,bnj->bn', delta, sp_inv, delta)
                metrics['bayesian/alpha_mean'] = alpha_vals.mean().item()
                metrics['bayesian/alpha_std'] = alpha_vals.std().item()
                metrics['bayesian/alpha_min'] = alpha_vals.min().item()
                metrics['bayesian/alpha_max'] = alpha_vals.max().item()
                metrics['bayesian/a0'] = a0.item()
                metrics['bayesian/b0'] = b0.item()
                metrics['bayesian/mahal_sq_mean'] = mahal_sq.mean().item()
                metrics['bayesian/mahal_sq_std'] = mahal_sq.std().item()
                break  # Only first layer for now

    if lambda_gamma > 0.0:
        metrics['attention/gamma_mean'] = gamma.mean().item()
        metrics['attention/kl_prior_mean'] = kl_prior.mean().item()

    # =================================================================
    # P-FLOW DATA: Include beliefs and per-position CE for optional P-flow updates
    # =================================================================
    # Compute per-position CE for weighting P-flow updates (low error = successful belief)
    with torch.no_grad():
        ce_per_position = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none',
            ignore_index=pad_token_id,
        ).reshape(targets.shape)  # (B, N)

    # Store in metrics for optional P-flow in training loop
    metrics['p_flow/mu_q'] = mu_q.detach()           # (B, N, K) final beliefs
    metrics['p_flow/ce_per_position'] = ce_per_position  # (B, N) per-position CE

    # Store attention info for RG metrics computation (detached)
    metrics['attention_info'] = {
        'beta': beta.detach(),
        'kl': kl.detach(),
        'mu': mu_q.detach(),
        'sigma': sigma_q.detach() if sigma_q is not None else None,
    }

    return total_loss, metrics


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """Training orchestration for gauge transformer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()

        # Get pad_token_id from dataset for proper loss masking
        # Default to -100 (PyTorch's default ignore_index) if not available
        self.pad_token_id = getattr(train_loader.dataset, 'pad_token_id', -100)

        # Move model to device
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.step = 0
        self.best_val_ce = float('inf')  # Track CE loss (not total loss) for best model

        # Create checkpoint directory
        if self.config.checkpoint_dir is not None:
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            wandb.watch(self.model, log='all', log_freq=1000)

        print("Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"  λ_β (attention-weighted KL): {self.config.lambda_beta}")
        print(f"  Max steps: {self.config.max_steps:,}")

        # Resume from checkpoint if specified
        if self.config.resume_from is not None:
            print(f"\n  Resuming from checkpoint: {self.config.resume_from}")
            self.load_checkpoint(self.config.resume_from)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create AdamW optimizer with configurable parameter grouping.

        Modes:
        - Simple (use_param_groups=False): 2 groups (decay vs no-decay) with single LR
        - Multi-group (use_param_groups=True): 6 groups (mu, sigma, phi, attention, output, other)
        """
        if self.config.use_param_groups:
            # Multi-group mode: Natural gradients with per-parameter-type learning rates
            return self._create_multigroup_optimizer()
        else:
            # Simple mode: Traditional 2-group optimizer (decay vs no-decay)
            return self._create_simple_optimizer()

    def _create_simple_optimizer(self) -> torch.optim.Optimizer:
        """Create simple 2-group optimizer (decay vs no-decay)."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2))

        return optimizer

    def _create_multigroup_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with per-parameter group learning rates.

        Parameter Groups:
            1. mu_embed: Mean embeddings
            2. sigma_embed: Covariance embeddings
            3. phi_embed: Gauge frame embeddings
            4. attention: Attention mechanism
            5. output: Output projection
            6. other: Layer norms, VFE hyperparams (base LR, no decay)

        This exploits natural gradient structure on statistical manifolds!
        """
        # Collect parameters by type
        mu_params = []
        sigma_params = []
        phi_params = []
        attention_params = []
        output_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Mean embeddings
            if 'mu_embed' in name:
                mu_params.append(param)
            # Covariance embeddings (sigma_embed, log_sigma_diag, base_log_sigma_diag)
            elif 'sigma_embed' in name or 'log_sigma' in name:
                sigma_params.append(param)
            # Gauge frame embeddings
            elif 'phi_embed' in name:
                phi_params.append(param)
            # Positional encoding (treat as gauge frames)
            elif 'pos_encoding' in name:
                phi_params.append(param)
            # Attention mechanism
            elif 'attention' in name or 'attn' in name:
                attention_params.append(param)
            # Output projection
            elif 'out_proj' in name:
                output_params.append(param)
            # Other: layer norms, VFE hyperparams (raw_a0, raw_b0, log_kappa_heads)
            else:
                other_params.append(param)

        # Create parameter groups
        param_groups = []

        if mu_params:
            param_groups.append({
                'params': mu_params,
                'lr': self.config.mu_lr,
                'weight_decay': 0.0,  # No decay for embeddings
                'name': 'mu_embed',
            })
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={self.config.mu_lr}")

        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': self.config.sigma_lr,
                'weight_decay': 0.0,
                'name': 'sigma_embed',
            })
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={self.config.sigma_lr}")

        if phi_params:
            param_groups.append({
                'params': phi_params,
                'lr': self.config.phi_lr,
                'weight_decay': 0.0,
                'name': 'phi_embed',
            })
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={self.config.phi_lr}")

        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.attention_lr,
                'weight_decay': self.config.weight_decay,
                'name': 'attention',
            })
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={self.config.attention_lr}")

        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': self.config.output_lr,
                'weight_decay': 0.0,  # Often tied to embeddings
                'name': 'output',
            })
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={self.config.output_lr}")

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.learning_rate,
                'weight_decay': 0.0,
                'name': 'other',
            })
            print(f"  Parameter group 'other': {len(other_params)} tensors @ lr={self.config.learning_rate}")

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_decay == 'constant':
            return None

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)

            if self.config.lr_decay == 'cosine':
                progress = min(1.0, (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps))
                return self.config.min_lr / self.config.learning_rate + \
                       0.5 * (1 - self.config.min_lr / self.config.learning_rate) * \
                       (1 + math.cos(progress * math.pi))
            elif self.config.lr_decay == 'linear':
                min_ratio = self.config.min_lr / self.config.learning_rate
                return max(min_ratio, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
            else:
                return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # model.train() is called once in train() method, not per-step

        token_ids, targets = batch
        token_ids = token_ids.to(self.device)
        targets = targets.to(self.device)

        # Forward + loss
        loss, metrics = compute_free_energy_loss(
            self.model,
            token_ids,
            targets,
            alpha=self.config.alpha,
            lambda_beta=self.config.lambda_beta,
            lambda_gamma=self.config.lambda_gamma,
            kappa_gamma=self.config.kappa_gamma,
            pad_token_id=self.pad_token_id,
        )

        # Scale loss for gradient accumulation
        loss = loss / self.config.accumulation_steps

        # Backward
        loss.backward()

        # =================================================================
        # Gradient Monitoring (only on logging steps to avoid GPU sync overhead)
        # =================================================================
        if self.step % self.config.log_every == 0:
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if 'mu_embed' in name:
                        grad_norms['grad/mu_embed'] = grad_norm
                    elif 'sigma_embed' in name or 'log_sigma' in name:
                        grad_norms['grad/sigma_embed'] = grad_norm
                    elif 'phi_embed' in name:
                        grad_norms['grad/phi_embed'] = grad_norm
                    elif 'out_proj' in name:
                        grad_norms['grad/out_proj'] = grad_norm
            metrics.update(grad_norms)

            if 'grad/mu_embed' in grad_norms and grad_norms['grad/mu_embed'] == 0.0:
                print("[WARNING] mu_embed gradient is ZERO - gradient flow may be broken!")

        # Optimizer step (if accumulation complete)
        if (self.step + 1) % self.config.accumulation_steps == 0:
            # Per-group gradient clipping for large gauge groups.
            # With SO(100), phi_embed has 4950 dims per token vs 100 for mu.
            # Global clipping at grad_clip=1.0 means phi dominates the norm,
            # starving mu/sigma of learning signal. Clip each param group
            # independently so all parameter types get sufficient gradients.
            #
            # phi gets a tighter clip (phi_grad_clip, default 0.1) because
            # gauge frame gradients spike 2-3 orders of magnitude above
            # mu/sigma, causing erratic effective LR.
            _phi_clip = getattr(self.config, 'phi_grad_clip', self.config.grad_clip)
            if self.config.use_param_groups:
                for group in self.optimizer.param_groups:
                    if group['params']:
                        clip = _phi_clip if 'phi' in group.get('name', '') else self.config.grad_clip
                        torch.nn.utils.clip_grad_norm_(
                            group['params'],
                            clip
                        )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        # Add learning rate to metrics
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        # Report sanitization metrics (sigma clamps, Cholesky fallbacks, etc.)
        san_metrics = san.report(step=self.step, log_every=self.config.log_every)
        metrics.update(san_metrics)

        return metrics

    @torch.no_grad()
    def validate(self, max_batches: int = 200) -> Dict[str, float]:
        """Validation pass.

        Args:
            max_batches: Maximum number of validation batches to evaluate.
                Prevents long stalls on large datasets like WikiText-103.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            token_ids, targets = batch
            token_ids = token_ids.to(self.device)
            targets = targets.to(self.device)

            loss, metrics = compute_free_energy_loss(
                self.model, token_ids, targets,
                alpha=self.config.alpha,
                lambda_beta=self.config.lambda_beta,
                lambda_gamma=self.config.lambda_gamma,
                kappa_gamma=self.config.kappa_gamma,
                pad_token_id=self.pad_token_id,
            )

            total_loss += loss.item()
            total_ce_loss += metrics['loss/ce']
            n_batches += 1

            if n_batches >= max_batches:
                break

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce_loss = total_ce_loss / max(n_batches, 1)
        perplexity = torch.exp(torch.tensor(avg_ce_loss)).item()

        return {
            'val/loss': avg_loss,
            'val/ce_loss': avg_ce_loss,
            'val/perplexity': perplexity,
        }

    def train(self):
        """Main training loop."""
        print("\n" + "="*70)
        print("STARTING TRAINING (Attention-Weighted Free Energy)")
        print("="*70)

        if TQDM_AVAILABLE:
            pbar = tqdm(total=self.config.max_steps, desc="Training")
        else:
            pbar = None

        start_time = time.time()
        self.model.train()

        try:
            while self.step < self.config.max_steps:
                for batch in self.train_loader:
                    # Training step
                    metrics = self.train_step(batch)

                    # Logging
                    if self.step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        tokens_per_sec = (self.step * self.config.batch_size * batch[0].shape[1]) / elapsed

                        # Basic metrics
                        print(f"\nStep {self.step:6d} | Loss: {metrics['loss/total']:.4f} | "
                              f"CE: {metrics['loss/ce']:.4f} | Align: {metrics['loss/belief_align']:.4f} | "
                              f"LR: {metrics['lr']:.2e}")

                        # Gradient norms (verify gradients are flowing!)
                        grad_mu = metrics.get('grad/mu_embed', 0.0)
                        grad_out = metrics.get('grad/out_proj', 0.0)
                        grad_phi = metrics.get('grad/phi_embed', 0.0)
                        print(f"         Grads | μ_embed: {grad_mu:.4f} | out_proj: {grad_out:.4f} | φ_embed: {grad_phi:.4f}")

                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(metrics, step=self.step)

                    # Validation
                    if self.step % self.config.eval_every == 0 and self.step > 0:
                        val_metrics = self.validate()
                        self.model.train()  # Restore training mode after validation
                        if val_metrics:
                            print(f"\nValidation | Loss: {val_metrics['val/loss']:.4f} | "
                                  f"PPL: {val_metrics['val/perplexity']:.2f}")

                            if self.config.use_wandb and WANDB_AVAILABLE:
                                wandb.log(val_metrics, step=self.step)

                            # Save best model based on CE loss (not total loss)
                            # CE loss is the proper metric since PPL = exp(CE)
                            if val_metrics['val/ce_loss'] < self.best_val_ce:
                                self.best_val_ce = val_metrics['val/ce_loss']
                                self.save_checkpoint('best_model.pt')

                    # Checkpointing
                    if self.step % self.config.save_every == 0 and self.step > 0:
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

                    # Update progress
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({'loss': f"{metrics['loss/total']:.4f}"})

                    self.step += 1

                    if self.step >= self.config.max_steps:
                        break


        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")

        finally:
            if pbar is not None:
                pbar.close()

        # Final validation
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)

        final_metrics = self.validate()
        if final_metrics:
            print(f"Final Validation Loss: {final_metrics['val/loss']:.4f}")
            print(f"Final Perplexity: {final_metrics['val/perplexity']:.2f}")

        # Save final model
        self.save_checkpoint('final_model.pt')

        print("="*70)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        if self.config.checkpoint_dir is None:
            return

        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'model_state': self.model.state_dict(),
            'best_val_ce': self.best_val_ce,
            'config': self.model.config,
        }

        if self.config.save_optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])

        if 'optimizer_state' in checkpoint and self.config.save_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'scheduler_state' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.step = checkpoint.get('step', 0)
        # Backward compatible: try new key first, fall back to old key
        self.best_val_ce = checkpoint.get('best_val_ce', checkpoint.get('best_val_loss', float('inf')))

        print(f"✓ Loaded checkpoint from step {self.step}")


# =============================================================================
# PURE FEP MODE: Backprop-Free Training via Prior Evolution
# =============================================================================
# This implements the BELIEF (Backprop-free Evolving Local Inference via Free Energy)
# training paradigm where learning happens through prior evolution, not backprop.
#
# Key differences from standard training:
# 1. Forward pass wrapped in torch.no_grad() - no gradient tracking
# 2. Targets ARE passed to VFE dynamics (observation term active)
# 3. Learning via update_priors_from_beliefs() based on prediction errors
# 4. Persistent priors in each FFN layer consolidate successful beliefs
#
# The theoretical basis:
# - VFE includes observation term: F = KL(q||p) + alignment + CE
# - Beliefs adjust to minimize CE during forward pass
# - Low-error beliefs update persistent priors (soft EM)
# - Priors consolidate knowledge without backprop
# - Embeddings update toward successful beliefs (tied with output projection)
# =============================================================================


def update_output_embeddings_pflow(
    model,
    targets: torch.Tensor,         # (B, N) target token IDs
    mu_beliefs: torch.Tensor,      # (B, N, K) final beliefs (posteriors)
    prediction_errors: torch.Tensor,  # (B, N) per-position CE
    lr: float = 0.1,
):
    """
    P-FLOW: Update OUTPUT embeddings toward beliefs that predict each target.

    CRITICAL FOR LEARNING in pure FEP mode with untied embeddings!

    When embeddings are untied:
    - Input embeddings (priors) = what we expect for input tokens
    - Output embeddings (W_out) = what beliefs should look like to predict tokens

    If we only update input embeddings, the output projection never learns
    which belief patterns correspond to which tokens -> random predictions!

    This function updates W_out[target_v] toward the belief at positions
    that should predict token v.

    Args:
        model: GaugeTransformerLM with out_proj
        targets: (B, N) target token IDs (which tokens to predict)
        mu_beliefs: (B, N, K) evolved belief means (posteriors)
        prediction_errors: (B, N) per-position cross-entropy loss
        lr: Base learning rate for output embedding updates
    """
    B, N, K = mu_beliefs.shape

    # Get OUTPUT projection weight
    if not hasattr(model, 'out_proj') or model.out_proj is None:
        return {'out_embed_updates': 0, 'out_embed_mode': 'none'}

    out_weight = model.out_proj.weight  # (V, K)

    with torch.no_grad():
        # Compute weights from prediction errors (low CE = high weight)
        errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
        weights = 1.0 / (1.0 + errors_clamped)  # (B, N) in range [0.05, 1]

        # Flatten for processing
        flat_targets = targets.view(-1)  # (B*N,) - TARGET token IDs
        flat_beliefs = mu_beliefs.view(-1, K)  # (B*N, K)
        flat_weights = weights.view(-1)  # (B*N,)

        # Mask out padding tokens (typically -100)
        valid_mask = flat_targets >= 0
        flat_targets = flat_targets[valid_mask]
        flat_beliefs = flat_beliefs[valid_mask]
        flat_weights = flat_weights[valid_mask]

        if len(flat_targets) == 0:
            return {'out_embed_updates': 0, 'out_embed_mode': 'no_valid_targets'}

        # For each unique TARGET token, compute weighted average posterior
        unique_tokens = flat_targets.unique()
        n_updates = 0

        for token_id in unique_tokens:
            mask = flat_targets == token_id
            token_beliefs = flat_beliefs[mask]  # (n_occurrences, K)
            token_weights = flat_weights[mask]  # (n_occurrences,)

            # Weighted average of posteriors for this target token
            weight_sum = token_weights.sum()
            if weight_sum > 0:
                weighted_posterior = (token_beliefs * token_weights.unsqueeze(-1)).sum(dim=0) / weight_sum

                # P-FLOW: output_embed <- (1-lr)*output_embed + lr*posterior
                # Scale lr by confidence (weight_sum / count)
                effective_lr = lr * (weight_sum / mask.sum()).item()
                effective_lr = min(effective_lr, 0.1)  # Cap for stability

                out_weight[token_id] = (
                    (1.0 - effective_lr) * out_weight[token_id] +
                    effective_lr * weighted_posterior
                )
                n_updates += 1

    return {
        'out_embed_updates': n_updates,
        'out_embed_mode': 'out_proj',
    }


def update_input_embeddings_pflow(
    model,
    input_ids: torch.Tensor,       # (B, N) input token IDs
    mu_beliefs: torch.Tensor,      # (B, N, K) final beliefs (posteriors)
    prediction_errors: torch.Tensor,  # (B, N) per-position CE
    lr: float = 0.1,
):
    """
    P-FLOW: Update INPUT embeddings toward posteriors (beliefs).

    This is the correct FEP learning rule:
    - Prior (input embedding) moves toward posterior (belief)
    - μ_p ← (1-lr) * μ_p + lr * μ_q
    - Weighted by prediction success (low CE = stronger update)

    With UNTIED embeddings:
    - Input embeddings (W_in) = priors, updated here
    - Output embeddings (W_out) = observation anchors, stay fixed

    Args:
        model: GaugeTransformerLM with untied embeddings
        input_ids: (B, N) input token IDs (which tokens to update)
        mu_beliefs: (B, N, K) evolved belief means (posteriors)
        prediction_errors: (B, N) per-position cross-entropy loss
        lr: Base learning rate for p-flow updates
    """
    B, N, K = mu_beliefs.shape

    # Get INPUT embedding weight (priors)
    token_embed = model.token_embed

    if hasattr(token_embed, 'mu_embed'):
        embed_weight = token_embed.mu_embed.weight  # (V, K) - INPUT embeddings
    elif hasattr(token_embed, 'base_mu'):
        # Gauge-fixed priors: update base_mu toward average belief
        with torch.no_grad():
            errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
            weights = 1.0 / (1.0 + errors_clamped)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weighted_belief = (mu_beliefs * weights.unsqueeze(-1)).sum(dim=(0, 1)) / weight_sum
                effective_lr = min(lr * 0.1, 0.01)
                token_embed.base_mu.data = (
                    (1.0 - effective_lr) * token_embed.base_mu.data +
                    effective_lr * weighted_belief
                )
        return {'embed_updates': 1, 'embed_mode': 'base_mu'}
    else:
        return {'embed_updates': 0, 'embed_mode': 'none'}

    with torch.no_grad():
        # Compute weights from prediction errors (low CE = high weight)
        errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
        weights = 1.0 / (1.0 + errors_clamped)  # (B, N) in range [0.05, 1]

        # Flatten for processing
        flat_inputs = input_ids.view(-1)  # (B*N,) - INPUT token IDs
        flat_beliefs = mu_beliefs.view(-1, K)  # (B*N, K)
        flat_weights = weights.view(-1)  # (B*N,)

        # For each unique INPUT token, compute weighted average posterior
        unique_tokens = flat_inputs.unique()
        n_updates = 0

        for token_id in unique_tokens:
            mask = flat_inputs == token_id
            token_beliefs = flat_beliefs[mask]  # (n_occurrences, K)
            token_weights = flat_weights[mask]  # (n_occurrences,)

            # Weighted average of posteriors for this input token
            weight_sum = token_weights.sum()
            if weight_sum > 0:
                weighted_posterior = (token_beliefs * token_weights.unsqueeze(-1)).sum(dim=0) / weight_sum

                # P-FLOW: prior <- (1-lr)*prior + lr*posterior
                effective_lr = lr * (weight_sum / mask.sum()).item()
                effective_lr = min(effective_lr, 0.1)  # Cap for stability

                embed_weight[token_id] = (
                    (1.0 - effective_lr) * embed_weight[token_id] +
                    effective_lr * weighted_posterior
                )
                n_updates += 1

        return {'embed_updates': n_updates, 'embed_unique_tokens': len(unique_tokens)}


def pure_fep_train_step(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Single training step using pure FEP (no backprop).

    Learning happens through prior evolution:
    1. Forward pass WITH targets (CE inside VFE)
    2. Beliefs adjust to minimize prediction error
    3. Priors update toward successful beliefs

    Args:
        model: GaugeTransformerLM with VFE_dynamic FFN in pure_fep_mode
        input_ids: (B, N) input token IDs
        targets: (B, N) target token IDs
        device: Target device

    Returns:
        Dict of training metrics
    """
    model.eval()  # No dropout etc during pure FEP

    input_ids = input_ids.to(device)
    targets = targets.to(device)

    B, N = input_ids.shape

    with torch.no_grad():
        # Forward pass WITH targets - observation term is now active!
        # This is the key change: targets flow into VFE dynamics
        logits, attn_info = model.forward_with_attention(input_ids, targets=targets)

        # Per-position cross-entropy for prior weighting
        ce_per_position = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none',
            ignore_index=-100,
        ).view(B, N)

        # Overall loss for logging
        valid_mask = targets != -100
        ce_loss = ce_per_position[valid_mask].mean() if valid_mask.any() else ce_per_position.mean()

        # Get evolved beliefs for prior updates
        mu_beliefs = attn_info['mu']  # (B, N, K)
        sigma_beliefs = attn_info['sigma']  # (B, N, K, K) or (B, N, K)

        # Convert full covariance to diagonal if needed
        if sigma_beliefs is not None and sigma_beliefs.dim() == 4:
            sigma_beliefs = torch.diagonal(sigma_beliefs, dim1=-2, dim2=-1)

        # Update priors in each transformer block's FFN
        prior_stats = {}
        for layer_idx, block in enumerate(model.transformer.blocks):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'update_priors_from_beliefs'):
                block.ffn.update_priors_from_beliefs(
                    mu_beliefs=mu_beliefs,
                    sigma_beliefs=sigma_beliefs if sigma_beliefs is not None else torch.ones_like(mu_beliefs),
                    prediction_errors=ce_per_position,
                )

                # Collect prior stats from last layer
                if layer_idx == len(model.transformer.blocks) - 1:
                    if hasattr(block.ffn, 'get_prior_stats'):
                        prior_stats = block.ffn.get_prior_stats()

        # P-FLOW: Update INPUT embeddings toward posteriors (beliefs)
        # With untied embeddings:
        # - Input embeddings (priors) get updated here
        embed_stats = update_input_embeddings_pflow(
            model=model,
            input_ids=input_ids,
            mu_beliefs=mu_beliefs,
            prediction_errors=ce_per_position,
            lr=0.1,
        )

        # P-FLOW: Update OUTPUT embeddings toward beliefs that predict targets
        # CRITICAL: With untied embeddings, output projection must also learn!
        # Otherwise logits stay random because W_out never changes.
        out_embed_stats = update_output_embeddings_pflow(
            model=model,
            targets=targets,
            mu_beliefs=mu_beliefs,
            prediction_errors=ce_per_position,
            lr=0.1,
        )
        embed_stats.update(out_embed_stats)

    # Compute metrics
    perplexity = torch.exp(ce_loss).item()

    metrics = {
        'loss/ce': ce_loss.item(),
        'loss/total': ce_loss.item(),  # No other loss terms in pure FEP
        'perplexity': perplexity,
        'attention/beta_mean': attn_info['beta'].mean().item(),
        'attention/kl_mean': attn_info['kl'].mean().item(),
        **{f'prior/{k}': v for k, v in prior_stats.items()},
        **{f'embed/{k}': v for k, v in embed_stats.items()},
    }

    return metrics


def pure_fep_validate(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate model in pure FEP mode.

    Note: Validation does NOT update priors - just measures performance.

    Args:
        model: GaugeTransformerLM
        dataloader: Validation data
        device: Target device

    Returns:
        Dict of validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch (tuple format from dataloader)
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward WITHOUT targets for fair evaluation
            # (beliefs don't get to see answers during eval)
            logits, _ = model.forward_with_attention(input_ids, targets=None)

            # Compute CE loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum',
                ignore_index=-100,
            )

            valid_tokens = (targets != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)

    return {
        'val/loss': avg_loss,
        'val/perplexity': perplexity,
    }


# NOTE: PureFEPConfig and PureFEPTrainer have been moved to transformer/experimental/
# For pure FEP training, use:
#   from transformer.experimental.pure_fep_transformer import PureFEPConfig, PureFEPTrainer
# Or use the pure_fep_train_step() and pure_fep_validate() functions above directly.