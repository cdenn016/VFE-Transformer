# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:34:45 2025

@author: chris and christine
"""

"""
Gauge-Theoretic Token Embeddings (0D Transformer)
==================================================

Maps discrete tokens → agent beliefs (μ_i, Σ_i, φ_i) at single base manifold point c*.

Key Insight from plan.py:
    "0D Transformer: All N tokens → N agents at the SAME base point c*
     Each token i → (μ_i, Σ_i, φ_i) where:
     - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
     - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
     - φ_i ∈ gl(K): gauge frame (Lie algebra element)"

GL(K) Gauge Structure (NEW):
    The VFE is invariant under GL(K) gauge transformations, not just SO(K)!
    This is because f-divergences are invariant under pushforward:
        D_KL(Ω·P || Ω·Q) = D_KL(P || Q) for any Ω ∈ GL(K)

    Parameterization options:
    - phi_dim=3: so(3) subalgebra (3 generators, rotation-only)
    - phi_dim=K(K-1)/2: so(K) subalgebra (skew-symmetric, orthogonal)
    - phi_dim=K²: gl(K) full algebra (all K×K matrices, maximum flexibility)

Author: Implementation from plan.py
Date: November 2025
Updated: GL(K) generalization - February 2026
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Import SO(N) BCH composition for proper Lie group operations
try:
    from math_utils.generators import soN_compose_bch_torch
    SON_BCH_AVAILABLE = True
except ImportError:
    SON_BCH_AVAILABLE = False


class GaugeTokenEmbedding(nn.Module):
    """
    Map discrete tokens to gauge-equivariant agent beliefs at single point.

    0D Transformer: All N tokens → N agents at the SAME base point c*
    Each token i → (μ_i, Σ_i, φ_i) where:
    - μ_i ∈ ℝ^K: mean belief vector (NO spatial dependence)
    - Σ_i ∈ SPD(K): covariance (scalar matrix per agent)
    - φ_i ∈ gl(K): gauge frame (Lie algebra element)

    GL(K) Gauge Structure:
        The gauge group can be SO(K) (orthogonal) or GL(K) (general linear).
        For VFE-based attention, GL(K) is sufficient because f-divergences
        are invariant under all invertible linear transformations.

        Parameterization:
        - phi_dim=3: so(3) subalgebra (rotation-only, legacy)
        - phi_dim=K(K-1)/2: so(K) subalgebra (orthogonal transformations)
        - phi_dim=K²: gl(K) full algebra (maximum flexibility)

    Architecture:
        token_id → [Embedding Layer] → (μ, Σ, φ)

        where:
        - μ: Learnable embedding (standard)
        - Σ: Initialized to small isotropic (σ²I), optionally learnable
        - φ: Initialized near zero (near-identity gauge frame)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        irrep_spec: list = None,
        init_std: float = None,  # Default: 2.0 for sharper attention
        init_sigma_scale: float = 1.0,  # Scaled to match init_std for O(1) KL
        learnable_sigma: bool = False,
        learnable_phi: bool = False,
        gauge_fixed_priors: bool = False,
        generators: Optional[torch.Tensor] = None,
        diagonal_covariance: bool = False,
        max_seq_len: int = 2048,
        use_positional_embedding: bool = False,
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        phi_scale: float = 0.3,  # Target ||φ|| norm for gauge frame initialization
        # Mean embedding normalization options
        mu_normalize: bool = False,  # If True, project μ to unit sphere
        mu_max_norm: Optional[float] = None,  # If set, clamp ||μ|| ≤ max_norm
    ):
        """
        Initialize gauge token embedding.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension K (fiber dimension)
            irrep_spec: List of (label, multiplicity, dim) for SO(3)/SO(N) irreps
            init_std: Std dev for initializing mean embeddings
            init_sigma_scale: Initial scale for covariance (σ in σ²I)
            learnable_sigma: If True, Σ evolves during training
            learnable_phi: If True, φ evolves during training
            gauge_fixed_priors: If True, priors are defined as GL(K) transformations of a
                               single base prior: p_i = G_i ▷ p_0. This guarantees
                               gauge covariance: p_i = Ω_ij[p_j] where Ω_ij = G_i G_j^{-1}.
                               Requires generators for computing transformations.
            generators: Lie algebra generators (n_gen, K, K), required if gauge_fixed_priors=True
            diagonal_covariance: If True, output sigma as (B,N,K) diagonal variances
                                instead of (B,N,K,K) full matrices. Saves O(K) memory.
            max_seq_len: Maximum sequence length for positional embeddings
            use_positional_embedding: If True, add learnable positional embeddings to μ
                                     (like standard transformers). This provides position
                                     info in the content while keeping gauge transport Ω_ij.
            phi_dim: Dimension of gauge frame φ. Options:
                    - 3: so(3) subalgebra (rotation-only, legacy)
                    - K(K-1)/2: so(K) subalgebra (orthogonal transformations)
                    - K²: gl(K) full algebra (maximum flexibility for GL(K) gauge)
            phi_scale: Target ||φ|| norm for gauge frame initialization. Higher values
                      (e.g., 1.0-2.0) encourage semantic clustering in gauge frames.
        """
        super().__init__()
        self.phi_scale = phi_scale
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.irrep_spec = irrep_spec
        self.learnable_sigma = learnable_sigma
        self.learnable_phi = learnable_phi
        self.gauge_fixed_priors = gauge_fixed_priors

        # Embedding initialization scale
        # OLD: 1/sqrt(K) keeps ||μ||² = O(1) but makes all embeddings equidistant!
        # NEW: Larger init_std creates more variance in pairwise distances,
        #      enabling sharper KL-based attention from the start.
        if init_std is None:
            init_std = 2.0  # Was: 1.0 / np.sqrt(embed_dim) ≈ 0.15 for K=40
        self.init_std = init_std

        # Mean embedding normalization options
        self.mu_normalize = mu_normalize
        self.mu_max_norm = mu_max_norm

        # CRITICAL: diagonal_covariance is incompatible with gauge_fixed_priors!
        # When gauge_fixed_priors=True, Σ_i = R_i Σ_0 R_i^T produces a FULL matrix
        # even if Σ_0 is diagonal. Extracting only diagonal loses correlations
        # induced by rotation, breaking gauge covariance: Σ_i ≠ Ω_ij Σ_j Ω_ij^T
        if gauge_fixed_priors and diagonal_covariance:
            import warnings
            warnings.warn(
                "gauge_fixed_priors=True is incompatible with diagonal_covariance=True. "
                "Rotation R_i Σ_0 R_i^T produces full matrices. "
                "Forcing diagonal_covariance=False to preserve gauge covariance.",
                UserWarning
            )
            diagonal_covariance = False

        self.diagonal_covariance = diagonal_covariance
        self.phi_dim = phi_dim

        if gauge_fixed_priors and generators is None:
            raise ValueError("gauge_fixed_priors=True requires generators to be provided")

        if generators is not None:
            self.register_buffer('generators', generators)

        # =================================================================
        # Mean Embeddings μ_i (or base prior μ_0 if gauge_fixed_priors)
        # =================================================================
        if gauge_fixed_priors:
            # Single base prior mean μ_0 - all token priors are rotations of this
            self.base_mu = nn.Parameter(torch.randn(embed_dim) * init_std)
        else:
            # Standard learnable embedding: vocab_size × embed_dim
            self.mu_embed = nn.Embedding(vocab_size, embed_dim)
            nn.init.normal_(self.mu_embed.weight, mean=0.0, std=init_std)

        # =================================================================
        # Covariance Embeddings Σ_i (or base prior Σ_0 if gauge_fixed_priors)
        # =================================================================
        # Parameterize via log-diagonal (ensures positivity):
        #   Σ = diag(exp(log_σ_diag))
        #
        # This is a simplified SPD parametrization. Future: full Cholesky.

        if gauge_fixed_priors:
            # Single base prior covariance Σ_0 - all token priors are rotations of this
            self.base_log_sigma_diag = nn.Parameter(
                torch.full((embed_dim,), np.log(init_sigma_scale))
            )
        elif learnable_sigma:
            # Per-token covariance
            self.log_sigma_diag = nn.Parameter(
                torch.full((vocab_size, embed_dim), np.log(init_sigma_scale))
            )
        else:
            # Shared isotropic covariance across all tokens
            self.register_buffer(
                'log_sigma_diag',
                torch.full((embed_dim,), np.log(init_sigma_scale))
            )

        # =================================================================
        # Gauge Frame Embeddings φ_i ∈ so(n)
        # =================================================================
        # CRITICAL: With gauge_fixed_priors=True, φ defines the token embedding!
        # μ_i = R(φ_i) @ μ_base, so different φ = different embeddings.
        # Zero init would make ALL tokens identical - must use random init!

        if learnable_phi or gauge_fixed_priors:
            # Per-token gauge frame
            self.phi_embed = nn.Embedding(vocab_size, phi_dim)  # so(n) has phi_dim components
            # RANDOM init for non-trivial gauge structure from the start!
            # NOTE: Random init is required BOTH for gauge_fixed_priors=True (where phi
            # defines token identity) AND for gauge_fixed_priors=False (where phi
            # provides gauge structure via transport Ω_ij). Zero init makes Ω=I,
            # completely disabling the gauge-theoretic attention mechanism!
            # Phi will grow during training via learnable_phi=True.
            #
            # IMPORTANT: Scale std inversely with sqrt(phi_dim) to maintain consistent
            # norm across different SO(N) dimensions. Target ||φ|| ≈ phi_scale regardless of N.
            # - SO(3): phi_dim=3, phi_scale=1.0 → std=0.577, ||φ||≈1.0
            # - SO(50): phi_dim=1225, phi_scale=1.0 → std=0.029, ||φ||≈1.0
            # - SO(100): phi_dim=4950, phi_scale=1.0 → std=0.014, ||φ||≈1.0
            phi_init_std = phi_scale / (phi_dim ** 0.5)
            nn.init.normal_(self.phi_embed.weight, mean=0.0, std=phi_init_std)
        else:
            # All tokens start at identity frame
            self.register_buffer('phi_base', torch.zeros(phi_dim))

        # =================================================================
        # Positional Embeddings (optional, like standard transformers)
        # =================================================================
        self.use_positional_embedding = use_positional_embedding
        self.max_seq_len = max_seq_len

        if use_positional_embedding:
            # Learnable positional embeddings added to μ
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=init_std)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed tokens as agent beliefs at single base manifold point c*.

        Args:
            token_ids: (batch, seq_len) integer token indices

        Returns:
            mu: (batch, num_agents, K) mean beliefs (one per agent, NOT per spatial point)
            sigma: (batch, num_agents, K, K) covariances if diagonal_covariance=False
                   (batch, num_agents, K) diagonal variances if diagonal_covariance=True
            phi: (batch, num_agents, phi_dim) gauge frames (one per agent)
                 phi_dim = 3 for SO(3), N(N-1)/2 for SO(N)

        NOTE: seq_len = number of agents at the single point c*
              This is NOT a spatial dimension!

        When gauge_fixed_priors=True:
            Priors are computed as p_i = R_i ▷ p_0 where R_i = exp(φ_i · generators).
            This guarantees p_i = Ω_ij[p_j] for all i,j, restoring gauge covariance.
        """
        batch_size, num_agents = token_ids.shape

        # =================================================================
        # Gauge Frame Embeddings (computed first for gauge_fixed_priors)
        # =================================================================
        if self.learnable_phi or self.gauge_fixed_priors:
            # Per-token gauge frame
            phi = self.phi_embed(token_ids)  # (B, N, 3)
        else:
            # All agents at identity frame
            phi = self.phi_base.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
            phi = phi.expand(batch_size, num_agents, -1)  # (B, N, 3)

        # =================================================================
        # Mean and Covariance Embeddings
        # =================================================================
        if self.gauge_fixed_priors:
            # Compute rotation matrices R_i = exp(φ_i · generators)
            # phi: (B, N, 3), generators: (3, K, K)
            phi_matrix = torch.einsum('bnc,ckl->bnkl', phi, self.generators)  # (B, N, K, K)
            R = torch.linalg.matrix_exp(phi_matrix)  # (B, N, K, K)

            # Rotate base prior mean: μ_i = R_i @ μ_0
            # base_mu: (K,), R: (B, N, K, K)
            mu = torch.einsum('bnkl,l->bnk', R, self.base_mu)  # (B, N, K)

            # Build base covariance Σ_0 = diag(exp(log_σ_0))
            sigma_diag_base = torch.exp(self.base_log_sigma_diag)  # (K,)
            # STABILITY: Clamp to prevent singular matrices in deep networks
            sigma_diag_base = torch.clamp(sigma_diag_base, min=0.01, max=5.0)
            Sigma_0 = torch.diag(sigma_diag_base)  # (K, K)

            # Rotate base prior covariance: Σ_i = R_i @ Σ_0 @ R_i^T
            # R: (B, N, K, K), Sigma_0: (K, K)
            # NOTE: diagonal_covariance is forced False when gauge_fixed_priors=True
            # (see __init__), so we always output full matrices here.
            sigma = torch.einsum('bnij,jk,bnlk->bnil', R, Sigma_0, R)  # (B, N, K, K)
        else:
            # Standard per-token embeddings
            # μ(token_i) for each agent i at c*
            mu = self.mu_embed(token_ids)  # (B, N, K) where N = num_agents

            # Build diagonal covariances: Σ = diag(exp(log_σ))
            if self.learnable_sigma:
                # Per-token covariance
                log_sigma = self.log_sigma_diag[token_ids]  # (B, N, K)
                sigma_diag = torch.exp(log_sigma)  # (B, N, K)
                # STABILITY: Clamp to prevent singular matrices in deep networks
                sigma_diag = torch.clamp(sigma_diag, min=0.01, max=5.0)
            else:
                # Shared covariance
                sigma_diag = torch.exp(self.log_sigma_diag)  # (K,)
                # STABILITY: Clamp to prevent singular matrices in deep networks
                sigma_diag = torch.clamp(sigma_diag, min=0.01, max=5.0)
                sigma_diag = sigma_diag.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
                sigma_diag = sigma_diag.expand(batch_size, num_agents, -1)  # (B, N, K)

            if self.diagonal_covariance:
                # Keep as diagonal variances (B, N, K)
                sigma = sigma_diag
            else:
                # Convert to full covariance matrices (diagonal)
                sigma = torch.diag_embed(sigma_diag)  # (B, N, K, K)

        # =================================================================
        # Add positional embeddings to μ (like standard transformers)
        # =================================================================
        if self.use_positional_embedding:
            # Create position indices [0, 1, 2, ..., N-1]
            positions = torch.arange(num_agents, device=token_ids.device)  # (N,)
            pos_emb = self.pos_embed(positions)  # (N, K)
            mu = mu + pos_emb.unsqueeze(0)  # (B, N, K) + (1, N, K) -> (B, N, K)

        # =================================================================
        # Apply μ normalization/clamping (for sharper KL-based attention)
        # =================================================================
        if self.mu_normalize:
            # Project to unit sphere: ||μ|| = 1
            mu = torch.nn.functional.normalize(mu, dim=-1)
        elif self.mu_max_norm is not None:
            # Clamp norm: ||μ|| ≤ max_norm (like gradient clipping for embeddings)
            mu_norm = mu.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.clamp(self.mu_max_norm / mu_norm, max=1.0)
            mu = mu * scale

        return mu, sigma, phi

    def extra_repr(self) -> str:
        """Pretty print for model summary."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable_sigma={self.learnable_sigma}, "
            f"learnable_phi={self.learnable_phi}, "
            f"gauge_fixed_priors={self.gauge_fixed_priors}, "
            f"use_positional_embedding={self.use_positional_embedding}"
        )

    # =========================================================================
    # P-FLOW: EMA update of token embeddings toward successful beliefs
    # =========================================================================
    def update_embeddings_from_beliefs(
        self,
        token_ids: torch.Tensor,           # (B, N) token IDs in this batch
        mu_beliefs: torch.Tensor,          # (B, N, K) final beliefs after VFE
        prediction_errors: torch.Tensor,   # (B, N) per-position CE loss
        ema_decay: float = 0.99,           # EMA decay rate (higher = slower update)
        min_weight: float = 0.01,          # Minimum weight to prevent dead tokens
    ):
        """
        P-flow: Update token embeddings toward successful beliefs via EMA.

        This is the key learning mechanism from fep_transformer.py:
        - Beliefs with low prediction error are "successful"
        - Token embeddings should drift toward successful beliefs
        - EMA provides stable, gradual updates

        Formula:
            μ_token ← (1 - η·w) · μ_token + η·w · μ_belief

        where w = softmax(-error) weights by prediction success.

        Args:
            token_ids: (B, N) token indices that were in this batch
            mu_beliefs: (B, N, K) final belief means after VFE dynamics
            prediction_errors: (B, N) per-position cross-entropy loss
            ema_decay: EMA decay rate (0.99 = slow, 0.9 = faster)
            min_weight: Minimum weight to ensure all tokens get some update
        """
        if self.gauge_fixed_priors:
            # For gauge_fixed_priors, we'd need to update phi_embed instead
            # This is more complex - skip for now
            return

        B, N, K = mu_beliefs.shape
        device = mu_beliefs.device
        lr = 1.0 - ema_decay  # Convert decay to learning rate

        with torch.no_grad():
            # Compute success weights from prediction errors
            # Low error = high weight (successful predictions should update more)
            errors_clamped = prediction_errors.clamp(min=1e-6, max=20.0)
            weights = torch.softmax(-errors_clamped, dim=-1)  # (B, N)
            weights = weights.clamp(min=min_weight)  # Ensure minimum update

            # For each unique token in batch, accumulate weighted belief updates
            # This handles repeated tokens correctly
            unique_tokens = token_ids.unique()

            for token_id in unique_tokens:
                # Find all occurrences of this token
                mask = (token_ids == token_id)  # (B, N)

                if mask.sum() == 0:
                    continue

                # Get beliefs and weights for this token
                token_beliefs = mu_beliefs[mask]  # (num_occurrences, K)
                token_weights = weights[mask]     # (num_occurrences,)

                # Weighted average belief for this token
                total_weight = token_weights.sum()
                weighted_belief = (token_beliefs * token_weights.unsqueeze(-1)).sum(dim=0) / total_weight

                # EMA update: prior ← (1 - lr) · prior + lr · belief
                current_embedding = self.mu_embed.weight[token_id]  # (K,)
                new_embedding = (1.0 - lr) * current_embedding + lr * weighted_belief
                self.mu_embed.weight[token_id] = new_embedding

    def get_embedding_stats(self) -> dict:
        """Get statistics about embeddings for logging."""
        with torch.no_grad():
            if hasattr(self, 'mu_embed'):
                mu_weight = self.mu_embed.weight
            elif hasattr(self, 'base_mu'):
                mu_weight = self.base_mu.unsqueeze(0)  # (1, K) for consistent stats
            else:
                return {}
            return {
                'embed_mu_mean': mu_weight.mean().item(),
                'embed_mu_std': mu_weight.std().item(),
                'embed_mu_norm_mean': mu_weight.norm(dim=-1).mean().item(),
            }


def so3_log_torch(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Logarithm map from SO(3) → so(3) (PyTorch version).

    Given R ∈ SO(3), find φ ∈ ℝ³ such that exp([φ]_×) = R.

    Formula:
        θ = arccos((tr(R) - 1) / 2)
        φ = (θ / (2 sin θ)) * vex(R - Rᵀ)

    Args:
        R: Rotation matrices, shape (..., 3, 3)
        eps: Threshold for small angle approximation

    Returns:
        phi: Lie algebra elements, shape (..., 3)
    """
    # Compute rotation angle from trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...)

    # Extract skew-symmetric part: vex(R - R^T) / 2
    # vex extracts [v_x, v_y, v_z] from skew-symmetric matrix
    skew = R - R.transpose(-1, -2)  # (..., 3, 3)
    v_x = (skew[..., 2, 1] - skew[..., 1, 2]) / 2.0
    v_y = (skew[..., 0, 2] - skew[..., 2, 0]) / 2.0
    v_z = (skew[..., 1, 0] - skew[..., 0, 1]) / 2.0
    vex_skew = torch.stack([v_x, v_y, v_z], dim=-1)  # (..., 3)

    # Coefficient: θ / (2 sin θ), handle small angles
    sin_theta = torch.sin(theta)
    # For small θ: θ/(2sinθ) ≈ 1/2 + θ²/12
    small_angle = theta < eps
    coeff = torch.where(
        small_angle,
        0.5 + theta**2 / 12.0,
        theta / (2.0 * sin_theta + eps)
    )

    phi = coeff.unsqueeze(-1) * vex_skew
    return phi


def so3_compose_bch(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """
    Compose two so(3) elements using Baker-Campbell-Hausdorff formula.

    log(exp(φ₁)·exp(φ₂)) = φ₁ + φ₂ + ½[φ₁,φ₂] + (1/12)[φ₁,[φ₁,φ₂]] - ...

    For so(3), the Lie bracket is: [X, Y] = X × Y (cross product)

    Args:
        phi1: First so(3) element, shape (..., 3)
        phi2: Second so(3) element, shape (..., 3)
        order: BCH expansion order (0=addition, 1=first correction, 2=second)

    Returns:
        phi_composed: Composed element in so(3), shape (..., 3)
    """
    if order == 0:
        # Simple addition (valid for small angles only)
        return phi1 + phi2

    # First-order BCH: φ₁ + φ₂ + ½[φ₁,φ₂]
    # In so(3): [φ₁,φ₂] = φ₁ × φ₂ (cross product)
    bracket_12 = torch.cross(phi1, phi2, dim=-1)
    result = phi1 + phi2 + 0.5 * bracket_12

    if order >= 2:
        # Second-order: + (1/12)[φ₁,[φ₁,φ₂]] - (1/12)[φ₂,[φ₁,φ₂]]
        bracket_1_12 = torch.cross(phi1, bracket_12, dim=-1)
        bracket_2_12 = torch.cross(phi2, bracket_12, dim=-1)
        result = result + (1.0/12.0) * bracket_1_12 - (1.0/12.0) * bracket_2_12

    return result


class GaugePositionalEncoding(nn.Module):
    """
    Agent-index-dependent gauge frame modulation (0D positional encoding).

    In 0D: Position encodes AGENT INDEX, not spatial location.
    All agents are at the same point c*, but need to distinguish
    their roles in the sequence via gauge frame modulation.

    Encoding modes:
        - 'none': No positional encoding in gauge space. Transport Ω_ij is purely
                  content-based. Use with use_positional_embedding=True to put
                  position info in μ instead, or for position-invariant attention.
        - 'learned': Learnable per-position gauge frames φ_pos[i] ∈ so(n)
        - 'sinusoidal': Fixed sinusoidal encoding (like original Transformer)

    Composition modes (for combining token φ with positional φ):
        - 'add': φ_combined = φ_base + φ_pos (valid for small angles)
        - 'bch1': φ_combined = φ_base + φ_pos + ½[φ_base, φ_pos] (BCH order 1) [SO(3) only]
        - 'bch2': Higher-order BCH correction [SO(3) only]
        - 'exact': Full SO(3) composition via exp → multiply → log [SO(3) only]

    WARNING: Positional encoding in gauge space creates ABSOLUTE position-dependent
    transport operators. This can cause attention to be dominated by position rather
    than content. For translation-invariant attention, use mode='none'.

    For SO(N) with N > 3, only 'add' composition is supported since the BCH formula
    requires Lie bracket computation which differs for so(N).
    """

    def __init__(
        self,
        max_seq_len: int,
        mode: str = 'none',  # Default: no positional encoding in gauge space
        scale: float = 0.1,
        composition: str = 'exact',  # Default: full SO(3) composition (most accurate)
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        generators: Optional[torch.Tensor] = None,  # Required for SO(N) BCH composition
    ):
        """
        Initialize positional encoding in gauge space.

        Args:
            max_seq_len: Maximum sequence length (max number of agents at c*)
            mode: 'none', 'learned', or 'sinusoidal'
                  - 'none': No positional gauge encoding (transport is content-only)
                  - 'learned': Learnable positional gauge frames
                  - 'sinusoidal': Fixed sinusoidal encoding
            scale: Scaling factor for positional encodings (ignored if mode='none')
            composition: How to combine token φ with positional φ:
                - 'add': Simple addition (φ_base + φ_pos) - fast but only valid for small angles
                - 'bch1': BCH order 1 correction (works for SO(3) and SO(N) with generators)
                - 'bch2': BCH order 2 correction (works for SO(3) and SO(N) with generators)
                - 'exact': Full SO(3) composition (SO(3) only)
            phi_dim: Dimension of gauge frame φ. 3 for SO(3), N(N-1)/2 for SO(N).
            generators: Lie algebra generators (n_gen, N, N). Required for SO(N) BCH when N > 3.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.scale = scale
        self.phi_dim = phi_dim

        # Store generators for SO(N) BCH composition
        if generators is not None:
            self.register_buffer('generators', generators)
        else:
            self.generators = None

        # For SO(N) with N > 3, BCH requires generators; 'exact' is SO(3)-only
        if phi_dim != 3:
            if composition == 'exact':
                print(f"[WARNING] Composition 'exact' only supported for SO(3) (phi_dim=3). "
                      f"Falling back to 'bch2' for phi_dim={phi_dim}.")
                composition = 'bch2'
            if composition in ['bch1', 'bch2'] and generators is None and not SON_BCH_AVAILABLE:
                print(f"[WARNING] SO(N) BCH requires generators. "
                      f"Falling back to 'add' for phi_dim={phi_dim}.")
                composition = 'add'
        self.composition = composition

        if mode == 'none':
            # No positional encoding in gauge space - φ stays content-only
            # Use this when position info comes from μ (use_positional_embedding)
            # or when you want purely content-based transport operators
            self.register_buffer('pos_phi', torch.zeros(max_seq_len, phi_dim))

        elif mode == 'learned':
            # Learnable agent-index-specific gauge biases
            # Each agent index i gets a unique φ_pos(i) ∈ so(n)
            self.pos_phi = nn.Parameter(torch.randn(max_seq_len, phi_dim) * scale)

        elif mode == 'sinusoidal':
            # Sinusoidal encoding projected to so(n)
            # Fixed (not learnable)
            self.register_buffer('pos_phi', self._make_sinusoidal(max_seq_len, scale, phi_dim))

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'none', 'learned', or 'sinusoidal'.")

    def _make_sinusoidal(self, max_len: int, scale: float, phi_dim: int = 3) -> torch.Tensor:
        """
        Create sinusoidal positional encoding in so(n).

        This encodes agent index i, not spatial position!

        Formula (adapted from Transformer):
            For each dimension d in [0, phi_dim):
            φ_pos[i, d] = scale * sin/cos(i / 10000^(d/phi_dim))

        Args:
            max_len: Maximum sequence length
            scale: Scaling factor
            phi_dim: Dimension of gauge frame (3 for SO(3), N(N-1)/2 for SO(N))

        Returns:
            pos_phi: (max_len, phi_dim) positional gauge frames
        """
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(torch.arange(0, phi_dim, 1, dtype=torch.float32) * -(np.log(10000.0) / phi_dim))

        phi = torch.zeros(max_len, phi_dim)
        for d in range(phi_dim):
            if d % 2 == 0:
                phi[:, d] = torch.sin(position.squeeze() * div_term[d])
            else:
                phi[:, d] = torch.cos(position.squeeze() * div_term[d])

        return phi * scale

    def forward(self, num_agents: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Get positional gauge frames for given number of agents.

        Args:
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            pos_phi: (num_agents, phi_dim) agent-index-dependent gauge frames

        NOTE: This is NOT a spatial field! Just one φ per agent index.
        """
        if num_agents > self.max_seq_len:
            raise ValueError(
                f"Sequence length {num_agents} exceeds max {self.max_seq_len}. "
                f"Increase max_seq_len in config."
            )

        pos_phi = self.pos_phi[:num_agents]  # (N, phi_dim)

        if device is not None:
            pos_phi = pos_phi.to(device)

        return pos_phi

    def compose(
        self,
        phi: torch.Tensor,
        num_agents: int,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compose token gauge frames with positional gauge frames using proper Lie group composition.

        For SO(3): Uses BCH formula or exact composition.
        For SO(N) with N > 3: Uses simple addition (valid for small angles).

        Args:
            phi: Token gauge frames, shape (B, N, phi_dim)
            num_agents: Number of agents (sequence length)
            device: Device to place output on

        Returns:
            phi_combined: Composed gauge frames, shape (B, N, phi_dim)

        Mathematical background:
            In SO(3), the correct composition is R_combined = exp(φ) · exp(φ_pos)
            In so(3), this is NOT simply φ + φ_pos. The BCH formula gives:
            log(exp(X)·exp(Y)) = X + Y + ½[X,Y] + (1/12)[X,[X,Y]] - (1/12)[Y,[X,Y]] + ...
            For so(3), the Lie bracket is the cross product: [X,Y] = X × Y

            For SO(N) with N > 3, the Lie bracket is [X,Y] = XY - YX, which is more
            complex. We use simple addition for these cases (valid for small angles).
        """
        # Short-circuit: if mode='none', φ_pos is all zeros, so return unchanged φ
        # This avoids unnecessary tensor operations and BCH computations
        if self.mode == 'none':
            return phi

        pos_phi = self.forward(num_agents, device)  # (N, phi_dim)
        pos_phi = pos_phi.unsqueeze(0).expand(phi.shape[0], -1, -1)  # (B, N, phi_dim)

        if self.composition == 'add':
            # Simple addition (original behavior, valid for small angles only)
            return phi + pos_phi

        elif self.composition == 'bch1':
            # First-order BCH correction
            if self.phi_dim == 3:
                # Use SO(3)-specific BCH (cross product)
                return so3_compose_bch(phi, pos_phi, order=1)
            elif SON_BCH_AVAILABLE and self.generators is not None:
                # Use general SO(N) BCH (matrix commutator)
                return soN_compose_bch_torch(phi, pos_phi, self.generators, order=1)
            else:
                # Fallback to addition
                return phi + pos_phi

        elif self.composition == 'bch2':
            # Second-order BCH correction
            if self.phi_dim == 3:
                # Use SO(3)-specific BCH (cross product)
                return so3_compose_bch(phi, pos_phi, order=2)
            elif SON_BCH_AVAILABLE and self.generators is not None:
                # Use general SO(N) BCH (matrix commutator)
                return soN_compose_bch_torch(phi, pos_phi, self.generators, order=2)
            else:
                # Fallback to addition
                return phi + pos_phi

        elif self.composition == 'exact':
            # Full SO(3) composition: log(exp(φ) · exp(φ_pos))
            # Build skew-symmetric matrices and exponentiate
            # [φ]_× for so(3) → SO(3)
            def skew_symmetric_batch(v):
                """v: (..., 3) -> (..., 3, 3) skew-symmetric"""
                zeros = torch.zeros_like(v[..., 0])
                return torch.stack([
                    torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1),
                    torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1),
                    torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1),
                ], dim=-2)

            phi_skew = skew_symmetric_batch(phi)  # (B, N, 3, 3)
            pos_phi_skew = skew_symmetric_batch(pos_phi)  # (B, N, 3, 3)

            R_phi = torch.linalg.matrix_exp(phi_skew)  # (B, N, 3, 3)
            R_pos = torch.linalg.matrix_exp(pos_phi_skew)  # (B, N, 3, 3)

            R_combined = R_phi @ R_pos  # (B, N, 3, 3)

            # Map back to so(3) via logarithm
            phi_combined = so3_log_torch(R_combined)  # (B, N, 3)
            return phi_combined

        else:
            raise ValueError(f"Unknown composition mode: {self.composition}")

    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, mode={self.mode}, scale={self.scale}, composition={self.composition}"


# =============================================================================
# Testing & Visualization
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TOKEN EMBEDDING TEST (0D Transformer)")
    print("="*70)

    # Test configuration
    vocab_size = 100
    embed_dim = 32
    batch_size = 4
    seq_len = 10

    # Create embedding layer
    print("\n[1] Creating GaugeTokenEmbedding...")
    embedder = GaugeTokenEmbedding(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        init_std=0.02,
        init_sigma_scale=0.1,
        learnable_sigma=False,  # Start simple
        learnable_phi=False,
    )
    print(embedder)

    # Create random tokens
    print(f"\n[2] Embedding random tokens...")
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"    Token IDs shape: {token_ids.shape}")

    # Forward pass
    mu, sigma, phi = embedder(token_ids)

    print(f"\n[3] Output shapes:")
    print(f"    μ (means):      {mu.shape}     # (B, N, K) where N=num_agents at c*")
    print(f"    Σ (covariances): {sigma.shape}   # (B, N, K, K)")
    print(f"    φ (gauge frames): {phi.shape}      # (B, N, 3) in so(3)")

    # Validate covariance is SPD
    print(f"\n[4] Validating covariances...")
    eigenvalues = torch.linalg.eigvalsh(sigma[0, 0])  # Check first agent
    print(f"    Eigenvalues of Σ[0,0]: {eigenvalues.numpy()}")
    assert torch.all(eigenvalues > 0), "Covariance not positive definite!"
    print("    ✓ All eigenvalues positive (SPD verified)")

    # Test positional encoding
    print(f"\n{'='*70}")
    print("GAUGE POSITIONAL ENCODING TEST")
    print('='*70)

    max_seq_len = 64

    # Test learned encoding
    print("\n[5] Testing learned positional encoding...")
    pos_enc_learned = GaugePositionalEncoding(max_seq_len, mode='learned', scale=0.1)
    pos_phi_learned = pos_enc_learned(seq_len)
    print(f"    Learned φ_pos shape: {pos_phi_learned.shape}  # (N, 3)")
    print(f"    φ_pos[0]: {pos_phi_learned[0].detach().numpy()}")
    print(f"    φ_pos[9]: {pos_phi_learned[9].detach().numpy()}")

    # Test sinusoidal encoding
    print("\n[6] Testing sinusoidal positional encoding...")
    pos_enc_sin = GaugePositionalEncoding(max_seq_len, mode='sinusoidal', scale=0.1)
    pos_phi_sin = pos_enc_sin(seq_len)
    print(f"    Sinusoidal φ_pos shape: {pos_phi_sin.shape}")
    print(f"    φ_pos[0]: {pos_phi_sin[0].numpy()}")
    print(f"    φ_pos[9]: {pos_phi_sin[9].numpy()}")

    # Combined: Embedding + Position
    print(f"\n[7] Combined embedding with positional encoding...")
    phi_combined = phi + pos_phi_learned.unsqueeze(0)  # (B, N, 3)
    print(f"    φ_total = φ_base + φ_pos: {phi_combined.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"\n[8] Parameter count:")
    print(f"    Token embedder: {total_params:,} parameters")
    pos_params = sum(p.numel() for p in pos_enc_learned.parameters())
    print(f"    Position encoder (learned): {pos_params:,} parameters")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
