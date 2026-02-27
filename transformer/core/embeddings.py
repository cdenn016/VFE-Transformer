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

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


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
            # Clamp in log-space to preserve gradients (clamping exp() kills grads at boundary)
            log_sigma_clamped = self.base_log_sigma_diag.clamp(min=math.log(0.01), max=math.log(5.0))
            sigma_diag_base = torch.exp(log_sigma_clamped)  # (K,)
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
                # Per-token covariance — clamp in log-space to preserve gradients
                log_sigma = self.log_sigma_diag[token_ids]  # (B, N, K)
                log_sigma = log_sigma.clamp(min=math.log(0.01), max=math.log(5.0))
                sigma_diag = torch.exp(log_sigma)  # (B, N, K)
            else:
                # Shared covariance — clamp in log-space to preserve gradients
                log_sigma = self.log_sigma_diag.clamp(min=math.log(0.01), max=math.log(5.0))
                sigma_diag = torch.exp(log_sigma)  # (K,)
                sigma_diag = sigma_diag.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
                sigma_diag = sigma_diag.expand(batch_size, num_agents, -1)  # (B, N, K)

            if self.diagonal_covariance:
                # Keep as diagonal variances (B, N, K)
                sigma = sigma_diag
            else:
                # Convert to full covariance matrices (diagonal)
                sigma = torch.diag_embed(sigma_diag)  # (B, N, K, K)

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
            f"gauge_fixed_priors={self.gauge_fixed_priors}"
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

    # Parameter count
    total_params = sum(p.numel() for p in embedder.parameters())
    print(f"\n[5] Parameter count:")
    print(f"    Token embedder: {total_params:,} parameters")

    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)
