# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 17:06:07 2025

@author: chris and christine
"""

"""
Token-Dependent Prior Bank for VFE Transformers
================================================

Unified prior bank that serves as BOTH embedding and output projection.
Each vocabulary token v has a prior belief distribution: π_v = N(μ_v, Σ_v)

This module enables:
1. ENCODING: Initialize beliefs from token priors (replaces nn.Embedding)
2. DECODING: Compute logits via KL to priors (replaces nn.Linear)
3. LEARNING: Update priors via prediction-error weighted EMA (pure FEP mode)

Author: Extracted from pure_fep_transformer.py for reusability
Date: December 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from transformer.core.sanitization import san


class PriorBank(nn.Module):
    """
    Token-dependent prior bank for pure FEP learning.

    Each vocabulary token v has a prior belief distribution:
        π_v = N(μ_v, Σ_v)

    GAUGE-FIXED PRIORS (default, theoretically principled):
        All token priors are rotations of a SINGLE base prior:
            π_v = R_v ▷ π_0   where R_v = exp(φ_v · G)

        This guarantees gauge covariance: π_i = Ω_ij[π_j] for all i,j
        The model learns:
        - base_prior_mu (K,): shared base prior mean
        - phi_embed (V, phi_dim): per-token gauge frames defining rotations

    NON-GAUGE-FIXED (backward compatibility):
        Each token has independent μ_v, Σ_v - breaks gauge covariance but
        allows more flexibility for pure FEP learning.

    ENCODING (replaces nn.Embedding):
        Given input token y_t, initialize belief from prior:
        q(z_t) ← π_{y_t}

    DECODING (replaces nn.Linear output projection):
        Given belief q = N(μ_q, Σ_q), compute observation likelihood:
        p(y = v | q) ∝ exp(-KL(q || π_v) / τ)

    Learning:
    - In pure FEP mode: Priors evolve via slow VFE pressure
    - In hybrid mode: Priors can be updated via backprop
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        init_std: float = None,  # Default: 1/sqrt(embed_dim) for O(1) KL
        init_sigma_scale: float = 1.0,  # Scaled to match init_std for O(1) KL
        learnable_sigma: bool = True,
        eps: float = 1e-6,
        # Gauge-fixed priors (principled approach)
        gauge_fixed_priors: bool = False,  # Default False for pure FEP flexibility
        generators: Optional[torch.Tensor] = None,  # (n_gen, K, K) Lie algebra generators
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
    ):
        """
        Initialize the prior bank.

        Args:
            vocab_size: Number of tokens in vocabulary
            embed_dim: Embedding dimension K
            init_std: Std for initializing prior means (default: 1/sqrt(embed_dim))
            init_sigma_scale: Initial scale for prior variances
            learnable_sigma: If True, Σ_v evolves during training
            eps: Numerical stability
            gauge_fixed_priors: If True, use single base prior with per-token rotations
            generators: Lie algebra generators for computing rotations (required if gauge_fixed_priors=True)
            phi_dim: Dimension of gauge frame (3 for SO(3))
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.eps = eps
        self.learnable_sigma = learnable_sigma
        self.gauge_fixed_priors = gauge_fixed_priors
        self.phi_dim = phi_dim

        # Dimension-aware initialization: 1/sqrt(K) keeps ||μ||² = O(1)
        if init_std is None:
            init_std = 1.0 / math.sqrt(embed_dim)

        if gauge_fixed_priors:
            # Validate generators
            if generators is None:
                raise ValueError("gauge_fixed_priors=True requires generators to be provided")
            self.register_buffer('generators', generators)

            # Single base prior mean μ_0 - all token priors are rotations of this
            self.base_prior_mu = nn.Parameter(torch.randn(embed_dim) * init_std)

            # Single base prior variance (diagonal)
            if learnable_sigma:
                self.base_log_prior_sigma = nn.Parameter(
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'base_log_prior_sigma',
                    torch.full((embed_dim,), math.log(init_sigma_scale))
                )

            # Per-token gauge frames φ_v ∈ so(n) - defines rotation R_v = exp(φ_v · G)
            self.phi_embed = nn.Embedding(vocab_size, phi_dim)
            nn.init.zeros_(self.phi_embed.weight)  # Start at identity rotation
        else:
            # Standard per-token priors (TOKEN-DEPENDENT, not position-dependent!)
            self.prior_mu = nn.Parameter(torch.randn(vocab_size, embed_dim) * init_std)

            if learnable_sigma:
                self.log_prior_sigma = nn.Parameter(
                    torch.full((vocab_size, embed_dim), math.log(init_sigma_scale))
                )
            else:
                self.register_buffer(
                    'log_prior_sigma',
                    torch.full((vocab_size, embed_dim), math.log(init_sigma_scale))
                )

    @property
    def base_prior_sigma(self) -> torch.Tensor:
        """Get base prior variance (always positive). Only for gauge_fixed_priors=True."""
        return torch.exp(self.base_log_prior_sigma).clamp(min=self.eps)

    @property
    def prior_sigma(self) -> torch.Tensor:
        """Get prior variances (always positive). Only for gauge_fixed_priors=False."""
        return torch.exp(self.log_prior_sigma).clamp(min=self.eps)

    def _compute_rotation(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Compute rotation matrix R = exp(φ · G) from gauge frame.

        Args:
            phi: (..., phi_dim) gauge frames

        Returns:
            R: (..., K, K) rotation matrices
        """
        # Compute φ · G = Σ_a φ_a G_a
        # generators: (n_gen, K, K), phi: (..., phi_dim)
        # Result: (..., K, K)
        phi_dot_G = torch.einsum('...a,aij->...ij', phi, self.generators)

        # Matrix exponential
        R = torch.linalg.matrix_exp(phi_dot_G)
        return R

    def _get_prior_for_tokens(
        self,
        token_ids: torch.Tensor,  # (B, N) or (V,) for all vocab
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prior (μ, σ, φ) for given tokens.

        CRITICAL: Indexing by TOKEN ID, not position!

        Returns:
            mu_p: prior means
            sigma_p: prior variances (diagonal)
            phi_p: gauge frames
        """
        if self.gauge_fixed_priors:
            # Get per-token gauge frames
            phi = self.phi_embed(token_ids)  # (..., phi_dim)

            # Compute rotation matrices R_v = exp(φ_v · G)
            R = self._compute_rotation(phi)  # (..., K, K)

            # Rotate base prior: μ_v = R_v @ μ_0
            mu_p = torch.einsum('...ij,j->...i', R, self.base_prior_mu)

            # Rotate base covariance (diagonal approximation):
            # For diagonal Σ_0, the rotated diagonal is (R @ diag(σ_0) @ R^T)_kk = Σ_l R_kl² σ_0[l]
            base_sigma = self.base_prior_sigma  # (K,)
            R_sq = R ** 2  # (..., K, K)
            sigma_p = torch.einsum('...kl,l->...k', R_sq, base_sigma)  # (..., K)

            return mu_p, sigma_p, phi
        else:
            # Standard per-token lookup (TOKEN-INDEXED!)
            mu_p = self.prior_mu[token_ids]  # Index by token ID
            sigma_p = self.prior_sigma[token_ids]
            # Return zero phi for non-gauge-fixed mode
            phi = torch.zeros(*token_ids.shape, self.phi_dim, device=token_ids.device)
            return mu_p, sigma_p, phi

    def encode(
        self,
        token_ids: torch.Tensor,  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode tokens by looking up their prior beliefs.

        This replaces the standard nn.Embedding lookup.

        Args:
            token_ids: (B, N) input token IDs

        Returns:
            mu_q: (B, N, K) belief means initialized from priors
            sigma_q: (B, N, K) belief variances initialized from priors
            phi: (B, N, phi_dim) gauge frames for tokens
        """
        return self._get_prior_for_tokens(token_ids)

    def decode(
        self,
        mu_q: torch.Tensor,      # (B, N, K) belief means
        sigma_q: torch.Tensor,   # (B, N, K) belief variances (diagonal)
        tau: float = 1.0,        # Temperature
    ) -> torch.Tensor:
        """
        Compute observation likelihood via KL to all token priors.

        p(y = v | q) ∝ exp(-KL(q || π_v) / τ)

        This is the PRINCIPLED output model - no learned W_out needed!

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances
            tau: Temperature for softmax

        Returns:
            logits: (B, N, vocab_size) log-probabilities (unnormalized)
        """
        B, N, K = mu_q.shape
        V = self.vocab_size
        device = mu_q.device

        # Get all token priors
        all_token_ids = torch.arange(V, device=device)
        mu_p, sigma_p, _ = self._get_prior_for_tokens(all_token_ids)  # (V, K)

        # Expand for broadcasting:
        # mu_q: (B, N, K) -> (B, N, 1, K)
        # mu_p: (V, K) -> (1, 1, V, K)
        mu_q_exp = mu_q.unsqueeze(2)  # (B, N, 1, K)
        sigma_q_exp = sigma_q.unsqueeze(2)  # (B, N, 1, K)
        mu_p_exp = mu_p.unsqueeze(0).unsqueeze(0)  # (1, 1, V, K)
        sigma_p_exp = sigma_p.unsqueeze(0).unsqueeze(0)  # (1, 1, V, K)

        # Compute KL(q || π_v) for all v
        # For diagonal Gaussians:
        # KL = 0.5 * (Σ_q/Σ_p + (μ_q-μ_p)²/Σ_p - 1 + log(Σ_p/Σ_q))
        variance_floor = 1e-6
        n_clamped = int((sigma_q_exp < variance_floor).sum().item()) + int((sigma_p_exp < variance_floor).sum().item())
        if n_clamped > 0:
            san.record('sigma_clamp', count=n_clamped)
        sigma_q_safe = sigma_q_exp.clamp(min=variance_floor)
        sigma_p_safe = sigma_p_exp.clamp(min=variance_floor)

        kl_per_dim = 0.5 * (
            sigma_q_safe / sigma_p_safe
            + (mu_q_exp - mu_p_exp)**2 / sigma_p_safe
            - 1.0
            + torch.log(sigma_p_safe / sigma_q_safe)
        )  # (B, N, V, K)

        # Sum over K dimensions
        kl_total = kl_per_dim.sum(dim=-1)  # (B, N, V)

        # Convert to logits: -KL/τ (negative because higher KL = lower probability)
        logits = -kl_total / tau  # (B, N, V)

        return logits

    def update_from_beliefs(
        self,
        token_ids: torch.Tensor,       # (B, N) token IDs
        mu_beliefs: torch.Tensor,      # (B, N, K) evolved belief means
        sigma_beliefs: torch.Tensor,   # (B, N, K) belief variances
        prediction_errors: torch.Tensor,  # (B, N) per-position CE loss
        lr: float = 0.01,
    ):
        """
        Update token priors via prediction-error weighted EMA.

        This is the pure FEP learning mechanism:
        - Beliefs with low prediction error are "good" - priors should move toward them
        - Beliefs with high prediction error are "bad" - priors should ignore them
        - For each token, aggregate across all its occurrences in the batch

        CRITICAL: Updates priors by TOKEN ID, not position!

        Args:
            token_ids: (B, N) token IDs in the batch
            mu_beliefs: (B, N, K) evolved belief means after VFE
            sigma_beliefs: (B, N, K) evolved belief variances
            prediction_errors: (B, N) per-position cross-entropy loss
            lr: Learning rate for prior updates
        """
        if self.gauge_fixed_priors:
            # For gauge-fixed priors, update phi_embed (rotation angles) and base prior
            # via prediction-error weighted EMA toward successful beliefs.
            with torch.no_grad():
                B, N, K = mu_beliefs.shape

                # Compute success weights from prediction errors
                weights = F.softmax(-prediction_errors.clamp(min=-10, max=10), dim=-1)

                unique_tokens = torch.unique(token_ids)
                for token_id in unique_tokens:
                    mask = (token_ids == token_id)
                    if mask.sum() == 0:
                        continue

                    token_beliefs_mu = mu_beliefs[mask]  # (num_occurrences, K)
                    token_weights = weights[mask]  # (num_occurrences,)
                    total_weight = token_weights.sum()
                    weighted_belief = (token_beliefs_mu * token_weights.unsqueeze(-1)).sum(0) / total_weight

                    # Compute the target phi that would rotate base_prior_mu to weighted_belief
                    # Update phi_embed via EMA toward inverse-rotated belief
                    mean_error = prediction_errors[mask].mean()
                    confidence = 1.0 / (1.0 + mean_error)
                    effective_lr = lr * confidence

                    # EMA update phi toward the direction that produces this belief
                    # Approximate: update base_prior_mu toward the mean of all inverse-rotated beliefs
                    token_id_int = int(token_id.item())
                    current_phi = self.phi_embed.weight.data[token_id_int]  # (phi_dim,)

                    # Compute current rotation and its inverse
                    R = self._compute_rotation(current_phi.unsqueeze(0)).squeeze(0)  # (K, K)
                    # Inverse-rotate the belief back to base frame
                    belief_in_base = R.T @ weighted_belief  # (K,)

                    # EMA update base prior toward this
                    self.base_prior_mu.data[:] = (
                        (1.0 - effective_lr * 0.1) * self.base_prior_mu.data +
                        effective_lr * 0.1 * belief_in_base.detach()
                    )
            return

        with torch.no_grad():
            B, N, K = mu_beliefs.shape

            # Get unique tokens in this batch
            unique_tokens = torch.unique(token_ids)

            for token_id in unique_tokens:
                # Find all occurrences of this token across batch
                mask = (token_ids == token_id)  # (B, N) boolean mask

                if mask.sum() == 0:
                    continue

                # Extract beliefs for this token across all occurrences
                token_mu = mu_beliefs[mask]  # (num_occurrences, K)
                token_sigma = sigma_beliefs[mask]  # (num_occurrences, K)
                token_errors = prediction_errors[mask]  # (num_occurrences,)

                # Weight by prediction quality (softmax over errors)
                # Low error = high weight
                weights = F.softmax(-token_errors.clamp(min=-10, max=10), dim=0)  # (num_occurrences,)

                # Compute weighted average of beliefs
                weighted_mu = (token_mu * weights.unsqueeze(-1)).sum(0)  # (K,)
                weighted_sigma = (token_sigma * weights.unsqueeze(-1)).sum(0)  # (K,)

                # Confidence-weighted learning rate
                mean_error = token_errors.mean()
                confidence = 1.0 / (1.0 + mean_error)  # High error = low confidence
                effective_lr = lr * confidence

                # EMA update: prior ← (1-lr)*prior + lr*belief
                token_id_int = int(token_id.item())
                self.prior_mu.data[token_id_int] = (
                    (1.0 - effective_lr) * self.prior_mu.data[token_id_int] +
                    effective_lr * weighted_mu.detach()
                )

                # Update sigma with smaller learning rate
                if self.learnable_sigma:
                    sigma_lr = effective_lr * 0.1
                    current_sigma = torch.exp(self.log_prior_sigma.data[token_id_int])
                    new_sigma = (1.0 - sigma_lr) * current_sigma + sigma_lr * weighted_sigma.detach()
                    self.log_prior_sigma.data[token_id_int] = torch.log(new_sigma.clamp(min=self.eps))

    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        mu_q: Optional[torch.Tensor] = None,
        sigma_q: Optional[torch.Tensor] = None,
        mode: str = 'encode',
        tau: float = 1.0,
    ):
        """
        Forward pass - encode or decode.

        Args:
            token_ids: (B, N) for encoding
            mu_q, sigma_q: (B, N, K) for decoding
            mode: 'encode' or 'decode'
            tau: Temperature for decoding
        """
        if mode == 'encode':
            assert token_ids is not None
            return self.encode(token_ids)
        elif mode == 'decode':
            assert mu_q is not None and sigma_q is not None
            return self.decode(mu_q, sigma_q, tau)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def extra_repr(self) -> str:
        """Pretty print for model summary."""
        return (
            f"vocab_size={self.vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable_sigma={self.learnable_sigma}, "
            f"gauge_fixed_priors={self.gauge_fixed_priors}"
        )