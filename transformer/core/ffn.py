"""
Feed-Forward Networks for Gauge Transformer
===========================================

VFE_dynamic mode: Dynamic attention-belief co-evolution
- Recomputes β at each VFE step
- Updates both μ AND Σ
- Enables emergent block structure through attention-belief coupling

Author: Extended architecture with VFE integration
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal

from transformer.core.variational_ffn import (
    VariationalFFNDynamic,  # Dynamic-β VFE with attention-belief co-evolution
)


class GaugeFFN(nn.Module):
    """
    FFN module for Gauge Transformer using VFE_dynamic mode.

    VFE_dynamic: Dynamic-β VFE with attention-belief co-evolution
    - Recomputes β at each VFE step
    - Updates both μ AND Σ
    - Enables emergent block structure

    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        dropout: float = 0.1,
        mode: Literal['VFE_dynamic'] = 'VFE_dynamic',
        # Variational parameters
        alpha: float = 0.001,
        kappa: float = 1.0,
        n_iterations: int = 1,
        learnable_lr: bool = True,
        lambda_belief: float = 1.0,
        update_sigma: bool = True,
        compute_sigma_align_grad: bool = True,  # Sigma gradient from alignment term
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Pure FEP mode: learning via prior evolution (no backprop)
        pure_fep_mode: bool = False,
        max_seq_len: int = 512,
        prior_lr: float = 0.01,
        prior_bank: Optional[nn.Module] = None,  # Token-dependent PriorBank
        use_prior_bank: bool = False,  # Use PriorBank vs position-dependent priors
        # Phi evolution via VFE gradients (principled approach)
        update_phi: bool = False,  # If True, update phi via ∂F/∂φ (after E-step loop)
        update_phi_per_iteration: bool = False,  # If True, update phi during each E-step iteration
        phi_lr: float = 0.05,      # Learning rate for phi updates
        phi_max_norm: float = 3.14159,  # Max norm for phi (π = 180° rotation)
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        # Bayesian precision (learned prior self-coupling)
        learnable_alpha: bool = False,  # If True, use Gamma-Normal conjugate precision
        # Multi-head VFE: per-block β through VFE iterations
        multihead_vfe: bool = False,  # If True, maintain per-head attention in VFE loop
        per_head_kappa: bool = False,  # If True, learn separate κ_h per head in VFE
        # Legacy parameters (ignored, kept for API compatibility)
        **kwargs,
    ):
        """
        Initialize VFE FFN module.

        Args:
            embed_dim: K - latent dimension
            hidden_dim: Hidden layer size (unused, kept for API compatibility)
            generators: SO(3) generators (3, K, K) - required
            dropout: Dropout rate (unused, kept for API compatibility)
            mode: FFN mode - only 'VFE_dynamic' is supported
            alpha: Prior weight
            kappa: Softmax temperature for attention
            n_iterations: VFE inference steps per forward pass
            learnable_lr: Learn step size for variational descent
            lambda_belief: Belief alignment weight
            update_sigma: Update covariances during inference
            compute_sigma_align_grad: Compute sigma gradient from alignment term
            diagonal_covariance: Use diagonal covariance for memory efficiency
            pure_fep_mode: If True, use persistent priors for backprop-free learning
            max_seq_len: Max sequence length for persistent priors (pure FEP mode)
            prior_lr: Learning rate for prior updates (pure FEP mode)
            irrep_dims: Block dimensions [d₁, d₂, ...] for memory-efficient block-diagonal KL.
                       Exploits O(N² × Σᵢdᵢ²) vs O(N² × K²) - massive savings for multi-irrep!
            chunk_size: Chunk size for memory-efficient processing. Processes N×N in C×C chunks.
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0 always.
            learnable_alpha: If True, use Bayesian precision via Gamma-Normal conjugacy.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = 'VFE_dynamic'
        # Note: pure_fep_mode accessed via property that delegates to variational_ffn

        if generators is None:
            raise ValueError("generators required for VFE_dynamic mode")

        # Initialize VFE_dynamic FFN
        self.variational_ffn = VariationalFFNDynamic(
            embed_dim=embed_dim,
            generators=generators,
            alpha=alpha,
            lambda_belief=lambda_belief,
            kappa=kappa,
            n_iterations=n_iterations,
            learnable_lr=learnable_lr,
            update_sigma=update_sigma,
            diagonal_covariance=diagonal_covariance,
            compute_sigma_align_grad=compute_sigma_align_grad,
            # Pure FEP mode parameters
            pure_fep_mode=pure_fep_mode,
            max_seq_len=max_seq_len,
            prior_lr=prior_lr,
            prior_bank=prior_bank,  # Pass PriorBank
            use_prior_bank=use_prior_bank,  # Enable token-dependent priors
            # Phi evolution via VFE gradients
            update_phi=update_phi,
            update_phi_per_iteration=update_phi_per_iteration,
            phi_lr=phi_lr,
            phi_max_norm=phi_max_norm,
            # Memory-efficient options
            irrep_dims=irrep_dims,
            chunk_size=chunk_size,
            # Self-attention masking
            mask_self_attention=mask_self_attention,
            # Bayesian precision
            learnable_alpha=learnable_alpha,
            # Multi-head VFE
            multihead_vfe=multihead_vfe,
            per_head_kappa=per_head_kappa,
        )

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - always required
        beta: Optional[torch.Tensor] = None,      # (B, n_heads, N, N) or (B, N, N)
        mu_prior: Optional[torch.Tensor] = None,  # (B, N, K)
        phi: Optional[torch.Tensor] = None,       # (B, N, 3)
        sigma: Optional[torch.Tensor] = None,     # (B, N, K, K)
        sigma_prior: Optional[torch.Tensor] = None,  # (B, N, K, K) - unused
        mask: Optional[torch.Tensor] = None,      # (B, N, N)
        token_ids: Optional[torch.Tensor] = None,  # (B, N) - For PriorBank lookup
        targets: Optional[torch.Tensor] = None,   # (B, N) - target tokens
        W_out: Optional[torch.Tensor] = None,     # (V, K) - output projection
        return_beta_history: bool = False,         # Return β evolution for RG analysis
    ):
        """
        Forward pass through VFE_dynamic FFN.

        Args:
            mu: Current beliefs (B, N, K)
            beta: Initial attention weights (will be recomputed each step)
            mu_prior: Embedding priors (B, N, K)
            phi: Gauge frames (B, N, phi_dim)
            sigma: Covariances (B, N, K, K)
            sigma_prior: Prior covariances (unused)
            mask: Causal mask (B, N, N)
            targets: Target token IDs (B, N)
            W_out: Output projection matrix (V, K)
            return_beta_history: If True, return (mu, sigma, phi, beta_history) tuple

        Returns:
            (mu_out, sigma_out, phi_out) or (mu_out, sigma_out, phi_out, beta_history)
        """
        # Check required inputs
        if mu_prior is None or phi is None:
            raise ValueError("VFE_dynamic requires mu_prior, phi")

        # Dynamic VFE returns (mu, sigma, phi, beta_history)
        mu_out, sigma_out, phi_out, beta_history = self.variational_ffn(
            mu=mu,
            beta=beta,          # Initial β (will be recomputed each step)
            mu_prior=mu_prior,
            phi=phi,
            sigma=sigma,
            mask=mask,
            token_ids=token_ids,  # For PriorBank lookup
            targets=targets,
            W_out=W_out,
            return_beta_history=return_beta_history,
        )
        if return_beta_history:
            return (mu_out, sigma_out, phi_out, beta_history)
        return (mu_out, sigma_out, phi_out)

    def get_mode(self) -> str:
        """Get current FFN mode."""
        return self.mode

    # =========================================================================
    # Pure FEP mode pass-through methods
    # =========================================================================

    @property
    def pure_fep_mode(self) -> bool:
        """Check if pure FEP mode is enabled."""
        return getattr(self.variational_ffn, 'pure_fep_mode', False)

    def update_priors_from_beliefs(self, *args, **kwargs):
        """Forward to variational_ffn for prior updates."""
        if hasattr(self.variational_ffn, 'update_priors_from_beliefs'):
            return self.variational_ffn.update_priors_from_beliefs(*args, **kwargs)

    def get_prior_stats(self):
        """Forward to variational_ffn for prior statistics."""
        if hasattr(self.variational_ffn, 'get_prior_stats'):
            return self.variational_ffn.get_prior_stats()
        return {}


# =============================================================================
# Convenience functions
# =============================================================================

def create_ffn(
    embed_dim: int,
    hidden_dim: int,
    generators: Optional[torch.Tensor] = None,
    mode: str = 'VFE_dynamic',
    **kwargs
) -> GaugeFFN:
    """
    Factory function for creating FFN.

    Example:
        >>> ffn = create_ffn(
        ...     embed_dim=11, hidden_dim=44,
        ...     generators=generators, alpha=0.001
        ... )
    """
    return GaugeFFN(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        generators=generators,
        mode=mode,
        **kwargs
    )