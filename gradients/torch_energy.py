# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:36:43 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Differentiable Free Energy Functions (PyTorch Autograd)
=======================================================

This module provides differentiable implementations of all energy terms
using PyTorch, enabling automatic gradient computation via autograd.

Key advantage: No manual gradient derivation needed! Just call .backward()
on the energy and autograd computes all gradients automatically.

This replaces ~1300 lines of hand-derived gradients in gradient_terms.py
and gradient_engine.py.

Author: Claude (refactoring)
Date: December 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# =============================================================================
# Core Differentiable Operations
# =============================================================================

def kl_divergence_gaussian(
    mu_q: torch.Tensor,    # (..., K)
    Sigma_q: torch.Tensor, # (..., K, K)
    mu_p: torch.Tensor,    # (..., K)
    Sigma_p: torch.Tensor, # (..., K, K)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    KL(q || p) for multivariate Gaussians - FULLY DIFFERENTIABLE.

    Autograd handles all gradients automatically:
    - d(KL)/d(mu_q), d(KL)/d(Sigma_q)
    - d(KL)/d(mu_p), d(KL)/d(Sigma_p)

    Formula:
    KL = 0.5 * (tr(Sigma_p^{-1} Sigma_q) + (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
                - K + log|Sigma_p| - log|Sigma_q|)

    Args:
        mu_q: Source (belief) mean, shape (..., K)
        Sigma_q: Source covariance, shape (..., K, K)
        mu_p: Target (prior) mean, shape (..., K)
        Sigma_p: Target covariance, shape (..., K, K)
        eps: Regularization for numerical stability

    Returns:
        KL divergence, shape (...)
    """
    K = mu_q.shape[-1]
    device = mu_q.device
    dtype = mu_q.dtype

    # Regularize for numerical stability
    eye = torch.eye(K, device=device, dtype=dtype)
    Sigma_q_reg = Sigma_q + eps * eye
    Sigma_p_reg = Sigma_p + eps * eye

    # Cholesky decomposition (differentiable!)
    L_q = torch.linalg.cholesky(Sigma_q_reg)
    L_p = torch.linalg.cholesky(Sigma_p_reg)

    # Log determinants: log|Sigma| = 2 * sum(log(diag(L)))
    # No +eps needed here: Cholesky of (Sigma + eps*I) already has positive diagonal
    logdet_q = 2 * torch.sum(
        torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)),
        dim=-1
    )
    logdet_p = 2 * torch.sum(
        torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)),
        dim=-1
    )

    # Trace term: tr(Sigma_p^{-1} Sigma_q)
    Sigma_p_inv = torch.cholesky_inverse(L_p)
    trace_term = torch.sum(Sigma_p_inv * Sigma_q_reg, dim=(-2, -1))

    # Quadratic term: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    delta = mu_p - mu_q  # (..., K)
    # Solve L_p @ y = delta for y, then ||y||^2 = delta^T Sigma_p^{-1} delta
    y = torch.linalg.solve_triangular(
        L_p, delta.unsqueeze(-1), upper=False
    )  # (..., K, 1)
    quad_term = torch.sum(y ** 2, dim=(-2, -1))

    # KL divergence
    kl = 0.5 * (trace_term + quad_term - K + logdet_p - logdet_q)

    return torch.clamp(kl, min=0.0)


def transport_gaussian(
    mu: torch.Tensor,      # (..., K)
    Sigma: torch.Tensor,   # (..., K, K)
    Omega: torch.Tensor    # (..., K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transport Gaussian by operator: (Omega @ mu, Omega @ Sigma @ Omega^T).

    DIFFERENTIABLE - autograd computes gradients through transport!

    Args:
        mu: Mean, shape (..., K)
        Sigma: Covariance, shape (..., K, K)
        Omega: Transport operator, shape (..., K, K)

    Returns:
        mu_t: Transported mean, shape (..., K)
        Sigma_t: Transported covariance, shape (..., K, K)
    """
    # Transport mean: Omega @ mu
    mu_t = torch.einsum('...ij,...j->...i', Omega, mu)

    # Transport covariance: Omega @ Sigma @ Omega^T
    Sigma_t = Omega @ Sigma @ Omega.transpose(-2, -1)

    # Symmetrize for numerical stability
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-2, -1))

    return mu_t, Sigma_t


def compute_transport_operator(
    phi_i: torch.Tensor,     # (..., 3)
    phi_j: torch.Tensor,     # (..., 3)
    generators: torch.Tensor # (3, K, K)
) -> torch.Tensor:
    """
    Compute transport operator Omega_ij = exp(phi_i . J) @ exp(-phi_j . J).

    DIFFERENTIABLE via torch.linalg.matrix_exp!

    Args:
        phi_i: Source gauge field, shape (..., 3)
        phi_j: Target gauge field, shape (..., 3)
        generators: Lie algebra generators, shape (3, K, K)

    Returns:
        Omega: Transport operator, shape (..., K, K)
    """
    # Contract gauge fields with generators: X = phi . J
    # X_i[..., k, l] = sum_a phi_i[..., a] * generators[a, k, l]
    X_i = torch.einsum('...a,akl->...kl', phi_i, generators)  # (..., K, K)
    X_j = torch.einsum('...a,akl->...kl', phi_j, generators)  # (..., K, K)

    # Matrix exponentials (differentiable!)
    exp_i = torch.linalg.matrix_exp(X_i)
    exp_neg_j = torch.linalg.matrix_exp(-X_j)

    # Compose: Omega = exp(phi_i . J) @ exp(-phi_j . J)
    Omega = exp_i @ exp_neg_j

    return Omega


def hat_map(phi: torch.Tensor) -> torch.Tensor:
    """
    Hat map: so(3) vector to skew-symmetric matrix.

    phi = [p1, p2, p3] -> [[0, -p3, p2], [p3, 0, -p1], [-p2, p1, 0]]

    Args:
        phi: so(3) element, shape (..., 3)

    Returns:
        Skew-symmetric matrix, shape (..., 3, 3)
    """
    batch_shape = phi.shape[:-1]
    device = phi.device
    dtype = phi.dtype

    # Extract components
    p1 = phi[..., 0]
    p2 = phi[..., 1]
    p3 = phi[..., 2]

    # Build skew-symmetric matrix
    zero = torch.zeros_like(p1)
    row1 = torch.stack([zero, -p3, p2], dim=-1)
    row2 = torch.stack([p3, zero, -p1], dim=-1)
    row3 = torch.stack([-p2, p1, zero], dim=-1)

    return torch.stack([row1, row2, row3], dim=-2)


# =============================================================================
# Free Energy Module (nn.Module for integration with PyTorch ecosystem)
# =============================================================================

class FreeEnergy(nn.Module):
    """
    Differentiable free energy functional.

    F = alpha * sum_i KL(q_i || p_i)                        [Self-coupling]
      + lambda_belief * sum_ij beta_ij * KL(q_i || Omega_ij[q_j])  [Belief alignment]
      + lambda_prior * sum_ij gamma_ij * KL(p_i || Omega_ij[p_j])  [Prior alignment]

    Call .backward() on the output to get all gradients via autograd!

    Example:
        free_energy = FreeEnergy(config)
        F = free_energy(system)  # Forward pass
        F.backward()             # Autograd computes ALL gradients!
        # Now each parameter has .grad populated
    """

    def __init__(
        self,
        lambda_self: float = 1.0,
        lambda_belief: float = 1.0,
        lambda_prior: float = 0.0,
        kappa_beta: float = 1.0,
        kappa_gamma: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        Initialize free energy module.

        Args:
            lambda_self: Weight for self-coupling KL(q||p)
            lambda_belief: Weight for belief alignment terms
            lambda_prior: Weight for prior alignment terms
            kappa_beta: Temperature for belief softmax weights
            kappa_gamma: Temperature for prior softmax weights
            eps: Numerical regularization
        """
        super().__init__()
        self.lambda_self = lambda_self
        self.lambda_belief = lambda_belief
        self.lambda_prior = lambda_prior
        self.kappa_beta = kappa_beta
        self.kappa_gamma = kappa_gamma
        self.eps = eps

    def forward(
        self,
        mu_q: torch.Tensor,      # (N, K) or (N, *S, K)
        Sigma_q: torch.Tensor,   # (N, K, K) or (N, *S, K, K)
        mu_p: torch.Tensor,      # (N, K) or (N, *S, K)
        Sigma_p: torch.Tensor,   # (N, K, K) or (N, *S, K, K)
        phi: torch.Tensor,       # (N, 3) or (N, *S, 3)
        generators: torch.Tensor, # (3, K, K)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total free energy (differentiable).

        Args:
            mu_q: All agent belief means, shape (N, ..., K)
            Sigma_q: All agent belief covariances, shape (N, ..., K, K)
            mu_p: All agent prior means, shape (N, ..., K)
            Sigma_p: All agent prior covariances, shape (N, ..., K, K)
            phi: All agent gauge fields, shape (N, ..., 3)
            generators: Lie algebra generators, shape (3, K, K)

        Returns:
            Dictionary with:
                'total': Total free energy (scalar)
                'self': Self-coupling energy
                'belief': Belief alignment energy
                'prior': Prior alignment energy
        """
        N = mu_q.shape[0]
        device = mu_q.device

        # === 1. Self-coupling: sum_i KL(q_i || p_i) ===
        kl_self = kl_divergence_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, self.eps)
        E_self = kl_self.sum()

        # === 2. Belief alignment: sum_ij beta_ij * KL(q_i || Omega_ij[q_j]) ===
        E_belief = torch.tensor(0.0, device=device)
        if self.lambda_belief > 0 and N > 1:
            E_belief = self._compute_alignment_energy(
                mu_q, Sigma_q, phi, generators, self.kappa_beta
            )

        # === 3. Prior alignment (similar structure) ===
        E_prior = torch.tensor(0.0, device=device)
        if self.lambda_prior > 0 and N > 1:
            E_prior = self._compute_alignment_energy(
                mu_p, Sigma_p, phi, generators, self.kappa_gamma
            )

        # === Total Free Energy ===
        F_total = (
            self.lambda_self * E_self +
            self.lambda_belief * E_belief +
            self.lambda_prior * E_prior
        )

        return {
            'total': F_total,
            'self': E_self,
            'belief': E_belief,
            'prior': E_prior,
        }

    def _compute_alignment_energy(
        self,
        mu: torch.Tensor,       # (N, K) or (N, *S, K)
        Sigma: torch.Tensor,    # (N, K, K) or (N, *S, K, K)
        phi: torch.Tensor,      # (N, 3) or (N, *S, 3)
        generators: torch.Tensor,
        kappa: float,
    ) -> torch.Tensor:
        """
        Compute alignment energy with softmax attention weights.

        For 0D (particle) agents, computes full N x N pairwise interactions.
        For spatial agents, uses mean values for efficiency.
        """
        N = mu.shape[0]
        device = mu.device
        dtype = mu.dtype

        # Handle spatial vs particle case
        if mu.ndim == 2:
            # Particle: shape (N, K)
            mu_eff = mu
            Sigma_eff = Sigma
            phi_eff = phi
        else:
            # Spatial: take mean over spatial dimensions for efficiency
            spatial_dims = tuple(range(1, mu.ndim - 1))
            mu_eff = mu.mean(dim=spatial_dims)  # (N, K)
            Sigma_eff = Sigma.mean(dim=spatial_dims)  # (N, K, K)
            phi_eff = phi.mean(dim=spatial_dims)  # (N, 3)

        # Compute all N x N pairwise KL divergences
        KL_matrix = torch.zeros(N, N, device=device, dtype=dtype)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Transport j to i's frame
                Omega_ij = compute_transport_operator(
                    phi_eff[i], phi_eff[j], generators
                )
                mu_j_t, Sigma_j_t = transport_gaussian(
                    mu_eff[j], Sigma_eff[j], Omega_ij
                )

                # KL(q_i || Omega_ij[q_j])
                KL_matrix[i, j] = kl_divergence_gaussian(
                    mu_eff[i], Sigma_eff[i],
                    mu_j_t, Sigma_j_t,
                    self.eps
                )

        # Softmax attention weights: beta_ij = softmax(-kappa * KL_ij)
        # Mask diagonal (i != j)
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        KL_masked = KL_matrix.clone()
        logits = -kappa * KL_masked
        logits[~mask] = float('-inf')  # Exclude self (masked_fill for softmax)

        beta = F.softmax(logits, dim=-1)  # (N, N)
        beta = beta * mask.float()  # Zero out diagonal

        # Weighted sum: sum_ij beta_ij * KL_ij
        E_align = (beta * KL_matrix).sum()

        return E_align


# =============================================================================
# Batched Operations (for N x N computations)
# =============================================================================

def batched_pairwise_kl(
    mu: torch.Tensor,      # (N, K)
    Sigma: torch.Tensor,   # (N, K, K)
    phi: torch.Tensor,     # (N, 3)
    generators: torch.Tensor,  # (3, K, K)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute all N x N pairwise KL divergences in one pass.

    KL[i, j] = KL(q_i || Omega_ij[q_j])

    This is more efficient than the loop version for large N.

    Args:
        mu: All means, shape (N, K)
        Sigma: All covariances, shape (N, K, K)
        phi: All gauge fields, shape (N, 3)
        generators: Lie algebra generators, shape (3, K, K)
        eps: Regularization

    Returns:
        KL matrix, shape (N, N)
    """
    N, K = mu.shape
    device = mu.device
    dtype = mu.dtype

    # Compute all N x N transport operators
    # Omega[i, j] = exp(phi_i . J) @ exp(-phi_j . J)
    phi_i = phi.unsqueeze(1)  # (N, 1, 3)
    phi_j = phi.unsqueeze(0)  # (1, N, 3)

    # X_i[i, j, k, l] = phi_i[i, :, a] * generators[a, k, l] (broadcast j)
    X_i = torch.einsum('nia,akl->nikl', phi_i.expand(N, N, 3), generators)  # (N, N, K, K)
    X_j = torch.einsum('nja,akl->njkl', phi_j.expand(N, N, 3), generators)  # (N, N, K, K)

    exp_i = torch.linalg.matrix_exp(X_i)      # (N, N, K, K)
    exp_neg_j = torch.linalg.matrix_exp(-X_j)  # (N, N, K, K)

    Omega = exp_i @ exp_neg_j  # (N, N, K, K)

    # Transport all beliefs: mu_j_t[i, j] = Omega[i, j] @ mu[j]
    mu_j_t = torch.einsum('ijkl,jl->ijk', Omega, mu)  # (N, N, K)

    # Sigma_j_t[i, j] = Omega[i, j] @ Sigma[j] @ Omega[i, j]^T
    Sigma_j = Sigma.unsqueeze(0).expand(N, N, K, K)  # (N, N, K, K)
    Sigma_j_t = Omega @ Sigma_j @ Omega.transpose(-2, -1)  # (N, N, K, K)
    Sigma_j_t = 0.5 * (Sigma_j_t + Sigma_j_t.transpose(-2, -1))  # Symmetrize

    # Broadcast source distributions
    mu_i = mu.unsqueeze(1).expand(N, N, K)           # (N, N, K)
    Sigma_i = Sigma.unsqueeze(1).expand(N, N, K, K)  # (N, N, K, K)

    # Vectorized KL computation
    eye = torch.eye(K, device=device, dtype=dtype)
    Sigma_j_t_reg = Sigma_j_t + eps * eye
    Sigma_i_reg = Sigma_i + eps * eye

    # Batched Cholesky
    L_j_t = torch.linalg.cholesky(Sigma_j_t_reg)  # (N, N, K, K)
    L_i = torch.linalg.cholesky(Sigma_i_reg)      # (N, N, K, K)

    # Log determinants
    logdet_j_t = 2 * torch.diagonal(L_j_t, dim1=-2, dim2=-1).log().sum(-1)  # (N, N)
    logdet_i = 2 * torch.diagonal(L_i, dim1=-2, dim2=-1).log().sum(-1)      # (N, N)

    # Trace term: tr(Sigma_j_t^{-1} Sigma_i)
    Sigma_j_t_inv = torch.cholesky_inverse(L_j_t)
    trace = (Sigma_j_t_inv * Sigma_i_reg).sum((-2, -1))  # (N, N)

    # Quadratic term
    delta = mu_j_t - mu_i  # (N, N, K)
    y = torch.linalg.solve_triangular(
        L_j_t, delta.unsqueeze(-1), upper=False
    )  # (N, N, K, 1)
    quad = (y ** 2).sum((-2, -1))  # (N, N)

    # KL divergence
    kl = 0.5 * (trace + quad - K + logdet_j_t - logdet_i)

    return kl.clamp(min=0.0)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_free_energy_simple(
    agents: list,
    generators: torch.Tensor,
    config: dict,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Simple interface to compute free energy from a list of TensorAgents.

    Args:
        agents: List of TensorAgent instances
        generators: Lie algebra generators
        config: Dictionary with lambda_self, lambda_belief, kappa_beta, etc.

    Returns:
        Total free energy (scalar tensor)
    """
    # Stack all agent states
    mu_q = torch.stack([a.mu_q for a in agents])
    Sigma_q = torch.stack([a.Sigma_q for a in agents])
    mu_p = torch.stack([a.mu_p for a in agents])
    Sigma_p = torch.stack([a.Sigma_p for a in agents])
    phi = torch.stack([a.phi for a in agents])

    # Create FreeEnergy module
    fe = FreeEnergy(
        lambda_self=config.get('lambda_self', 1.0),
        lambda_belief=config.get('lambda_belief', 1.0),
        lambda_prior=config.get('lambda_prior', 0.0),
        kappa_beta=config.get('kappa_beta', 1.0),
        kappa_gamma=config.get('kappa_gamma', 1.0),
    )

    # Compute
    result = fe(mu_q, Sigma_q, mu_p, Sigma_p, phi, generators)
    return result['total']