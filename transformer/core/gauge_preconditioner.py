"""
Gauge-Theoretic Gradient Preconditioning
=========================================

Implements principled gradient preconditioning for gauge frame (φ) parameters
based on the Cartan decomposition of gl(K).

Background:
    gl(K) = so(K) ⊕ sym(K)

    - so(K): compact (antisymmetric generators). The matrix exponential
      has bounded derivatives: ||d exp(X)/dX|| = O(1) for X ∈ so(K).
    - sym(K): non-compact (symmetric generators). exp(X) grows without bound,
      and ||d exp(X)/dX|| ~ exp(||X||) for X ∈ sym(K).

    This non-compact amplification is why phi gradients can spike to ~1e3
    when using GL(K) gauge groups: the backward pass through matrix_exp
    amplifies gradients exponentially in the symmetric directions.

Solution (Killing-form-motivated preconditioning):
    The Killing form B(X,Y) = 2K tr(XY) - 2 tr(X)tr(Y) on gl(K) is:
    - Negative definite on so(K) (compact part)
    - Positive definite on sym₀(K) (non-compact traceless part)
    - Zero on the center ℝ·I

    This sign structure reflects exactly which directions are dangerous.
    We use the Cartan decomposition to project the gradient into so(K)
    and sym(K) components, then dampen the sym(K) component.

    This is equivalent to using a Riemannian metric on gl(K) that assigns
    higher cost to non-compact directions — the natural metric for gradient
    descent on a non-compact Lie group.

Also implements:
    - SL(K) projection: projects φ to the traceless subalgebra sl(K),
      removing the single most dangerous degree of freedom (uniform scaling).

Author: Theoretical foundation from Cartan decomposition of gl(K)
Date: March 2026
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def build_cartan_projector(
    generators: torch.Tensor,  # (n_gen, K, K)
    sym_dampening: float = 0.1,
) -> torch.Tensor:
    """
    Build Cartan decomposition preconditioner for gl(K) phi gradients.

    Decomposes the Lie algebra gl(K) = so(K) ⊕ sym(K) and constructs a
    preconditioning matrix that dampens the non-compact (symmetric) directions.

    For generators {T_a}, the symmetric projection in coordinate space is:
        [P_sym]_{ab} = (1/2)(⟨T_a, T_b⟩ + tr(T_a T_b))

    where ⟨T_a, T_b⟩ = tr(T_aᵀ T_b) is the Frobenius inner product.
    For Frobenius-orthonormal generators, ⟨T_a, T_b⟩ = δ_ab.

    The preconditioner is:
        C = P_so + c · P_sym = I - (1-c) · P_sym

    where c ∈ (0, 1] dampens the symmetric directions:
        c = 1.0: no preconditioning (Euclidean gradient)
        c = 0.0: project out symmetric components entirely (restrict to so(K))
        c = 0.1: dampen symmetric by 10× (recommended for GL(K))

    Args:
        generators: Lie algebra generators (n_gen, K, K)
        sym_dampening: Dampening factor for symmetric directions.
                      0.1 means sym directions get 10× smaller steps.

    Returns:
        preconditioner: (n_gen, n_gen) matrix to left-multiply phi gradients
    """
    n_gen, K, _ = generators.shape
    device = generators.device
    dtype = generators.dtype

    # Gram matrix: G_ab = tr(T_a^T T_b) (Frobenius inner product)
    # For standard E_ij basis this is the identity.
    gram = torch.einsum('aij,bij->ab', generators.transpose(-2, -1), generators)

    # Trace product: tr(T_a T_b) — captures symmetric/antisymmetric structure
    # For antisymmetric T: tr(T^2) < 0. For symmetric T: tr(T^2) > 0.
    trace_prod = torch.einsum('aij,bji->ab', generators, generators)

    # Symmetric projection in generator coordinate space:
    # P_sym = G^{-1} · (G + trace_prod) / 2
    # For orthonormal generators (G = I): P_sym = (I + trace_prod) / 2

    # Use pseudoinverse for numerical stability (G may be ill-conditioned
    # for non-standard generator bases)
    gram_inv = torch.linalg.pinv(gram)
    P_sym = gram_inv @ (gram + trace_prod) / 2.0

    # Preconditioner: C = I - (1 - c) * P_sym
    # This keeps so(K) components at full strength and dampens sym(K) by factor c
    I_gen = torch.eye(n_gen, device=device, dtype=dtype)
    preconditioner = I_gen - (1.0 - sym_dampening) * P_sym

    return preconditioner


def build_slk_projector(
    generators: torch.Tensor,  # (n_gen, K, K)
) -> torch.Tensor:
    """
    Build projector that removes the trace component of phi, restricting to sl(K).

    For φ parameterizing M = Σ_a φ_a T_a ∈ gl(K), the trace is:
        tr(M) = Σ_a φ_a tr(T_a) = v^T φ

    where v_a = tr(T_a). The SL(K) projection removes this component:
        φ_sl = φ - (v^T φ / ||v||²) v
        φ_sl = (I - v v^T / ||v||²) φ

    This ensures det(exp(M)) = exp(tr(M)) = exp(0) = 1, so Ω_ij ∈ SL(K).

    Args:
        generators: Lie algebra generators (n_gen, K, K)

    Returns:
        trace_proj: (n_gen,) vector v where v_a = tr(T_a).
                   To project: phi -= (phi @ v) / (v @ v) * v
    """
    # v_a = tr(T_a) = sum of diagonal elements
    trace_vec = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (n_gen,)
    return trace_vec


def apply_slk_projection(
    phi: torch.Tensor,           # (..., n_gen) — phi embedding weights
    trace_vec: torch.Tensor,     # (n_gen,) — from build_slk_projector
) -> torch.Tensor:
    """
    Project phi to the traceless subalgebra sl(K) in-place.

    Removes the component of phi along the trace direction, ensuring
    that M = Σ_a φ_a T_a has tr(M) = 0, so exp(M) ∈ SL(K).

    Args:
        phi: Gauge frame coordinates (..., n_gen)
        trace_vec: Trace vector v_a = tr(T_a), shape (n_gen,)

    Returns:
        phi_projected: Same shape as phi, with trace component removed
    """
    v_norm_sq = (trace_vec @ trace_vec).clamp(min=1e-12)
    # Component of phi along trace direction
    phi_trace = (phi @ trace_vec).unsqueeze(-1)  # (..., 1)
    # Remove it
    return phi - (phi_trace / v_norm_sq) * trace_vec


def apply_cartan_preconditioning(
    grad_phi: torch.Tensor,         # (..., n_gen) — gradient w.r.t. phi
    preconditioner: torch.Tensor,   # (n_gen, n_gen) — from build_cartan_projector
) -> torch.Tensor:
    """
    Apply Cartan decomposition preconditioning to phi gradients.

    Dampens the non-compact (symmetric) gradient components while
    preserving the compact (antisymmetric) components at full strength.

    This is the "Killing form natural gradient" — using the Lie algebra
    structure to define the proper metric for gradient descent on GL(K).

    Args:
        grad_phi: Raw gradient w.r.t. phi (..., n_gen)
        preconditioner: Preconditioning matrix (n_gen, n_gen)

    Returns:
        preconditioned gradient, same shape as grad_phi
    """
    return grad_phi @ preconditioner.T
