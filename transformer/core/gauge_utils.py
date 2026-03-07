"""
Shared Gauge-Geometry Utilities
================================

Shared utilities for gauge transport computations used across
attention.py, variational_ffn.py, and embeddings.py.

Consolidates duplicated matrix exponential and KL divergence patterns.
"""

import torch
from typing import Tuple


def stable_matrix_exp_pair(
    matrix: torch.Tensor,
    dim_threshold: int = 8,
    max_norm: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exp(M) and exp(-M) with norm clamping and float64 upcasting.

    Two stability measures:
    1. Frobenius norm clamping: caps ||M||_F at max_norm to prevent the
       Pade scaling-squaring algorithm from overflowing. For GL+(K),
       exp(M) with ||M|| >> 1 produces extreme condition numbers that
       make downstream Omega Sigma Omega^T numerically non-positive-definite.
    2. Float64 upcasting for K >= dim_threshold.

    Note on surjectivity:
        exp(M) always has det > 0 (since det(exp(M)) = exp(tr(M))), so the
        outputs live in GL⁺(K), the identity component.

        Even within GL⁺(K), a single exp(M) is NOT surjective for K > 1.
        By Culver (1966), A ∈ GL(K,ℝ) has a real log iff for each negative
        real eigenvalue, the number of Jordan blocks of each size is even.
        E.g. diag(-2, -3) has det = 6 > 0 but no real logarithm.

        This does not limit transport: Ω_ij = exp(M_i)·exp(-M_j) is a free
        product of two exponentials, which covers all of GL⁺(K) (by polar
        decomposition: A = exp(log P)·exp(log O) where P sym.pos.def., O ∈ SO).
        For SO(K), exp: so(K) → SO(K) is surjective — no issues.

    Args:
        matrix: (..., d, d) matrix to exponentiate.
        dim_threshold: Upcast to float64 when d >= this value. Default 8.
        max_norm: Maximum Frobenius norm for input matrix. Default 10.0.

    Returns:
        (exp_pos, exp_neg): Tuple of exp(M) and exp(-M), both same dtype as input.
    """
    # Clamp Frobenius norm to prevent overflow in matrix_exp.
    # Gradient flows through the scaling factor, so phi still gets
    # signal to shrink when it exceeds the cap.
    mat_norm = matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    scale = (max_norm / mat_norm).clamp(max=1.0)
    matrix = matrix * scale

    d = matrix.shape[-1]
    if d >= dim_threshold:
        matrix_f64 = matrix.double()
        exp_pos = torch.linalg.matrix_exp(matrix_f64).to(matrix.dtype)
        exp_neg = torch.linalg.matrix_exp(-matrix_f64).to(matrix.dtype)
    else:
        exp_pos = torch.linalg.matrix_exp(matrix)
        exp_neg = torch.linalg.matrix_exp(-matrix)
    return exp_pos, exp_neg
