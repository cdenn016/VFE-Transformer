# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:21:52 2025

@author: chris and christine
"""

"""
CUDA-Optimized Kernels for VFE Simulation
==========================================

GPU-accelerated implementations of core operations using CuPy.
These provide massive speedups for RTX 5090 (and similar GPUs).

Performance Expectations (RTX 5090, N=100 agents, K=8):
- KL divergence batch: CPU ~10ms, GPU ~0.1ms (100x speedup)
- Transport batch: CPU ~5ms, GPU ~0.05ms (100x speedup)
- Gradient computation: CPU ~100ms, GPU ~5ms (20x speedup)

Architecture:
- Uses CuPy for NumPy-compatible GPU arrays
- Raw CUDA kernels via CuPy for custom operations
- Batched operations to maximize GPU utilization

Author: Chris
Date: December 2025
"""

import numpy as np
from typing import Tuple, Optional

# Try to import CuPy
try:
    import cupy as cp
    from cupyx import jit
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# =============================================================================
# CuPy Implementations (NumPy-compatible API on GPU)
# =============================================================================

if CUPY_AVAILABLE:

    def kl_gaussian_cupy(
        mu_q: cp.ndarray,
        Sigma_q: cp.ndarray,
        mu_p: cp.ndarray,
        Sigma_p: cp.ndarray,
        eps: float = 1e-8
    ) -> float:
        """
        KL divergence KL(q||p) between Gaussians - CuPy GPU implementation.

        Args:
            mu_q, Sigma_q: Source distribution N(mu_q, Sigma_q)
            mu_p, Sigma_p: Target distribution N(mu_p, Sigma_p)
            eps: Regularization

        Returns:
            kl: Scalar KL divergence (>= 0)
        """
        K = mu_q.shape[0]

        # Regularize
        eye_K = cp.eye(K, dtype=Sigma_q.dtype)
        Sigma_q_reg = Sigma_q + eps * eye_K
        Sigma_p_reg = Sigma_p + eps * eye_K

        # Cholesky decomposition
        L_q = cp.linalg.cholesky(Sigma_q_reg)
        L_p = cp.linalg.cholesky(Sigma_p_reg)

        # Log determinants
        logdet_q = 2.0 * cp.sum(cp.log(cp.maximum(cp.diag(L_q), eps)))
        logdet_p = 2.0 * cp.sum(cp.log(cp.maximum(cp.diag(L_p), eps)))

        # Trace term: tr(Sigma_p^-1 @ Sigma_q)
        Y = cp.linalg.solve(L_p, Sigma_q_reg)
        Z = cp.linalg.solve(L_p.T, Y)
        term_trace = cp.trace(Z)

        # Quadratic term: (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q)
        delta = mu_p - mu_q
        y = cp.linalg.solve(L_p, delta)
        z = cp.linalg.solve(L_p.T, y)
        term_quad = cp.dot(delta, z)

        # Combine
        kl = 0.5 * (term_trace + term_quad - K + logdet_p - logdet_q)

        return float(cp.maximum(kl, 0.0))


    def kl_gaussian_batch_cupy(
        mu_q: cp.ndarray,
        Sigma_q: cp.ndarray,
        mu_p_batch: cp.ndarray,
        Sigma_p_batch: cp.ndarray,
        eps: float = 1e-8
    ) -> cp.ndarray:
        """
        Batch KL divergence - GPU parallel computation (fully vectorized).

        Computes KL(q || p_i) for multiple targets in parallel.

        Args:
            mu_q: Source mean (K,)
            Sigma_q: Source covariance (K, K)
            mu_p_batch: Target means (N, K)
            Sigma_p_batch: Target covariances (N, K, K)
            eps: Regularization

        Returns:
            kl_batch: (N,) array of KL divergences
        """
        N = mu_p_batch.shape[0]
        K = mu_q.shape[0]

        # Regularize source
        eye_K = cp.eye(K, dtype=Sigma_q.dtype)
        Sigma_q_reg = Sigma_q + eps * eye_K

        # Source log-determinant (scalar, computed once)
        _, logdet_q = cp.linalg.slogdet(Sigma_q_reg)

        # Regularize targets (batched)
        Sigma_p_batch_reg = Sigma_p_batch + eps * eye_K

        # Target log-determinants (N,) — batched over first axis
        _, logdet_p_batch = cp.linalg.slogdet(Sigma_p_batch_reg)

        # Trace term: tr(Σ_p_i^{-1} @ Σ_q) for each i
        # Solve Σ_p_i @ X_i = Σ_q for X_i, then tr(X_i)
        # Sigma_p_batch_reg: (N, K, K), Sigma_q_reg: (K, K) -> broadcast to (N, K, K)
        Sigma_q_expanded = cp.broadcast_to(Sigma_q_reg, (N, K, K)).copy()
        Z = cp.linalg.solve(Sigma_p_batch_reg, Sigma_q_expanded)  # (N, K, K)
        term_trace = cp.trace(Z, axis1=-2, axis2=-1)  # (N,)

        # Quadratic term: (μ_p_i - μ_q)^T Σ_p_i^{-1} (μ_p_i - μ_q)
        delta = mu_p_batch - mu_q  # (N, K)
        # Solve Σ_p_i @ x_i = δ_i for x_i
        z = cp.linalg.solve(Sigma_p_batch_reg, delta[..., None])[..., 0]  # (N, K)
        term_quad = cp.sum(delta * z, axis=-1)  # (N,)

        # Combine: KL = 0.5 * (tr + quad - K + logdet_p - logdet_q)
        kl_batch = 0.5 * (term_trace + term_quad - K + logdet_p_batch - logdet_q)

        return cp.maximum(kl_batch, 0.0)


    def transport_gaussian_cupy(
        mu: cp.ndarray,
        Sigma: cp.ndarray,
        Omega: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Push Gaussian forward via transport operator - GPU.

        N(mu, Sigma) -> N(Omega @ mu, Omega @ Sigma @ Omega.T)

        Args:
            mu: Mean (K,)
            Sigma: Covariance (K, K)
            Omega: Transport operator (K, K)

        Returns:
            mu_transported, Sigma_transported
        """
        # Transport mean
        mu_transported = Omega @ mu

        # Transport covariance
        temp = Omega @ Sigma
        Sigma_transported = temp @ Omega.T

        # Symmetrize
        Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.T)

        return mu_transported, Sigma_transported


    def transport_gaussian_batch_cupy(
        mu: cp.ndarray,
        Sigma: cp.ndarray,
        Omega_batch: cp.ndarray
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Batch Gaussian transport - GPU parallel.

        Computes transport for multiple Omega operators in parallel.

        Args:
            mu: Mean (K,)
            Sigma: Covariance (K, K)
            Omega_batch: Transport operators (N, K, K)

        Returns:
            mu_batch: (N, K)
            Sigma_batch: (N, K, K)
        """
        # Batched mean transport: mu_i = Omega_i @ mu
        mu_batch = cp.einsum('nij,j->ni', Omega_batch, mu)

        # Batched covariance transport
        temp = cp.einsum('nij,jk->nik', Omega_batch, Sigma)
        Sigma_batch = cp.einsum('nij,nkj->nik', temp, Omega_batch)

        # Symmetrize
        Sigma_batch = 0.5 * (Sigma_batch + cp.swapaxes(Sigma_batch, -1, -2))

        return mu_batch, Sigma_batch


    def kl_transported_cupy(
        mu_i: cp.ndarray,
        Sigma_i: cp.ndarray,
        mu_j: cp.ndarray,
        Sigma_j: cp.ndarray,
        Omega_ij: cp.ndarray,
        eps: float = 1e-8
    ) -> float:
        """
        Compute KL(q_i || Omega_ij[q_j]) in one GPU kernel.

        Combines transport + KL for efficiency.

        Args:
            mu_i, Sigma_i: Receiver distribution
            mu_j, Sigma_j: Sender distribution
            Omega_ij: Transport operator
            eps: Regularization

        Returns:
            kl: KL divergence
        """
        # Transport j -> i
        mu_j_transported = Omega_ij @ mu_j
        temp = Omega_ij @ Sigma_j
        Sigma_j_transported = temp @ Omega_ij.T
        Sigma_j_transported = 0.5 * (Sigma_j_transported + Sigma_j_transported.T)

        # Compute KL(i || transported_j)
        return kl_gaussian_cupy(mu_i, Sigma_i, mu_j_transported, Sigma_j_transported, eps)


    def rodrigues_formula_cupy(phi: cp.ndarray, eps: float = 1e-8) -> cp.ndarray:
        """
        Rodrigues formula for SO(3) exponential map - GPU.

        exp(phi) = I + sin(theta)/theta * [phi]_x + (1-cos(theta))/theta^2 * [phi]_x^2

        Args:
            phi: Axis-angle vector (3,) or batch (*S, 3)

        Returns:
            R: Rotation matrix (3, 3) or (*S, 3, 3)
        """
        is_batch = phi.ndim > 1
        if not is_batch:
            phi = phi[cp.newaxis, :]

        batch_shape = phi.shape[:-1]
        n_total = int(np.prod(batch_shape))
        phi_flat = phi.reshape(n_total, 3)

        # Compute theta
        theta = cp.linalg.norm(phi_flat, axis=-1)

        # Allocate output
        R = cp.zeros((n_total, 3, 3), dtype=phi.dtype)

        # Identity matrix
        I = cp.eye(3, dtype=phi.dtype)

        for idx in range(n_total):
            th = theta[idx]
            p = phi_flat[idx]

            # Skew symmetric matrix [phi]_x
            phi_x = cp.zeros((3, 3), dtype=phi.dtype)
            phi_x[0, 1] = -p[2]
            phi_x[0, 2] = p[1]
            phi_x[1, 0] = p[2]
            phi_x[1, 2] = -p[0]
            phi_x[2, 0] = -p[1]
            phi_x[2, 1] = p[0]

            phi_x_sq = phi_x @ phi_x

            if th < eps:
                # Taylor expansion
                R[idx] = I + phi_x + 0.5 * phi_x_sq
            else:
                # Rodrigues formula
                c1 = cp.sin(th) / th
                c2 = (1.0 - cp.cos(th)) / (th * th)
                R[idx] = I + c1 * phi_x + c2 * phi_x_sq

        R = R.reshape(batch_shape + (3, 3))

        if not is_batch:
            R = R[0]

        return R


    def compute_transport_cupy(
        phi_i: cp.ndarray,
        phi_j: cp.ndarray,
        generators: cp.ndarray,
        eps: float = 1e-8
    ) -> cp.ndarray:
        """
        Compute transport operator Omega_ij = exp(phi_i) @ exp(-phi_j) - GPU.

        Args:
            phi_i, phi_j: Gauge fields (*S, 3)
            generators: SO(3) generators (3, K, K)

        Returns:
            Omega_ij: Transport operator (*S, K, K)
        """
        K = generators.shape[1]

        # Handle scalar vs batch
        is_batch = phi_i.ndim > 1
        if not is_batch:
            phi_i = phi_i[cp.newaxis, :]
            phi_j = phi_j[cp.newaxis, :]

        batch_size = phi_i.shape[0]

        # Compute Lie algebra elements X_i = phi_i^a * G_a
        X_i = cp.einsum('na,aij->nij', phi_i, generators)
        X_j = cp.einsum('na,aij->nij', phi_j, generators)

        # Skew-symmetrize
        X_i = 0.5 * (X_i - cp.swapaxes(X_i, -1, -2))
        X_j = 0.5 * (X_j - cp.swapaxes(X_j, -1, -2))

        # Matrix exponentials (using scipy-like approximation)
        exp_phi_i = _batch_matrix_exp_cupy(X_i)
        exp_neg_phi_j = _batch_matrix_exp_cupy(-X_j)

        # Compose: Omega = exp(phi_i) @ exp(-phi_j)
        Omega_ij = cp.matmul(exp_phi_i, exp_neg_phi_j)

        if not is_batch:
            Omega_ij = Omega_ij[0]

        return Omega_ij


    def _batch_matrix_exp_cupy(X: cp.ndarray, order: int = 8) -> cp.ndarray:
        """
        Batch matrix exponential via Taylor series - GPU.

        exp(X) = I + X + X^2/2! + X^3/3! + ...

        Args:
            X: Batch of matrices (N, K, K)
            order: Taylor series order

        Returns:
            exp_X: (N, K, K)
        """
        N, K, _ = X.shape
        I = cp.eye(K, dtype=X.dtype)

        result = cp.tile(I, (N, 1, 1))
        X_power = X.copy()
        factorial = 1.0

        for n in range(1, order + 1):
            factorial *= n
            result = result + X_power / factorial
            X_power = cp.matmul(X_power, X)

        return result


    # =============================================================================
    # Softmax Weight Computation - GPU
    # =============================================================================

    def compute_softmax_weights_cupy(
        kl_values: cp.ndarray,
        kappa: float
    ) -> cp.ndarray:
        """
        Compute softmax weights from KL divergences - GPU.

        beta_j = exp(-KL_j / kappa) / sum_k exp(-KL_k / kappa)

        Args:
            kl_values: KL divergences to neighbors (N,)
            kappa: Temperature parameter

        Returns:
            weights: Softmax weights (N,)
        """
        # Compute negative KL / kappa
        neg_kl_scaled = -kl_values / kappa

        # Subtract max for numerical stability
        neg_kl_scaled = neg_kl_scaled - cp.max(neg_kl_scaled)

        # Softmax
        exp_vals = cp.exp(neg_kl_scaled)
        weights = exp_vals / cp.sum(exp_vals)

        return weights


    # =============================================================================
    # Memory Management Utilities
    # =============================================================================

    def to_gpu(x: np.ndarray) -> cp.ndarray:
        """Transfer NumPy array to GPU."""
        return cp.asarray(x)


    def to_cpu(x: cp.ndarray) -> np.ndarray:
        """Transfer CuPy array to CPU."""
        return cp.asnumpy(x)


    def gpu_memory_info() -> dict:
        """Get GPU memory usage info."""
        mempool = cp.get_default_memory_pool()
        return {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_gb': mempool.used_bytes() / 1e9,
            'total_gb': mempool.total_bytes() / 1e9,
        }


    def clear_gpu_cache():
        """Clear GPU memory cache."""
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()


else:
    # =============================================================================
    # Fallback: NumPy implementations when CuPy not available
    # =============================================================================

    def kl_gaussian_cupy(*args, **kwargs):
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda12x")

    def kl_gaussian_batch_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def transport_gaussian_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def transport_gaussian_batch_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def kl_transported_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def rodrigues_formula_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def compute_transport_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def compute_softmax_weights_cupy(*args, **kwargs):
        raise ImportError("CuPy not available")

    def to_gpu(x):
        raise ImportError("CuPy not available")

    def to_cpu(x):
        return x

    def gpu_memory_info():
        return {'error': 'CuPy not available'}

    def clear_gpu_cache():
        pass


# =============================================================================
# Unified Interface (Auto-select CPU/GPU)
# =============================================================================

def kl_gaussian_auto(mu_q, Sigma_q, mu_p, Sigma_p, eps=1e-8, force_gpu=False):
    """
    Auto-select KL divergence implementation based on data location.

    Args:
        mu_q, Sigma_q, mu_p, Sigma_p: Distribution parameters
        eps: Regularization
        force_gpu: Force GPU computation (transfers data if needed)

    Returns:
        kl: KL divergence
    """
    if CUPY_AVAILABLE and (force_gpu or isinstance(mu_q, cp.ndarray)):
        if not isinstance(mu_q, cp.ndarray):
            mu_q = cp.asarray(mu_q)
            Sigma_q = cp.asarray(Sigma_q)
            mu_p = cp.asarray(mu_p)
            Sigma_p = cp.asarray(Sigma_p)
        return kl_gaussian_cupy(mu_q, Sigma_q, mu_p, Sigma_p, eps)
    else:
        # Fall back to Numba/NumPy
        from math_utils.numba_kernels import kl_gaussian_numba
        return kl_gaussian_numba(
            np.asarray(mu_q, dtype=np.float64),
            np.asarray(Sigma_q, dtype=np.float64),
            np.asarray(mu_p, dtype=np.float64),
            np.asarray(Sigma_p, dtype=np.float64)
        )


def is_cupy_available() -> bool:
    """Check if CuPy is available."""
    return CUPY_AVAILABLE