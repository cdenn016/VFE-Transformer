# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:47:07 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
PyTorch GPU Backend for VFE Simulation
=======================================

High-performance GPU backend using PyTorch with:
- torch.compile for kernel fusion (PyTorch 2.0+)
- Persistent GPU tensors (no CPU↔GPU transfers)
- Batched operations for maximum throughput
- FP32/FP16 mixed precision support

Performance Target (RTX 5090):
- 50-100x speedup over CPU for large systems
- Sub-millisecond gradient computation per step

Usage:
------
    from math_utils.torch_backend import TorchBackend

    backend = TorchBackend(device='cuda', dtype=torch.float32)

    # Transfer beliefs to GPU once at simulation start
    mu_gpu = backend.to_device(mu_np)
    Sigma_gpu = backend.to_device(Sigma_np)

    # All operations stay on GPU
    kl = backend.kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p)
    mu_t, Sigma_t = backend.transport_gaussian(mu, Sigma, Omega)

Author: Chris
Date: December 2025
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import warnings

# Check PyTorch availability
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = (0, 0)


def is_torch_available() -> bool:
    """Check if PyTorch CUDA is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


# =============================================================================
# Core Backend Class
# =============================================================================

class TorchBackend:
    """
    PyTorch-native GPU backend for VFE simulation.

    Key design principles:
    1. Keep data on GPU throughout simulation step
    2. Use compiled functions for kernel fusion
    3. Batch operations across agents when possible
    4. Minimize synchronization points
    """

    def __init__(
        self,
        device: str = 'cuda',
        dtype: 'torch.dtype' = None,
        compile_mode: str = 'reduce-overhead',  # 'default', 'reduce-overhead', 'max-autotune'
        use_compile: bool = True
    ):
        """
        Initialize PyTorch backend.

        Args:
            device: 'cuda' or 'cuda:N' for specific GPU
            dtype: torch.float32 or torch.float64 (float32 faster on RTX)
            compile_mode: torch.compile optimization mode
            use_compile: Whether to use torch.compile (requires PyTorch 2.0+)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.compile_mode = compile_mode

        # Only use compile if PyTorch 2.0+ and requested
        self.use_compile = use_compile and TORCH_VERSION >= (2, 0)

        # Compile core functions
        self._compiled_kl = None
        self._compiled_transport = None
        self._compiled_rodrigues = None

        if self.use_compile:
            self._compile_functions()

        # Print info
        if torch.cuda.is_available():
            print(f"[TorchBackend] Using {torch.cuda.get_device_name(self.device)}")
            print(f"[TorchBackend] dtype={self.dtype}, compile={self.use_compile}")

    def _compile_functions(self):
        """Compile core functions using torch.compile."""
        try:
            self._compiled_kl = torch.compile(
                self._kl_gaussian_impl,
                mode=self.compile_mode,
                fullgraph=False  # Allow graph breaks for flexibility
            )
            self._compiled_transport = torch.compile(
                self._transport_gaussian_impl,
                mode=self.compile_mode,
                fullgraph=False
            )
            self._compiled_rodrigues = torch.compile(
                self._rodrigues_formula_impl,
                mode=self.compile_mode,
                fullgraph=True  # This one is simple enough for full graph
            )
            print(f"[TorchBackend] Compiled functions with mode={self.compile_mode}")
        except RuntimeError as e:
            warnings.warn(f"torch.compile failed: {e}. Falling back to eager mode.")
            self.use_compile = False

    # =========================================================================
    # Device Transfer
    # =========================================================================

    def to_device(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Transfer array to GPU."""
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.tensor(x, device=self.device, dtype=self.dtype)

    def to_numpy(self, x: 'torch.Tensor') -> np.ndarray:
        """Transfer tensor to CPU as NumPy array."""
        return x.detach().cpu().numpy()

    def synchronize(self):
        """Block until all GPU operations complete."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    # =========================================================================
    # KL Divergence - Core Operation
    # =========================================================================

    def _kl_gaussian_impl(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        KL divergence KL(q || p) between Gaussians.

        Pure PyTorch implementation for GPU compilation.
        """
        K = mu_q.shape[-1]

        # Regularize covariances
        eye = torch.eye(K, device=self.device, dtype=self.dtype)
        Sigma_q_reg = Sigma_q + eps * eye
        Sigma_p_reg = Sigma_p + eps * eye

        # Cholesky decomposition (stable)
        L_p = torch.linalg.cholesky(Sigma_p_reg)
        L_q = torch.linalg.cholesky(Sigma_q_reg)

        # Log determinants: log|Σ| = 2 * Σ log(L_ii)
        logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)

        # Trace term: tr(Σ_p^{-1} Σ_q)
        # Solve L_p @ X = Σ_q for X, then tr(L_p^{-T} @ X) = tr(Σ_p^{-1} @ Σ_q)
        Sigma_p_inv = torch.cholesky_inverse(L_p)
        trace_term = torch.sum(Sigma_p_inv * Sigma_q_reg, dim=(-2, -1))

        # Quadratic term: (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q)
        delta = mu_p - mu_q
        # Solve L_p @ y = delta, then ||y||^2 = delta^T Σ_p^{-1} delta
        y = torch.linalg.solve_triangular(L_p, delta.unsqueeze(-1), upper=False)
        quad_term = torch.sum(y ** 2, dim=(-2, -1))

        # Combine: KL = 0.5 * (tr + quad - K + logdet_p - logdet_q)
        kl = 0.5 * (trace_term + quad_term - K + logdet_p - logdet_q)

        return torch.clamp(kl, min=0.0)

    def kl_gaussian(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(q || p).

        Uses compiled version if available.
        """
        if self._compiled_kl is not None:
            return self._compiled_kl(mu_q, Sigma_q, mu_p, Sigma_p, eps)
        return self._kl_gaussian_impl(mu_q, Sigma_q, mu_p, Sigma_p, eps)

    def kl_gaussian_batch(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p_batch: torch.Tensor,
        Sigma_p_batch: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Batched KL divergence: KL(q || p_i) for multiple targets.

        Args:
            mu_q: (K,) source mean
            Sigma_q: (K, K) source covariance
            mu_p_batch: (N, K) target means
            Sigma_p_batch: (N, K, K) target covariances

        Returns:
            kl_batch: (N,) KL divergences
        """
        N = mu_p_batch.shape[0]
        K = mu_q.shape[-1]

        # Broadcast source to batch
        mu_q_batch = mu_q.unsqueeze(0).expand(N, -1)
        Sigma_q_batch = Sigma_q.unsqueeze(0).expand(N, -1, -1)

        # Vectorized KL computation
        eye = torch.eye(K, device=self.device, dtype=self.dtype)
        Sigma_q_reg = Sigma_q_batch + eps * eye
        Sigma_p_reg = Sigma_p_batch + eps * eye

        # Batched Cholesky
        L_p = torch.linalg.cholesky(Sigma_p_reg)
        L_q = torch.linalg.cholesky(Sigma_q_reg)

        # Batched log determinants
        logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1)), dim=-1)
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)

        # Batched trace term
        Sigma_p_inv = torch.cholesky_inverse(L_p)
        trace_term = torch.sum(Sigma_p_inv * Sigma_q_reg, dim=(-2, -1))

        # Batched quadratic term
        delta = mu_p_batch - mu_q_batch
        y = torch.linalg.solve_triangular(L_p, delta.unsqueeze(-1), upper=False)
        quad_term = torch.sum(y ** 2, dim=(-2, -1))

        # Combine
        kl_batch = 0.5 * (trace_term + quad_term - K + logdet_p - logdet_q)

        return torch.clamp(kl_batch, min=0.0)

    # =========================================================================
    # Gaussian Transport
    # =========================================================================

    def _transport_gaussian_impl(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        Omega: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transport Gaussian by operator: (Ω μ, Ω Σ Ω^T).
        """
        # Mean transport
        mu_t = torch.matmul(Omega, mu)

        # Covariance transport: Σ' = Ω Σ Ω^T
        Sigma_t = torch.matmul(torch.matmul(Omega, Sigma), Omega.transpose(-2, -1))

        # Symmetrize for numerical stability
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.transpose(-2, -1))

        return mu_t, Sigma_t

    def transport_gaussian(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        Omega: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transport Gaussian distribution by operator Ω.

        Uses compiled version if available.
        """
        if self._compiled_transport is not None:
            return self._compiled_transport(mu, Sigma, Omega)
        return self._transport_gaussian_impl(mu, Sigma, Omega)

    def transport_gaussian_batch(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        Omega_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch transport: apply N transport operators to one Gaussian.

        Args:
            mu: (K,) mean
            Sigma: (K, K) covariance
            Omega_batch: (N, K, K) transport operators

        Returns:
            mu_batch: (N, K) transported means
            Sigma_batch: (N, K, K) transported covariances
        """
        # Batched mean: μ' = Ω @ μ
        mu_batch = torch.einsum('nij,j->ni', Omega_batch, mu)

        # Batched covariance: Σ' = Ω @ Σ @ Ω^T
        temp = torch.einsum('nij,jk->nik', Omega_batch, Sigma)
        Sigma_batch = torch.einsum('nij,nkj->nik', temp, Omega_batch)

        # Symmetrize
        Sigma_batch = 0.5 * (Sigma_batch + Sigma_batch.transpose(-2, -1))

        return mu_batch, Sigma_batch

    # =========================================================================
    # SO(3) Exponential Map (Rodrigues Formula)
    # =========================================================================

    def _rodrigues_formula_impl(
        self,
        phi: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Rodrigues formula: exp(φ) for φ ∈ so(3).

        R = I + sin(θ)/θ * [φ]_× + (1-cos(θ))/θ² * [φ]_×²

        Args:
            phi: (..., 3) axis-angle vector

        Returns:
            R: (..., 3, 3) rotation matrix
        """
        # Compute angle
        theta = torch.norm(phi, dim=-1, keepdim=True)  # (..., 1)
        theta_sq = theta ** 2

        # Small angle approximation coefficients
        # sin(θ)/θ ≈ 1 - θ²/6 for small θ
        # (1-cos(θ))/θ² ≈ 0.5 - θ²/24 for small θ
        small_angle = theta.squeeze(-1) < eps

        # Safe division
        theta_safe = torch.where(theta < eps, torch.ones_like(theta), theta)
        theta_sq_safe = torch.where(theta_sq < eps**2, torch.ones_like(theta_sq), theta_sq)

        # Coefficients
        c1 = torch.where(
            theta < eps,
            1.0 - theta_sq / 6.0,
            torch.sin(theta) / theta_safe
        )
        c2 = torch.where(
            theta < eps,
            0.5 - theta_sq / 24.0,
            (1.0 - torch.cos(theta)) / theta_sq_safe
        )

        # Build skew-symmetric matrix [φ]_×
        batch_shape = phi.shape[:-1]
        phi_x = torch.zeros(*batch_shape, 3, 3, device=self.device, dtype=self.dtype)

        phi_x[..., 0, 1] = -phi[..., 2]
        phi_x[..., 0, 2] = phi[..., 1]
        phi_x[..., 1, 0] = phi[..., 2]
        phi_x[..., 1, 2] = -phi[..., 0]
        phi_x[..., 2, 0] = -phi[..., 1]
        phi_x[..., 2, 1] = phi[..., 0]

        # [φ]_×²
        phi_x_sq = torch.matmul(phi_x, phi_x)

        # R = I + c1 * [φ]_× + c2 * [φ]_×²
        eye = torch.eye(3, device=self.device, dtype=self.dtype)
        R = eye + c1.unsqueeze(-1) * phi_x + c2.unsqueeze(-1) * phi_x_sq

        return R

    def rodrigues_formula(
        self,
        phi: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        SO(3) exponential map via Rodrigues formula.

        Uses compiled version if available.
        """
        if self._compiled_rodrigues is not None:
            return self._compiled_rodrigues(phi, eps)
        return self._rodrigues_formula_impl(phi, eps)

    # =========================================================================
    # Transport Operator Computation
    # =========================================================================

    def compute_transport(
        self,
        phi_i: torch.Tensor,
        phi_j: torch.Tensor,
        generators: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute transport operator Ω_ij = exp(φ_i) · exp(-φ_j).

        For SO(3) gauge fields in K-dimensional representation.

        Args:
            phi_i: (..., 3) source gauge field
            phi_j: (..., 3) target gauge field
            generators: (3, K, K) Lie algebra generators

        Returns:
            Omega: (..., K, K) transport operator
        """
        # Get rotation matrices in 3D
        R_i = self.rodrigues_formula(phi_i, eps)      # (..., 3, 3)
        R_j = self.rodrigues_formula(-phi_j, eps)     # (..., 3, 3)

        # Compose: R_ij = R_i @ R_j
        R_ij = torch.matmul(R_i, R_j)  # (..., 3, 3)

        # Extract axis-angle from composed rotation (for K-dim representation)
        # Using logarithm map
        phi_ij = self._rotation_to_axis_angle(R_ij)  # (..., 3)

        # Compute K-dimensional representation: Ω = exp(φ · J)
        # where J are the generators
        K = generators.shape[1]
        Omega = self._exponential_map_representation(phi_ij, generators)

        return Omega

    def _rotation_to_axis_angle(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to axis-angle representation.

        Uses the logarithm map: log(R) = (θ / 2sin(θ)) * (R - R^T)
        """
        # Trace gives angle: tr(R) = 1 + 2*cos(θ)
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)  # (...,)

        # Extract axis from skew part
        # R - R^T = 2 sin(θ) [n]_×
        skew = R - R.transpose(-2, -1)

        # Small angle: use linear approximation
        small_angle = theta < 1e-6

        # Axis components from skew matrix
        axis = torch.stack([
            skew[..., 2, 1],
            skew[..., 0, 2],
            skew[..., 1, 0]
        ], dim=-1)  # (..., 3)

        # Normalize by 2*sin(θ)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        sin_theta_safe = torch.where(
            sin_theta.abs() < 1e-8,
            torch.ones_like(sin_theta),
            sin_theta
        )

        axis = axis / (2.0 * sin_theta_safe)

        # Scale by angle
        phi = theta.unsqueeze(-1) * axis

        # For small angles, use direct extraction
        phi = torch.where(
            small_angle.unsqueeze(-1),
            axis / 2.0,  # First-order approximation
            phi
        )

        return phi

    def _exponential_map_representation(
        self,
        phi: torch.Tensor,
        generators: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute matrix exponential in representation space.

        Ω = exp(φ_a J^a) where J^a are generators.

        Args:
            phi: (..., 3) Lie algebra coordinates
            generators: (3, K, K) generators

        Returns:
            Omega: (..., K, K) group element
        """
        K = generators.shape[1]

        # Contract: X = φ_a J^a
        # generators: (3, K, K), phi: (..., 3)
        X = torch.einsum('...a,aij->...ij', phi, generators)  # (..., K, K)

        # Matrix exponential
        Omega = torch.linalg.matrix_exp(X)

        return Omega

    # =========================================================================
    # Softmax Coupling Weights
    # =========================================================================

    def compute_softmax_weights(
        self,
        kl_values: torch.Tensor,
        kappa: float
    ) -> torch.Tensor:
        """
        Compute softmax coupling weights: β_j = exp(-κ * KL_j) / Σ exp(-κ * KL_k)

        Args:
            kl_values: (N,) KL divergences
            kappa: Temperature parameter (higher = sharper)

        Returns:
            weights: (N,) normalized weights
        """
        # Use log-sum-exp trick for numerical stability
        scaled = -kappa * kl_values
        weights = F.softmax(scaled, dim=-1)
        return weights

    # =========================================================================
    # Batch Operations for Full System
    # =========================================================================

    def compute_all_pairwise_kl(
        self,
        mu_all: torch.Tensor,
        Sigma_all: torch.Tensor,
        Omega_all: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute all pairwise KL divergences with transport.

        Args:
            mu_all: (N, K) all agent means
            Sigma_all: (N, K, K) all agent covariances
            Omega_all: (N, N, K, K) transport operators [i,j] = Ω_ij

        Returns:
            kl_matrix: (N, N) where [i,j] = KL(q_i || T_ij q_j)
        """
        N, K = mu_all.shape
        kl_matrix = torch.zeros(N, N, device=self.device, dtype=self.dtype)

        for i in range(N):
            # Transport all j's beliefs to i's frame
            # Ω_ij @ μ_j, Ω_ij @ Σ_j @ Ω_ij^T
            mu_transported = torch.einsum('jkl,jl->jk', Omega_all[i], mu_all)  # (N, K)
            temp = torch.einsum('jkl,jlm->jkm', Omega_all[i], Sigma_all)       # (N, K, K)
            Sigma_transported = torch.einsum('jkl,jml->jkm', temp, Omega_all[i])  # (N, K, K)

            # Symmetrize
            Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.transpose(-2, -1))

            # Compute KL(q_i || transported q_j) for all j
            kl_matrix[i] = self.kl_gaussian_batch(
                mu_all[i], Sigma_all[i],
                mu_transported, Sigma_transported
            )

        return kl_matrix

    # =========================================================================
    # Free Energy Computation
    # =========================================================================

    def compute_free_energy(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute variational free energy F = KL(q||p) = E_q[log q] - E_q[log p]

        Returns:
            F: Total free energy
            energy: -E_q[log p] (energy term)
            entropy: -E_q[log q] (entropy term)
        """
        K = mu_q.shape[-1]

        # KL divergence
        kl = self.kl_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, eps)

        # Entropy of q: H(q) = 0.5 * (K * (1 + log(2π)) + log|Σ_q|)
        eye = torch.eye(K, device=self.device, dtype=self.dtype)
        L_q = torch.linalg.cholesky(Sigma_q + eps * eye)
        logdet_q = 2.0 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)
        entropy = 0.5 * (K * (1.0 + np.log(2 * np.pi)) + logdet_q)

        # Energy = F - (-H) = F + H
        energy = kl + entropy

        return kl, energy, -entropy


# =============================================================================
# Singleton Instance for Global Use
# =============================================================================

_GLOBAL_BACKEND: Optional[TorchBackend] = None


def get_torch_backend(
    device: str = 'cuda',
    dtype: 'torch.dtype' = None,
    reinitialize: bool = False
) -> TorchBackend:
    """
    Get or create global TorchBackend instance.

    Args:
        device: GPU device
        dtype: Data type (default float32 for speed)
        reinitialize: Force re-creation of backend

    Returns:
        TorchBackend instance
    """
    global _GLOBAL_BACKEND

    if _GLOBAL_BACKEND is None or reinitialize:
        _GLOBAL_BACKEND = TorchBackend(device=device, dtype=dtype)

    return _GLOBAL_BACKEND


# =============================================================================
# NumPy-Compatible Wrapper Functions
# =============================================================================
# These can be called directly with NumPy arrays and handle GPU transfer

def kl_gaussian_torch(
    mu_q: np.ndarray,
    Sigma_q: np.ndarray,
    mu_p: np.ndarray,
    Sigma_p: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    KL divergence using PyTorch backend.

    Accepts NumPy arrays, returns Python float.
    """
    backend = get_torch_backend()

    mu_q_t = backend.to_device(mu_q)
    Sigma_q_t = backend.to_device(Sigma_q)
    mu_p_t = backend.to_device(mu_p)
    Sigma_p_t = backend.to_device(Sigma_p)

    kl = backend.kl_gaussian(mu_q_t, Sigma_q_t, mu_p_t, Sigma_p_t, eps)

    return kl.item()


def transport_gaussian_torch(
    mu: np.ndarray,
    Sigma: np.ndarray,
    Omega: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transport Gaussian using PyTorch backend.

    Accepts NumPy arrays, returns NumPy arrays.
    """
    backend = get_torch_backend()

    mu_t = backend.to_device(mu)
    Sigma_t = backend.to_device(Sigma)
    Omega_t = backend.to_device(Omega)

    mu_out, Sigma_out = backend.transport_gaussian(mu_t, Sigma_t, Omega_t)

    return backend.to_numpy(mu_out), backend.to_numpy(Sigma_out)


def compute_transport_torch(
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    generators: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute transport operator using PyTorch backend.

    Accepts NumPy arrays, returns NumPy array.
    """
    backend = get_torch_backend()

    phi_i_t = backend.to_device(phi_i)
    phi_j_t = backend.to_device(phi_j)
    gen_t = backend.to_device(generators)

    Omega = backend.compute_transport(phi_i_t, phi_j_t, gen_t, eps)

    return backend.to_numpy(Omega)