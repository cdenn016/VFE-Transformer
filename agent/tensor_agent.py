# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:36:30 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
TensorAgent: PyTorch GPU-Accelerated Agent
==========================================

Agent with PyTorch tensor state for GPU acceleration + autograd.

This mirrors the NumPy Agent class but uses PyTorch tensors with
requires_grad=True for automatic differentiation.

Key differences from NumPy Agent:
- All state stored as torch.Tensor on GPU
- Supports autograd for gradient computation
- Batched operations via broadcasting
- Cholesky parameterization option for unconstrained optimization

Author:
Date: December 2024
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any


class TensorAgent(nn.Module):
    """
    Agent as smooth section with PyTorch tensor state.

    State tensors (all require_grad=True for autograd):
        mu_q: (*S, K) belief mean
        Sigma_q: (*S, K, K) belief covariance (SPD)
        mu_p: (*S, K) prior mean
        Sigma_p: (*S, K, K) prior covariance (SPD)
        phi: (*S, 3) gauge field in so(3)

    Where S = spatial_shape (empty tuple for 0D particle agents).

    Design choice: We store Sigma directly (not Cholesky L) to preserve
    gauge covariance: Under g in SO(3), Sigma -> g @ Sigma @ g.T is covariant,
    but cholesky(g @ Sigma @ g.T) != g @ cholesky(Sigma).

    For optimization, we provide L_q_param and L_p_param as alternative
    parameterizations that guarantee SPD via Sigma = L @ L.T.
    """

    def __init__(
        self,
        K: int,
        spatial_shape: Tuple[int, ...] = (),
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        use_cholesky_param: bool = False,
    ):
        """
        Initialize TensorAgent.

        Args:
            K: Latent dimension
            spatial_shape: Spatial dimensions, () for 0D particle
            device: 'cuda' or 'cpu'
            dtype: torch.float32 or torch.float64
            use_cholesky_param: If True, parameterize covariance via Cholesky
                               factor L where Sigma = L @ L.T. This guarantees
                               SPD but breaks gauge covariance.
        """
        super().__init__()
        self.K = K
        self.spatial_shape = spatial_shape
        self.device = device
        self.dtype = dtype
        self.use_cholesky_param = use_cholesky_param
        self.is_particle = (len(spatial_shape) == 0)

        # Determine tensor shapes
        if self.is_particle:
            mu_shape = (K,)
            sigma_shape = (K, K)
            phi_shape = (3,)
        else:
            mu_shape = (*spatial_shape, K)
            sigma_shape = (*spatial_shape, K, K)
            phi_shape = (*spatial_shape, 3)

        # === TRAINABLE PARAMETERS ===
        # Belief distribution q = N(mu_q, Sigma_q)
        self.mu_q = nn.Parameter(
            torch.zeros(mu_shape, device=device, dtype=dtype)
        )

        # Prior distribution p = N(mu_p, Sigma_p)
        self.mu_p = nn.Parameter(
            torch.zeros(mu_shape, device=device, dtype=dtype)
        )

        # Gauge field phi in so(3)
        self.phi = nn.Parameter(
            torch.zeros(phi_shape, device=device, dtype=dtype)
        )

        # Covariance parameterization
        if use_cholesky_param:
            # Parameterize via Cholesky: Sigma = L @ L.T
            # L is lower triangular, stored as full matrix (only lower used)
            self._L_q = nn.Parameter(
                torch.eye(K, device=device, dtype=dtype).expand(*sigma_shape).clone()
            )
            self._L_p = nn.Parameter(
                torch.eye(K, device=device, dtype=dtype).expand(*sigma_shape).clone()
            )
        else:
            # Direct Sigma parameterization (gauge-covariant)
            self._Sigma_q = nn.Parameter(
                torch.eye(K, device=device, dtype=dtype).expand(*sigma_shape).clone()
            )
            self._Sigma_p = nn.Parameter(
                torch.eye(K, device=device, dtype=dtype).expand(*sigma_shape).clone()
            )

        # Store shapes for reference
        self._mu_shape = mu_shape
        self._sigma_shape = sigma_shape
        self._phi_shape = phi_shape

    @property
    def Sigma_q(self) -> torch.Tensor:
        """Belief covariance Sigma_q, shape (*S, K, K)."""
        if self.use_cholesky_param:
            L = torch.tril(self._L_q)
            return L @ L.transpose(-1, -2)
        else:
            # Ensure symmetric (numerical stability)
            return 0.5 * (self._Sigma_q + self._Sigma_q.transpose(-1, -2))

    @property
    def Sigma_p(self) -> torch.Tensor:
        """Prior covariance Sigma_p, shape (*S, K, K)."""
        if self.use_cholesky_param:
            L = torch.tril(self._L_p)
            return L @ L.transpose(-1, -2)
        else:
            return 0.5 * (self._Sigma_p + self._Sigma_p.transpose(-1, -2))

    @property
    def L_q(self) -> torch.Tensor:
        """Belief Cholesky factor (computed if not parameterized)."""
        if self.use_cholesky_param:
            return torch.tril(self._L_q)
        else:
            return torch.linalg.cholesky(self.Sigma_q + 1e-6 * torch.eye(
                self.K, device=self.device, dtype=self.dtype
            ))

    @property
    def L_p(self) -> torch.Tensor:
        """Prior Cholesky factor (computed if not parameterized)."""
        if self.use_cholesky_param:
            return torch.tril(self._L_p)
        else:
            return torch.linalg.cholesky(self.Sigma_p + 1e-6 * torch.eye(
                self.K, device=self.device, dtype=self.dtype
            ))

    def initialize(
        self,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        phi_scale: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize with random state.

        Args:
            mu_scale: Scale for mean initialization
            sigma_scale: Scale for covariance initialization
            phi_scale: Scale for gauge field initialization
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            # Means: small random
            self.mu_q.normal_(0, mu_scale * 0.5)
            self.mu_p.normal_(0, mu_scale * 0.5)

            # Gauge field: small random in principal ball
            self.phi.uniform_(-phi_scale, phi_scale)

            # Covariances
            if self.use_cholesky_param:
                # Initialize L as scaled identity
                eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
                self._L_q.copy_(sigma_scale * eye.expand(*self._sigma_shape))
                self._L_p.copy_(1.5 * sigma_scale * eye.expand(*self._sigma_shape))
            else:
                # Initialize Sigma as scaled identity
                eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
                self._Sigma_q.copy_(sigma_scale * eye.expand(*self._sigma_shape))
                self._Sigma_p.copy_(1.5 * sigma_scale * eye.expand(*self._sigma_shape))

    def get_state_dict_numpy(self) -> Dict[str, Any]:
        """
        Get state as NumPy arrays for compatibility with existing code.

        Returns:
            Dictionary with NumPy arrays:
                mu_q, Sigma_q, mu_p, Sigma_p, phi
        """
        return {
            'mu_q': self.mu_q.detach().cpu().numpy(),
            'Sigma_q': self.Sigma_q.detach().cpu().numpy(),
            'mu_p': self.mu_p.detach().cpu().numpy(),
            'Sigma_p': self.Sigma_p.detach().cpu().numpy(),
            'phi': self.phi.detach().cpu().numpy(),
        }

    @classmethod
    def from_numpy_agent(
        cls,
        agent: 'Agent',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        use_cholesky_param: bool = False,
    ) -> 'TensorAgent':
        """
        Create TensorAgent from existing NumPy Agent.

        Args:
            agent: NumPy-based Agent instance
            device: Target device
            dtype: Target dtype
            use_cholesky_param: Whether to use Cholesky parameterization

        Returns:
            TensorAgent with copied state
        """
        t_agent = cls(
            K=agent.K,
            spatial_shape=agent.geometry.spatial_shape,
            device=device,
            dtype=dtype,
            use_cholesky_param=use_cholesky_param,
        )

        with torch.no_grad():
            # Copy means
            t_agent.mu_q.copy_(torch.tensor(agent.mu_q, device=device, dtype=dtype))
            t_agent.mu_p.copy_(torch.tensor(agent.mu_p, device=device, dtype=dtype))

            # Copy gauge field
            t_agent.phi.copy_(torch.tensor(agent.phi, device=device, dtype=dtype))

            # Copy covariances
            Sigma_q = torch.tensor(agent.Sigma_q, device=device, dtype=dtype)
            Sigma_p = torch.tensor(agent.Sigma_p, device=device, dtype=dtype)

            if use_cholesky_param:
                # Convert Sigma to Cholesky
                eye = 1e-6 * torch.eye(agent.K, device=device, dtype=dtype)
                L_q = torch.linalg.cholesky(Sigma_q + eye)
                L_p = torch.linalg.cholesky(Sigma_p + eye)
                t_agent._L_q.copy_(L_q)
                t_agent._L_p.copy_(L_p)
            else:
                t_agent._Sigma_q.copy_(Sigma_q)
                t_agent._Sigma_p.copy_(Sigma_p)

        return t_agent

    def to_numpy_agent(self, agent: 'Agent'):
        """
        Copy state back to a NumPy Agent.

        Args:
            agent: Target NumPy Agent to update in-place
        """
        import numpy as np

        state = self.get_state_dict_numpy()

        agent.mu_q = state['mu_q'].astype(np.float32)
        agent.mu_p = state['mu_p'].astype(np.float32)
        agent.Sigma_q = state['Sigma_q'].astype(np.float32)
        agent.Sigma_p = state['Sigma_p'].astype(np.float32)
        agent.gauge.phi = state['phi'].astype(np.float32)

        # Invalidate Cholesky caches
        agent._L_q_cache = None
        agent._L_p_cache = None

    def check_constraints(self, eps: float = 1e-6) -> Dict[str, Any]:
        """
        Verify manifold constraints are satisfied.

        Returns:
            Dictionary with constraint status and any violations
        """
        status = {
            'valid': True,
            'violations': [],
        }

        # Check Sigma_q positive-definite
        try:
            eigvals_q = torch.linalg.eigvalsh(self.Sigma_q)
            min_eig_q = eigvals_q.min().item()
            if min_eig_q <= eps:
                status['valid'] = False
                status['violations'].append(
                    f"Sigma_q not SPD: min eigenvalue = {min_eig_q:.3e}"
                )
        except RuntimeError as e:
            status['valid'] = False
            status['violations'].append(f"Sigma_q eigenvalue error: {e}")

        # Check Sigma_p positive-definite
        try:
            eigvals_p = torch.linalg.eigvalsh(self.Sigma_p)
            min_eig_p = eigvals_p.min().item()
            if min_eig_p <= eps:
                status['valid'] = False
                status['violations'].append(
                    f"Sigma_p not SPD: min eigenvalue = {min_eig_p:.3e}"
                )
        except RuntimeError as e:
            status['valid'] = False
            status['violations'].append(f"Sigma_p eigenvalue error: {e}")

        # Check phi in principal ball (|phi| < pi)
        phi_norms = torch.linalg.norm(self.phi, dim=-1)
        max_norm = phi_norms.max().item()
        if max_norm >= torch.pi - 0.01:
            status['valid'] = False
            status['violations'].append(
                f"phi violates principal ball: max ||phi|| = {max_norm:.3f}"
            )

        # Check for NaNs
        if torch.any(torch.isnan(self.mu_q)):
            status['valid'] = False
            status['violations'].append("NaN in mu_q")
        if torch.any(torch.isnan(self.Sigma_q)):
            status['valid'] = False
            status['violations'].append("NaN in Sigma_q")
        if torch.any(torch.isnan(self.phi)):
            status['valid'] = False
            status['violations'].append("NaN in phi")

        return status

    def count_parameters(self) -> int:
        """Count total learnable parameters."""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        mode = "Cholesky" if self.use_cholesky_param else "Direct"
        return (
            f"TensorAgent(K={self.K}, spatial_shape={self.spatial_shape}, "
            f"device={self.device}, param_mode={mode})"
        )