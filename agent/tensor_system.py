# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:43:14 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
TensorSystem: PyTorch GPU Multi-Agent System
=============================================

GPU-accelerated version of MultiAgentSystem using PyTorch tensors.
All agent states are batched into single tensors for efficient GPU computation.

Key differences from NumPy MultiAgentSystem:
- All N agents stored as batched tensors (N, *shape)
- Energy computation via FreeEnergy module with autograd
- No explicit gradient computation - use .backward()
- Support for spatial agents or 0D particle agents

Author: Claude (refactoring)
Date: December 2024
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class TensorSystem(nn.Module):
    """
    Multi-agent system with batched PyTorch tensors.

    All agent state is stored in batched tensors of shape (N, ...).
    This enables efficient GPU parallel computation across all agents.

    State tensors:
        mu_q: (N, K) or (N, *S, K) - belief means
        Sigma_q: (N, K, K) or (N, *S, K, K) - belief covariances
        mu_p: (N, K) or (N, *S, K) - prior means
        Sigma_p: (N, K, K) or (N, *S, K, K) - prior covariances
        phi: (N, 3) or (N, *S, 3) - gauge fields
        generators: (3, K, K) - Lie algebra generators (shared)

    Usage:
        system = TensorSystem(N=4, K=3, device='cuda')
        system.initialize()

        # Forward pass computes energy
        energy = system.compute_energy()

        # Autograd computes all gradients
        energy['total'].backward()

        # Gradients available in .grad attributes
        print(system.mu_q.grad)
    """

    def __init__(
        self,
        N: int,
        K: int,
        spatial_shape: Tuple[int, ...] = (),
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        use_cholesky_param: bool = False,
        lambda_self: float = 1.0,
        lambda_belief: float = 1.0,
        lambda_prior: float = 0.0,
        kappa_beta: float = 1.0,
        kappa_gamma: float = 1.0,
    ):
        """
        Initialize tensor system.

        Args:
            N: Number of agents
            K: Latent dimension
            spatial_shape: Spatial dimensions, () for 0D particles
            device: 'cuda' or 'cpu'
            dtype: torch.float32 or torch.float64
            use_cholesky_param: If True, parameterize Sigma via Cholesky
            lambda_self: Weight for self-coupling KL(q||p)
            lambda_belief: Weight for belief alignment
            lambda_prior: Weight for prior alignment
            kappa_beta: Temperature for belief softmax
            kappa_gamma: Temperature for prior softmax
        """
        super().__init__()

        self.N = N
        self.K = K
        self.spatial_shape = spatial_shape
        self.device = device
        self.dtype = dtype
        self.use_cholesky_param = use_cholesky_param
        self.is_particle = (len(spatial_shape) == 0)

        # Energy weights
        self.lambda_self = lambda_self
        self.lambda_belief = lambda_belief
        self.lambda_prior = lambda_prior
        self.kappa_beta = kappa_beta
        self.kappa_gamma = kappa_gamma

        # Determine shapes
        if self.is_particle:
            mu_shape = (N, K)
            sigma_shape = (N, K, K)
            phi_shape = (N, 3)
        else:
            mu_shape = (N, *spatial_shape, K)
            sigma_shape = (N, *spatial_shape, K, K)
            phi_shape = (N, *spatial_shape, 3)

        # === TRAINABLE PARAMETERS (batched across N agents) ===
        self.mu_q = nn.Parameter(
            torch.zeros(mu_shape, device=device, dtype=dtype)
        )
        self.mu_p = nn.Parameter(
            torch.zeros(mu_shape, device=device, dtype=dtype)
        )
        self.phi = nn.Parameter(
            torch.zeros(phi_shape, device=device, dtype=dtype)
        )

        if use_cholesky_param:
            eye = torch.eye(K, device=device, dtype=dtype)
            self._L_q = nn.Parameter(
                eye.unsqueeze(0).expand(*sigma_shape).clone()
            )
            self._L_p = nn.Parameter(
                eye.unsqueeze(0).expand(*sigma_shape).clone()
            )
        else:
            eye = torch.eye(K, device=device, dtype=dtype)
            self._Sigma_q = nn.Parameter(
                eye.unsqueeze(0).expand(*sigma_shape).clone()
            )
            self._Sigma_p = nn.Parameter(
                eye.unsqueeze(0).expand(*sigma_shape).clone()
            )

        # === GENERATORS (shared, non-trainable) ===
        self.register_buffer(
            'generators',
            self._create_so3_generators(K, device, dtype)
        )

        # Store shapes
        self._mu_shape = mu_shape
        self._sigma_shape = sigma_shape
        self._phi_shape = phi_shape

    def _create_so3_generators(
        self,
        K: int,
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create SO(3) Lie algebra generators.

        Returns generators J_a (a=1,2,3) where exp(phi . J) gives SO(3) element.
        For K > 3, we embed SO(3) in top-left 3x3 block.
        """
        generators = torch.zeros(3, K, K, device=device, dtype=dtype)

        # Standard so(3) generators (antisymmetric)
        # J_1 generates rotation around x-axis
        generators[0, 1, 2] = -1.0
        generators[0, 2, 1] = 1.0

        # J_2 generates rotation around y-axis
        generators[1, 0, 2] = 1.0
        generators[1, 2, 0] = -1.0

        # J_3 generates rotation around z-axis
        generators[2, 0, 1] = -1.0
        generators[2, 1, 0] = 1.0

        return generators

    @property
    def Sigma_q(self) -> torch.Tensor:
        """Belief covariances, shape (N, ..., K, K)."""
        if self.use_cholesky_param:
            L = torch.tril(self._L_q)
            return L @ L.transpose(-1, -2)
        else:
            return 0.5 * (self._Sigma_q + self._Sigma_q.transpose(-1, -2))

    @property
    def Sigma_p(self) -> torch.Tensor:
        """Prior covariances, shape (N, ..., K, K)."""
        if self.use_cholesky_param:
            L = torch.tril(self._L_p)
            return L @ L.transpose(-1, -2)
        else:
            return 0.5 * (self._Sigma_p + self._Sigma_p.transpose(-1, -2))

    def initialize(
        self,
        mu_scale: float = 1.0,
        sigma_scale: float = 1.0,
        phi_scale: float = 0.1,
        seed: Optional[int] = None,
        diversity: float = 1.0,
    ):
        """
        Initialize with random state.

        Args:
            mu_scale: Scale for mean initialization
            sigma_scale: Scale for covariance initialization
            phi_scale: Scale for gauge field initialization
            seed: Random seed
            diversity: Scale for agent-to-agent variation
        """
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            # Means: random with agent diversity
            self.mu_q.normal_(0, mu_scale * diversity)
            self.mu_p.normal_(0, mu_scale * diversity)

            # Gauge fields: small random in principal ball
            self.phi.uniform_(-phi_scale, phi_scale)

            # Covariances
            if self.use_cholesky_param:
                eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
                base_L = sigma_scale * eye
                self._L_q.copy_(
                    base_L.unsqueeze(0).expand(*self._sigma_shape)
                )
                self._L_p.copy_(
                    (1.5 * sigma_scale) * eye.unsqueeze(0).expand(*self._sigma_shape)
                )
            else:
                eye = torch.eye(self.K, device=self.device, dtype=self.dtype)
                base_Sigma = sigma_scale * eye
                self._Sigma_q.copy_(
                    base_Sigma.unsqueeze(0).expand(*self._sigma_shape)
                )
                self._Sigma_p.copy_(
                    (1.5 * sigma_scale) * eye.unsqueeze(0).expand(*self._sigma_shape)
                )

    def compute_energy(self, eps: float = 1e-6) -> Dict[str, torch.Tensor]:
        """
        Compute total free energy (differentiable).

        Returns:
            Dictionary with:
                'total': Total free energy
                'self': Self-coupling energy (sum of KL(q_i || p_i))
                'belief': Belief alignment energy
                'prior': Prior alignment energy
        """
        from gradients.torch_energy import FreeEnergy

        fe = FreeEnergy(
            lambda_self=self.lambda_self,
            lambda_belief=self.lambda_belief,
            lambda_prior=self.lambda_prior,
            kappa_beta=self.kappa_beta,
            kappa_gamma=self.kappa_gamma,
            eps=eps,
        )

        return fe(
            self.mu_q, self.Sigma_q,
            self.mu_p, self.Sigma_p,
            self.phi, self.generators
        )

    def compute_kl_self(self, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute per-agent KL(q_i || p_i).

        Returns:
            kl: (N,) tensor of KL divergences
        """
        from gradients.torch_energy import kl_divergence_gaussian

        return kl_divergence_gaussian(
            self.mu_q, self.Sigma_q,
            self.mu_p, self.Sigma_p,
            eps=eps
        )

    def get_state_dict_numpy(self) -> Dict[str, np.ndarray]:
        """
        Get all state as NumPy arrays.

        Returns:
            Dictionary with NumPy arrays for all state tensors
        """
        return {
            'mu_q': self.mu_q.detach().cpu().numpy(),
            'Sigma_q': self.Sigma_q.detach().cpu().numpy(),
            'mu_p': self.mu_p.detach().cpu().numpy(),
            'Sigma_p': self.Sigma_p.detach().cpu().numpy(),
            'phi': self.phi.detach().cpu().numpy(),
            'generators': self.generators.detach().cpu().numpy(),
        }

    @classmethod
    def from_multi_agent_system(
        cls,
        system: 'MultiAgentSystem',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        use_cholesky_param: bool = False,
    ) -> 'TensorSystem':
        """
        Create TensorSystem from existing NumPy MultiAgentSystem.

        Args:
            system: NumPy-based MultiAgentSystem
            device: Target device
            dtype: Target dtype
            use_cholesky_param: Whether to use Cholesky parameterization

        Returns:
            TensorSystem with copied state
        """
        agents = system.agents
        N = len(agents)
        K = agents[0].K
        spatial_shape = agents[0].geometry.spatial_shape

        # Get config values if available
        config = system.config if hasattr(system, 'config') else None

        t_system = cls(
            N=N,
            K=K,
            spatial_shape=spatial_shape,
            device=device,
            dtype=dtype,
            use_cholesky_param=use_cholesky_param,
            lambda_self=getattr(config, 'alpha', 1.0) if config else 1.0,
            lambda_belief=getattr(config, 'lambda_belief', 1.0) if config else 1.0,
            lambda_prior=getattr(config, 'lambda_prior', 0.0) if config else 0.0,
            kappa_beta=getattr(config, 'kappa_beta', 1.0) if config else 1.0,
            kappa_gamma=getattr(config, 'kappa_gamma', 1.0) if config else 1.0,
        )

        # Copy state from all agents
        with torch.no_grad():
            mu_q_list = [torch.tensor(a.mu_q, device=device, dtype=dtype)
                         for a in agents]
            mu_p_list = [torch.tensor(a.mu_p, device=device, dtype=dtype)
                         for a in agents]
            phi_list = [torch.tensor(a.phi, device=device, dtype=dtype)
                        for a in agents]
            Sigma_q_list = [torch.tensor(a.Sigma_q, device=device, dtype=dtype)
                            for a in agents]
            Sigma_p_list = [torch.tensor(a.Sigma_p, device=device, dtype=dtype)
                            for a in agents]

            t_system.mu_q.copy_(torch.stack(mu_q_list))
            t_system.mu_p.copy_(torch.stack(mu_p_list))
            t_system.phi.copy_(torch.stack(phi_list))

            if use_cholesky_param:
                # Convert Sigma to Cholesky
                eye = 1e-6 * torch.eye(K, device=device, dtype=dtype)
                for i, (Sq, Sp) in enumerate(zip(Sigma_q_list, Sigma_p_list)):
                    L_q = torch.linalg.cholesky(Sq + eye)
                    L_p = torch.linalg.cholesky(Sp + eye)
                    t_system._L_q.data[i] = L_q
                    t_system._L_p.data[i] = L_p
            else:
                t_system._Sigma_q.copy_(torch.stack(Sigma_q_list))
                t_system._Sigma_p.copy_(torch.stack(Sigma_p_list))

            # Copy generators from first agent
            if hasattr(agents[0], 'generators'):
                t_system.generators.copy_(
                    torch.tensor(agents[0].generators, device=device, dtype=dtype)
                )

        return t_system

    def to_multi_agent_system(
        self,
        system: 'MultiAgentSystem'
    ):
        """
        Copy state back to NumPy MultiAgentSystem.

        Args:
            system: Target MultiAgentSystem to update in-place
        """
        state = self.get_state_dict_numpy()

        for i, agent in enumerate(system.agents):
            agent.mu_q = state['mu_q'][i].astype(np.float32)
            agent.mu_p = state['mu_p'][i].astype(np.float32)
            agent.Sigma_q = state['Sigma_q'][i].astype(np.float32)
            agent.Sigma_p = state['Sigma_p'][i].astype(np.float32)
            agent.gauge.phi = state['phi'][i].astype(np.float32)

            # Invalidate caches
            if hasattr(agent, '_L_q_cache'):
                agent._L_q_cache = None
            if hasattr(agent, '_L_p_cache'):
                agent._L_p_cache = None

    def check_constraints(self, eps: float = 1e-6) -> Dict[str, Any]:
        """
        Verify manifold constraints for all agents.

        Returns:
            Dictionary with constraint status
        """
        status = {
            'valid': True,
            'violations': [],
            'min_eig_q': None,
            'min_eig_p': None,
            'max_phi_norm': None,
        }

        # Check Sigma_q positive-definite
        try:
            eigvals_q = torch.linalg.eigvalsh(self.Sigma_q)
            min_eig_q = eigvals_q.min().item()
            status['min_eig_q'] = min_eig_q
            if min_eig_q <= eps:
                status['valid'] = False
                status['violations'].append(
                    f"Sigma_q not SPD: min eigenvalue = {min_eig_q:.3e}"
                )
        except RuntimeError as e:
            status['valid'] = False
            status['violations'].append(f"Sigma_q error: {e}")

        # Check Sigma_p positive-definite
        try:
            eigvals_p = torch.linalg.eigvalsh(self.Sigma_p)
            min_eig_p = eigvals_p.min().item()
            status['min_eig_p'] = min_eig_p
            if min_eig_p <= eps:
                status['valid'] = False
                status['violations'].append(
                    f"Sigma_p not SPD: min eigenvalue = {min_eig_p:.3e}"
                )
        except RuntimeError as e:
            status['valid'] = False
            status['violations'].append(f"Sigma_p error: {e}")

        # Check phi in principal ball
        phi_norms = torch.linalg.norm(self.phi, dim=-1)
        max_norm = phi_norms.max().item()
        status['max_phi_norm'] = max_norm
        if max_norm >= torch.pi - 0.01:
            status['valid'] = False
            status['violations'].append(
                f"phi outside principal ball: max ||phi|| = {max_norm:.3f}"
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
            f"TensorSystem(N={self.N}, K={self.K}, "
            f"spatial_shape={self.spatial_shape}, "
            f"device={self.device}, param_mode={mode})"
        )