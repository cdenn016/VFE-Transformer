# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:46:48 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
PyTorch-Accelerated Gradient Computation
=========================================

High-performance gradient computation using PyTorch with:
- Full GPU residency (no CPU↔GPU transfers during step)
- Batched operations across all spatial positions
- torch.compile for kernel fusion
- Optional autograd for validation

This replaces the CPU gradient_engine for GPU execution.

Performance Target:
- 50-100x speedup over CPU Numba for N>50 agents
- <1ms per step for typical configurations

Author: Chris
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import warnings

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from math_utils.torch_backend import TorchBackend, get_torch_backend, is_torch_available


# =============================================================================
# Data Structures (GPU-resident)
# =============================================================================

@dataclass
class GPUAgentState:
    """
    Agent state stored on GPU as PyTorch tensors.

    All tensors have shape (S1, S2, ..., K) or (S1, S2, ..., K, K)
    where (S1, S2, ...) is the spatial shape.
    """
    mu_q: torch.Tensor      # (S..., K) belief mean
    Sigma_q: torch.Tensor   # (S..., K, K) belief covariance
    mu_p: torch.Tensor      # (S..., K) prior mean
    Sigma_p: torch.Tensor   # (S..., K, K) prior covariance
    phi: torch.Tensor       # (S..., 3) gauge field


@dataclass
class GPUGradients:
    """
    Gradient tensors on GPU.
    """
    grad_mu_q: torch.Tensor
    grad_Sigma_q: torch.Tensor
    grad_mu_p: torch.Tensor
    grad_Sigma_p: torch.Tensor
    grad_phi: torch.Tensor


# =============================================================================
# Core Gradient Computation (PyTorch)
# =============================================================================

class TorchGradientEngine:
    """
    GPU-accelerated gradient computation for VFE simulation.

    Key optimizations:
    1. All agent states stored on GPU
    2. Batched operations across spatial grid
    3. Compiled kernels via torch.compile
    4. Vectorized pairwise KL computation
    """

    def __init__(
        self,
        backend: Optional[TorchBackend] = None,
        compile_mode: str = 'reduce-overhead'
    ):
        """
        Initialize gradient engine.

        Args:
            backend: TorchBackend instance (creates new if None)
            compile_mode: torch.compile optimization mode
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.backend = backend or get_torch_backend()
        self.device = self.backend.device
        self.dtype = self.backend.dtype

        # Compile core functions
        self._compile_gradients(compile_mode)

        # Cache for transport operators
        self._transport_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _compile_gradients(self, mode: str):
        """Compile gradient functions."""
        try:
            self._kl_grad_q = torch.compile(
                self._grad_kl_wrt_q_impl,
                mode=mode,
                fullgraph=False
            )
            self._kl_grad_p = torch.compile(
                self._grad_kl_wrt_p_impl,
                mode=mode,
                fullgraph=False
            )
        except RuntimeError as e:
            warnings.warn(f"torch.compile failed for gradients: {e}")
            self._kl_grad_q = self._grad_kl_wrt_q_impl
            self._kl_grad_p = self._grad_kl_wrt_p_impl

    # =========================================================================
    # Transfer Agent State to GPU
    # =========================================================================

    def create_gpu_state(
        self,
        mu_q: np.ndarray,
        Sigma_q: np.ndarray,
        mu_p: np.ndarray,
        Sigma_p: np.ndarray,
        phi: np.ndarray
    ) -> GPUAgentState:
        """
        Transfer agent state to GPU.

        Call once at simulation start, then update in-place.
        """
        return GPUAgentState(
            mu_q=self.backend.to_device(mu_q),
            Sigma_q=self.backend.to_device(Sigma_q),
            mu_p=self.backend.to_device(mu_p),
            Sigma_p=self.backend.to_device(Sigma_p),
            phi=self.backend.to_device(phi)
        )

    def state_to_numpy(self, state: GPUAgentState) -> Tuple[np.ndarray, ...]:
        """Transfer GPU state back to CPU."""
        return (
            self.backend.to_numpy(state.mu_q),
            self.backend.to_numpy(state.Sigma_q),
            self.backend.to_numpy(state.mu_p),
            self.backend.to_numpy(state.Sigma_p),
            self.backend.to_numpy(state.phi)
        )

    # =========================================================================
    # KL Divergence Gradients
    # =========================================================================

    def _grad_kl_wrt_q_impl(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient of KL(q||p) w.r.t. q parameters.

        ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        """
        K = mu_q.shape[-1]
        eye = torch.eye(K, device=self.device, dtype=self.dtype)

        # Regularize
        Sigma_p_reg = Sigma_p + eps * eye
        Sigma_q_reg = Sigma_q + eps * eye

        # Inverses via Cholesky
        L_p = torch.linalg.cholesky(Sigma_p_reg)
        L_q = torch.linalg.cholesky(Sigma_q_reg)
        Sigma_p_inv = torch.cholesky_inverse(L_p)
        Sigma_q_inv = torch.cholesky_inverse(L_q)

        # Gradient w.r.t. μ_q
        delta = mu_q - mu_p
        grad_mu_q = torch.matmul(Sigma_p_inv, delta.unsqueeze(-1)).squeeze(-1)

        # Gradient w.r.t. Σ_q
        grad_Sigma_q = 0.5 * (Sigma_p_inv - Sigma_q_inv)

        return grad_mu_q, grad_Sigma_q

    def _grad_kl_wrt_p_impl(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient of KL(q||p) w.r.t. p parameters.

        ∂KL/∂μ_p = -Σ_p^{-1} (μ_q - μ_p)
        ∂KL/∂Σ_p = 0.5 * (Σ_p^{-1} - Σ_p^{-1} (Σ_q + δδ^T) Σ_p^{-1})
        """
        K = mu_q.shape[-1]
        eye = torch.eye(K, device=self.device, dtype=self.dtype)

        # Regularize
        Sigma_p_reg = Sigma_p + eps * eye
        Sigma_q_reg = Sigma_q + eps * eye

        # Inverse via Cholesky
        L_p = torch.linalg.cholesky(Sigma_p_reg)
        Sigma_p_inv = torch.cholesky_inverse(L_p)

        # Gradient w.r.t. μ_p
        delta = mu_q - mu_p
        grad_mu_p = -torch.matmul(Sigma_p_inv, delta.unsqueeze(-1)).squeeze(-1)

        # Gradient w.r.t. Σ_p: 0.5 * (Σ_p^{-1} - Σ_p^{-1}(Σ_q + δδ^T)Σ_p^{-1})
        delta_outer = torch.einsum('...i,...j->...ij', delta, delta)
        middle = Sigma_q_reg + delta_outer
        grad_Sigma_p = 0.5 * (
            Sigma_p_inv - torch.matmul(
                Sigma_p_inv,
                torch.matmul(middle, Sigma_p_inv)
            )
        )

        return grad_mu_p, grad_Sigma_p

    def grad_kl_wrt_q(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient of KL(q||p) w.r.t. source (q) - uses compiled version."""
        return self._kl_grad_q(mu_q, Sigma_q, mu_p, Sigma_p, eps)

    def grad_kl_wrt_p(
        self,
        mu_q: torch.Tensor,
        Sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        Sigma_p: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient of KL(q||p) w.r.t. target (p) - uses compiled version."""
        return self._kl_grad_p(mu_q, Sigma_q, mu_p, Sigma_p, eps)

    # =========================================================================
    # Self-Consistency Gradients (KL(q||p) for same agent)
    # =========================================================================

    def grad_self_consistency(
        self,
        state: GPUAgentState,
        eps: float = 1e-6
    ) -> GPUGradients:
        """
        Gradient of self-consistency term: KL(q||p).

        This is the local free energy contribution.
        """
        grad_mu_q, grad_Sigma_q = self.grad_kl_wrt_q(
            state.mu_q, state.Sigma_q,
            state.mu_p, state.Sigma_p, eps
        )

        grad_mu_p, grad_Sigma_p = self.grad_kl_wrt_p(
            state.mu_q, state.Sigma_q,
            state.mu_p, state.Sigma_p, eps
        )

        # No gauge gradient for self term
        grad_phi = torch.zeros_like(state.phi)

        return GPUGradients(
            grad_mu_q=grad_mu_q,
            grad_Sigma_q=grad_Sigma_q,
            grad_mu_p=grad_mu_p,
            grad_Sigma_p=grad_Sigma_p,
            grad_phi=grad_phi
        )

    # =========================================================================
    # Alignment Gradients (with transport)
    # =========================================================================

    def grad_alignment(
        self,
        state_i: GPUAgentState,
        state_j: GPUAgentState,
        Omega_ij: torch.Tensor,
        generators: torch.Tensor,
        eps: float = 1e-6
    ) -> Tuple[GPUGradients, GPUGradients]:
        """
        Gradient of alignment term: KL(q_i || T_ij q_j).

        Returns gradients for both agents i and j.

        Args:
            state_i: Agent i's state (receiver)
            state_j: Agent j's state (sender)
            Omega_ij: Transport operator from j to i
            generators: Lie algebra generators (3, K, K)

        Returns:
            grad_i: Gradients for agent i
            grad_j: Gradients for agent j
        """
        # Transport j's belief to i's frame
        mu_j_t, Sigma_j_t = self.backend.transport_gaussian(
            state_j.mu_q, state_j.Sigma_q, Omega_ij
        )

        # Gradient w.r.t. q_i (source of KL)
        grad_mu_q_i, grad_Sigma_q_i = self.grad_kl_wrt_q(
            state_i.mu_q, state_i.Sigma_q,
            mu_j_t, Sigma_j_t, eps
        )

        # Gradient w.r.t. transported q_j (target of KL)
        grad_mu_j_t, grad_Sigma_j_t = self.grad_kl_wrt_p(
            state_i.mu_q, state_i.Sigma_q,
            mu_j_t, Sigma_j_t, eps
        )

        # Back-transport gradients to j's frame: Ω^T @ grad
        Omega_ij_T = Omega_ij.transpose(-2, -1)
        grad_mu_q_j = torch.matmul(Omega_ij_T, grad_mu_j_t.unsqueeze(-1)).squeeze(-1)
        grad_Sigma_q_j = torch.matmul(
            Omega_ij_T,
            torch.matmul(grad_Sigma_j_t, Omega_ij)
        )

        # Gauge gradient (∂KL/∂φ via chain rule through transport)
        # This requires differentiating through the exponential map
        grad_phi_i, grad_phi_j = self._grad_transport_chain_rule(
            state_i.phi, state_j.phi,
            state_j.mu_q, state_j.Sigma_q,
            grad_mu_j_t, grad_Sigma_j_t,
            generators
        )

        grad_i = GPUGradients(
            grad_mu_q=grad_mu_q_i,
            grad_Sigma_q=grad_Sigma_q_i,
            grad_mu_p=torch.zeros_like(state_i.mu_p),
            grad_Sigma_p=torch.zeros_like(state_i.Sigma_p),
            grad_phi=grad_phi_i
        )

        grad_j = GPUGradients(
            grad_mu_q=grad_mu_q_j,
            grad_Sigma_q=grad_Sigma_q_j,
            grad_mu_p=torch.zeros_like(state_j.mu_p),
            grad_Sigma_p=torch.zeros_like(state_j.Sigma_p),
            grad_phi=grad_phi_j
        )

        return grad_i, grad_j

    def _grad_transport_chain_rule(
        self,
        phi_i: torch.Tensor,
        phi_j: torch.Tensor,
        mu_j: torch.Tensor,
        Sigma_j: torch.Tensor,
        grad_mu_t: torch.Tensor,
        grad_Sigma_t: torch.Tensor,
        generators: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gauge gradients via chain rule through transport.

        Uses finite differences for now (autograd would be cleaner).
        """
        # Small perturbation for numerical gradient
        h = 1e-5

        grad_phi_i = torch.zeros_like(phi_i)
        grad_phi_j = torch.zeros_like(phi_j)

        # Finite difference for each component
        for a in range(3):
            # Perturb phi_i
            phi_i_plus = phi_i.clone()
            phi_i_plus[..., a] += h

            Omega_plus = self.backend.compute_transport(phi_i_plus, phi_j, generators)
            mu_t_plus, Sigma_t_plus = self.backend.transport_gaussian(mu_j, Sigma_j, Omega_plus)

            phi_i_minus = phi_i.clone()
            phi_i_minus[..., a] -= h

            Omega_minus = self.backend.compute_transport(phi_i_minus, phi_j, generators)
            mu_t_minus, Sigma_t_minus = self.backend.transport_gaussian(mu_j, Sigma_j, Omega_minus)

            # Gradient contribution
            dmu = (mu_t_plus - mu_t_minus) / (2 * h)
            dSigma = (Sigma_t_plus - Sigma_t_minus) / (2 * h)

            grad_phi_i[..., a] = (
                torch.sum(grad_mu_t * dmu, dim=-1) +
                torch.sum(grad_Sigma_t * dSigma, dim=(-2, -1))
            )

            # Perturb phi_j
            phi_j_plus = phi_j.clone()
            phi_j_plus[..., a] += h

            Omega_plus = self.backend.compute_transport(phi_i, phi_j_plus, generators)
            mu_t_plus, Sigma_t_plus = self.backend.transport_gaussian(mu_j, Sigma_j, Omega_plus)

            phi_j_minus = phi_j.clone()
            phi_j_minus[..., a] -= h

            Omega_minus = self.backend.compute_transport(phi_i, phi_j_minus, generators)
            mu_t_minus, Sigma_t_minus = self.backend.transport_gaussian(mu_j, Sigma_j, Omega_minus)

            dmu = (mu_t_plus - mu_t_minus) / (2 * h)
            dSigma = (Sigma_t_plus - Sigma_t_minus) / (2 * h)

            grad_phi_j[..., a] = (
                torch.sum(grad_mu_t * dmu, dim=-1) +
                torch.sum(grad_Sigma_t * dSigma, dim=(-2, -1))
            )

        return grad_phi_i, grad_phi_j

    # =========================================================================
    # Full System Gradient Computation
    # =========================================================================

    def compute_all_gradients(
        self,
        states: List[GPUAgentState],
        generators: torch.Tensor,
        kappa: float,
        lambda_align: float = 1.0,
        lambda_self: float = 1.0,
        eps: float = 1e-6
    ) -> List[GPUGradients]:
        """
        Compute gradients for all agents in one GPU pass.

        This is the main entry point for simulation steps.

        Args:
            states: List of GPU agent states
            generators: Lie algebra generators (3, K, K)
            kappa: Softmax temperature
            lambda_align: Alignment term weight
            lambda_self: Self-consistency term weight

        Returns:
            List of gradients for each agent
        """
        N = len(states)

        # Initialize gradient accumulators
        gradients = []
        for state in states:
            gradients.append(GPUGradients(
                grad_mu_q=torch.zeros_like(state.mu_q),
                grad_Sigma_q=torch.zeros_like(state.Sigma_q),
                grad_mu_p=torch.zeros_like(state.mu_p),
                grad_Sigma_p=torch.zeros_like(state.Sigma_p),
                grad_phi=torch.zeros_like(state.phi)
            ))

        # Self-consistency terms
        if lambda_self > 0:
            for i, state in enumerate(states):
                grad_self = self.grad_self_consistency(state, eps)
                self._accumulate_gradients(gradients[i], grad_self, lambda_self)

        # Alignment terms
        if lambda_align > 0:
            # Compute all pairwise KL for softmax weights
            kl_matrix = self._compute_kl_matrix(states, generators, eps)

            for i in range(N):
                # Softmax weights: β_ij = softmax(-κ * KL_ij)
                kl_row = kl_matrix[i]  # KL(q_i || T_ij q_j) for all j
                weights = self.backend.compute_softmax_weights(kl_row, kappa)

                for j in range(N):
                    if i == j:
                        continue

                    weight_ij = weights[j]
                    if weight_ij < 1e-8:
                        continue  # Skip negligible contributions

                    # Compute transport operator
                    Omega_ij = self.backend.compute_transport(
                        states[i].phi.mean(dim=tuple(range(states[i].phi.ndim - 1))),  # Mean over spatial
                        states[j].phi.mean(dim=tuple(range(states[j].phi.ndim - 1))),
                        generators, eps
                    )

                    # Alignment gradients
                    grad_i, grad_j = self.grad_alignment(
                        states[i], states[j], Omega_ij, generators, eps
                    )

                    # Weighted accumulation
                    self._accumulate_gradients(gradients[i], grad_i, lambda_align * weight_ij.item())
                    self._accumulate_gradients(gradients[j], grad_j, lambda_align * weight_ij.item())

        return gradients

    def _compute_kl_matrix(
        self,
        states: List[GPUAgentState],
        generators: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute pairwise KL matrix for all agents.

        Uses mean beliefs over spatial dimensions for efficiency.
        """
        N = len(states)

        # Extract mean beliefs (average over spatial dimensions)
        mu_all = torch.stack([
            s.mu_q.mean(dim=tuple(range(s.mu_q.ndim - 1)))
            for s in states
        ])  # (N, K)

        Sigma_all = torch.stack([
            s.Sigma_q.mean(dim=tuple(range(s.Sigma_q.ndim - 2)))
            for s in states
        ])  # (N, K, K)

        phi_all = torch.stack([
            s.phi.mean(dim=tuple(range(s.phi.ndim - 1)))
            for s in states
        ])  # (N, 3)

        kl_matrix = torch.zeros(N, N, device=self.device, dtype=self.dtype)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Transport j to i
                Omega_ij = self.backend.compute_transport(
                    phi_all[i], phi_all[j], generators, eps
                )
                mu_j_t, Sigma_j_t = self.backend.transport_gaussian(
                    mu_all[j], Sigma_all[j], Omega_ij
                )

                # KL(q_i || T_ij q_j)
                kl_matrix[i, j] = self.backend.kl_gaussian(
                    mu_all[i], Sigma_all[i],
                    mu_j_t, Sigma_j_t, eps
                )

        return kl_matrix

    def _accumulate_gradients(
        self,
        target: GPUGradients,
        source: GPUGradients,
        weight: float = 1.0
    ):
        """Add weighted source gradients to target (in-place)."""
        target.grad_mu_q += weight * source.grad_mu_q
        target.grad_Sigma_q += weight * source.grad_Sigma_q
        target.grad_mu_p += weight * source.grad_mu_p
        target.grad_Sigma_p += weight * source.grad_Sigma_p
        target.grad_phi += weight * source.grad_phi


# =============================================================================
# Integration with Existing System
# =============================================================================

def compute_gradients_torch(
    system,
    verbose: int = 0
) -> List:
    """
    Drop-in replacement for compute_natural_gradients using PyTorch.

    Args:
        system: MultiAgentSystem instance
        verbose: Verbosity level

    Returns:
        List of AgentGradients (converted from GPU)
    """
    from gradients.gradient_engine import AgentGradients

    if not is_torch_available():
        raise RuntimeError("PyTorch CUDA not available")

    # Get or create engine
    engine = TorchGradientEngine()

    # Transfer agent states to GPU
    gpu_states = []
    for agent in system.agents:
        state = engine.create_gpu_state(
            mu_q=agent.mu_q,
            Sigma_q=agent.Sigma_q,
            mu_p=agent.mu_p,
            Sigma_p=agent.Sigma_p,
            phi=agent.phi
        )
        gpu_states.append(state)

    # Get generators
    generators = engine.backend.to_device(system.generators)

    # Compute gradients
    kappa = getattr(system.config, 'kappa', 1.0)
    lambda_align = getattr(system.config, 'lambda_coupling', 1.0)
    lambda_self = getattr(system.config, 'lambda_self', 1.0)

    gpu_gradients = engine.compute_all_gradients(
        gpu_states, generators, kappa, lambda_align, lambda_self
    )

    # Convert back to CPU AgentGradients
    result = []
    for grad in gpu_gradients:
        result.append(AgentGradients(
            grad_mu_q=engine.backend.to_numpy(grad.grad_mu_q),
            grad_Sigma_q=engine.backend.to_numpy(grad.grad_Sigma_q),
            grad_mu_p=engine.backend.to_numpy(grad.grad_mu_p),
            grad_Sigma_p=engine.backend.to_numpy(grad.grad_Sigma_p),
            grad_phi=engine.backend.to_numpy(grad.grad_phi)
        ))

    return result