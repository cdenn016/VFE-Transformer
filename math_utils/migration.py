# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:45:59 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Migration Utilities: NumPy <-> PyTorch Tensor Conversion
========================================================

Utilities for converting between NumPy-based agents/systems and
PyTorch tensor-based agents/systems.

This enables:
1. Migrating existing NumPy simulations to GPU
2. Validating GPU implementation against NumPy reference
3. Hybrid workflows (e.g., train on GPU, analyze with NumPy)

Author: Claude (refactoring)
Date: December 2024
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple, Union


# =============================================================================
# Core Conversion Functions
# =============================================================================

def numpy_to_tensor(
    arr: np.ndarray,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        arr: NumPy array to convert
        device: Target device ('cuda', 'cpu')
        dtype: Target dtype (torch.float32, torch.float64)
        requires_grad: Whether to enable gradient tracking

    Returns:
        PyTorch tensor on specified device
    """
    tensor = torch.tensor(arr, device=device, dtype=dtype)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def tensor_to_numpy(
    tensor: torch.Tensor,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.

    Args:
        tensor: PyTorch tensor to convert
        dtype: NumPy dtype for result

    Returns:
        NumPy array (detached from computation graph)
    """
    return tensor.detach().cpu().numpy().astype(dtype)


# =============================================================================
# Agent Conversion
# =============================================================================

def agent_to_tensor_dict(
    agent: 'Agent',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Extract agent state as dictionary of tensors.

    Args:
        agent: NumPy-based Agent instance
        device: Target device
        dtype: Target dtype

    Returns:
        Dictionary with tensors:
            mu_q, Sigma_q, mu_p, Sigma_p, phi, generators
    """
    return {
        'mu_q': numpy_to_tensor(agent.mu_q, device, dtype),
        'Sigma_q': numpy_to_tensor(agent.Sigma_q, device, dtype),
        'mu_p': numpy_to_tensor(agent.mu_p, device, dtype),
        'Sigma_p': numpy_to_tensor(agent.Sigma_p, device, dtype),
        'phi': numpy_to_tensor(agent.phi, device, dtype),
        'generators': numpy_to_tensor(agent.generators, device, dtype),
    }


def tensor_dict_to_agent(
    tensors: Dict[str, torch.Tensor],
    agent: 'Agent',
):
    """
    Copy tensor state back to NumPy agent (in-place).

    Args:
        tensors: Dictionary of tensors
        agent: Target NumPy Agent to update
    """
    agent.mu_q = tensor_to_numpy(tensors['mu_q'])
    agent.mu_p = tensor_to_numpy(tensors['mu_p'])
    agent.Sigma_q = tensor_to_numpy(tensors['Sigma_q'])
    agent.Sigma_p = tensor_to_numpy(tensors['Sigma_p'])

    if 'phi' in tensors:
        if hasattr(agent, 'gauge'):
            agent.gauge.phi = tensor_to_numpy(tensors['phi'])
        else:
            agent.phi = tensor_to_numpy(tensors['phi'])

    # Invalidate caches
    if hasattr(agent, '_L_q_cache'):
        agent._L_q_cache = None
    if hasattr(agent, '_L_p_cache'):
        agent._L_p_cache = None


# =============================================================================
# System Conversion
# =============================================================================

def system_to_batched_tensors(
    system: 'MultiAgentSystem',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Convert MultiAgentSystem to batched tensors.

    Stacks all N agents into tensors of shape (N, ...).

    Args:
        system: NumPy-based MultiAgentSystem
        device: Target device
        dtype: Target dtype

    Returns:
        Dictionary with batched tensors:
            mu_q: (N, K) or (N, *S, K)
            Sigma_q: (N, K, K) or (N, *S, K, K)
            mu_p: (N, K) or (N, *S, K)
            Sigma_p: (N, K, K) or (N, *S, K, K)
            phi: (N, 3) or (N, *S, 3)
            generators: (3, K, K)
    """
    agents = system.agents

    return {
        'mu_q': torch.stack([
            numpy_to_tensor(a.mu_q, device, dtype) for a in agents
        ]),
        'Sigma_q': torch.stack([
            numpy_to_tensor(a.Sigma_q, device, dtype) for a in agents
        ]),
        'mu_p': torch.stack([
            numpy_to_tensor(a.mu_p, device, dtype) for a in agents
        ]),
        'Sigma_p': torch.stack([
            numpy_to_tensor(a.Sigma_p, device, dtype) for a in agents
        ]),
        'phi': torch.stack([
            numpy_to_tensor(a.phi, device, dtype) for a in agents
        ]),
        'generators': numpy_to_tensor(agents[0].generators, device, dtype),
    }


def batched_tensors_to_system(
    tensors: Dict[str, torch.Tensor],
    system: 'MultiAgentSystem',
):
    """
    Copy batched tensor state back to MultiAgentSystem (in-place).

    Args:
        tensors: Dictionary of batched tensors
        system: Target MultiAgentSystem to update
    """
    for i, agent in enumerate(system.agents):
        agent.mu_q = tensor_to_numpy(tensors['mu_q'][i])
        agent.mu_p = tensor_to_numpy(tensors['mu_p'][i])
        agent.Sigma_q = tensor_to_numpy(tensors['Sigma_q'][i])
        agent.Sigma_p = tensor_to_numpy(tensors['Sigma_p'][i])

        if 'phi' in tensors:
            if hasattr(agent, 'gauge'):
                agent.gauge.phi = tensor_to_numpy(tensors['phi'][i])
            else:
                agent.phi = tensor_to_numpy(tensors['phi'][i])

        # Invalidate caches
        if hasattr(agent, '_L_q_cache'):
            agent._L_q_cache = None
        if hasattr(agent, '_L_p_cache'):
            agent._L_p_cache = None


# =============================================================================
# TensorAgent <-> Agent Conversion
# =============================================================================

def create_tensor_agent_from_agent(
    agent: 'Agent',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    use_cholesky_param: bool = False,
) -> 'TensorAgent':
    """
    Create TensorAgent from NumPy Agent.

    Args:
        agent: NumPy-based Agent
        device: Target device
        dtype: Target dtype
        use_cholesky_param: Whether to use Cholesky parameterization

    Returns:
        TensorAgent with copied state
    """
    from agent.tensor_agent import TensorAgent

    return TensorAgent.from_numpy_agent(
        agent,
        device=device,
        dtype=dtype,
        use_cholesky_param=use_cholesky_param,
    )


def create_tensor_system_from_system(
    system: 'MultiAgentSystem',
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    use_cholesky_param: bool = False,
) -> 'TensorSystem':
    """
    Create TensorSystem from NumPy MultiAgentSystem.

    Args:
        system: NumPy-based MultiAgentSystem
        device: Target device
        dtype: Target dtype
        use_cholesky_param: Whether to use Cholesky parameterization

    Returns:
        TensorSystem with copied state
    """
    from agent.tensor_system import TensorSystem

    return TensorSystem.from_multi_agent_system(
        system,
        device=device,
        dtype=dtype,
        use_cholesky_param=use_cholesky_param,
    )


# =============================================================================
# Gradient Conversion
# =============================================================================

def extract_autograd_gradients(
    system: 'TensorSystem',
) -> List[Dict[str, np.ndarray]]:
    """
    Extract gradients from TensorSystem after backward pass.

    Call this after energy.backward() to get gradients in NumPy format,
    compatible with the gradient format from gradient_engine.py.

    Args:
        system: TensorSystem with gradients computed

    Returns:
        List of gradient dicts (one per agent), each containing:
            delta_mu_q, delta_Sigma_q, delta_mu_p, delta_Sigma_p, delta_phi
    """
    gradients = []

    for i in range(system.N):
        grad = {}

        # Mean gradients
        if system.mu_q.grad is not None:
            grad['delta_mu_q'] = tensor_to_numpy(system.mu_q.grad[i])
        else:
            grad['delta_mu_q'] = np.zeros(system.K, dtype=np.float32)

        if system.mu_p.grad is not None:
            grad['delta_mu_p'] = tensor_to_numpy(system.mu_p.grad[i])
        else:
            grad['delta_mu_p'] = np.zeros(system.K, dtype=np.float32)

        # Covariance gradients
        if system.use_cholesky_param:
            if system._L_q.grad is not None:
                # Convert L gradient to Sigma gradient: dSigma = L @ dL^T + dL @ L^T
                L = system._L_q[i].detach()
                dL = system._L_q.grad[i]
                dSigma = L @ dL.transpose(-1, -2) + dL @ L.transpose(-1, -2)
                grad['delta_Sigma_q'] = tensor_to_numpy(dSigma)
            else:
                grad['delta_Sigma_q'] = np.zeros((system.K, system.K), dtype=np.float32)

            if system._L_p.grad is not None:
                L = system._L_p[i].detach()
                dL = system._L_p.grad[i]
                dSigma = L @ dL.transpose(-1, -2) + dL @ L.transpose(-1, -2)
                grad['delta_Sigma_p'] = tensor_to_numpy(dSigma)
            else:
                grad['delta_Sigma_p'] = np.zeros((system.K, system.K), dtype=np.float32)
        else:
            if system._Sigma_q.grad is not None:
                grad['delta_Sigma_q'] = tensor_to_numpy(system._Sigma_q.grad[i])
            else:
                grad['delta_Sigma_q'] = np.zeros((system.K, system.K), dtype=np.float32)

            if system._Sigma_p.grad is not None:
                grad['delta_Sigma_p'] = tensor_to_numpy(system._Sigma_p.grad[i])
            else:
                grad['delta_Sigma_p'] = np.zeros((system.K, system.K), dtype=np.float32)

        # Phi gradient
        if system.phi.grad is not None:
            grad['delta_phi'] = tensor_to_numpy(system.phi.grad[i])
        else:
            grad['delta_phi'] = np.zeros(3, dtype=np.float32)

        gradients.append(grad)

    return gradients


# =============================================================================
# Validation Utilities
# =============================================================================

def compare_states(
    numpy_system: 'MultiAgentSystem',
    tensor_system: 'TensorSystem',
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compare states between NumPy and Tensor systems.

    Useful for validating that migration preserved state correctly.

    Args:
        numpy_system: NumPy-based MultiAgentSystem
        tensor_system: TensorSystem
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with comparison results
    """
    results = {
        'match': True,
        'max_diff': {},
        'mismatches': [],
    }

    state_t = tensor_system.get_state_dict_numpy()

    for i, agent in enumerate(numpy_system.agents):
        # Compare mu_q
        diff = np.abs(agent.mu_q - state_t['mu_q'][i]).max()
        results['max_diff'][f'agent_{i}_mu_q'] = diff
        if diff > atol + rtol * np.abs(agent.mu_q).max():
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_mu_q')

        # Compare Sigma_q
        diff = np.abs(agent.Sigma_q - state_t['Sigma_q'][i]).max()
        results['max_diff'][f'agent_{i}_Sigma_q'] = diff
        if diff > atol + rtol * np.abs(agent.Sigma_q).max():
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_Sigma_q')

        # Compare mu_p
        diff = np.abs(agent.mu_p - state_t['mu_p'][i]).max()
        results['max_diff'][f'agent_{i}_mu_p'] = diff
        if diff > atol + rtol * np.abs(agent.mu_p).max():
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_mu_p')

        # Compare Sigma_p
        diff = np.abs(agent.Sigma_p - state_t['Sigma_p'][i]).max()
        results['max_diff'][f'agent_{i}_Sigma_p'] = diff
        if diff > atol + rtol * np.abs(agent.Sigma_p).max():
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_Sigma_p')

        # Compare phi
        phi_numpy = agent.phi if hasattr(agent, 'phi') else agent.gauge.phi
        diff = np.abs(phi_numpy - state_t['phi'][i]).max()
        results['max_diff'][f'agent_{i}_phi'] = diff
        if diff > atol + rtol * np.abs(phi_numpy).max():
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_phi')

    return results


def validate_gradients(
    numpy_system: 'MultiAgentSystem',
    tensor_system: 'TensorSystem',
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """
    Validate that autograd gradients match hand-derived gradients.

    This is crucial for ensuring the PyTorch implementation is correct.

    Args:
        numpy_system: NumPy system (for reference gradients)
        tensor_system: TensorSystem (for autograd gradients)
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with validation results
    """
    from gradients.gradient_engine import compute_natural_gradients

    # Compute reference gradients (hand-derived)
    ref_grads = compute_natural_gradients(numpy_system)

    # Compute autograd gradients
    tensor_system.zero_grad()
    energies = tensor_system.compute_energy()
    energies['total'].backward()
    auto_grads = extract_autograd_gradients(tensor_system)

    results = {
        'match': True,
        'max_diff': {},
        'mismatches': [],
    }

    for i in range(len(ref_grads)):
        ref = ref_grads[i]
        auto = auto_grads[i]

        # Compare mu_q gradient
        diff = np.abs(ref.delta_mu_q - auto['delta_mu_q']).max()
        results['max_diff'][f'agent_{i}_grad_mu_q'] = diff
        if diff > atol + rtol * (np.abs(ref.delta_mu_q).max() + 1e-8):
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_grad_mu_q')

        # Compare phi gradient
        diff = np.abs(ref.delta_phi - auto['delta_phi']).max()
        results['max_diff'][f'agent_{i}_grad_phi'] = diff
        if diff > atol + rtol * (np.abs(ref.delta_phi).max() + 1e-8):
            results['match'] = False
            results['mismatches'].append(f'agent_{i}_grad_phi')

    return results


# =============================================================================
# Device Management
# =============================================================================

def get_device(prefer_cuda: bool = True) -> str:
    """
    Get best available device.

    Args:
        prefer_cuda: If True, prefer CUDA over CPU

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cpu',
    }

    if torch.cuda.is_available():
        info['current_device'] = f'cuda:{torch.cuda.current_device()}'
        info['cuda_device_name'] = torch.cuda.get_device_name()
        info['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        info['cuda_memory_reserved'] = torch.cuda.memory_reserved()

    return info


def move_to_device(
    system: 'TensorSystem',
    device: str,
) -> 'TensorSystem':
    """
    Move TensorSystem to specified device.

    Args:
        system: TensorSystem to move
        device: Target device ('cuda' or 'cpu')

    Returns:
        System on new device
    """
    return system.to(device)


# =============================================================================
# Batch Operations
# =============================================================================

def batch_numpy_arrays(
    arrays: List[np.ndarray],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Stack list of NumPy arrays into batched tensor.

    Args:
        arrays: List of NumPy arrays with same shape
        device: Target device
        dtype: Target dtype

    Returns:
        Batched tensor of shape (N, *array_shape)
    """
    stacked = np.stack(arrays, axis=0)
    return numpy_to_tensor(stacked, device, dtype)


def unbatch_tensor(
    tensor: torch.Tensor,
    dtype: np.dtype = np.float32,
) -> List[np.ndarray]:
    """
    Split batched tensor into list of NumPy arrays.

    Args:
        tensor: Batched tensor of shape (N, ...)
        dtype: NumPy dtype for output

    Returns:
        List of N NumPy arrays
    """
    return [tensor_to_numpy(tensor[i], dtype) for i in range(tensor.shape[0])]