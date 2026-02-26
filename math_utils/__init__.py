# -*- coding: utf-8 -*-
"""
Math Utilities Module
=====================

Mathematical utilities for both NumPy (CPU) and PyTorch (GPU) backends.

NumPy (CPU):
    - transport: Gauge transport operators
    - push_pull: Gaussian push-pull operations
    - sigma: Covariance utilities
    - generators: SO(3) Lie algebra generators

PyTorch (GPU):
    - migration: NumPy <-> Tensor conversion utilities
    - batched_ops: Compiled batched operations for GPU
    - torch_backend: PyTorch backend utilities
"""

# NumPy utilities
from .transport import compute_transport as np_transport_operator
from .push_pull import push_gaussian as np_transport_gaussian
from .generators import (
    generate_so3_generators,
    generate_soN_generators,
    generate_wedge2_generators,
    generate_sym2_traceless_generators,
    generate_multi_irrep_soN_generators,
)
from .numerical_utils import safe_inv

# Simple NumPy utilities (not in separate module)
import numpy as np

def symmetrize(M: np.ndarray) -> np.ndarray:
    """Symmetrize a matrix: (M + M^T) / 2"""
    return 0.5 * (M + np.swapaxes(M, -1, -2))

def np_ensure_spd(Sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ensure matrix is symmetric positive definite (NumPy version)."""
    Sigma = symmetrize(Sigma)
    K = Sigma.shape[-1]
    Sigma = Sigma + eps * np.eye(K, dtype=Sigma.dtype)
    return Sigma

# PyTorch utilities (optional - only if torch is available)
try:
    from .migration import (
        numpy_to_tensor,
        tensor_to_numpy,
        create_tensor_agent_from_agent,
        create_tensor_system_from_system,
        get_device,
        get_device_info,
    )
    from .batched_ops import (
        batched_kl_divergence,
        batched_transport_operator,
        batched_transport_gaussian,
        compute_all_pairwise_kl,
        compute_softmax_attention,
        ensure_spd,
        project_to_principal_ball,
        is_compiled,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    # NumPy
    'np_transport_operator',
    'np_transport_gaussian',
    'generate_so3_generators',
    'generate_soN_generators',
    'generate_wedge2_generators',
    'generate_sym2_traceless_generators',
    'generate_multi_irrep_soN_generators',
    'safe_inv',
    'symmetrize',
    'np_ensure_spd',
]

if _TORCH_AVAILABLE:
    __all__.extend([
        # PyTorch migration
        'numpy_to_tensor',
        'tensor_to_numpy',
        'create_tensor_agent_from_agent',
        'create_tensor_system_from_system',
        'get_device',
        'get_device_info',
        # PyTorch batched ops
        'batched_kl_divergence',
        'batched_transport_operator',
        'batched_transport_gaussian',
        'compute_all_pairwise_kl',
        'compute_softmax_attention',
        'ensure_spd',
        'project_to_principal_ball',
        'is_compiled',
    ])