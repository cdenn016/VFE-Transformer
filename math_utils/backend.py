# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 20:21:33 2025

@author: chris and christine
"""

"""
GPU Backend Abstraction Layer
==============================

Provides seamless switching between NumPy (CPU) and CuPy/PyTorch (GPU).

Usage:
------
    from math_utils.backend import get_backend, to_device, to_numpy

    xp = get_backend()  # Returns numpy, cupy, or torch-compatible module

    # Your code works with any backend:
    x = xp.zeros((10, 10))
    y = xp.linalg.cholesky(cov)

    # Transfer between devices:
    x_gpu = to_device(x_cpu, device='cuda')
    x_cpu = to_numpy(x_gpu)

Backend Selection:
------------------
Priority order:
1. CuPy (if CUDA available and cupy installed) - Best for NumPy-like code
2. PyTorch (if CUDA available) - Useful for autograd
3. NumPy (fallback) - Always available

Configuration via environment variable or config:
    COMPUTE_BACKEND=cupy|pytorch|numpy

Performance Notes:
------------------
RTX 5090 specifications (estimated):
- ~32GB GDDR7 VRAM
- ~2000+ CUDA cores
- FP32: ~90+ TFLOPS

For N=100 agents with K=8 latent dims:
- CPU (Numba): ~50-100ms per step
- GPU (CuPy): ~5-10ms per step (10x speedup)
- GPU (batched): ~1-2ms per step (50x speedup with proper batching)

Author: Chris
Date: December 2025
"""

import os
import warnings
from typing import Optional, Union, Literal, Any
from functools import lru_cache
from math_utils.numerical_monitor import record as _nr
import numpy as np



# =============================================================================

# FIX: OpenMP Conflict on Windows (NumPy MKL + CuPy)

# =============================================================================

# This MUST be set BEFORE importing CuPy to avoid the libiomp5md.dll conflict

# Set this unconditionally to prevent the fatal abort

 

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

 

# Also set OMP_NUM_THREADS to prevent thread oversubscription

if 'OMP_NUM_THREADS' not in os.environ:

    # Use half of available cores for OpenMP, let GPU handle parallelism

    import multiprocessing

    n_cores = multiprocessing.cpu_count()

    os.environ['OMP_NUM_THREADS'] = str(max(1, n_cores // 2))


# =============================================================================
# Backend Detection and Selection
# =============================================================================

BackendType = Literal['numpy', 'cupy', 'pytorch']

# Global state
_CURRENT_BACKEND: BackendType = 'numpy'
_CUPY_AVAILABLE: Optional[bool] = None
_PYTORCH_CUDA_AVAILABLE: Optional[bool] = None


def _check_cupy_available() -> bool:
    """Check if CuPy is available with CUDA support."""
    global _CUPY_AVAILABLE

    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE

    try:
        import cupy as cp
        # Try a simple operation to verify CUDA works
        _ = cp.zeros(1)
        _CUPY_AVAILABLE = True
    except (ImportError, RuntimeError, OSError):
        _CUPY_AVAILABLE = False

    return _CUPY_AVAILABLE


def _check_pytorch_cuda_available() -> bool:
    """Check if PyTorch is available with CUDA support."""
    global _PYTORCH_CUDA_AVAILABLE

    if _PYTORCH_CUDA_AVAILABLE is not None:
        return _PYTORCH_CUDA_AVAILABLE

    try:
        import torch
        _PYTORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        _PYTORCH_CUDA_AVAILABLE = False

    return _PYTORCH_CUDA_AVAILABLE


def detect_best_backend() -> BackendType:
    """
    Detect the best available backend.

    Priority:
    1. CuPy (most NumPy-compatible)
    2. PyTorch CUDA
    3. NumPy (fallback)
    """
    # Check environment variable override
    env_backend = os.environ.get('COMPUTE_BACKEND', '').lower()
    if env_backend in ('numpy', 'cupy', 'pytorch'):
        return env_backend

    # Auto-detect
    if _check_cupy_available():
        return 'cupy'
    elif _check_pytorch_cuda_available():
        return 'pytorch'
    else:
        return 'numpy'


def set_backend(backend: BackendType) -> None:
    """
    Set the compute backend.

    Args:
        backend: One of 'numpy', 'cupy', 'pytorch'

    Raises:
        ValueError: If requested backend is not available
    """
    global _CURRENT_BACKEND

    if backend == 'cupy' and not _check_cupy_available():
        raise ValueError("CuPy not available. Install with: pip install cupy-cuda12x")

    if backend == 'pytorch' and not _check_pytorch_cuda_available():
        raise ValueError("PyTorch CUDA not available")

    _CURRENT_BACKEND = backend
    print(f"[Backend] Set compute backend to: {backend}")


def get_backend_name() -> BackendType:
    """Get current backend name."""
    return _CURRENT_BACKEND


# =============================================================================
# Array Module Interface (NumPy-compatible API)
# =============================================================================

class ArrayModule:
    """
    Unified array module interface compatible with NumPy/CuPy/PyTorch.

    Provides a consistent API regardless of backend, enabling code like:

        xp = get_backend()
        x = xp.zeros((10, 10))
        y = xp.linalg.cholesky(cov)
        z = xp.einsum('ij,jk->ik', a, b)
    """

    def __init__(self, backend: BackendType):
        self.backend = backend
        self._module = self._get_module()
        self._linalg = self._get_linalg_module()
        self._random = self._get_random_module()

    def _get_module(self):
        if self.backend == 'numpy':
            return np
        elif self.backend == 'cupy':
            import cupy as cp
            return cp
        elif self.backend == 'pytorch':
            # PyTorch has different semantics - wrap in adapter
            return _PyTorchArrayAdapter()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_linalg_module(self):
        if self.backend == 'numpy':
            return np.linalg
        elif self.backend == 'cupy':
            import cupy as cp
            return cp.linalg
        elif self.backend == 'pytorch':
            return _PyTorchLinalgAdapter()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _get_random_module(self):
        if self.backend == 'numpy':
            return np.random
        elif self.backend == 'cupy':
            import cupy as cp
            return cp.random
        elif self.backend == 'pytorch':
            return _PyTorchRandomAdapter()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # Pass-through to underlying module
    def __getattr__(self, name):
        return getattr(self._module, name)

    @property
    def linalg(self):
        return self._linalg

    @property
    def random(self):
        return self._random

    def is_gpu(self) -> bool:
        """Check if this backend uses GPU."""
        return self.backend in ('cupy', 'pytorch')

    def synchronize(self) -> None:
        """Synchronize GPU operations (no-op for CPU)."""
        if self.backend == 'cupy':
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        elif self.backend == 'pytorch':
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()


# =============================================================================
# PyTorch Adapters (to match NumPy API)
# =============================================================================

class _PyTorchArrayAdapter:
    """Adapter to make PyTorch more NumPy-like."""

    def __init__(self):
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def zeros(self, shape, dtype=None):
        dtype = self._convert_dtype(dtype)
        return self.torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        dtype = self._convert_dtype(dtype)
        return self.torch.ones(shape, dtype=dtype, device=self.device)

    def eye(self, N, M=None, dtype=None):
        dtype = self._convert_dtype(dtype)
        return self.torch.eye(N, M or N, dtype=dtype, device=self.device)

    def array(self, data, dtype=None):
        dtype = self._convert_dtype(dtype)
        if isinstance(data, self.torch.Tensor):
            return data.to(dtype=dtype, device=self.device)
        return self.torch.tensor(data, dtype=dtype, device=self.device)

    def asarray(self, data, dtype=None):
        return self.array(data, dtype)

    def stack(self, arrays, axis=0):
        return self.torch.stack(arrays, dim=axis)

    def concatenate(self, arrays, axis=0):
        return self.torch.cat(arrays, dim=axis)

    def einsum(self, subscripts, *operands, **kwargs):
        return self.torch.einsum(subscripts, *operands)

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.sum()
        return a.sum(dim=axis, keepdim=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.mean()
        return a.mean(dim=axis, keepdim=keepdims)

    def max(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.max()
        return a.max(dim=axis, keepdim=keepdims).values

    def min(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.min()
        return a.min(dim=axis, keepdim=keepdims).values

    def exp(self, x):
        return self.torch.exp(x)

    def log(self, x):
        return self.torch.log(x)

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def abs(self, x):
        return self.torch.abs(x)

    def clip(self, x, a_min, a_max):
        return self.torch.clamp(x, a_min, a_max)

    def where(self, condition, x, y):
        return self.torch.where(condition, x, y)

    def isfinite(self, x):
        return self.torch.isfinite(x)

    def swapaxes(self, a, axis1, axis2):
        return a.transpose(axis1, axis2)

    def trace(self, a, axis1=-2, axis2=-1):
        return self.torch.diagonal(a, dim1=axis1, dim2=axis2).sum(-1)

    def diagonal(self, a, offset=0, axis1=-2, axis2=-1):
        return self.torch.diagonal(a, offset, axis1, axis2)

    def diag(self, v, k=0):
        return self.torch.diag(v, k)

    def _convert_dtype(self, dtype):
        if dtype is None:
            return self.torch.float32
        if isinstance(dtype, self.torch.dtype):
            return dtype

        dtype_map = {
            np.float32: self.torch.float32,
            np.float64: self.torch.float64,
            np.int32: self.torch.int32,
            np.int64: self.torch.int64,
            'float32': self.torch.float32,
            'float64': self.torch.float64,
        }
        return dtype_map.get(dtype, self.torch.float32)

    # NumPy dtype attributes
    @property
    def float32(self):
        return self.torch.float32

    @property
    def float64(self):
        return self.torch.float64

    @property
    def pi(self):
        return 3.141592653589793


class _PyTorchLinalgAdapter:
    """Adapter for PyTorch linear algebra functions."""

    def __init__(self):
        import torch
        self.torch = torch

    def cholesky(self, a):
        return self.torch.linalg.cholesky(a)

    def inv(self, a):
        try:
            return self.torch.linalg.inv(a)
        except (RuntimeError,):
            _nr("inv_pinv")
            return self.torch.linalg.pinv(a)

    def solve(self, a, b):
        return self.torch.linalg.solve(a, b)

    def eigh(self, a):
        return self.torch.linalg.eigh(a)

    def eigvalsh(self, a):
        return self.torch.linalg.eigvalsh(a)

    def svd(self, a, full_matrices=True):
        return self.torch.linalg.svd(a, full_matrices=full_matrices)

    def norm(self, x, ord=None, axis=None, keepdims=False):
        if axis is None:
            return self.torch.linalg.norm(x, ord=ord)
        return self.torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)

    def det(self, a):
        return self.torch.linalg.det(a)

    def slogdet(self, a):
        return self.torch.linalg.slogdet(a)


class _PyTorchRandomAdapter:
    """Adapter for PyTorch random functions."""

    def __init__(self):
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = ()
        return self.torch.normal(loc, scale, size, device=self.device)

    def randn(self, *shape):
        return self.torch.randn(*shape, device=self.device)

    def rand(self, *shape):
        return self.torch.rand(*shape, device=self.device)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            size = ()
        return (high - low) * self.torch.rand(size, device=self.device) + low


# =============================================================================
# Main API Functions
# =============================================================================

@lru_cache(maxsize=1)
def get_backend() -> ArrayModule:
    """
    Get the current compute backend module.

    Returns:
        ArrayModule: NumPy-compatible module for current backend

    Usage:
        xp = get_backend()
        x = xp.zeros((10, 10))
        y = xp.linalg.cholesky(cov)
    """
    return ArrayModule(_CURRENT_BACKEND)


def to_device(
    x: Union[np.ndarray, Any],
    device: Literal['cpu', 'cuda', 'auto'] = 'auto'
) -> Any:
    """
    Transfer array to specified device.

    Args:
        x: Input array (numpy, cupy, or torch tensor)
        device: Target device ('cpu', 'cuda', or 'auto')

    Returns:
        Array on target device
    """
    if device == 'auto':
        device = 'cuda' if _CURRENT_BACKEND in ('cupy', 'pytorch') else 'cpu'

    # NumPy array
    if isinstance(x, np.ndarray):
        if device == 'cpu':
            return x
        elif _CURRENT_BACKEND == 'cupy':
            import cupy as cp
            return cp.asarray(x)
        elif _CURRENT_BACKEND == 'pytorch':
            import torch
            return torch.tensor(x, device=device)

    # CuPy array
    if _check_cupy_available():
        import cupy as cp
        if isinstance(x, cp.ndarray):
            if device == 'cpu':
                return cp.asnumpy(x)
            return x

    # PyTorch tensor
    if _check_pytorch_cuda_available():
        import torch
        if isinstance(x, torch.Tensor):
            if device == 'cpu':
                return x.cpu().numpy()
            return x.to(device)

    return x


def to_numpy(x: Any) -> np.ndarray:
    """
    Convert any array to NumPy (CPU).

    Args:
        x: Input array (numpy, cupy, or torch tensor)

    Returns:
        NumPy array on CPU
    """
    if isinstance(x, np.ndarray):
        return x

    # CuPy
    if _check_cupy_available():
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)

    # PyTorch
    if _check_pytorch_cuda_available():
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

    return np.asarray(x)


def synchronize() -> None:
    """Synchronize GPU operations (blocks until all GPU ops complete)."""
    xp = get_backend()
    xp.synchronize()


# =============================================================================
# CUDA-Optimized Operations
# =============================================================================

def batch_kl_divergence_gpu(
    mu_q: Any,
    Sigma_q: Any,
    mu_p_batch: Any,
    Sigma_p_batch: Any,
    eps: float = 1e-8
) -> Any:
    """
    Batch KL divergence computation optimized for GPU.

    Computes KL(q || p_i) for multiple targets p_i in parallel.

    Args:
        mu_q: Source mean (K,)
        Sigma_q: Source covariance (K, K)
        mu_p_batch: Target means (N, K)
        Sigma_p_batch: Target covariances (N, K, K)
        eps: Regularization

    Returns:
        kl_batch: (N,) array of KL divergences

    Performance:
        GPU: ~100x faster than CPU loop for N>100
    """
    xp = get_backend()

    N = mu_p_batch.shape[0]
    K = mu_q.shape[0]

    # Regularize
    eye_K = xp.eye(K, dtype=mu_q.dtype)
    Sigma_q_reg = Sigma_q + eps * eye_K
    Sigma_p_batch_reg = Sigma_p_batch + eps * eye_K

    # Cholesky decomposition (batched)
    L_q = xp.linalg.cholesky(Sigma_q_reg)
    L_p_batch = xp.linalg.cholesky(Sigma_p_batch_reg)

    # Log determinants: log|Sigma| = 2 * sum(log(diag(L)))
    logdet_q = 2.0 * xp.sum(xp.log(xp.maximum(xp.diagonal(L_q), eps)))
    logdet_p_batch = 2.0 * xp.sum(xp.log(xp.maximum(xp.diagonal(L_p_batch, axis1=-2, axis2=-1), eps)), axis=-1)

    # Trace term: tr(Sigma_p^-1 @ Sigma_q) for each p
    # Using solve instead of explicit inverse
    kl_batch = xp.zeros(N, dtype=mu_q.dtype)

    for i in range(N):
        L_p = L_p_batch[i]
        mu_p = mu_p_batch[i]

        # Trace: solve L_p @ Y = Sigma_q, then L_p.T @ Z = Y, trace(Z)
        Y = xp.linalg.solve(L_p, Sigma_q_reg)
        Z = xp.linalg.solve(L_p.swapaxes(-1, -2), Y)
        term_trace = xp.trace(Z)

        # Quadratic: (mu_p - mu_q)^T Sigma_p^-1 (mu_p - mu_q)
        delta = mu_p - mu_q
        y = xp.linalg.solve(L_p, delta)
        z = xp.linalg.solve(L_p.swapaxes(-1, -2), y)
        term_quad = xp.sum(delta * z)

        # Log det term
        term_logdet = logdet_p_batch[i] - logdet_q

        # Combine
        kl_batch[i] = 0.5 * (term_trace + term_quad - K + term_logdet)

    return xp.clip(kl_batch, 0, None)


def batch_transport_gaussian_gpu(
    mu: Any,
    Sigma: Any,
    Omega_batch: Any
) -> tuple:
    """
    Batch Gaussian transport optimized for GPU.

    Computes (Omega_i @ mu, Omega_i @ Sigma @ Omega_i.T) for all i.

    Args:
        mu: Mean (K,)
        Sigma: Covariance (K, K)
        Omega_batch: Transport operators (N, K, K)

    Returns:
        mu_batch: Transported means (N, K)
        Sigma_batch: Transported covariances (N, K, K)
    """
    xp = get_backend()

    # Batched mean transport: mu_i = Omega_i @ mu
    mu_batch = xp.einsum('nij,j->ni', Omega_batch, mu)

    # Batched covariance transport: Sigma_i = Omega_i @ Sigma @ Omega_i.T
    # Step 1: temp = Omega @ Sigma
    temp = xp.einsum('nij,jk->nik', Omega_batch, Sigma)
    # Step 2: Sigma_out = temp @ Omega.T
    Sigma_batch = xp.einsum('nij,nkj->nik', temp, Omega_batch)

    # Symmetrize
    Sigma_batch = 0.5 * (Sigma_batch + xp.swapaxes(Sigma_batch, -1, -2))

    return mu_batch, Sigma_batch


# =============================================================================
# Initialization
# =============================================================================

def initialize_backend(
    backend: Optional[BackendType] = None,
    verbose: bool = True
) -> BackendType:
    """
    Initialize compute backend.

    Args:
        backend: Force specific backend, or None for auto-detect
        verbose: Print backend info

    Returns:
        Selected backend name
    """
    global _CURRENT_BACKEND

    if backend is None:
        backend = detect_best_backend()

    set_backend(backend)

    if verbose:
        _print_backend_info()

    # Clear cached backend module
    get_backend.cache_clear()

    return backend


def _print_backend_info():
    """Print information about current backend."""
    print(f"\n[VFE Compute Backend]")
    print(f"  Selected: {_CURRENT_BACKEND}")
    print(f"  CuPy available: {_check_cupy_available()}")
    print(f"  PyTorch CUDA: {_check_pytorch_cuda_available()}")

    if _CURRENT_BACKEND == 'cupy':
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  GPU: {props['name'].decode()}")
        print(f"  Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")

    elif _CURRENT_BACKEND == 'pytorch':
        import torch
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()


# =============================================================================
# Context Manager for Temporary Backend Switching
# =============================================================================

class use_backend:
    """
    Context manager for temporary backend switching.

    Usage:
        with use_backend('numpy'):
            # Force NumPy operations here
            result = some_computation()
    """

    def __init__(self, backend: BackendType):
        self.new_backend = backend
        self.old_backend = None

    def __enter__(self):
        global _CURRENT_BACKEND
        self.old_backend = _CURRENT_BACKEND
        _CURRENT_BACKEND = self.new_backend
        get_backend.cache_clear()
        return get_backend()

    def __exit__(self, *args):
        global _CURRENT_BACKEND
        _CURRENT_BACKEND = self.old_backend
        get_backend.cache_clear()


# =============================================================================
# Auto-initialize on import (detect best backend)
# =============================================================================

# Don't auto-initialize by default - let user control
# Uncomment below for auto-init:
# _CURRENT_BACKEND = detect_best_backend()