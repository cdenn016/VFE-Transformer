# -*- coding: utf-8 -*-
"""
Pytest Configuration and Shared Fixtures
=========================================

Provides common fixtures for transformer tests.
"""

import pytest
import sys
from pathlib import Path

# Suppress warnings before any imports
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device (CPU for CI, GPU if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cpu_device():
    """Force CPU device."""
    return torch.device('cpu')


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def minimal_config():
    """Minimal model config for fast tests."""
    return {
        'vocab_size': 100,
        'embed_dim': 15,
        'n_layers': 1,
        'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
        'hidden_dim': 32,
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'use_diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }


@pytest.fixture
def small_config():
    """Small but realistic config for integration tests."""
    return {
        'vocab_size': 256,
        'embed_dim': 25,
        'n_layers': 2,
        'irrep_spec': [('l0', 5, 1), ('l1', 3, 3), ('l2', 1, 5)],
        'hidden_dim': 64,
        'max_seq_len': 64,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'use_diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }


@pytest.fixture
def vfe_config():
    """Config for VFE dynamic mode tests."""
    return {
        'vocab_size': 100,
        'embed_dim': 16,
        'n_layers': 1,
        'hidden_dim': 32,
        'max_seq_len': 32,
        'kappa_beta': 1.0,
        'dropout': 0.0,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'use_diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
        'vfe_steps': 3,
    }


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def batch_tensors(minimal_config):
    """Create batch of input tensors."""
    B, N = 2, 16
    vocab_size = minimal_config['vocab_size']

    input_ids = torch.randint(0, vocab_size, (B, N))
    targets = torch.randint(0, vocab_size, (B, N))

    return {
        'input_ids': input_ids,
        'targets': targets,
        'batch_size': B,
        'seq_len': N,
    }


@pytest.fixture
def belief_tensors():
    """Create belief state tensors (mu, sigma)."""
    B, N, K = 2, 8, 16

    mu = torch.randn(B, N, K)
    # Sigma must be positive definite
    sigma_diag = torch.abs(torch.randn(B, N, K)) + 0.1

    return {
        'mu': mu,
        'sigma_diag': sigma_diag,
        'batch_size': B,
        'seq_len': N,
        'embed_dim': K,
    }


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def gauge_model(minimal_config, cpu_device):
    """Create a minimal GaugeTransformerLM for testing."""
    from transformer.core.model import GaugeTransformerLM

    model = GaugeTransformerLM(minimal_config)
    model = model.to(cpu_device)
    model.eval()
    return model


@pytest.fixture
def standard_model(minimal_config, cpu_device):
    """Create a StandardTransformerLM for baseline comparison."""
    from transformer.baselines.standard_transformer import StandardTransformerLM

    # StandardTransformer uses different config keys
    config = {
        'vocab_size': minimal_config['vocab_size'],
        'embed_dim': minimal_config['embed_dim'],
        'n_heads': 3,
        'n_layers': minimal_config['n_layers'],
        'hidden_dim': minimal_config['hidden_dim'],
        'max_seq_len': minimal_config['max_seq_len'],
        'dropout': minimal_config['dropout'],
    }

    model = StandardTransformerLM(config)
    model = model.to(cpu_device)
    model.eval()
    return model


# =============================================================================
# Helper Functions
# =============================================================================

def assert_tensor_finite(tensor, name="tensor"):
    """Assert tensor contains no NaN or Inf values."""
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def assert_tensor_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} has shape {tensor.shape}, expected {expected_shape}"


def assert_probabilities(tensor, name="tensor", dim=-1):
    """Assert tensor represents valid probabilities (sum to 1, non-negative)."""
    assert (tensor >= 0).all(), f"{name} has negative values"
    sums = tensor.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"{name} doesn't sum to 1"


# Export helpers for use in tests
pytest.assert_tensor_finite = assert_tensor_finite
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_probabilities = assert_probabilities
