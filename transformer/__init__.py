# -*- coding: utf-8 -*-
"""
Gauge-Theoretic Transformer Package
====================================

Implements gauge-theoretic transformers with KL-divergence based attention
and variational free energy dynamics.

Package Structure:
    transformer/
    ├── core/           # Core model components
    ├── training/       # Training infrastructure
    ├── data/           # Data loading
    ├── analysis/       # Analysis and metrics
    ├── visualization/  # Plotting and visualization
    ├── utils/          # Utilities
    ├── baselines/      # Baseline models
    └── experimental/   # Experimental code
"""

# Suppress noisy Triton warnings
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", category=UserWarning, module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", category=UserWarning, module="triton")

# =============================================================================
# Core Model (from transformer.core)
# =============================================================================
from transformer.core.model import GaugeTransformerLM

# =============================================================================
# Training (from transformer.train and transformer.training)
# =============================================================================
from transformer.train import Trainer, TrainingConfig
from transformer.training import (
    create_optimizer,
    create_param_groups,
    MetricsTracker,
)
from transformer.training.config import (
    get_standard_config,
    get_vfe_dynamic_config,
    get_pure_fep_config,
)

# =============================================================================
# Data Loading (from transformer.data)
# =============================================================================
from transformer.data import (
    create_dataloaders,
    create_char_dataloaders,
    create_byte_dataloaders,
)

__all__ = [
    # Core model
    'GaugeTransformerLM',

    # Training
    'Trainer',
    'TrainingConfig',
    'create_optimizer',
    'create_param_groups',
    'MetricsTracker',
    'get_standard_config',
    'get_vfe_dynamic_config',
    'get_pure_fep_config',

    # Data loading
    'create_dataloaders',
    'create_char_dataloaders',
    'create_byte_dataloaders',
]
