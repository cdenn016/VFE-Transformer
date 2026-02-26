"""
Unified Training Module for Gauge Transformer
==============================================

This module consolidates training infrastructure previously scattered
across multiple files (train.py, train_fast.py, train_publication.py).

Components:
    - TrainingConfig: Unified configuration dataclass
    - create_optimizer: Parameter group-aware optimizer creation
    - create_scheduler: Learning rate scheduling
    - MetricsTracker: CSV logging for training metrics
    - compute_free_energy_loss: Free energy loss computation

Usage:
    from transformer.training import TrainingConfig, create_optimizer
    from transformer.train import Trainer  # Main trainer class
"""

from transformer.training.config import TrainingConfig
from transformer.training.optimizer import (
    create_optimizer,
    create_param_groups,
)
from transformer.training.metrics import MetricsTracker

__all__ = [
    'TrainingConfig',
    'create_optimizer',
    'create_param_groups',
    'MetricsTracker',
]
