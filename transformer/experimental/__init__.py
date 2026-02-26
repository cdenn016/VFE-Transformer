"""
Experimental modules - not part of main training pipeline.

Contains:
    - hamiltonian_ffn: Symplectic Hamiltonian dynamics on belief space
    - pure_fep_transformer: Pure Free Energy Principle transformer (backprop-free)
    - fep_transformer: Free Energy Principle transformer variant
    - train_pure_FEP: Training script for PureFEPTransformer
    - train_fep: Training script for FEPTransformer

These implementations are experimental and not recommended for production use.
For standard training, use the main transformer module:
    from transformer import GaugeTransformerLM, Trainer, TrainingConfig
"""

from transformer.experimental.hamiltonian_ffn import HamiltonianFFN

# Lazy imports to avoid import errors if dependencies change
# Use: from transformer.experimental.pure_fep_transformer import PureFEPTransformer
# Use: from transformer.experimental.fep_transformer import FEPTransformer

__all__ = [
    'HamiltonianFFN',
    # Available via direct import:
    # 'PureFEPTransformer', 'PureFEPConfig', 'PureFEPTrainer',
    # 'FEPTransformer',
]
