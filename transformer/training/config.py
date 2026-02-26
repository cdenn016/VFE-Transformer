"""
Unified Training Configuration
==============================

Single source of truth for all training configuration parameters.
Supports multiple training modes through configuration rather than
separate classes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TrainingConfig:
    """
    Unified training configuration supporting all training modes.

    Modes:
    - Simple (use_param_groups=False): Single learning rate for all parameters
    - Multi-group (use_param_groups=True): Separate learning rates for mu, sigma, phi, etc.

    Training Types:
    - 'standard': Standard transformer (no gauge theory)
    - 'vfe_dynamic': VFE-based transformer with gauge theory
    - 'pure_fep': Pure Free Energy Principle (no backprop)
    """

    # ==========================================================================
    # Training Mode
    # ==========================================================================
    training_mode: str = 'vfe_dynamic'  # 'standard', 'vfe_dynamic', 'pure_fep'

    # ==========================================================================
    # Parameter Grouping Strategy
    # ==========================================================================
    use_param_groups: bool = True  # If True, use multi-group learning rates

    # Simple mode: Single learning rate (used when use_param_groups=False)
    learning_rate: float = 3e-4

    # Multi-group mode: Per-parameter group learning rates
    mu_lr: float = 0.1           # Mean embeddings (natural gradient scale)
    sigma_lr: float = 0.005      # Covariance embeddings (smaller for stability)
    phi_lr: float = 0.01         # Gauge frames
    attention_lr: float = 0.01   # Attention parameters
    ffn_lr: float = 0.001        # FFN parameters (standard)
    output_lr: float = 0.001     # Output projection

    # ==========================================================================
    # Optimizer Hyperparameters
    # ==========================================================================
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # ==========================================================================
    # Learning Rate Schedule
    # ==========================================================================
    warmup_steps: int = 1000
    max_steps: int = 50000
    lr_decay: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr: float = 3e-5

    # ==========================================================================
    # Free Energy Weights
    # ==========================================================================
    # NOTE: alpha > 0 is CRITICAL for gradient flow to embeddings!
    alpha: float = 0.1           # Self-consistency: KL(q||p) to embedding priors
    lambda_beta: float = 1.0     # Belief alignment: Σβ_ij·KL (CRUCIAL!)
    lambda_gamma: float = 0.0    # Model alignment (disabled by default)
    kappa_gamma: float = 1.0     # Temperature for γ_ij coupling weights

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    batch_size: int = 16
    max_seq_len: int = 256
    num_epochs: Optional[int] = None  # If set, overrides max_steps
    accumulation_steps: int = 1

    # ==========================================================================
    # Logging & Evaluation
    # ==========================================================================
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500

    # Aliases for backward compatibility
    @property
    def log_every(self) -> int:
        return self.log_interval

    @property
    def eval_every(self) -> int:
        return self.eval_interval

    @property
    def save_every(self) -> int:
        return self.checkpoint_interval

    # ==========================================================================
    # Early Stopping
    # ==========================================================================
    patience: int = 0  # If > 0, stop if no improvement for this many evals
    min_improvement: float = 0.001  # Minimum improvement to reset patience

    # ==========================================================================
    # Checkpointing
    # ==========================================================================
    checkpoint_dir: Optional[Path] = None
    save_optimizer: bool = True
    save_total_limit: int = 3
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # ==========================================================================
    # Weights & Biases
    # ==========================================================================
    use_wandb: bool = False
    wandb_project: str = 'gauge-transformer'
    wandb_run_name: Optional[str] = None

    # ==========================================================================
    # Hardware
    # ==========================================================================
    device: str = 'cuda'
    use_amp: bool = False  # Automatic mixed precision

    # ==========================================================================
    # Gauge Group
    # ==========================================================================
    # When True, trivialize the gauge group by setting all transport operators
    # to identity (Ω_ij = I). This bypasses matrix exponentials in attention,
    # reducing to raw KL(q_i || q_j) without frame alignment.
    # Maps to 'use_identity_transport' in the model config dict.
    use_identity_group: bool = False

    # ==========================================================================
    # Model Architecture (for creation, not training)
    # ==========================================================================
    embed_dim: int = 128
    n_layers: int = 4
    vocab_size: int = 50257  # GPT-2 tokenizer size

    def __post_init__(self):
        """Convert checkpoint_dir to Path if string."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_standard_config(**overrides) -> TrainingConfig:
    """Get configuration for standard transformer baseline."""
    config = TrainingConfig(
        training_mode='standard',
        use_param_groups=False,
        learning_rate=3e-4,
        alpha=0.0,
        lambda_beta=0.0,
        lambda_gamma=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def get_vfe_dynamic_config(**overrides) -> TrainingConfig:
    """
    Get configuration for VFE-dynamic transformer.

    Supports use_identity_group=True to trivialize gauge transport
    (sets all Ω_ij = I, bypassing matrix exponentials in attention).
    """
    config = TrainingConfig(
        training_mode='vfe_dynamic',
        use_param_groups=True,
        mu_lr=0.1,
        sigma_lr=0.005,
        phi_lr=0.01,
        attention_lr=0.01,
        ffn_lr=0.001,
        output_lr=0.001,
        alpha=0.1,
        lambda_beta=1.0,
        lambda_gamma=0.0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def get_pure_fep_config(**overrides) -> TrainingConfig:
    """Get configuration for pure FEP (backprop-free) training."""
    config = TrainingConfig(
        training_mode='pure_fep',
        use_param_groups=False,  # No optimizer in pure FEP
        alpha=1.0,
        lambda_beta=1.0,
        lambda_gamma=0.1,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
