"""
Optimizer Creation with Parameter Grouping
==========================================

Extracted from train.py and train_fast.py to eliminate duplication.
Provides parameter-group-aware optimizer creation for natural gradient
optimization on statistical manifolds.

Parameter Groups:
    1. mu_embed: Mean embeddings (higher LR for natural gradients)
    2. sigma_embed: Covariance embeddings (lower LR for stability)
    3. phi_embed: Gauge frame embeddings
    4. attention: Attention mechanism parameters
    5. ffn: Feed-forward network parameters
    6. output: Output projection parameters
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from transformer.training.config import TrainingConfig


def create_param_groups(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for multi-group optimization.

    This implements natural gradient structure on statistical manifolds
    by assigning different learning rates to different parameter types.

    Args:
        model: The model to create parameter groups for
        config: Training configuration with per-group learning rates
        verbose: If True, print parameter group information

    Returns:
        List of parameter group dicts for torch.optim
    """
    # Collect parameters by type
    mu_params = []
    sigma_params = []
    phi_params = []
    attention_params = []
    ffn_params = []
    output_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Mean embeddings
        if 'mu_embed' in name or 'mu_prior' in name:
            mu_params.append(param)
        # Covariance embeddings
        elif 'sigma_embed' in name or 'log_sigma' in name or 'sigma_prior' in name:
            sigma_params.append(param)
        # Gauge frame embeddings
        elif 'phi_embed' in name or 'phi_prior' in name:
            phi_params.append(param)
        # Positional encoding (treat as gauge frames)
        elif 'pos_encoding' in name or 'position' in name:
            phi_params.append(param)
        # Attention mechanism
        elif 'attention' in name or 'attn' in name:
            attention_params.append(param)
        # Output projection
        elif 'out_proj' in name or 'lm_head' in name:
            output_params.append(param)
        # FFN (default for everything else)
        else:
            ffn_params.append(param)

    # Create parameter groups
    param_groups = []

    if mu_params:
        param_groups.append({
            'params': mu_params,
            'lr': config.mu_lr,
            'weight_decay': 0.0,  # No decay for embeddings
            'name': 'mu_embed',
        })
        if verbose:
            print(f"  Parameter group 'mu_embed': {len(mu_params)} tensors @ lr={config.mu_lr}")

    if sigma_params:
        param_groups.append({
            'params': sigma_params,
            'lr': config.sigma_lr,
            'weight_decay': 0.0,
            'name': 'sigma_embed',
        })
        if verbose:
            print(f"  Parameter group 'sigma_embed': {len(sigma_params)} tensors @ lr={config.sigma_lr}")

    if phi_params:
        param_groups.append({
            'params': phi_params,
            'lr': config.phi_lr,
            'weight_decay': 0.0,
            'name': 'phi_embed',
        })
        if verbose:
            print(f"  Parameter group 'phi_embed': {len(phi_params)} tensors @ lr={config.phi_lr}")

    if attention_params:
        param_groups.append({
            'params': attention_params,
            'lr': config.attention_lr,
            'weight_decay': config.weight_decay,
            'name': 'attention',
        })
        if verbose:
            print(f"  Parameter group 'attention': {len(attention_params)} tensors @ lr={config.attention_lr}")

    if ffn_params:
        param_groups.append({
            'params': ffn_params,
            'lr': config.ffn_lr,
            'weight_decay': config.weight_decay,
            'name': 'ffn',
        })
        if verbose:
            print(f"  Parameter group 'ffn': {len(ffn_params)} tensors @ lr={config.ffn_lr}")

    if output_params:
        param_groups.append({
            'params': output_params,
            'lr': config.output_lr,
            'weight_decay': 0.0,  # Often tied to embeddings
            'name': 'output',
        })
        if verbose:
            print(f"  Parameter group 'output': {len(output_params)} tensors @ lr={config.output_lr}")

    return param_groups


def create_simple_param_groups(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create simple 2-group parameter groups (decay vs no-decay).

    Args:
        model: The model to create parameter groups for
        config: Training configuration
        verbose: If True, print parameter group information

    Returns:
        List of parameter group dicts for torch.optim
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if verbose:
        print(f"  Parameter groups: {len(decay_params)} with decay, {len(no_decay_params)} without")

    return param_groups


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with configurable parameter grouping.

    Args:
        model: The model to optimize
        config: Training configuration
        verbose: If True, print optimizer information

    Returns:
        Configured AdamW optimizer
    """
    if config.use_param_groups:
        # Multi-group mode: Natural gradients with per-parameter-type learning rates
        if verbose:
            print("Creating multi-group optimizer (natural gradients):")
        param_groups = create_param_groups(model, config, verbose=verbose)
    else:
        # Simple mode: Traditional 2-group optimizer (decay vs no-decay)
        if verbose:
            print("Creating simple optimizer:")
        param_groups = create_simple_param_groups(model, config, verbose=verbose)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,  # Base LR (overridden by group-specific LRs)
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule
        config: Training configuration

    Returns:
        LR scheduler or None if constant
    """
    if config.lr_decay == 'constant':
        return None

    def lr_lambda(step):
        # Warmup phase
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)

        # Decay phase
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, progress)  # Clamp to [0, 1]

        if config.lr_decay == 'cosine':
            import math
            return config.min_lr / config.learning_rate + \
                   0.5 * (1 - config.min_lr / config.learning_rate) * \
                   (1 + math.cos(progress * math.pi))
        elif config.lr_decay == 'linear':
            return max(config.min_lr / config.learning_rate, 1 - progress)
        else:
            return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
