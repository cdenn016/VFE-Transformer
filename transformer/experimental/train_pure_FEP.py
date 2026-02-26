#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure FEP Transformer Training Script
=====================================

Language modeling with PURE Free Energy Principle learning.
NO Adam, NO backprop through VFE - just hierarchical belief/prior dynamics.

Usage:
    # Just edit the config below and run:
    python transformer/train_pure_fep.py

Author: Chris
Date: December 2025
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
import math
from datetime import datetime

from transformer.experimental.pure_fep_transformer import PureFEPConfig, PureFEPTransformer


# =============================================================================
# EDIT THIS CONFIG - Just click Run!
# =============================================================================

CONFIG = {
    # Architecture
    'embed_dim': 11,             # K - embedding dimension
    'num_layers': 2,              # Hierarchical scales
    'seq_length': 24,            # Max sequence length
    'vocab_size': 5000,          # Will be overwritten if using dataset

    # =========================================================================
    # GAUGE GROUP: Choose the symmetry structure
    # =========================================================================
    # 'SO3': Standard SO(3) with 3 generators, φ ∈ ℝ³
    #        - embed_dim MUST be ODD for single irrep
    #        - Irrep dims must be 1, 3, 5, 7, ... (2ℓ+1)
    # 'SON': General SO(N) with N(N-1)/2 generators, φ ∈ ℝ^{N(N-1)/2}
    #        - Uses N-dim fundamental representation
    #        - More generators = richer gauge structure
    'gauge_group': 'SO3',         # 'SO3' or 'SON'
    'gauge_dim': 8,               # N for SO(N) - only used when gauge_group='SON'

    # Irrep structure for SO(3) decomposition
    # Each tuple: (label, multiplicity, dim) where dim is 1,3,5,7,...
    # Total must equal embed_dim. Set None for auto-generation.
    # Example for K=127: 32×1 + 15×3 + 10×5 = 32 + 45 + 50 = 127
    #
    # For SO(N), supported irreps:
    #   - 'scalar': dim = 1              (gauge-invariant)
    #   - 'fund':   dim = N              (fundamental/vector)
    #   - 'wedge2': dim = N*(N-1)/2      (antisymmetric 2-tensor ∧²V)
    #   - 'sym2':   dim = N*(N+1)/2 - 1  (symmetric traceless Sym²₀V)
    #
    # Example for SO(5) with K=132:
    #   [('scalar', 10, 1), ('fund', 8, 5), ('wedge2', 4, 10), ('sym2', 3, 14)]
    #   = 10 + 40 + 40 + 42 = 132
    #
    # Different irreps have different Casimir eigenvalues (transformation rates):
    #   fund: 1.0x, wedge2: ~1.5x, sym2: ~2.5x
    # This provides genuine representational diversity (like SO(3) spin-ℓ irreps)
    'irrep_spec': [
       # ('ℓ0', 32, 1),   # 32 scalars
       # ('ℓ1', 15, 3),   # 45 dims (vectors)
        ('ℓ5', 1, 11),   # 50 dims (rank-2 tensors)
    ],  # = 127 total

    # =========================================================================
    # PURE FEP MODE - TRUE FREE ENERGY PRINCIPLE LEARNING
    # =========================================================================
    # When True: NO backprop on embeddings, ALL learning via prior evolution
    # When False: Hybrid mode with backprop + prior evolution
    'pure_fep_mode': True,        # THE KEY SETTING!

    # VFE parameters
    'alpha': 0.1,                 # Self-coupling weight KL(q||p)
    'lambda_belief': 1.0,         # Belief alignment weight
    'lambda_obs': 1.0,            # Observation likelihood weight (CE in VFE)
    'kappa': 0.1,                 # Attention temperature (LOWER = sharper attention!)

    # Learning rates
    'mu_lr': 0.1,                 # Belief mean update
    'sigma_lr': 0.025,            # Belief variance update
    'prior_lr': 0.01,             # Prior update (SLOWER for stability)
    'phi_lr': 0.05,               # Gauge frame update

    # =========================================================================
    # TIMESCALES - CRITICAL FOR FEP LEARNING!
    # =========================================================================
    # Fast timescale: VFE gradient descent (perception)
    # Slow timescale: Prior updates (learning)
    # Rule: More VFE steps = better belief convergence before prior update
    'n_vfe_steps': 20,            # VFE iterations per forward (MORE for convergence)
    'prior_update_interval': 1,   # Update priors every batch

    # Stability
    'grad_clip': 1.0,

    # VFE mode - only used when pure_fep_mode=False
    'differentiable_vfe': True,

    # =========================================================================
    # ADVANCED FEP FEATURES (experimental)
    # =========================================================================
    # Prior coupling: priors learn from each other via KL(p_i || Ω_ij·p_j)
    # CRITICAL: Must be True for gauge evolution! (∂F/∂φ includes prior term)
    'prior_coupling_enabled': True,  # Was False - MUST be True!
    'lambda_prior': 0.1,

    # Gradient-based prior updates: use VFE gradient instead of EMA
    'gradient_prior_updates': True,
    'prior_grad_lr': 0.01,

    # Gauge field evolution: φ evolves via dφ/dt = -∂F/∂φ (pure VFE!)
    'gauge_evolution_enabled': True,
    'gauge_lr': 0.01,
    'phi_max_norm': 3.14159,  # Max norm for φ (π radians)

    # Dynamic layers: spawn/merge layers based on VFE (experimental)
    'dynamic_layers_enabled': False,
    'layer_spawn_threshold': 0.5,
    'max_layers': 8,

    # Ouroboros Tower: Multi-level hyperpriors from ALL ancestors
    # Creates non-Markovian memory by collecting priors from grandparent, great-grandparent, etc.
    # Each hyperprior contributes with decaying weight: F += Σ_d decay^d · KL(p || h^d)
    'enable_ouroboros_tower': False,
    'tower_max_depth': 3,             # How many ancestor levels to collect
    'tower_decay': 0.3,               # Weight decay per level (0.3^d)

    # Training
    'batch_size': 24,
    'epochs': 5,                  # More epochs for pure FEP learning
    'log_interval': 50,

    # Data
    'dataset': 'synthetic',       # 'synthetic' or 'wikitext2'
    'data_dir': './data',

    # Output
    'save_dir': './checkpoints/pure_fep',

    # Device
    'device': 'auto',             # 'auto', 'cuda', 'cpu'

    # Reproducibility
    'seed': 42,
}

# Debug mode - uncomment to use smaller config for testing
# CONFIG.update({
#     'embed_dim': 63,
#     'num_layers': 1,
#     'seq_length': 32,
#     'batch_size': 4,
#     'epochs': 2,
#     'log_interval': 10,
# })


# =============================================================================
# DATA LOADING
# =============================================================================

class SyntheticDataLoader:
    """Simple synthetic data for testing."""

    def __init__(self, n_samples, seq_length, vocab_size, batch_size):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.data = torch.randint(0, vocab_size, (n_samples, seq_length + 1))

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = torch.randperm(self.n_samples)
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_data = self.data[batch_idx]
            yield {
                'input_ids': batch_data[:, :-1],
                'targets': batch_data[:, 1:],
            }


def load_data(config):
    """Load training and validation data."""
    dataset = config['dataset']
    batch_size = config['batch_size']
    seq_length = config['seq_length']

    if dataset == 'wikitext2':
        try:
            from transformer.data import create_dataloaders
            train_loader, val_loader = create_dataloaders(
                batch_size=batch_size,
                seq_length=seq_length,
                data_dir=config['data_dir'],
            )
            vocab_size = len(train_loader.dataset.vocab_mapping)
            return train_loader, val_loader, vocab_size
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print(f"  Warning: Could not load WikiText-2: {e}")
            print(f"  Falling back to synthetic data")

    # Synthetic data
    vocab_size = config['vocab_size']
    train_loader = SyntheticDataLoader(5000, seq_length, vocab_size, batch_size)
    val_loader = SyntheticDataLoader(500, seq_length, vocab_size, batch_size)
    return train_loader, val_loader, vocab_size


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ppl = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        metrics = model.train_step(
            input_ids, targets,
            n_vfe_steps=config['n_vfe_steps']
        )

        total_loss += metrics['ce_loss']
        total_ppl += metrics['perplexity']
        n_batches += 1

        if batch_idx % config['log_interval'] == 0 and batch_idx > 0:
            avg_loss = total_loss / n_batches
            avg_ppl = total_ppl / n_batches
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")

    return {
        'loss': total_loss / n_batches,
        'perplexity': total_ppl / n_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, vocab_size):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow (exp(20) ≈ 485M)

    return {'loss': avg_loss, 'perplexity': ppl}


def save_checkpoint(model, config, history, save_dir, name):
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, save_dir / f'{name}.pt')

    with open(save_dir / f'{name}_config.json', 'w') as f:
        json.dump(config, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = CONFIG.copy()

    # Validate embed_dim based on gauge group
    gauge_group = config.get('gauge_group', 'SO3')
    gauge_dim = config.get('gauge_dim', 8)
    irrep_spec = config.get('irrep_spec')

    if gauge_group == 'SO3':
        # For SO(3) without explicit irrep_spec, embed_dim must be odd
        if irrep_spec is None and config['embed_dim'] % 2 == 0:
            raise ValueError(
                f"embed_dim must be ODD for single SO(3) irrep (got {config['embed_dim']}). "
                f"Try {config['embed_dim'] - 1} or {config['embed_dim'] + 1}, "
                f"OR provide an irrep_spec, OR use gauge_group='SON'."
            )
    elif gauge_group == 'SON':
        if gauge_dim < 2:
            raise ValueError(f"gauge_dim must be >= 2 for SO(N), got {gauge_dim}")
        # For SO(N), auto-generate irrep_spec if not provided
        if irrep_spec is None:
            n_copies = config['embed_dim'] // gauge_dim
            remainder = config['embed_dim'] % gauge_dim
            if remainder != 0:
                config['irrep_spec'] = [
                    ('scalar', remainder, 1),
                    ('fund', n_copies, gauge_dim),
                ]
            else:
                config['irrep_spec'] = [('fund', n_copies, gauge_dim)]
            print(f"  Auto-generated SO({gauge_dim}) irrep_spec: {config['irrep_spec']}")

    # Set seed for full reproducibility
    seed = config['seed']
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

    # Device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"\n{'='*60}")
    print("PURE FEP TRANSFORMER TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dataset: {config['dataset']}")

    # Load data
    print(f"\nLoading data...")
    train_loader, val_loader, vocab_size = load_data(config)
    config['vocab_size'] = vocab_size

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Vocab size: {vocab_size}")

    # Create model config
    model_config = PureFEPConfig(
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        seq_length=config['seq_length'],
        vocab_size=vocab_size,
        irrep_spec=config.get('irrep_spec'),  # None = auto-generate
        # Gauge group
        gauge_group=config.get('gauge_group', 'SO3'),
        gauge_dim=config.get('gauge_dim', 8),
        # VFE parameters
        alpha=config['alpha'],
        lambda_belief=config['lambda_belief'],
        lambda_obs=config.get('lambda_obs', 1.0),  # Observation weight in VFE
        kappa=config['kappa'],
        # Learning rates
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        prior_lr=config['prior_lr'],
        phi_lr=config['phi_lr'],
        # Timescales
        belief_steps=config['n_vfe_steps'],
        prior_update_interval=config['prior_update_interval'],
        # Stability
        grad_clip=config['grad_clip'],
        # PURE FEP MODE - the key setting!
        pure_fep_mode=config.get('pure_fep_mode', True),
        differentiable_vfe=config.get('differentiable_vfe', True),
        # Advanced FEP features
        prior_coupling_enabled=config.get('prior_coupling_enabled', False),
        lambda_prior=config.get('lambda_prior', 0.1),
        gradient_prior_updates=config.get('gradient_prior_updates', False),
        prior_grad_lr=config.get('prior_grad_lr', 0.01),
        gauge_evolution_enabled=config.get('gauge_evolution_enabled', False),
        gauge_lr=config.get('gauge_lr', 0.01),
        dynamic_layers_enabled=config.get('dynamic_layers_enabled', False),
        layer_spawn_threshold=config.get('layer_spawn_threshold', 0.5),
        max_layers=config.get('max_layers', 8),
        # Ouroboros Tower
        enable_ouroboros_tower=config.get('enable_ouroboros_tower', False),
        tower_max_depth=config.get('tower_max_depth', 3),
        tower_decay=config.get('tower_decay', 0.3),
    )

    print(f"\nModel config:")
    print(f"  embed_dim (K): {model_config.embed_dim}")
    # Show gauge group info
    if model_config.gauge_group == 'SO3':
        print(f"  gauge_group: SO(3) - 3 generators, φ ∈ ℝ³")
    else:
        N = model_config.gauge_dim
        n_gen = N * (N - 1) // 2
        print(f"  gauge_group: SO({N}) - {n_gen} generators, φ ∈ ℝ^{n_gen}")
    print(f"  irrep_spec: {model_config.irrep_spec}")
    print(f"  num_layers: {model_config.num_layers}")
    print(f"  mu_lr: {model_config.mu_lr}")
    print(f"  prior_lr: {model_config.prior_lr}")
    print(f"  alpha: {model_config.alpha}")
    print(f"  lambda_obs: {model_config.lambda_obs}")
    print(f"  n_vfe_steps: {config['n_vfe_steps']}")

    # Show learning mode
    if model_config.pure_fep_mode:
        print(f"\n  [PURE FEP MODE] No backprop - learning via prior evolution only!")
        print(f"    - Position-dependent priors: ({model_config.seq_length}, {model_config.embed_dim})")
        print(f"    - Observations INSIDE VFE (λ_obs={model_config.lambda_obs})")
    else:
        print(f"\n  [HYBRID MODE] Backprop + prior evolution")

    # Print advanced features if enabled
    if model_config.prior_coupling_enabled:
        print(f"  [ENABLED] Prior coupling: λ_γ={model_config.lambda_prior}")
    if model_config.gradient_prior_updates:
        print(f"  [ENABLED] Gradient prior updates: lr={model_config.prior_grad_lr}")
    if model_config.gauge_evolution_enabled:
        print(f"  [ENABLED] Gauge evolution: lr={model_config.gauge_lr}")
    if model_config.enable_ouroboros_tower:
        print(f"  [ENABLED] Ouroboros Tower: depth={model_config.tower_max_depth}, decay={model_config.tower_decay}")
    if model_config.dynamic_layers_enabled:
        print(f"  [ENABLED] Dynamic layers: threshold={model_config.layer_spawn_threshold}, max={model_config.max_layers}")

    # Create model
    print(f"\nCreating model...")
    model = PureFEPTransformer(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    best_val_ppl = float('inf')
    history = []

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, device, config, epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, device, vocab_size)

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | PPL: {train_metrics['perplexity']:.2f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | PPL: {val_metrics['perplexity']:.2f}")

        # Track history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_ppl': train_metrics['perplexity'],
            'val_loss': val_metrics['loss'],
            'val_ppl': val_metrics['perplexity'],
            'time': epoch_time,
        })

        # Save best model
        if val_metrics['perplexity'] < best_val_ppl:
            best_val_ppl = val_metrics['perplexity']
            save_checkpoint(model, config, history, config['save_dir'], 'best')
            print(f"  [NEW BEST] Saved checkpoint")

    # Save final model
    save_checkpoint(model, config, history, config['save_dir'], 'final')

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val PPL: {best_val_ppl:.2f}")
    print(f"Checkpoints saved to: {config['save_dir']}")


if __name__ == '__main__':
    main()