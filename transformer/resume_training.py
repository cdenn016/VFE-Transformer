#!/usr/bin/env python3
"""
Resume Training from Checkpoint
===============================

Simple script to resume training from a checkpoint after power failure or interruption.
No CLI arguments needed - just edit the configuration below.

Usage:
    1. Set CHECKPOINT_PATH to your checkpoint file
    2. Set EXPERIMENT_DIR to the experiment directory (contains experiment_config.json)
    3. Optionally adjust TARGET_STEPS if you want to train longer
    4. Run: python transformer/resume_training.py
"""

import sys
import os

# Ensure project root is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import json
from pathlib import Path

from transformer.core.model import GaugeTransformerLM
from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer._archive.train_fast import FastTrainingConfig
from transformer.train_publication import run_test_evaluation, PublicationTrainer
from transformer.analysis.publication_metrics import PublicationMetrics


# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Path to checkpoint file (e.g., checkpoint_step_179999.pt)
CHECKPOINT_PATH = "checkpoints_publication/ffn_VFE_dynamic/checkpoint_step_179999.pt"

# Experiment directory (contains experiment_config.json)
# If None, will use checkpoint's parent directory
EXPERIMENT_DIR = None

# Target total steps (set higher than checkpoint step to continue training)
# If None, will use original max_steps from config
TARGET_STEPS = None

# Override batch size (set to reduce memory usage if needed)
# If None, will use original batch_size from config
BATCH_SIZE = None

# Gradient accumulation steps (effective_batch = batch_size * grad_accum)
# Set higher to compensate for smaller batch size
GRAD_ACCUMULATION = 1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# RESUME TRAINING LOGIC
# =============================================================================


def load_experiment_config(experiment_dir: Path) -> dict:
    """Load experiment configuration from JSON file."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        return {}  # Return empty dict if not found

    with open(config_path, 'r') as f:
        data = json.load(f)

    # Handle nested config structure
    if 'config' in data and isinstance(data['config'], dict):
        return data['config']
    return data


def extract_config_from_checkpoint(checkpoint: dict) -> dict:
    """Extract model config from checkpoint."""
    config = {}

    # Try 'config' key (most common)
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        if isinstance(ckpt_config, dict):
            config.update(ckpt_config)
        elif hasattr(ckpt_config, '__dict__'):
            # It's a dataclass or similar
            config.update(vars(ckpt_config))

    # Also check for individual keys that might be stored at top level
    for key in ['embed_dim', 'n_layers', 'vocab_size', 'max_seq_len', 'irrep_spec',
                'hidden_dim', 'n_heads', 'dropout', 'ffn_mode', 'gauge_group', 'gauge_dim']:
        if key in checkpoint and key not in config:
            config[key] = checkpoint[key]

    return config


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model architecture from state_dict tensor shapes.

    This is the most reliable way to get the correct architecture
    when checkpoint config is missing or incorrect.
    """
    config = {}

    # Infer embed_dim from generator shape or embedding weight
    if 'generators' in state_dict:
        # generators shape: (n_gen, embed_dim, embed_dim)
        config['embed_dim'] = state_dict['generators'].shape[1]
    elif 'token_embed.mu_embed.weight' in state_dict:
        # mu_embed.weight shape: (vocab_size, embed_dim)
        config['embed_dim'] = state_dict['token_embed.mu_embed.weight'].shape[1]

    # Infer vocab_size from embedding
    if 'token_embed.mu_embed.weight' in state_dict:
        config['vocab_size'] = state_dict['token_embed.mu_embed.weight'].shape[0]
    elif 'out_proj.weight' in state_dict:
        config['vocab_size'] = state_dict['out_proj.weight'].shape[0]

    # Infer max_seq_len from positional encoding
    if 'pos_encoding.pos_phi' in state_dict:
        config['max_seq_len'] = state_dict['pos_encoding.pos_phi'].shape[0]

    # Infer n_layers by counting transformer blocks
    n_layers = 0
    for key in state_dict.keys():
        if 'transformer.blocks.' in key:
            # Extract block number from key like "transformer.blocks.0.attention..."
            parts = key.split('.')
            try:
                block_idx = int(parts[2])
                n_layers = max(n_layers, block_idx + 1)
            except (IndexError, ValueError):
                pass
    if n_layers > 0:
        config['n_layers'] = n_layers

    # Infer irrep_spec from attention head generators
    # Keys like: transformer.blocks.0.attention.head_generators.0.gen
    head_dims = []
    for key in state_dict.keys():
        if 'attention.head_generators.' in key and key.endswith('.gen'):
            # Shape: (n_gen, head_dim, head_dim)
            head_dim = state_dict[key].shape[1]
            # Extract head index
            parts = key.split('.')
            try:
                head_idx = int(parts[parts.index('head_generators') + 1])
                while len(head_dims) <= head_idx:
                    head_dims.append(None)
                head_dims[head_idx] = head_dim
            except (IndexError, ValueError):
                pass

    if head_dims and all(d is not None for d in head_dims):
        # Convert to irrep_spec format
        # head_dim = 2*ell + 1 for SO(3)
        irrep_spec = []
        for i, dim in enumerate(head_dims):
            ell = (dim - 1) // 2
            irrep_spec.append([f'ℓ{ell}', 1, dim])
        config['irrep_spec'] = irrep_spec

    # Infer diagonal_covariance from sigma storage format
    # log_sigma_diag = diagonal mode, log_sigma or sigma_embed = full mode
    if 'token_embed.log_sigma_diag' in state_dict:
        config['diagonal_covariance'] = True
        config['use_diagonal_covariance'] = True
    elif 'token_embed.log_sigma' in state_dict or 'token_embed.sigma_embed' in state_dict:
        config['diagonal_covariance'] = False
        config['use_diagonal_covariance'] = False

    return config


def resume_training():
    """Resume training from checkpoint."""

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        for pt_file in Path(".").rglob("checkpoint_step_*.pt"):
            print(f"  {pt_file}")
        return

    # Determine experiment directory
    if EXPERIMENT_DIR is not None:
        experiment_dir = Path(EXPERIMENT_DIR)
    else:
        experiment_dir = checkpoint_path.parent

    print("=" * 70)
    print("RESUMING TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Device: {DEVICE}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    start_step = checkpoint.get('step', 0)
    best_val_ce = checkpoint.get('best_val_ce', float('inf'))
    print(f"  Checkpoint step: {start_step}")
    print(f"  Best val CE: {best_val_ce:.4f}")

    # Load config - infer from state_dict first (most reliable), then merge others
    print("\nLoading config...")

    # Get model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = {}

    # FIRST: Infer architecture from actual tensor shapes (most reliable!)
    inferred_config = infer_config_from_state_dict(state_dict)
    if inferred_config:
        print(f"  Inferred from state_dict: {inferred_config}")

    # SECOND: Get config stored in checkpoint
    ckpt_config = extract_config_from_checkpoint(checkpoint)
    if ckpt_config:
        print(f"  Checkpoint config: {len(ckpt_config)} keys")

    # THIRD: Load experiment_config.json
    json_config = load_experiment_config(experiment_dir)
    if json_config:
        print(f"  experiment_config.json: {len(json_config)} keys")

    # Merge configs: inferred > checkpoint > json > defaults
    # Start with json, then checkpoint, then inferred (last wins for conflicts)
    config = {}
    config.update(json_config)
    config.update(ckpt_config)
    config.update(inferred_config)  # Inferred values override everything

    if not config:
        print("  WARNING: No config found!")
        print("  Will use defaults - this may not match your original model.")

    # Print key architecture params for verification
    print(f"\n  Model architecture (from state_dict):")
    print(f"    embed_dim: {config.get('embed_dim', 'NOT SET')}")
    print(f"    n_layers: {config.get('n_layers', 'NOT SET')}")
    print(f"    max_seq_len: {config.get('max_seq_len', 'NOT SET')}")
    print(f"    irrep_spec: {config.get('irrep_spec', 'NOT SET')}")

    # Override batch_size if specified
    if BATCH_SIZE is not None:
        original_batch = config.get('batch_size', 32)
        config['batch_size'] = BATCH_SIZE
        print(f"  Overriding batch_size: {original_batch} -> {BATCH_SIZE}")

    # Print memory-critical settings
    print(f"\n  Memory-critical settings:")
    print(f"    batch_size: {config.get('batch_size', 32)}")
    print(f"    hidden_dim: {config.get('hidden_dim', 'NOT SET')}")
    print(f"    diagonal_covariance: {config.get('diagonal_covariance', 'NOT SET')}")
    print(f"    use_diagonal_covariance: {config.get('use_diagonal_covariance', 'NOT SET')}")

    # Override max_steps if specified
    original_max_steps = config.get('max_steps', 200000)
    if TARGET_STEPS is not None:
        config['max_steps'] = TARGET_STEPS
        print(f"  Overriding max_steps: {original_max_steps} -> {TARGET_STEPS}")
    else:
        config['max_steps'] = original_max_steps

    if start_step >= config['max_steps']:
        print(f"\nWARNING: Checkpoint step ({start_step}) >= max_steps ({config['max_steps']})")
        print("Set TARGET_STEPS higher to continue training.")
        return

    remaining_steps = config['max_steps'] - start_step
    print(f"  Remaining steps: {remaining_steps}")

    # Create data loaders
    print("\nCreating data loaders...")
    # Try multiple keys for dataset name
    dataset_name = (config.get('dataset') or
                    config.get('dataset_name') or
                    'wikitext-103')  # Default to 103, not 2
    print(f"  Dataset: {dataset_name}")

    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config.get('vocab_size', 50257) <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    if use_char:
        print(f"  Using character-level tokenizer")
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config.get('max_seq_len', 256),
            batch_size=config.get('batch_size', 32),
            num_workers=config.get('num_workers', 0),
        )
        test_loader = None
    else:
        print(f"  Using BPE tokenizer")
        train_loader, val_loader, test_loader, actual_vocab_size = create_dataloaders(
            max_seq_len=config.get('max_seq_len', 256),
            batch_size=config.get('batch_size', 32),
            vocab_size=config.get('vocab_size', 50257),
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,
        )

    config['vocab_size'] = actual_vocab_size
    print(f"  Vocab size: {actual_vocab_size}")

    # Ensure required model config keys have defaults
    model_defaults = {
        'kappa_beta': 1.0,
        'lambda_beta': 1.0,
        'alpha': 0.1,
        'beta': 1.0,
        'lambda_gamma': 0.0,
        'dropout': 0.1,
        'n_layers': 4,
        'n_heads': 1,
        'hidden_dim': config.get('embed_dim', 128) * 4,
        'ffn_mode': 'VFE_dynamic',
        'diagonal_covariance': True,
        'use_diagonal_covariance': True,
        'evolve_sigma': True,
        'evolve_phi': True,
        'tie_embeddings': True,
        'gauge_group': 'SO3',
        'gauge_dim': 3,
    }
    for key, default in model_defaults.items():
        if key not in config:
            config[key] = default

    # Create model
    print("\nCreating model...")
    ffn_mode = config.get('ffn_mode', 'VFE_dynamic')
    print(f"  FFN mode: {ffn_mode}")

    if ffn_mode == 'standard':
        model_config = {
            'vocab_size': actual_vocab_size,
            'embed_dim': config.get('embed_dim', 128),
            'n_layers': config.get('n_layers', 4),
            'n_heads': config.get('n_heads', 1),
            'hidden_dim': config.get('hidden_dim', config.get('embed_dim', 128) * 4),
            'max_seq_len': config.get('max_seq_len', 256),
            'dropout': config.get('dropout', 0.1),
        }
        model = StandardTransformerLM(model_config)
    else:
        model = GaugeTransformerLM(config)

    device = torch.device(DEVICE)
    model = model.to(device)

    # Load model weights
    print("Loading model weights from checkpoint...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        # Try loading directly (checkpoint might be just state dict)
        model.load_state_dict(checkpoint)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create training config
    print("\nCreating training config...")

    # Print key performance settings
    print(f"  Performance settings:")
    print(f"    batch_size: {config.get('batch_size', 32)}")
    print(f"    grad_accumulation: {GRAD_ACCUMULATION}")
    print(f"    diagonal_covariance: {config.get('diagonal_covariance', True)}")

    train_config = FastTrainingConfig(
        max_steps=config['max_steps'],
        warmup_steps=config.get('warmup_steps', 1000),

        # Learning rates
        mu_lr=config.get('mu_lr', 0.1),
        sigma_lr=config.get('sigma_lr', 0.005),
        phi_lr=config.get('phi_lr', 0.01),
        attention_lr=config.get('phi_lr', 0.01),
        ffn_lr=config.get('ffn_lr', 0.001),
        output_lr=config.get('ffn_lr', 0.001),

        weight_decay=config.get('weight_decay', 0.01),
        grad_clip=config.get('grad_clip', 1.0),
        grad_accumulation_steps=GRAD_ACCUMULATION,

        # Free energy weights
        alpha=config.get('alpha', 0.1),
        beta=config.get('beta', 1.0),
        lambda_gamma=config.get('lambda_gamma', 0.0),

        # Intervals
        log_interval=config.get('log_interval', 100),
        eval_interval=config.get('eval_interval', 500),
        checkpoint_interval=config.get('checkpoint_interval', 5000),

        # Checkpointing
        checkpoint_dir=experiment_dir,

        # P-FLOW and delta rule
        use_p_flow=config.get('use_p_flow', False),
        p_flow_ema_decay=config.get('p_flow_ema_decay', 0.99),
        use_delta_rule_w_out=config.get('use_delta_rule_w_out', False),
        delta_rule_lr=config.get('delta_rule_lr', 0.001),
    )

    # Create publication metrics tracker for figures
    import time
    experiment_name = f"resumed_{time.strftime('%Y%m%d_%H%M%S')}"
    pub_metrics = PublicationMetrics(
        experiment_name=experiment_name,
        base_dir=experiment_dir / "publication_outputs"
    )

    # Create trainer (PublicationTrainer for metrics logging)
    print("\nInitializing trainer...")
    trainer = PublicationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        publication_metrics=pub_metrics,
    )

    # Restore training state
    print("\nRestoring training state...")
    trainer.global_step = start_step
    trainer.best_val_ce = best_val_ce

    # Restore optimizer state if available
    if 'optimizer_state_dict' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Restored optimizer state")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"  Warning: Could not restore optimizer state: {e}")
            print("  Training will continue with fresh optimizer")
    elif 'optimizer_state' in checkpoint:
        try:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("  Restored optimizer state")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"  Warning: Could not restore optimizer state: {e}")

    print(f"\nResuming from step {start_step} -> {config['max_steps']}")
    print("=" * 70)

    # Train! (PublicationTrainer handles metrics logging and figure generation)
    trainer.train()

    # Run test set evaluation (not done by PublicationTrainer)
    if test_loader is not None:
        test_metrics = run_test_evaluation(
            model=model,
            test_loader=test_loader,
            device=device,
            vocab_size=config['vocab_size'],
        )
        # Save test metrics
        test_metrics_path = experiment_dir / 'test_metrics.json'
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"Saved test metrics to: {test_metrics_path}")

    print("\n" + "=" * 70)
    print("ALL DONE!")
    print("=" * 70)


if __name__ == '__main__':
    resume_training()
