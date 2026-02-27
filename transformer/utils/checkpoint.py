"""
Checkpoint Loading Utilities
=============================

Shared utilities for loading trained model checkpoints.
Used by visualization and analysis scripts.
"""

import torch
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from transformer.core.model import GaugeTransformerLM


def save_checkpoint(
    model: GaugeTransformerLM,
    optimizer,
    config: Dict[str, Any],
    epoch: int,
    step: int,
    save_path: str,
    **kwargs
):
    """
    Save a model checkpoint.

    Args:
        model: The model to save
        optimizer: Optimizer state to save
        config: Model configuration
        epoch: Current epoch
        step: Current step
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config,
        'epoch': epoch,
        'step': step,
        **kwargs
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load a raw checkpoint dictionary.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        checkpoint: Dictionary with model_state_dict, config, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_model(checkpoint_path: str) -> Tuple[GaugeTransformerLM, Dict[str, Any]]:
    """
    Load a trained GaugeTransformerLM model from checkpoint.

    Handles both:
    - experiment_config.json (preferred)
    - Config embedded in checkpoint file (fallback)

    Args:
        checkpoint_path: Path to best_model.pt or similar checkpoint

    Returns:
        model: Loaded GaugeTransformerLM in eval mode
        config: Configuration dictionary used to create the model

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If config cannot be determined
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_dir = checkpoint_path.parent
    config_json_path = checkpoint_dir / "experiment_config.json"

    # Default config (fallback values)
    config = {
        'vocab_size': 50257,
        'embed_dim': 25,
        'n_layers': 1,
        'irrep_spec': [('ℓ0', 5, 1), ('ℓ1', 3, 3), ('ℓ2', 1, 5)],
        'hidden_dim': 112,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'use_diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }

    # Load checkpoint once (reused for config extraction and weight loading)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Try loading from experiment_config.json first (more reliable)
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        with open(config_json_path, 'r') as f:
            json_data = json.load(f)

        # Check if config is nested under a 'config' key
        if 'config' in json_data and isinstance(json_data['config'], dict):
            config.update(json_data['config'])
            print("Loaded nested config from experiment_config.json")
        else:
            config.update(json_data)
            print("Loaded config from experiment_config.json")
    else:
        # Try to extract config from checkpoint pickle
        print(f"Warning: {config_json_path} not found, trying to extract from checkpoint...")

        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                config.update(ckpt_config)
            elif hasattr(ckpt_config, '__dict__'):
                config.update(vars(ckpt_config))
            print("Extracted config from checkpoint")
        else:
            print("Warning: No config found, using defaults")

    # Handle config key translations for backward compatibility
    # Legacy checkpoint compat: old configs stored kappa_beta_base instead of kappa_beta
    if 'kappa_beta' not in config:
        config['kappa_beta'] = config.pop('kappa_beta_base', 1.0)
    # Remove defunct auto-scale keys if present
    config.pop('kappa_beta_auto_scale', None)
    config.pop('kappa_beta_base', None)
    config.pop('kappa_beta_k_ref', None)
    if 'use_diagonal_covariance' not in config and 'diagonal_covariance' in config:
        config['use_diagonal_covariance'] = config['diagonal_covariance']

    print(f"Config: K={config['embed_dim']}, vocab={config['vocab_size']}, "
          f"layers={config['n_layers']}")

    # Create model
    model = GaugeTransformerLM(config)

    # Load checkpoint weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    return model, config


def get_tokenizer(config: Dict[str, Any], dataset_name: Optional[str] = None):
    """
    Get tokenizer for a given config.

    Tries multiple tokenizer backends in order:
    1. tiktoken (GPT-2 BPE, fast and lightweight)
    2. WikiTextDataset (full dataset with tokenizer)

    Args:
        config: Model configuration dict
        dataset_name: Dataset name override (default: from config or 'wikitext-103')

    Returns:
        tokenizer: Object with encode/decode methods
    """
    if dataset_name is None:
        dataset_name = config.get('dataset', 'wikitext-103')

    vocab_size = config.get('vocab_size', 50257)

    # Try tiktoken first (faster, lighter)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print(f"Using tiktoken GPT-2 tokenizer (vocab_size={enc.n_vocab})")
        return enc
    except ImportError:
        pass

    # Fall back to WikiTextDataset
    try:
        from transformer.data.datasets import WikiTextDataset
        dataset = WikiTextDataset(
            split='train',
            max_seq_len=128,
            dataset=dataset_name,
        )
        print(f"Using WikiTextDataset tokenizer")
        return dataset
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f"Warning: Could not load dataset tokenizer: {e}")

    print("Warning: No tokenizer available. Install tiktoken: pip install tiktoken")
    return None


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load metadata from a checkpoint without loading the full model.

    Useful for inspecting checkpoints before loading.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dict with config, epoch, step, and other metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    info = {}

    # Extract config
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']

    # Extract training state
    info['epoch'] = checkpoint.get('epoch', 'unknown')
    info['step'] = checkpoint.get('step', 'unknown')

    # Check for optimizer state (indicates training checkpoint vs inference)
    info['has_optimizer'] = 'optimizer_state_dict' in checkpoint

    # Model parameter count
    if 'model_state_dict' in checkpoint:
        n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        info['n_parameters'] = n_params

    return info
