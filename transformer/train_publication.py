
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 19:24:37 2025

@author: chris and christine
"""

# =============================================================================
# PATH SETUP - Ensure the project root is in the Python path
# This allows the script to be run from any directory (including the transformer/ folder)
# =============================================================================
import sys
import os

# Get the directory containing this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of transformer/)
_project_root = os.path.dirname(_script_dir)
# Add project root to path if not already there
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

"""
Publication Proof-of-Principle Training Script
===============================================

Language modeling on WikiText-2/103 with byte-level encoding for minimal publishable claim.

Demonstrates:
1. Variational FFN works - inference comparable to learned MLP
2. Architecture is trainable - converges to reasonable performance
3. Theoretical framework is sound - gauge-invariant inference holds

  

Comprehensive Metrics Tracking:
    - Free energy components (α, β, γ terms)
    - Gradient norms (total, μ, FFN)
    - All learning rates (μ, σ, φ, FFN)
    - Bits-per-character (BPC)
    - Attention statistics (β_mean, KL_mean)
    - Performance (step time, tokens/sec)
Output Files:
    - checkpoints_publication/ffn_{mode}/metrics.csv - comprehensive training metrics
    - checkpoints_publication/ffn_{mode}/best_model.pt - best model checkpoint
    - checkpoints_publication/result_{mode}.json - final summary (if single mode)
    - checkpoints_publication/ablation_results.json - comparison (if --run_ablation)

Usage:
    # Just click Run (edit defaults below)
    python transformer/train_publication.py


Author: Designed for minimal publishable claim
Date: December 2025
"""

import torch
import torch.nn.functional as F
import argparse
import json
import csv
import time
import math
import subprocess
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any


from transformer.core.model import GaugeTransformerLM
from transformer.baselines.standard_transformer import StandardTransformerLM
from transformer.data import create_dataloaders, create_char_dataloaders
from transformer.train import (
    compute_free_energy_loss,
    compute_rg_metrics_from_attention,
    compute_dynamic_rg_metrics,
)
from transformer._archive.train_fast import FastTrainer, FastTrainingConfig
from transformer.analysis.publication_metrics import PublicationMetrics, ExperimentResult

# Import the principled PureFEPTransformer (KL-to-prior output, no backprop)
from transformer.experimental.pure_fep_transformer import (
    PureFEPTransformer,
    PureFEPConfig as PureFEPTransformerConfig,
)


# ============================================================================
# EDIT THESE DEFAULTS TO RUN WITHOUT COMMAND-LINE ARGS (just click Run!)
# ============================================================================
# Three modes available:
#   'standard'    - Standard transformer baseline (dot-product attention + MLP)
#   'VFE_dynamic' - VFE with EM-step dynamics (backprop training)
#   'pure_fep'    - Pure FEP transformer (KL-to-prior output, NO backprop!)
DEFAULT_MODE = 'VFE_dynamic'      # Which mode to run
DEFAULT_ENABLE_SIGMA_PHI = True   # Enable learning Σ and φ (full geometric learning)

# Pure FEP learning rates
DEFAULT_PRIOR_LR = 0.1            # Learning rate for prior updates in pure FEP mode
DEFAULT_EMBED_LR = 0.1            # Learning rate for embedding updates in pure FEP mode

# Dataset
DEFAULT_DATASET = 'wikitext-103'  # 'wikitext-2' (~2M tokens) or 'wikitext-103' (~103M tokens)
# ============================================================================



# =============================================================================
# CONFIG 1: STANDARD TRANSFORMER (Baseline)
# =============================================================================
# Standard dot-product attention + learned MLP for fair comparison.
# This is the BASELINE to beat!
#
# Architecture:
#   - Attention: Q·K^T / √d (standard dot-product softmax)
#   - FFN: Linear → GELU → Linear (learned MLP)
#   - Output: Linear projection to vocab
#   - Learning: Backpropagation (standard)
#   - Position: Learned positional embeddings
# =============================================================================
STANDARD_CONFIG = {
    # Model architecture
    'vocab_size': 50257,
    'embed_dim': 320,             # Increased from 256
    'n_layers': 6,
    'hidden_dim': 1280,           # 4×embed_dim (standard ratio)
    'max_seq_len': 128,
    'n_heads': 8,                 # 320/8 = 40 per head ✓


   

    # Training
    'batch_size': 16,          # 16 × 128 = 2048 tokens/step (standard for small models)
    'use_amp': False,
    'num_workers': 6,
    'epochs': None,            # Set to 1-3 for WikiText-2, None for WikiText-103 (use max_steps)
    'max_steps': 200000,       # ~0.5 epochs on WikiText-103, ~50 epochs on WikiText-2
    'warmup_steps': 2000,      # 1% of max_steps (standard practice for Adam)

    # Standard transformer settings
    'ffn_mode': 'standard',        # Learned MLP (NOT VFE)
    'attention_type': 'standard', # Dot-product attention (NOT KL)
    'tie_embeddings': True,

    # Disable gauge features (not used in standard mode)
    'evolve_sigma': False,
    'evolve_phi': False,
    'diagonal_covariance': True,

    # Learning rates (standard Adam rates)
    'mu_lr': 3e-4,              #1e-3 - 1e-4 or it wont work well
    'sigma_lr': 0.0001,
    'phi_lr': 0.0001,
    'ffn_lr': 3e-4,

    # Free energy weights (NOT USED in standard mode)
    'alpha': 0,
    'beta': 0,
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,

    # Regularization
    'weight_decay': 0.01,
    'dropout': 0.1,
    'grad_clip': 1.0,
    'phi_grad_clip': 0.1,

    # Logging
    'log_interval': 1000,
    'eval_interval': 5000,
    'checkpoint_interval': 50000,

    # Unused in standard mode
    'kappa_beta': 1.0,
    'attention_pattern': 'full',
    'attention_window': 24,
    'gauge_group': 'SO3',
    'gauge_dim': 3,
    'gauge_mode': 'learned',  # 'learned' or 'trivial' (Ω=I, standard attention limit)
    'use_multi_irrep': False,
    'gauge_fixed_priors': True,
    'irrep_spec': [('ℓ0', 5, 1)],
    'compute_rg_metrics': False,
}



# =============================================================================
# CONFIG 2: VFE_EM (VFE with EM-step dynamics, uses backprop)
# =============================================================================
# Gauge-equivariant transformer with Variational Free Energy dynamics.
# Uses EM-step belief updates with backprop for training.
#
# Architecture:
#   - Attention: KL-divergence based (gauge-equivariant)
#   - FFN: VFE EM-step dynamics (belief inference)
#   - Output: Linear projection to vocab
#   - Learning: Backpropagation
#   - Position: None (emergent from data)
# =============================================================================
SEED = 6

VFE_EM_CONFIG = {
    # Model architecture
    'vocab_size': 50257,          # Will be overridden by tokenizer
    'embed_dim': 20,              # Embedding dimension K
    'n_layers': 1,                # Transformer depth
    'hidden_dim': 508,            # Only used if ffn_mode='learned'
    'max_seq_len': 64,            # Context length N

    'learnable_alpha': False,


    # Training
    'batch_size': 32,
    'use_amp': False,             # FP32 for precision
    'num_workers': 6,
    'epochs': None,               # Set to 1-3 for WikiText-2, None for WikiText-103 (use max_steps)
    'max_steps': 50000,          # ~0.5 epochs on WikiText-103
    'warmup_steps': 100,

    # VFE transformer settings
    'ffn_mode': 'VFE_dynamic',    # VFE EM-step dynamics
    'mask_self_attention': True,  # Prevent attention collapse
    'tie_embeddings': False,

    # Gauge geometry
    'evolve_sigma': True,         # Learn covariances Σ
    'evolve_phi': True,           # Learn gauge frames φ (M-step, via backprop)
    'evolve_phi_e_step': True,    # Update φ during E-step iterations (dynamical gauge frames)
                                  # When True: φ evolves via ∂F/∂φ at each VFE iteration
                                  # When False: φ only updated via backprop (M-step)
    'diagonal_covariance': False,
    'use_identity_transport': False,

    # Temperature: κ is a scalar sharpness dial; dimension scaling (√K) is hardcoded in attention
    'kappa_beta': 2,

    # Embedding initialization
    'mu_init_std': 1.0,
    'mu_normalize': False,
    'mu_max_norm': None,
    'phi_scale': 1.0,             # Gauge frame initialization scale (try 1.0-2.0 for clustering)

    # VFE dynamics
    'ffn_n_iterations': 1,
    'ffn_learnable_lr': True,
    'ffn_chunk_size': 64,

    # Learning rates
    'mu_lr':     0.05,
    'sigma_lr':  0.005,
    'phi_lr':    0.005,
    'ffn_lr':    0.05 ,

    # Free energy weights
    'alpha':        1,                   # Self-consistency in training loss
    'beta':         0,                    # Belief alignment in training loss?
    'lambda_gamma': 0,            # Model alignment
    'kappa_gamma':  1.0,
    
    'ffn_lambda_belief': 1,
    'ffn_alpha': 1,
    
    # Regularization
    'weight_decay': 0.01,
    'dropout':      0.1,
    'grad_clip':    1.0,
    'phi_grad_clip': 0.1,

    'use_layernorm': True,      # Critical!
    'use_residual':  True,       # Gradient flow
    'use_dropout':   False,

    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 10000,
    'semantic_analysis_interval': 5000,

    # =================================================================
    # GAUGE GROUP SELECTION (Generators from so(N), Transport in GL(K))
    # =================================================================
    # NOTE: The VFE is invariant under GL(K), not just SO(K)!
    # We use so(N) generators to parameterize φ, but transport operators
    # Ω = exp(φ·G) live in GL(K). No orthogonality constraint is needed.
    #
    # SO3: so(3) generators with 3 parameters (rotation-only subalgebra)
    #      Requires embed_dim = sum(mult * dim) for irrep_spec or odd embed_dim
    # SON: so(N) generators with N(N-1)/2 parameters
    #      Supports multiple irrep types for representational diversity:
    #        - 'scalar': dim = 1              (gauge-invariant)
    #        - 'fund':   dim = N              (fundamental/vector)
    #        - 'wedge2': dim = N*(N-1)/2      (antisymmetric 2-tensor ∧²V)
    #        - 'sym2':   dim = N*(N+1)/2 - 1  (symmetric traceless Sym²₀V)
    #
    #      Different irreps have different Casimir eigenvalues:
    #        fund ~1.0x, wedge2 ~1.5x, sym2 ~2.5x
    #      This provides genuine transformation diversity (like SO(3) spin-ℓ)
    # =================================================================
    
    'gauge_group': 'GLK',  # 'SO3', 'SON', or 'GLK'
    'gauge_dim': 10,        # N for SO(N) - only used when gauge_group='SON'
    'gauge_mode': 'learned',  # 'learned': per-token φ, Ω_ij = exp(φ_i)·exp(-φ_j)
                                # 'trivial': global frame, φ = 0, Ω = I (standard attention)
    
    'use_multi_irrep': True,  # Use block-diagonal generators from irrep_spec
    'enforce_orthogonal': False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
                                 # Set False for GL(K) (faster, still gauge-invariant)

    # P-FLOW: EMA update of token embeddings toward successful beliefs
    # This is the key learning mechanism from fep_transformer.py
    'use_p_flow': False,           # Enable P-flow updates on token embeddings
    'p_flow_ema_decay': 0.95,     # EMA decay (higher = slower update, 0.99 = 1% per step)

    # DELTA RULE: Backprop-free learning for W_out
    # If True, W_out is updated via delta rule instead of backpropagation
    # Combined with P-flow, this makes learning fully backprop-free!
    'use_delta_rule_w_out': False,  # Enable delta rule for W_out (instead of backprop)
    'delta_rule_lr': 0.1,         # Learning rate for delta rule updates


    # Irrep structure for SO(N)
    # Example for SO(5) with K=132:
    #   [('scalar', 10, 1), ('fund', 8, 5), ('wedge2', 4, 10), ('sym2', 3, 14)]
    #   = 10 + 40 + 40 + 42 = 132
    'irrep_spec': [
      # ('ℓ0', 50, 1),   # 75 dimensions (scalars)
      # ('ℓ1', 1, 3),   # 90 dimensions (vectors)
      # ('ℓ2', 2, 5),   # 90 dimensions (rank-2 tensors)
     #  ('ℓ3', 1, 7),
      # ('ℓ4', 1, 9),
      #('ℓ5', 9, 11),
     # ('ℓ6', 1, 13),
     # ('ℓ7', 1, 15),
      # ('ℓ50', 1, 101),
      ('fund', 2, 10)  #For SO(8)
     # ('fund', 10, 5),   # SO(5)
       
     # SO(5) multi-irrep example:
     # ('scalar', 10, 1),   # 10 dims (invariant)
     # ('fund', 8, 5),      # 40 dims (vector)
     # ('wedge2', 4, 10),   # 40 dims (∧² - angular momentum)
     # ('sym2', 3, 14),     # 42 dims (Sym²₀ - quadrupolar)
    ],
     
         
    # Option A: couple just 0↔1, head 2 stays independent
    #'cross_couplings': [(0, 1), (1, 0)],
    # → super-blocks: [20, 10]  (heads 0,1 merged into GL(20), head 2 alone)


    # Per-head specialization & multi-head VFE
    'per_head_kappa': False,         # Learn separate κ_h per head (attention + VFE)
    'use_output_projection': True, # W_O cross-head mixing after attention (toggle)
    'multihead_vfe': False,          # Maintain per-head β_h through VFE iterations


    'use_prior_bank': True,

}


# =============================================================================
# CONFIG 3: PURE_FEP (Pure Free Energy Principle, NO backprop!)
# =============================================================================
# The most theoretically principled mode!
# Learning happens through prior evolution (P-flow), not gradients.
#
# Architecture:
#   - Attention: KL-divergence based (gauge-equivariant)
#   - FFN: VFE dynamics with CE inside (beliefs minimize prediction error)
#   - Output: -KL(q||π_v)/τ (KL-to-prior, most principled!)
#   - Learning: P-flow only (priors ← EMA of successful beliefs)
#   - Position: None (emergent from data)
#
# Key insight: Cross-entropy is INSIDE the VFE, not a separate loss!
# Priors evolve toward beliefs that minimize prediction error.
# =============================================================================
PURE_FEP_CONFIG = {
    # Model architecture (same as VFE_EM for fair comparison)
    'vocab_size': 50257,          # Will be overridden by tokenizer
    'embed_dim': 30,              # Embedding dimension K
    'n_layers': 1,                # Transformer depth
    'hidden_dim': 508,            # Not used in pure FEP
    'max_seq_len': 128,           # Context length N

    # Training
    'batch_size': 6,
    'use_amp': False,             # FP32 for precision
    'num_workers': 4,
    'epochs': None,               # Set for WikiText-2, None for WikiText-103 (use max_steps)
    'max_steps': 5000,            # For quick pure FEP experiments
    'warmup_steps': 0,            # No warmup for P-flow

    # Pure FEP transformer settings
    'ffn_mode': 'VFE_dynamic',    # VFE dynamics (but with pure_fep_mode=True)
    'mask_self_attention': True,
    'tie_embeddings': False,

    # Gauge geometry
    'evolve_sigma': True,
    'evolve_phi': True,
    'evolve_phi_e_step': False,   # Update φ during E-step iterations (dynamical gauge frames)
    'diagonal_covariance': True,

    'use_identity_transport': False,

    # Temperature: κ is a scalar sharpness dial; dimension scaling (√K) is hardcoded in attention
    'kappa_beta': 0.25,

    # Embedding initialization
    'mu_init_std': 7.0,
    'mu_normalize': False,
    'mu_max_norm': None,
    'phi_scale': 1.0,             # Gauge frame initialization scale (try 1.0-2.0 for clustering)

    # VFE dynamics (more iterations for belief convergence)
    'ffn_n_iterations': 10,       # More belief updates per step
    'ffn_learnable_lr': False,    # Fixed learning rates
    'ffn_chunk_size': 64,

    # PURE FEP: Learning rates for P-flow (scaled conservatively)
    # These are BASE rates - will be scaled in run_single_experiment
    'prior_lr': 0.1,              # Base prior learning rate
    'mu_lr': 0.05,                # Belief mean update (slower)
    'sigma_lr': 0.01,             # Belief variance (much slower)
    'phi_lr': 0.01,               # Gauge frame (slow)
    'ffn_lr': 0.01,               # Not used in pure FEP

    # Free energy weights (used inside VFE dynamics)
    'alpha': 0.1,                 # Self-consistency (lower for stability)
    'beta': 1.0,                  # Belief alignment
    'lambda_gamma': 0,
    'kappa_gamma': 1.0,
    'lambda_obs': 1.0,            # Observation term weight

    # Regularization
    'weight_decay': 0.0,          # No weight decay for P-flow
    'dropout': 0.0,               # No dropout for P-flow
    'grad_clip': 1.0,
    'phi_grad_clip': 0.1,

    # Logging
    'log_interval': 100,
    'eval_interval': 1000,
    'checkpoint_interval': 5000,

    # =================================================================
    # GAUGE GROUP SELECTION
    # =================================================================
    # SO3: Standard SO(3) gauge group with 3 generators
    #      Requires embed_dim = sum(mult * dim) for irrep_spec or odd embed_dim
    # SON: SO(N) gauge group with N(N-1)/2 generators
    #      Supports multiple irrep types for representational diversity:
    #        - 'scalar': dim = 1              (gauge-invariant)
    #        - 'fund':   dim = N              (fundamental/vector)
    #        - 'wedge2': dim = N*(N-1)/2      (antisymmetric 2-tensor ∧²V)
    #        - 'sym2':   dim = N*(N+1)/2 - 1  (symmetric traceless Sym²₀V)
    # =================================================================
    'gauge_group': 'SON',  # 'SO3' or 'SON'
    'gauge_dim': 10,        # N for SO(N) - only used when gauge_group='SON'
    'gauge_mode': 'learned',  # 'learned': per-token φ, Ω_ij = exp(φ_i)·exp(-φ_j)
                              # 'trivial': global frame, φ = 0, Ω = I (standard attention)
    'use_multi_irrep': True,  # Use block-diagonal generators from irrep_spec

    # Irrep structure for SO(N)
    # Example for SO(10) with diverse irreps:
    #   [('scalar', 5, 1), ('fund', 6, 10), ('wedge2', 2, 45)]
    #   = 5 + 60 + 90 = 155
    'irrep_spec': [
      # ('ℓ0', 50, 1),   # 75 dimensions (scalars)
      # ('ℓ1', 1, 3),   # 90 dimensions (vectors)
      # ('ℓ2', 2, 5),   # 90 dimensions (rank-2 tensors)
     #  ('ℓ3', 1, 7),
      # ('ℓ4', 1, 9),
      #('ℓ5', 9, 11),
     # ('ℓ6', 1, 13),
     # ('ℓ7', 1, 15),
      # ('ℓ50', 1, 101),
      ('fund', 3, 10)  # SO(10) fundamental
     # ('fund', 10, 5),   # SO(5)
     # SO(10) multi-irrep example:
     # ('scalar', 5, 1),    # 5 dims (invariant)
     # ('fund', 6, 10),     # 60 dims (vector)
     # ('wedge2', 2, 45),   # 90 dims (∧² - angular momentum)
    ],

    # Attention
    'attention_pattern': 'full',
    'attention_window': 24,
    'use_prior_bank': True,       # Token-dependent priors

    # PURE FEP MODE FLAGS
    'ffn_pure_fep_mode': True,    # Enable pure FEP in VFE dynamics
    'ffn_prior_lr': 0.1,          # Prior update rate

    # RG metrics
    'compute_rg_metrics': True,   # Enable RG flow analysis
    'rg_metrics_interval': 100,   # Compute every 100 steps (not too frequent)
    'rg_auto_cluster': True,
    'rg_n_clusters': None,
    'track_dynamic_rg': True,  # Track RG flow across VFE iterations (requires n_iterations > 1)
}


def get_git_info() -> Dict[str, str]:
    """Get current git commit info."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        dirty = len(status) > 0

        return {
            'commit': commit,
            'branch': branch,
            'dirty': dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': 'unknown', 'branch': 'unknown', 'dirty': False}


def get_system_info() -> Dict[str, Any]:
    """Get system/hardware information."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def run_test_evaluation(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    vocab_size: int,
    max_batches: int = 2000,
    config: dict = None,
) -> Dict[str, float]:
    """
    Run final evaluation on test set.

    Uses the same code path as validation (compute_free_energy_loss with
    forward_with_attention) to ensure consistent evaluation.

    Args:
        model: Trained model
        test_loader: Test set dataloader
        device: Device to run evaluation on
        vocab_size: Vocabulary size for random baseline comparison
        max_batches: Maximum number of batches to evaluate (default: 2000)
        config: Training config dict (for alpha/beta/lambda values).
                If None, uses pure CE evaluation.

    Returns:
        Dictionary with test metrics:
            - test_loss: Cross-entropy loss on test set
            - test_ppl: Perplexity on test set
            - test_bpc: Bits per character
            - random_ppl: Random baseline perplexity
            - improvement: Factor improvement over random
    """
    print("\n" + "="*70)
    print("FINAL TEST SET EVALUATION")
    print("="*70)

    total_batches = len(test_loader)
    eval_batches = min(max_batches, total_batches)
    print(f"  Evaluating {eval_batches} / {total_batches} batches...")

    # Use same code path as validation: compute_free_energy_loss
    # which calls model.forward_with_attention() internally.
    is_standard = isinstance(model, StandardTransformerLM)

    # Extract config values (default to 0 for pure CE if no config)
    alpha = config.get('alpha', 0) if config else 0
    beta = config.get('beta', 0) if config else 0
    lambda_gamma = config.get('lambda_gamma', 0) if config else 0
    kappa_gamma = config.get('kappa_gamma', 1.0) if config else 1.0

    model.eval()
    total_ce = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            if is_standard:
                output = model(input_ids, labels=target_ids)
                ce_loss = output['loss'].item()
            else:
                loss, metrics = compute_free_energy_loss(
                    model,
                    input_ids,
                    target_ids,
                    alpha=alpha,
                    lambda_beta=beta,
                    lambda_gamma=lambda_gamma,
                    kappa_gamma=kappa_gamma,
                )
                ce_loss = metrics['loss/ce']

            total_ce += ce_loss
            num_batches += 1

            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                print(f"  Evaluated {batch_idx + 1}/{eval_batches} batches...")

    # Compute metrics (same averaging as validation: mean of batch means)
    test_ce = total_ce / max(1, num_batches)
    test_ppl = math.exp(min(test_ce, 20))  # Clamp to prevent overflow
    test_bpc = test_ce / math.log(2)
    random_ppl = vocab_size
    improvement = random_ppl / test_ppl if test_ppl > 0 else 0

    print(f"\nTest Set Results ({num_batches} batches evaluated):")
    print(f"  Cross-entropy loss: {test_ce:.4f}")
    print(f"  Perplexity:         {test_ppl:.2f}")
    print(f"  Bits per character: {test_bpc:.3f}")
    print(f"  Random baseline:    {random_ppl:.0f}")
    print(f"  Improvement:        {improvement:.1f}x better than random")
    print("="*70 + "\n")

    model.train()

    return {
        'test_loss': test_ce,
        'test_ppl': test_ppl,
        'test_bpc': test_bpc,
        'random_ppl': random_ppl,
        'improvement': improvement,
    }


def save_experiment_config(
    config: Dict[str, Any],
    ffn_mode: str,
    checkpoint_dir: Path,
    args: argparse.Namespace = None,
) -> Path:
    """
    Save complete experiment configuration to JSON.

    Args:
        config: Model/training configuration dictionary
        ffn_mode: FFN mode being used
        checkpoint_dir: Directory to save config
        args: Command-line arguments (if available)

    Returns:
        Path to saved config file
    """
    experiment_config = {
        # Metadata
        'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'timestamp': datetime.now().isoformat(),
        'ffn_mode': ffn_mode,

        # Full model/training config
        'config': config,

        # Command-line args (if available)
        'args': vars(args) if args else None,

        # Git info for reproducibility
        'git': get_git_info(),

        # System info
        'system': get_system_info(),
    }

    # Save to checkpoint directory
    config_path = checkpoint_dir / 'experiment_config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2, default=str)

    print(f"📋 Saved experiment config: {config_path}")

    return config_path




class PublicationMetricsTracker:
    """Track ALL metrics needed for publication."""

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []

        # Create CSV with comprehensive headers
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.headers = [
            # Core
            'step', 'timestamp',

            # Losses
            'train_loss_total', 'train_loss_ce', 'train_loss_belief_align',
            'train_loss_self_consistency', 'train_loss_model_align',
            'val_loss', 'val_ce',

            # Metrics
            'train_ppl', 'train_bpc', 'val_ppl', 'val_bpc',

            # Attention stats (crucial for interpretability!)
            'beta_mean', 'beta_std', 'kl_mean', 'kl_std',
            'attention_entropy', 'attention_concentration',

            # RG Metrics (meta-agent emergence!)
            'rg_modularity', 'rg_effective_rank', 'rg_n_clusters',
            'rg_kl_within_mean', 'rg_kl_within_std',
            'rg_kl_between_mean', 'rg_kl_between_std',
            'rg_beta_entropy',

            # Dynamic RG (across VFE iterations)
            'rg_dynamic_n_iterations',
            'rg_dynamic_modularity_init', 'rg_dynamic_modularity_final', 'rg_dynamic_modularity_change',
            'rg_dynamic_rank_init', 'rg_dynamic_rank_final', 'rg_dynamic_rank_change',

            # Learning rates
            'mu_lr', 'sigma_lr', 'phi_lr', 'ffn_lr',

            # Gradient norms
            'grad_norm_total', 'grad_norm_mu', 'grad_norm_ffn',

            # Bayesian alpha diagnostics
            'alpha_mean', 'alpha_std', 'alpha_min', 'alpha_max',
            'alpha_a0', 'alpha_b0', 'alpha_mahal_sq_mean', 'alpha_mahal_sq_std',

            # Performance
            'step_time', 'tokens_per_sec',
        ]

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, step: int, metrics: Dict, lrs: Dict, grad_norms: Dict,
                 step_time: float, batch_size: int, seq_len: int):
        """Log training step with full metrics."""

        # Compute tokens/sec
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        # Bits per character (convert from nats)
        train_bpc = metrics.get('train_loss_ce', 0) / math.log(2)

        entry = {
            'step': step,
            'timestamp': time.time(),

            # Losses
            'train_loss_total': metrics.get('train_loss_total'),
            'train_loss_ce': metrics.get('train_loss_ce'),
            'train_loss_belief_align': metrics.get('train_loss_belief_align', 0),
            'train_loss_self_consistency': metrics.get('train_loss_self_consistency', 0),
            'train_loss_model_align': metrics.get('train_loss_model_align', 0),
            'val_loss': None,
            'val_ce': None,

            # Metrics
            'train_ppl': metrics.get('train_ppl'),
            'train_bpc': train_bpc,
            'val_ppl': None,
            'val_bpc': None,

            # Attention (crucial for interpretability!)
            'beta_mean': metrics.get('beta_mean'),
            'beta_std': metrics.get('beta_std'),
            'kl_mean': metrics.get('kl_mean'),
            'kl_std': metrics.get('kl_std'),
            'attention_entropy': metrics.get('attention_entropy'),
            'attention_concentration': metrics.get('attention_concentration'),

            # RG Metrics (meta-agent emergence!)
            'rg_modularity': metrics.get('rg/modularity'),
            'rg_effective_rank': metrics.get('rg/effective_rank'),
            'rg_n_clusters': metrics.get('rg/n_clusters'),
            'rg_kl_within_mean': metrics.get('rg/kl_within_mean'),
            'rg_kl_within_std': metrics.get('rg/kl_within_std'),
            'rg_kl_between_mean': metrics.get('rg/kl_between_mean'),
            'rg_kl_between_std': metrics.get('rg/kl_between_std'),
            'rg_beta_entropy': metrics.get('rg/beta_entropy'),

            # Dynamic RG (across VFE iterations)
            'rg_dynamic_n_iterations': metrics.get('rg/dynamic/n_iterations'),
            'rg_dynamic_modularity_init': metrics.get('rg/dynamic/modularity_init'),
            'rg_dynamic_modularity_final': metrics.get('rg/dynamic/modularity_final'),
            'rg_dynamic_modularity_change': metrics.get('rg/dynamic/modularity_change'),
            'rg_dynamic_rank_init': metrics.get('rg/dynamic/rank_init'),
            'rg_dynamic_rank_final': metrics.get('rg/dynamic/rank_final'),
            'rg_dynamic_rank_change': metrics.get('rg/dynamic/rank_change'),

            # Learning rates
            'mu_lr': lrs.get('mu_embed', 0),
            'sigma_lr': lrs.get('sigma_embed', 0),
            'phi_lr': lrs.get('phi_embed', 0),
            'ffn_lr': lrs.get('ffn', 0),

            # Gradients
            'grad_norm_total': grad_norms.get('total', 0) if grad_norms else 0,
            'grad_norm_mu': grad_norms.get('mu', 0) if grad_norms else 0,
            'grad_norm_ffn': grad_norms.get('ffn', 0) if grad_norms else 0,

            # Bayesian alpha diagnostics
            'alpha_mean': metrics.get('bayesian/alpha_mean'),
            'alpha_std': metrics.get('bayesian/alpha_std'),
            'alpha_min': metrics.get('bayesian/alpha_min'),
            'alpha_max': metrics.get('bayesian/alpha_max'),
            'alpha_a0': metrics.get('bayesian/a0'),
            'alpha_b0': metrics.get('bayesian/b0'),
            'alpha_mahal_sq_mean': metrics.get('bayesian/mahal_sq_mean'),
            'alpha_mahal_sq_std': metrics.get('bayesian/mahal_sq_std'),

            # Performance
            'step_time': step_time,
            'tokens_per_sec': tokens_per_sec,
        }

        self.history.append(entry)

    def log_val(self, step: int, val_metrics: Dict):
        """Update entry with validation metrics."""
        for entry in reversed(self.history):
            if entry['step'] == step:
                entry['val_loss'] = val_metrics.get('loss')
                entry['val_ce'] = val_metrics.get('ce_loss', val_metrics.get('loss'))
                entry['val_ppl'] = val_metrics.get('perplexity')
                entry['val_bpc'] = entry['val_ce'] / math.log(2) if entry['val_ce'] else None
                break

    def save(self):
        """Save to CSV."""
        if not self.history:
            return

        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()
            writer.writerows(self.history)


class PublicationTrainer(FastTrainer):
    """Enhanced trainer with publication-quality metrics."""

    def __init__(self, *args, publication_metrics: PublicationMetrics = None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Basic CSV metrics tracker
        metrics_path = self.config.checkpoint_dir / 'metrics.csv'
        self.metrics_tracker = PublicationMetricsTracker(metrics_path)
        print(f"[INFO] Logging publication metrics to: {metrics_path}")

        # Comprehensive publication metrics (optional)
        self.pub_metrics = publication_metrics
        if self.pub_metrics:
            print(f"[INFO] Comprehensive metrics enabled: {self.pub_metrics.experiment_dir}")

        # Tokenizer for decoding sequences in interpretability outputs
        self.tokenizer = tokenizer

        # Track attention visualization count
        self._attention_viz_count = 0

    def _get_head_irrep_labels(self) -> list:
        """
        Map head indices to irrep types for diagnostic labeling.

        Returns:
            List of strings like "ℓ0", "ℓ1", "ℓ2" for each head.
        """
        irrep_spec = self.config.irrep_spec
        labels = []
        for irrep_name, num_heads, dim in irrep_spec:
            for _ in range(num_heads):
                labels.append(irrep_name)
        return labels

    def save_attention_visualization(self, step: int, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Save attention pattern visualization for interpretability analysis.

        CRITICAL FIXES (based on visualization analysis):
        1. Save PER-HEAD attention (not averaged - averaging destroys patterns!)
        2. Show WHAT sequence is being visualized (token IDs + decoded text)
        3. Label each head with its irrep type (ℓ0, ℓ1, ℓ2, etc.)
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import numpy as np
        except ImportError:
            return  # Skip if matplotlib unavailable

        self.model.eval()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)

        # Get attention from forward pass
        with torch.no_grad():
            if hasattr(self.model, 'forward_with_attention'):
                _, attn_info = self.model.forward_with_attention(input_ids, targets=None)
                beta = attn_info.get('beta')
                kl = attn_info.get('kl')  # Get per-head KL divergences
                # Average over heads to get aggregate KL matrix
                kl_matrix = kl.mean(dim=1) if kl is not None else None

                if beta is not None:
                    # Get shape info
                    if beta.dim() == 4:
                        B, n_heads, N, N = beta.shape
                    else:
                        B, N, N = beta.shape
                        n_heads = 1
                        beta = beta.unsqueeze(1)  # Add head dimension

                    beta_np = beta[0].cpu().numpy()  # (n_heads, N, N)

                    # Get irrep labels for each head
                    try:
                        head_labels = self._get_head_irrep_labels()
                    except (AttributeError, KeyError):
                        # Fallback if irrep_spec not available
                        head_labels = [f"H{i}" for i in range(n_heads)]

                    # Show what sequence we're visualizing
                    seq_info = f"Step {step}, Tokens: {input_ids[0, :20].tolist()}..."

                    # Try to decode if tokenizer available
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        try:
                            decoded = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
                            seq_info += f"\nText: {decoded[:100]}..."
                        except (KeyError, IndexError, TypeError, UnicodeDecodeError):
                            pass

                    # Save directory
                    save_dir = self.config.checkpoint_dir / 'attention_patterns'
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # ============================================================
                    # SAVE INDIVIDUAL HEAD VISUALIZATIONS (NOT AVERAGED!)
                    # ============================================================
                    for head_idx in range(n_heads):
                        fig, ax = plt.subplots(figsize=(8, 6))

                        attn_head = beta_np[head_idx]  # (N, N)
                        attn_plot = attn_head.copy()
                        np.fill_diagonal(attn_plot, np.nan)  # Mask diagonal
                        attn_plot = np.log10(np.maximum(attn_plot, 1e-6))  # Log scale

                        im = ax.imshow(attn_plot, cmap='viridis', aspect='auto', vmin=-3, vmax=0)
                        ax.set_xlabel('Key Position (j)')
                        ax.set_ylabel('Query Position (i)')

                        irrep_label = head_labels[head_idx] if head_idx < len(head_labels) else f"H{head_idx}"
                        ax.set_title(f'Head {head_idx} ({irrep_label}) - {seq_info}',
                                    fontsize=10)
                        plt.colorbar(im, ax=ax, label='log₁₀(β)')

                        fig.savefig(save_dir / f'attention_step_{step:06d}_head{head_idx}.png',
                                   dpi=100, bbox_inches='tight')
                        plt.close(fig)

                    
                    # ============================================================
                    # LOG INFO
                    # ============================================================
                    self._attention_viz_count += 1
                    if self._attention_viz_count == 1:
                        print(f"\n[INFO] Attention patterns saved to: {save_dir}/")
                        print(f"  Saving per-head visualizations (NOT averaged)")

        self.model.train()

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Train step with comprehensive metrics and AMP support."""
        self.model.train()

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Check if we should compute RG metrics this step
        # NOTE: Use (step + 1) to align with eval_interval which also uses (step + 1)
        compute_rg = (
            getattr(self.config, 'compute_rg_metrics', False) and
            (self.global_step + 1) % getattr(self.config, 'rg_metrics_interval', 100) == 0
        )

        # Check if using standard transformer (no VFE loss)
        is_standard = isinstance(self.model, StandardTransformerLM)

        # Check if using delta rule for W_out (backprop-free)
        use_delta_rule = getattr(self.config, 'use_delta_rule_w_out', False) and not is_standard

        # If delta rule is enabled, exclude W_out from backprop
        if use_delta_rule and hasattr(self.model, 'out_proj'):
            self.model.out_proj.weight.requires_grad = False

        # Forward pass with full metrics (with optional AMP)
        if self.scaler is not None:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                if is_standard:
                    # Standard transformer: simple cross-entropy loss
                    output = self.model(input_ids, labels=target_ids)
                    loss = output['loss']
                    full_metrics = {
                        'loss/total': loss.item(),
                        'loss/ce': loss.item(),
                    }
                else:
                    loss, full_metrics = compute_free_energy_loss(
                        self.model,
                        input_ids,
                        target_ids,
                        alpha=self.config.alpha,
                        lambda_beta=self.config.beta,
                        lambda_gamma=self.config.lambda_gamma,
                        kappa_gamma=self.config.kappa_gamma,

                    )
            # Scaled backward
            self.scaler.scale(loss).backward()
        else:
            # Standard forward pass
            if is_standard:
                # Standard transformer: simple cross-entropy loss
                output = self.model(input_ids, labels=target_ids)
                loss = output['loss']
                full_metrics = {
                    'loss/total': loss.item(),
                    'loss/ce': loss.item(),
                }
            else:
                loss, full_metrics = compute_free_energy_loss(
                    self.model,
                    input_ids,
                    target_ids,
                    alpha=self.config.alpha,
                    lambda_beta=self.config.beta,
                    lambda_gamma=self.config.lambda_gamma,
                    kappa_gamma=self.config.kappa_gamma,

                )
            loss.backward()

        # Compute gradient norms BEFORE clipping
        # Check if this is a log step (need to check global_step here)
        is_log_step = (self.global_step + 1) % self.config.log_interval == 0
        grad_norms = self._compute_gradient_norms() if is_log_step else None

        # Clip and step (with scaler if AMP enabled)
        # Per-group clipping for large gauge groups (SO(N>3)):
        # phi_embed gradients dominate global norm, starving mu/sigma.
        # FastTrainingConfig always uses param groups; TrainingConfig has use_param_groups flag.
        #
        # phi gets a tighter clip (phi_grad_clip, default 0.1) because gauge
        # frame gradients spike 2-3 orders of magnitude above mu/sigma,
        # causing erratic effective LR when the uniform clip crushes them.
        _use_param_groups = getattr(self.config, 'use_param_groups', True)
        _phi_clip = getattr(self.config, 'phi_grad_clip', self.config.grad_clip)

        def _clip_value(group):
            """Return clip threshold: tighter for phi, default for others."""
            name = group.get('name', '')
            if 'phi' in name:
                return _phi_clip
            return self.config.grad_clip

        if self.scaler is not None:
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                if _use_param_groups:
                    for group in self.optimizer.param_groups:
                        if group['params']:
                            torch.nn.utils.clip_grad_norm_(
                                group['params'],
                                _clip_value(group),
                            )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.config.grad_clip > 0:
                if _use_param_groups:
                    for group in self.optimizer.param_groups:
                        if group['params']:
                            torch.nn.utils.clip_grad_norm_(
                                group['params'],
                                _clip_value(group),
                            )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        # Re-enable requires_grad for W_out if it was disabled
        if use_delta_rule and hasattr(self.model, 'out_proj'):
            self.model.out_proj.weight.requires_grad = True

        # =================================================================
        # P-FLOW: EMA update of token embeddings toward successful beliefs
        # =================================================================
        # This is the key learning mechanism from fep_transformer.py
        # After backprop updates W_out, P-flow updates token embeddings
        # toward beliefs that predicted successfully (low CE)
        use_p_flow = getattr(self.config, 'use_p_flow', False)
        if use_p_flow and not is_standard and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            ce_per_position = full_metrics['p_flow/ce_per_position']
            ema_decay = getattr(self.config, 'p_flow_ema_decay', 0.99)

            # Call P-flow update on the model
            if hasattr(self.model, 'p_flow_update'):
                self.model.p_flow_update(
                    token_ids=input_ids,
                    mu_beliefs=mu_beliefs,
                    prediction_errors=ce_per_position,
                    ema_decay=ema_decay,
                )

        # =================================================================
        # DELTA RULE: Backprop-free update of W_out
        # =================================================================
        # Uses local learning rule: ΔW = η · (target - prediction) ⊗ μ^T
        # Combined with P-flow, this makes learning fully backprop-free!
        if use_delta_rule and 'p_flow/mu_q' in full_metrics:
            mu_beliefs = full_metrics['p_flow/mu_q']
            delta_lr = getattr(self.config, 'delta_rule_lr', 0.001)

            # Call delta rule update on the model
            if hasattr(self.model, 'delta_rule_update_w_out'):
                self.model.delta_rule_update_w_out(
                    mu_beliefs=mu_beliefs,
                    targets=target_ids,
                    lr=delta_lr,
                )

        # Format comprehensive metrics
        metrics = {
            'train_loss_total': full_metrics['loss/total'],
            'train_loss_ce': full_metrics['loss/ce'],
            'train_loss_belief_align': full_metrics.get('loss/belief_align', 0),
            'train_loss_self_consistency': full_metrics.get('loss/self_consistency', 0),
            'train_loss_model_align': full_metrics.get('loss/model_align', 0),
            'train_ppl': math.exp(min(full_metrics['loss/ce'], 20)),  # Clamp to prevent overflow
            'beta_mean': full_metrics.get('attention/beta_mean', 0),
            'beta_std': 0,  # Could compute if needed
            'kl_mean': full_metrics.get('attention/kl_mean', 0),
            'kl_std': 0,
            # Crucial attention interpretability metrics
            'attention_entropy': full_metrics.get('attention/entropy', 0),
            'attention_concentration': full_metrics.get('attention/concentration', 0),
        }

        # Carry over Bayesian alpha diagnostics
        for key in ['bayesian/alpha_mean', 'bayesian/alpha_std', 'bayesian/alpha_min',
                     'bayesian/alpha_max', 'bayesian/a0', 'bayesian/b0',
                     'bayesian/mahal_sq_mean', 'bayesian/mahal_sq_std']:
            if key in full_metrics:
                metrics[key] = full_metrics[key]

        # Compute RG metrics if enabled and attention info was returned
        if compute_rg and 'attention_info' in full_metrics:
            rg_metrics = compute_rg_metrics_from_attention(
                attn_info=full_metrics['attention_info'],
                step=self.global_step,
                auto_cluster=getattr(self.config, 'rg_auto_cluster', True),
                n_clusters=getattr(self.config, 'rg_n_clusters', None),
            )
            # Add RG metrics with proper key mapping for CSV
            metrics['rg/modularity'] = rg_metrics.get('rg/modularity')
            metrics['rg/effective_rank'] = rg_metrics.get('rg/effective_rank')
            metrics['rg/n_clusters'] = rg_metrics.get('rg/n_clusters')
            metrics['rg/kl_within_mean'] = rg_metrics.get('rg/kl_within_mean')
            metrics['rg/kl_within_std'] = rg_metrics.get('rg/kl_within_std')
            metrics['rg/kl_between_mean'] = rg_metrics.get('rg/kl_between_mean')
            metrics['rg/kl_between_std'] = rg_metrics.get('rg/kl_between_std')
            metrics['rg/beta_entropy'] = rg_metrics.get('rg/beta_entropy')

            # Dynamic RG tracking (across VFE iterations within forward pass)
            track_dynamic = getattr(self.config, 'track_dynamic_rg', False)
            if track_dynamic and hasattr(self.model, 'forward_with_rg_tracking'):
                try:
                    # Run a separate forward pass with RG tracking
                    # This captures beta_history across VFE iterations
                    with torch.no_grad():
                        _, rg_info = self.model.forward_with_rg_tracking(
                            token_ids=input_ids,
                            targets=target_ids,
                        )
                    dynamic_metrics = compute_dynamic_rg_metrics(rg_info, self.global_step)

                    # Add dynamic RG metrics
                    for key, value in dynamic_metrics.items():
                        metrics[key] = value
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    # Don't crash training on RG tracking errors
                    print(f"[WARNING] Dynamic RG tracking failed: {e}")

        return metrics, grad_norms

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        norms = {'total': 0, 'mu': 0, 'sigma': 0, 'phi': 0, 'ffn': 0}

        total_norm = 0
        mu_norm = 0
        sigma_norm = 0
        phi_norm = 0
        ffn_norm = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                if 'mu_embed' in name or 'mu' in name.lower():
                    mu_norm += param_norm ** 2
                elif 'sigma_embed' in name or 'sigma' in name.lower() or 'L_embed' in name:
                    sigma_norm += param_norm ** 2
                elif 'phi_embed' in name or 'phi' in name.lower():
                    phi_norm += param_norm ** 2
                elif 'ffn' in name:
                    ffn_norm += param_norm ** 2

        norms['total'] = math.sqrt(total_norm)
        norms['mu'] = math.sqrt(mu_norm)
        norms['sigma'] = math.sqrt(sigma_norm)
        norms['phi'] = math.sqrt(phi_norm)
        norms['ffn'] = math.sqrt(ffn_norm)

        return norms


    def sample_text(
        self,
        prompt: str = "The",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        """
        Generate text to verify the model is learning.

        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling

        Returns:
            Generated text string
        """
        self.model.eval()

        # Get dataset which has encode/decode methods
        dataset = self.train_loader.dataset

        # Encode prompt using dataset's method
        prompt_ids = dataset.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)

        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode using dataset's method
        generated_text = dataset.decode(generated[0])

        self.model.train()
        return generated_text

    def train(self):
        """Training loop with publication metrics."""
        print(f"{'='*70}")
        print("PUBLICATION-QUALITY TRAINING")
        print(f"{'='*70}\n")

        # Support resuming from a checkpoint
        start_step = self.global_step
        if start_step > 0:
            print(f"  Resuming from step {start_step}")

        print(f"  Training for {self.config.max_steps:,} steps (~{self.config.max_steps / len(self.train_loader):.1f} epochs)")

        start_time = time.time()
        train_iterator = iter(self.train_loader)

        # Calculate total steps: epochs takes precedence over max_steps
        epochs = getattr(self.config, 'epochs', None)
        if epochs is not None and epochs > 0:
            steps_per_epoch = len(self.train_loader)
            total_steps = epochs * steps_per_epoch
            print(f"  Training for {epochs} epoch(s) ({steps_per_epoch} steps/epoch = {total_steps:,} total steps)")
        else:
            total_steps = self.config.max_steps
            steps_per_epoch = len(self.train_loader)
            equiv_epochs = total_steps / steps_per_epoch if steps_per_epoch > 0 else 0
            print(f"  Training for {total_steps:,} steps (~{equiv_epochs:.1f} epochs)")

        try:
            from tqdm import tqdm
            pbar = tqdm(
                range(start_step, total_steps),
                desc="Training",
                initial=start_step,
                total=total_steps
            )
            use_tqdm = True
        except ImportError:
            pbar = range(start_step, total_steps)
            use_tqdm = False

        # Run initial gauge frame semantic analysis (only if starting fresh)
        if start_step == 0 and self.pub_metrics:
            try:
                print("[Semantic] Running initial analysis (step 0)...")
                self.pub_metrics.run_semantic_analysis(
                    model=self.model,
                    step=0,
                    verbose=True,
                )
            except (ValueError, RuntimeError, TypeError, OSError) as e:
                print(f"[WARN] Initial semantic analysis failed: {e}")

        for step in pbar:
            self.global_step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            # Train step with full metrics (grad_norms computed inside before zero_grad)
            metrics, grad_norms = self.train_step(batch)

            step_time = time.time() - step_start

            is_log_step = (step + 1) % self.config.log_interval == 0
            has_rg = metrics.get('rg/modularity') is not None

            # Get learning rates
            lrs = {group['name']: group['lr'] for group in self.optimizer.param_groups}

            # Log to basic tracker and console at log intervals OR when RG metrics were computed
            if is_log_step or has_rg:
                batch_size = batch[0].shape[0]
                seq_len = batch[0].shape[1]
                self.metrics_tracker.log_step(
                    step + 1, metrics, lrs, grad_norms, step_time, batch_size, seq_len
                )

                # Log to comprehensive publication metrics (if enabled)
                if self.pub_metrics:
                    self.pub_metrics.record_training_step(
                        step=step + 1,
                        epoch=(step + 1) / len(self.train_loader),
                        train_metrics={
                            'loss': metrics['train_loss_total'],
                            'ce_loss': metrics['train_loss_ce'],
                            'attention_entropy': metrics.get('attention_entropy', 0),
                            'attention_concentration': metrics.get('attention_concentration', 0),
                        },
                        grad_norms=grad_norms,
                        lrs=lrs,
                        step_time=step_time,
                        batch_size=batch_size,
                        seq_len=seq_len,
                    )

                # Console logging
                log_msg = (
                    f"Step {step+1}/{total_steps} | "
                    f"Loss: {metrics['train_loss_total']:.4f} | "
                    f"CE: {metrics['train_loss_ce']:.4f} | "
                    f"KL: {metrics['kl_mean']:.4f} | "
                    f"PPL: {metrics['train_ppl']:.1f}"
                )

                # RG metrics console output
                _rg_msg = None
                if has_rg:
                    _rg_msg = (
                        f"  [RG] Q={metrics['rg/modularity']:.4f} | "
                        f"rank={metrics['rg/effective_rank']:.1f} | "
                        f"clusters={metrics['rg/n_clusters']} | "
                        f"H={metrics['rg/beta_entropy']:.3f}"
                    )
                    if metrics.get('rg/dynamic/n_iterations') is not None and metrics['rg/dynamic/n_iterations'] > 1:
                        _rg_msg += (
                            f" | dyn({metrics['rg/dynamic/n_iterations']}it): "
                            f"Q {metrics.get('rg/dynamic/modularity_init', 0):.3f}->{metrics.get('rg/dynamic/modularity_final', 0):.3f}"
                        )

                if use_tqdm:
                    pbar.set_description(log_msg)
                    # Print gradient norms using tqdm.write for proper display
                    if grad_norms:
                        tqdm.write(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                                   f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                                   f"phi: {grad_norms['phi']:.3e}")
                    # Print Bayesian alpha diagnostics
                    if metrics.get('bayesian/alpha_mean') is not None:
                        tqdm.write(f"  [ALPHA] mean: {metrics['bayesian/alpha_mean']:.4f} | "
                                   f"std: {metrics['bayesian/alpha_std']:.4f} | "
                                   f"range: [{metrics['bayesian/alpha_min']:.4f}, {metrics['bayesian/alpha_max']:.4f}] | "
                                   f"a0: {metrics['bayesian/a0']:.4f} | b0: {metrics['bayesian/b0']:.4f} | "
                                   f"mahal: {metrics['bayesian/mahal_sq_mean']:.4f}")
                    if _rg_msg:
                        tqdm.write(_rg_msg)
                else:
                    print(log_msg)
                    if grad_norms:
                        print(f"  [GRAD] total: {grad_norms['total']:.3e} | "
                              f"mu: {grad_norms['mu']:.3e} | sigma: {grad_norms['sigma']:.3e} | "
                              f"phi: {grad_norms['phi']:.3e}")
                    if metrics.get('bayesian/alpha_mean') is not None:
                        print(f"  [ALPHA] mean: {metrics['bayesian/alpha_mean']:.4f} | "
                              f"std: {metrics['bayesian/alpha_std']:.4f} | "
                              f"range: [{metrics['bayesian/alpha_min']:.4f}, {metrics['bayesian/alpha_max']:.4f}] | "
                              f"a0: {metrics['bayesian/a0']:.4f} | b0: {metrics['bayesian/b0']:.4f} | "
                              f"mahal: {metrics['bayesian/mahal_sq_mean']:.4f}")
                    if _rg_msg:
                        print(_rg_msg)

            # Validation
            if (step + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.metrics_tracker.log_val(step + 1, val_metrics)

                # Log to comprehensive metrics
                if self.pub_metrics:
                    self.pub_metrics.record_validation(step + 1, val_metrics)

                # Log attention entropy/concentration for interpretability
                attn_entropy = metrics.get('attention_entropy', 0)
                attn_concentration = metrics.get('attention_concentration', 0)

                print(f"\n  Validation @ step {step+1}:")
                print(f"    Loss: {val_metrics['loss']:.4f}")
                print(f"    CE: {val_metrics['ce_loss']:.4f}")
                print(f"    PPL: {val_metrics['perplexity']:.2f}")
                print(f"    BPC: {val_metrics['ce_loss']/math.log(2):.3f}")
                print(f"    Attn entropy: {attn_entropy:.3f} | concentration: {attn_concentration:.3f}")

                # Log RG metrics if available (meta-agent emergence!)
                if metrics.get('rg/modularity') is not None:
                    print(f"    RG Metrics (meta-agent emergence):")
                    print(f"      Modularity Q: {metrics['rg/modularity']:.4f} (higher = more structure)")
                    print(f"      Effective rank: {metrics['rg/effective_rank']:.2f} (lower = concentrated)")
                    print(f"      Clusters (meta-agents): {metrics['rg/n_clusters']}")
                    print(f"      KL within: {metrics['rg/kl_within_mean']:.4f} (lower = tighter)")
                    print(f"      KL between: {metrics['rg/kl_between_mean']:.4f}")

                    # Dynamic RG flow (within forward pass)
                    if metrics.get('rg/dynamic/n_iterations') is not None:
                        n_iters = metrics['rg/dynamic/n_iterations']
                        if n_iters > 1:
                            mod_change = metrics.get('rg/dynamic/modularity_change', 0)
                            rank_change = metrics.get('rg/dynamic/rank_change', 0)
                            print(f"    Dynamic RG ({n_iters} VFE iterations):")
                            print(f"      Modularity: {metrics.get('rg/dynamic/modularity_init', 0):.4f} → {metrics.get('rg/dynamic/modularity_final', 0):.4f} (Δ={mod_change:+.4f})")
                            print(f"      Eff. Rank:  {metrics.get('rg/dynamic/rank_init', 0):.1f} → {metrics.get('rg/dynamic/rank_final', 0):.1f} (Δ={rank_change:+.1f})")

                # Generate sample text to verify learning (varied prompts for diversity)
                try:
                    import random
                    prompts = ["The", "In", "A", "It", "This", "As", "One", "When", "For",
                               "After", "Before", "During", "While", "Although", "However"]
                    prompt = random.choice(prompts)
                    # Use temperature 0.9 and lower top_k for more diversity
                    sample = self.sample_text(prompt=prompt, max_new_tokens=30, temperature=0.9, top_k=30)
                    print(f"    Sample: {sample[:100]}...")
                except (RuntimeError, ValueError, IndexError) as e:
                    import traceback
                    print(f"    Sample generation failed: {e}")
                    traceback.print_exc()
                print()

                # Save attention visualization periodically
                try:
                    sample_batch = next(iter(self.val_loader))
                    self.save_attention_visualization(step + 1, sample_batch)
                except StopIteration:
                    pass

                # Save best model based on CE loss (not total loss)
                # CE loss is the proper metric since PPL = exp(CE)
                if val_metrics['ce_loss'] < self.best_val_ce:
                    self.best_val_ce = val_metrics['ce_loss']
                    self.save_checkpoint(is_best=True)

            # Checkpointing
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(is_best=False)
                self.metrics_tracker.save()

            # Periodic gauge frame semantic analysis
            if self.pub_metrics and self.pub_metrics.should_run_semantic_analysis(step + 1):
                try:
                    self.pub_metrics.run_semantic_analysis(
                        model=self.model,
                        step=step + 1,
                        verbose=False,  # Minimal output during training
                    )
                except (ValueError, RuntimeError, TypeError, OSError) as e:
                    print(f"[WARN] Semantic analysis failed at step {step+1}: {e}")

        # Save final metrics
        self.metrics_tracker.save()
        print(f"\n[INFO] Final metrics saved to: {self.metrics_tracker.save_path}")

        # Save comprehensive publication metrics
        if self.pub_metrics:
            self.pub_metrics.save_all()
            self.pub_metrics.generate_all_figures()

            # Run final gauge frame semantic analysis
            try:
                self.pub_metrics.run_final_semantic_analysis(
                    model=self.model,
                    verbose=True,
                )
            except (ValueError, RuntimeError, TypeError, OSError) as e:
                print(f"[WARN] Final semantic analysis failed: {e}")

            # Generate interpretability outputs using a sample batch from validation
            try:
                sample_batch = next(iter(self.val_loader))
                self.pub_metrics.generate_interpretability_outputs(
                    model=self.model,
                    sample_batch=sample_batch,
                    tokenizer=self.tokenizer,  # Dataset with .decode() method
                    device=self.device,
                )
            except (ValueError, RuntimeError, TypeError, OSError) as e:
                import traceback
                print(f"[WARNING] Could not generate interpretability outputs: {e}")
                print(f"  Traceback: {traceback.format_exc()}")

            self.pub_metrics.print_summary()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Time: {elapsed/3600:.2f} hours")
        print(f"Best val CE: {self.best_val_ce:.4f} (PPL: {math.exp(self.best_val_ce):.2f})")
        print(f"{'='*70}\n")


def run_single_experiment(
    config: dict,
    ffn_mode: str,
    device: torch.device,
    checkpoint_dir: Path,
    use_wandb: bool = False,
    args: argparse.Namespace = None,
    enable_publication_metrics: bool = True,
    pure_fep: bool = False,
    prior_lr: float = 0.01,
) -> Dict:
    """
    Run a single training experiment.

    Args:
        config: Configuration dictionary
        ffn_mode: FFN mode ('VFE_dynamic')
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use Weights & Biases logging
        args: Command-line arguments for logging
        enable_publication_metrics: Whether to enable comprehensive publication metrics
        pure_fep: If True, use backprop-free learning via prior evolution
        prior_lr: Learning rate for prior updates in pure FEP mode

    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    if pure_fep:
        print(f"EXPERIMENT: PURE FEP MODE (Backprop-Free)")
    else:
        print(f"EXPERIMENT: FFN_MODE = {ffn_mode}")
    print("="*70)

    # Override FFN mode in config
    config = config.copy()
    config['ffn_mode'] = ffn_mode

    # Create experiment-specific checkpoint directory
    exp_checkpoint_dir = checkpoint_dir / f"ffn_{ffn_mode}"
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration at the START
    save_experiment_config(config, ffn_mode, exp_checkpoint_dir, args)

    # =================================================================
    # Data Loading (BPE tokenization using GPT-2 tokenizer)
    # =================================================================

    dataset_name = config.get('dataset', 'wikitext-2')
    print("\n" + "="*70)
    print(f"LOADING {dataset_name.upper()} DATA")
    print("="*70)

    # Tokenizer selection: 'char', 'bpe', or 'auto' (default)
    # 'auto' uses char for vocab_size <= 256, bpe otherwise
    tokenizer_mode = config.get('tokenizer', 'auto')
    if tokenizer_mode == 'auto':
        use_char = config['vocab_size'] <= 256
    else:
        use_char = (tokenizer_mode == 'char')

    test_loader = None  # Will be set if available
    if use_char:
        print(f"Using CHARACTER-LEVEL tokenizer (vocab_size={config['vocab_size']})")
        # Note: create_char_dataloaders doesn't support test set yet
        train_loader, val_loader, actual_vocab_size = create_char_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
        )
        tokenizer = None  # Character-level doesn't need tokenizer for decode
    else:
        print(f"Using BPE tokenizer (vocab_size={config['vocab_size']})")
        train_loader, val_loader, test_loader, actual_vocab_size, tokenizer = create_dataloaders(
            max_seq_len=config['max_seq_len'],
            batch_size=config['batch_size'],
            vocab_size=config['vocab_size'],  # Top K BPE tokens
            num_workers=config.get('num_workers', 0),
            dataset=dataset_name,
            include_test=True,  # Include test set for final evaluation
            return_tokenizer=True,  # Get tokenizer for interpretability outputs
        )

    config['vocab_size'] = actual_vocab_size

    # =================================================================
    # Model Creation - Three distinct modes
    # =================================================================

    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    print(f"  N (seq len): {config['max_seq_len']}")
    print(f"  K (embed): {config['embed_dim']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Vocab: {actual_vocab_size} ({'char' if use_char else 'BPE'})")

    # =====================================================================
    # MODE 1: STANDARD TRANSFORMER (baseline)
    # =====================================================================
    if ffn_mode == 'standard':
        print("  Model type: STANDARD TRANSFORMER (dot-product attention)")
        print("  - Attention: Q·K softmax")
        print("  - FFN: Learned MLP (GELU)")
        print("  - Output: Linear projection")
        print("  - Learning: Backprop")
        model_config = {
            'vocab_size': actual_vocab_size,
            'embed_dim': config['embed_dim'],
            'n_layers': config['n_layers'],
            'n_heads': config.get('n_heads', 1),
            'hidden_dim': config.get('hidden_dim', config['embed_dim'] * 4),
            'max_seq_len': config['max_seq_len'],
            'dropout': config.get('dropout', 0.1),
        }
        model = StandardTransformerLM(model_config)

    # =====================================================================
    # MODE 2: PURE FEP TRANSFORMER (most principled)
    # =====================================================================
    elif pure_fep:
        print("  Model type: PURE FEP TRANSFORMER (KL-to-prior output)")
        print("  - Attention: KL-divergence based")
        print("  - FFN: VFE dynamics (CE inside)")
        print("  - Output: -KL(q||π_v)/τ (principled!)")
        print("  - Learning: P-flow only (NO backprop)")
        print("  - Position: None (emergent from data)")

        # Create PureFEPTransformer config with CONSERVATIVE learning rates
        pure_fep_model_config = PureFEPTransformerConfig(
            # Architecture
            vocab_size=actual_vocab_size,
            embed_dim=config['embed_dim'],
            num_layers=config['n_layers'],
            seq_length=config['max_seq_len'],

            # Gauge group
            gauge_group=config.get('gauge_group', 'SON'),
            gauge_dim=config.get('gauge_dim', 10),
            irrep_spec=config.get('irrep_spec'),

            # VFE parameters
            alpha=config.get('alpha', 0.1),
            lambda_belief=config.get('beta', 1.0),
            lambda_obs=config.get('lambda_obs', 1.0),
            kappa=config.get('kappa_beta', 0.1),

            # Learning rates - less conservative to allow actual learning
            # The error-scaled update already provides stability
            mu_lr=prior_lr * 0.5,       # Belief mean update
            sigma_lr=prior_lr * 0.1,    # Belief variance update
            prior_lr=prior_lr,          # Prior update (NO 0.1x scaling - let error scaling handle it)
            phi_lr=prior_lr * 0.1,      # Gauge frame update

            # VFE dynamics
            belief_steps=config.get('ffn_n_iterations', 10),
            prior_update_interval=1,

            # Stability
            grad_clip=config.get('grad_clip', 1.0),

            # PURE FEP MODE
            pure_fep_mode=True,

            # Principled settings
            embedding_mode='prior_bank',   # Unified PriorBank
            output_mode='kl_to_prior',     # KL-based output (principled!)
            position_mode='none',          # No position encoding (emergent)
        )

        print(f"\n  Learning rates (conservative to prevent NaN):")
        print(f"    mu_lr:    {pure_fep_model_config.mu_lr:.4f}")
        print(f"    sigma_lr: {pure_fep_model_config.sigma_lr:.4f}")
        print(f"    prior_lr: {pure_fep_model_config.prior_lr:.4f}")
        print(f"    phi_lr:   {pure_fep_model_config.phi_lr:.4f}")

        model = PureFEPTransformer(pure_fep_model_config)

        # Verify gauge_fixed_priors setting
        if model.prior_bank is not None:
            print(f"    gauge_fixed_priors: {model.prior_bank.gauge_fixed_priors}")
        print(f"    embedding_mode: {pure_fep_model_config.embedding_mode}")
        print(f"    output_mode: {pure_fep_model_config.output_mode}")

    # =====================================================================
    # MODE 3: VFE_DYNAMIC TRANSFORMER (EM-step, uses backprop)
    # =====================================================================
    else:
        print("  Model type: GAUGE VFE TRANSFORMER (KL-divergence attention)")
        print("  - Attention: KL-divergence based")
        print("  - FFN: VFE EM-step dynamics")
        print("  - Output: Linear projection")
        print("  - Learning: Backprop")
        print("  - Position: None (emergent)")

        # kappa_beta: scalar sharpness dial (dimension scaling τ=√K is hardcoded in attention)
        if 'kappa_beta' not in config:
            config['kappa_beta'] = 1.0
        print(f"  kappa_beta: {config['kappa_beta']}")

        model = GaugeTransformerLM(config)

    model = model.to(device)

    # Get parameter counts
    if hasattr(model, 'get_num_params'):
        total_params = model.get_num_params(non_embedding=False)
        non_embed_params = model.get_num_params(non_embedding=True)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        non_embed_params = sum(p.numel() for p in model.parameters() if 'embed' not in str(p))

    print(f"\nModel Parameters:")
    print(f"  Total:         {total_params:,}")
    print(f"  Non-embedding: {non_embed_params:,}")
    print(f"  Embedding:     {total_params - non_embed_params:,}")

    # =================================================================
    # Training Configuration
    # =================================================================

    train_config = FastTrainingConfig(
        epochs=config.get('epochs', None),
        max_steps=config['max_steps'],
        warmup_steps=config['warmup_steps'],

        # Learning rates
        # For standard transformer: attention_lr should match ffn_lr (all standard Adam)
        # For gauge transformer: attention_lr matches phi_lr (natural gradient scale)
        mu_lr=config['mu_lr'],
        sigma_lr=config['sigma_lr'],
        phi_lr=config['phi_lr'],
        attention_lr=config.get('attention_lr', config['ffn_lr'] if ffn_mode == 'standard' else config['phi_lr']),
        ffn_lr=config['ffn_lr'],
        output_lr=config['ffn_lr'],

        weight_decay=config['weight_decay'],
        grad_clip=config['grad_clip'],
        phi_grad_clip=config.get('phi_grad_clip', 0.1),

        alpha=config['alpha'],
        beta=config['beta'],
        lambda_gamma=config['lambda_gamma'],

        log_interval=config['log_interval'],
        eval_interval=config['eval_interval'],
        checkpoint_interval=config['checkpoint_interval'],

        use_wandb=use_wandb,
        checkpoint_dir=exp_checkpoint_dir,

        # GPU optimizations
        use_amp=config.get('use_amp', False),

        # P-FLOW: EMA update of token embeddings toward successful beliefs
        use_p_flow=config.get('use_p_flow', False),
        p_flow_ema_decay=config.get('p_flow_ema_decay', 0.99),

        # DELTA RULE: Backprop-free learning for W_out
        use_delta_rule_w_out=config.get('use_delta_rule_w_out', False),
        delta_rule_lr=config.get('delta_rule_lr', 0.001),

        # RG METRICS: Track renormalization group flow
        compute_rg_metrics=config.get('compute_rg_metrics', False),
        rg_metrics_interval=config.get('rg_metrics_interval', 100),
        rg_auto_cluster=config.get('rg_auto_cluster', True),
        rg_n_clusters=config.get('rg_n_clusters', None),
        track_dynamic_rg=config.get('track_dynamic_rg', False),
    )

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    # Calculate training duration metrics
    steps_per_epoch = len(train_loader)
    batch_size = config['batch_size']
    seq_len = config['max_seq_len']
    tokens_per_step = batch_size * seq_len

    # Get dataset size for coverage calculation
    try:
        dataset_tokens = len(train_loader.dataset.tokens)
    except AttributeError:
        dataset_tokens = None

    if train_config.epochs is not None and train_config.epochs > 0:
        effective_steps = train_config.epochs * steps_per_epoch
        total_tokens = effective_steps * tokens_per_step
        print(f"  Epochs:         {train_config.epochs}")
        print(f"  Steps/epoch:    {steps_per_epoch:,}")
        print(f"  Total steps:    {effective_steps:,}")
        print(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            print(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    else:
        equiv_epochs = train_config.max_steps / steps_per_epoch
        total_tokens = train_config.max_steps * tokens_per_step
        print(f"  Max steps:      {train_config.max_steps:,}")
        print(f"  Steps/epoch:    {steps_per_epoch:,}")
        print(f"  *** EPOCHS:     {equiv_epochs:.4f} ***")
        print(f"  Tokens seen:    {total_tokens:,} ({total_tokens/1e6:.1f}M)")
        if dataset_tokens:
            coverage = total_tokens / dataset_tokens * 100
            print(f"  Dataset:        {dataset_tokens:,} ({dataset_tokens/1e6:.1f}M) - {coverage:.1f}% coverage")
    print(f"  Warmup:         {train_config.warmup_steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Seq length:     {seq_len}")
    print(f"  Use AMP:        {train_config.use_amp}")
    print(f"  Num workers:    {config.get('num_workers', 0)}")
    print(f"\nFree Energy Weights:")
    print(f"  α (self-consistency): {train_config.alpha}")
    print(f"  β (belief align):     {train_config.beta}")
    print(f"  γ (model align):      {train_config.lambda_gamma}")

    # P-FLOW configuration
    if train_config.use_p_flow:
        print(f"\nP-FLOW (EMA prior updates): ENABLED")
        print(f"  EMA decay: {train_config.p_flow_ema_decay} ({(1-train_config.p_flow_ema_decay)*100:.1f}% update per step)")
    else:
        print(f"\nP-FLOW: disabled")

    # DELTA RULE configuration
    if train_config.use_delta_rule_w_out:
        print(f"\nDELTA RULE (backprop-free W_out): ENABLED")
        print(f"  Learning rate: {train_config.delta_rule_lr}")
        if train_config.use_p_flow:
            print(f"  ** FULLY BACKPROP-FREE MODE **")
    else:
        print(f"\nDELTA RULE: disabled (using backprop for W_out)")

    # RG METRICS configuration
    if train_config.compute_rg_metrics:
        print(f"\nRG METRICS (meta-agent emergence): ENABLED")
        print(f"  Compute interval: every {train_config.rg_metrics_interval} steps")
        print(f"  Dynamic RG tracking: {train_config.track_dynamic_rg}")
    else:
        print(f"\nRG METRICS: disabled")

    # =================================================================
    # Create Trainer (Pure FEP or Standard)
    # =================================================================

    print("\n" + "="*70)
    print("INITIALIZING TRAINER")
    print("="*70)

    if pure_fep:
        # =========================================================
        # PURE FEP MODE: Using PureFEPTransformer with train_step()
        # =========================================================
        print("Mode: PURE FEP TRANSFORMER (Principled, No Backprop)")
        print(f"  Output: KL-to-prior (most principled)")
        print(f"  Position: None (emergent)")

        print("\n" + "="*70)
        print("STARTING PURE FEP TRAINING")
        print("="*70)
        print(f"Device: {device}")

        # Calculate total steps: epochs takes precedence
        epochs = config.get('epochs', None)
        steps_per_epoch = len(train_loader)
        if epochs is not None and epochs > 0:
            total_steps = epochs * steps_per_epoch
            print(f"Epochs: {epochs} ({steps_per_epoch:,} steps/epoch = {total_steps:,} total)")
        else:
            total_steps = config['max_steps']
            print(f"Total steps: {total_steps:,} (~{total_steps/steps_per_epoch:.1f} epochs)")

        print("\nLearning via P-flow: beliefs → priors (no backprop)")
        print("="*70 + "\n")

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        train_iterator = iter(train_loader)
        best_val_ppl = float('inf')
        log_interval = config['log_interval']
        eval_interval = config['eval_interval']

        pbar = tqdm(range(total_steps), desc="Training") if use_tqdm else range(total_steps)

        try:
            for step in pbar:
                # Get batch
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                input_ids, targets = batch
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                # Train step using PureFEPTransformer's built-in method
                metrics = model.train_step(input_ids, targets)

                # Check for NaN
                if math.isnan(metrics['ce_loss']):
                    print(f"\n❌ NaN detected at step {step}! Try lower learning rates.")
                    break

                # Logging
                if (step + 1) % log_interval == 0:
                    ppl = metrics['perplexity']
                    ce = metrics['ce_loss']
                    msg = f"Step {step+1:5d} | CE: {ce:.4f} | PPL: {ppl:.2f}"
                    if use_tqdm:
                        pbar.set_postfix_str(f"ppl={ppl:.2f}")
                        tqdm.write(msg)
                    else:
                        print(msg)

                # Validation (limit batches to avoid long waits)
                if (step + 1) % eval_interval == 0:
                    model.eval()
                    val_loss = 0.0
                    val_tokens = 0
                    max_val_batches = 50  # Limit validation to 50 batches
                    with torch.no_grad():
                        for batch_idx, val_batch in enumerate(val_loader):
                            if batch_idx >= max_val_batches:
                                break
                            v_input, v_targets = val_batch
                            v_input = v_input.to(device)
                            v_targets = v_targets.to(device)
                            logits, _ = model(v_input)
                            loss = F.cross_entropy(
                                logits.view(-1, actual_vocab_size),
                                v_targets.view(-1),
                                reduction='sum'
                            )
                            val_loss += loss.item()
                            val_tokens += v_targets.numel()

                    val_ce = val_loss / val_tokens if val_tokens > 0 else float('inf')
                    val_ppl = math.exp(min(val_ce, 20))

                    print(f"\n  Validation @ step {step+1}:")
                    print(f"    CE:  {val_ce:.4f}")
                    print(f"    PPL: {val_ppl:.2f}")

                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        # Save best checkpoint
                        ckpt_path = exp_checkpoint_dir / 'best_model.pt'
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'step': step + 1,
                            'val_ppl': val_ppl,
                        }, ckpt_path, pickle_protocol=4,
                            _use_new_zipfile_serialization=False)
                        print(f"    ✓ New best! Saved to {ckpt_path}")

                    model.train()

            print("\n" + "="*70)
            print("✓ PURE FEP TRAINING COMPLETE!")
            print("="*70)

            # Final validation metrics
            final_ppl = best_val_ppl
            random_ppl = actual_vocab_size
            improvement = random_ppl / final_ppl if final_ppl > 0 else 0

            print(f"\nValidation Results:")
            print(f"  Best Val PPL: {final_ppl:.2f}")
            print(f"  Random PPL:   {random_ppl:.0f}")
            print(f"  Improvement:  {improvement:.1f}x better!")

            # Run test set evaluation if test loader is available
            test_metrics = None
            if test_loader is not None:
                test_metrics = run_test_evaluation(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    vocab_size=actual_vocab_size,
                    config=config,
                )

            result = {
                'ffn_mode': 'pure_fep',
                'pure_fep': True,
                'final_loss': math.log(final_ppl) if final_ppl > 0 else float('inf'),
                'final_ppl': final_ppl,
                'random_ppl': random_ppl,
                'improvement': improvement,
                'total_params': total_params,
                'vocab_size': actual_vocab_size,
                'checkpoint': str(exp_checkpoint_dir / 'best_model.pt'),
                # Training duration stats
                'total_steps': total_steps,
                'tokens_seen': total_tokens,
                'dataset_tokens': dataset_tokens,
                'dataset_coverage': total_tokens / dataset_tokens if dataset_tokens else None,
                'batch_size': batch_size,
                'seq_len': seq_len,
            }

            # Add test metrics if available
            if test_metrics is not None:
                result['test_loss'] = test_metrics['test_loss']
                result['test_ppl'] = test_metrics['test_ppl']
                result['test_bpc'] = test_metrics['test_bpc']
                result['test_improvement'] = test_metrics['improvement']

            return result

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("⚠ Training interrupted by user")
            print("="*70)
            return None

        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    else:
        # =========================================================
        # STANDARD MODE: Backprop-based training
        # =========================================================
        # Safety check: warn if model has pure_fep_mode but we're using standard training
        # Get blocks from either model type
        blocks = model.blocks if isinstance(model, StandardTransformerLM) else model.transformer.blocks
        for block in blocks:
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'pure_fep_mode'):
                if block.ffn.pure_fep_mode:
                    print("\n⚠ WARNING: Model has pure_fep_mode=True but using standard training!")
                    print("  This may cause issues. Use --pure_fep flag for backprop-free training.")
                    print("  Or set ffn_pure_fep_mode=False in config.\n")
                break

        # Create comprehensive publication metrics tracker
        pub_metrics = None
        if enable_publication_metrics:
            experiment_name = f"{ffn_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
            pub_metrics = PublicationMetrics(
                experiment_name=experiment_name,
                base_dir=exp_checkpoint_dir / "publication_outputs"
            )

            # Configure gauge frame semantic analysis interval
            # Priority: config dict > CLI args > default (10000)
            semantic_interval = config.get('semantic_analysis_interval',
                                           getattr(args, 'semantic_analysis_interval', 10000) if args else 10000)
            pub_metrics.set_semantic_analysis_interval(semantic_interval)
            print(f"[Config] Gauge frame semantic analysis every {semantic_interval} steps")

        trainer = PublicationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            publication_metrics=pub_metrics,
            tokenizer=tokenizer,  # For decoding in interpretability outputs
        )

        # =================================================================
        # Training (Standard Backprop)
        # =================================================================

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Device: {device}")
        print(f"FFN mode: {ffn_mode}")
        # Show epochs-based info if set
        if train_config.epochs is not None and train_config.epochs > 0:
            eff_steps = train_config.epochs * steps_per_epoch
            print(f"Epochs: {train_config.epochs} ({steps_per_epoch:,} steps/epoch = {eff_steps:,} total)")
        else:
            print(f"Total steps: {train_config.max_steps:,}")
        print("\nNOTE: First few batches may be slow (JIT compilation)")
        print("="*70 + "\n")

        try:
            trainer.train()

            print("\n" + "="*70)
            print("✓ TRAINING COMPLETE!")
            print("="*70)

            # Final evaluation
            final_metrics = trainer.validate()

            print(f"\nFinal Validation Metrics:")
            print(f"  Loss:       {final_metrics['loss']:.4f}")
            print(f"  Perplexity: {final_metrics['perplexity']:.2f}")

            # vs random baseline
            random_ppl = actual_vocab_size
            improvement = random_ppl / final_metrics['perplexity']
            print(f"\nValidation improvement over random:")
            print(f"  Random:     {random_ppl:.0f}")
            print(f"  Model:      {final_metrics['perplexity']:.2f}")
            print(f"  Factor:     {improvement:.1f}x better!")

            # Save final checkpoint
            final_ckpt = trainer.save_checkpoint(is_best=True)
            print(f"\n✓ Saved: {final_ckpt}")

            # Run test set evaluation if test loader is available
            test_metrics = None
            if test_loader is not None:
                test_metrics = run_test_evaluation(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    vocab_size=actual_vocab_size,
                    config=config,
                )

            # Return metrics
            result = {
                'ffn_mode': ffn_mode,
                'pure_fep': False,
                'final_loss': final_metrics['loss'],
                'final_ppl': final_metrics['perplexity'],
                'random_ppl': random_ppl,
                'improvement': improvement,
                'total_params': total_params,
                'vocab_size': actual_vocab_size,
                'checkpoint': str(final_ckpt),
                # Training duration stats
                'total_steps': train_config.max_steps if train_config.epochs is None else train_config.epochs * steps_per_epoch,
                'tokens_seen': total_tokens,
                'dataset_tokens': dataset_tokens,
                'dataset_coverage': total_tokens / dataset_tokens if dataset_tokens else None,
                'batch_size': batch_size,
                'seq_len': seq_len,
            }

            # Add test metrics if available
            if test_metrics is not None:
                result['test_loss'] = test_metrics['test_loss']
                result['test_ppl'] = test_metrics['test_ppl']
                result['test_bpc'] = test_metrics['test_bpc']
                result['test_improvement'] = test_metrics['improvement']

            return result

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("TRAINING INTERRUPTED")
            print("="*70)
            ckpt = trainer.save_checkpoint(is_best=False)
            print(f"✓ Saved: {ckpt}")
            return None

        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Publication Training Script')

    # Mode selection (three distinct modes)
    parser.add_argument('--mode', type=str, default=DEFAULT_MODE,
                        choices=['standard', 'VFE_dynamic', 'pure_fep'],
                        help='Training mode: standard (baseline), VFE_dynamic (EM-step), pure_fep (no backprop)')

    # Legacy alias for backwards compatibility
    parser.add_argument('--ffn_mode', type=str, default=None,
                        choices=['VFE_dynamic', 'standard'],
                        help='DEPRECATED: Use --mode instead')
    parser.add_argument('--pure_fep', action='store_true', default=False,
                        help='DEPRECATED: Use --mode pure_fep instead')

    # Enable full geometric learning (Σ and φ)
    parser.add_argument('--enable_sigma_phi', action='store_true', default=DEFAULT_ENABLE_SIGMA_PHI,
                        help='Enable learning covariances (Σ) and gauge frames (φ)')

    # Pure FEP learning rates
    parser.add_argument('--prior_lr', type=float, default=DEFAULT_PRIOR_LR,
                        help='Learning rate for prior updates in pure FEP mode')
    parser.add_argument('--embed_lr', type=float, default=DEFAULT_EMBED_LR,
                        help='Learning rate for embedding updates in pure FEP mode')

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_publication')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        choices=['wikitext-2', 'wikitext-103'],
                        help='Dataset to use: wikitext-2 (~2M tokens) or wikitext-103 (~103M tokens)')
    parser.add_argument('--semantic_analysis_interval', type=int, default=10000,
                        help='Run gauge frame semantic analysis every N steps (0 to disable)')

    args = parser.parse_args()

    # Handle legacy --ffn_mode and --pure_fep arguments
    if args.ffn_mode is not None:
        print("⚠ WARNING: --ffn_mode is deprecated. Use --mode instead.")
        args.mode = args.ffn_mode
    if args.pure_fep:
        print("⚠ WARNING: --pure_fep is deprecated. Use --mode pure_fep instead.")
        args.mode = 'pure_fep'

    # Set random seed for reproducibility
    # Default to seed=42 if not specified, for consistent results
    seed = SEED
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic CUDA operations (may slow down training slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*70)
    print("PUBLICATION PROOF-OF-PRINCIPLE TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)

    # =================================================================
    # SELECT CONFIG BASED ON MODE
    # =================================================================
    # Three distinct configs for clarity:
    #   STANDARD_CONFIG  - Baseline transformer (dot-product + MLP)
    #   VFE_EM_CONFIG    - VFE with EM-step dynamics (backprop)
    #   PURE_FEP_CONFIG  - Pure FEP (KL-to-prior output, NO backprop)
    # =================================================================

    mode = args.mode
    is_pure_fep = (mode == 'pure_fep')

    if mode == 'standard':
        print("\n" + "="*70)
        print("MODE: STANDARD TRANSFORMER (Baseline)")
        print("="*70)
        print("   Attention: Q·K^T / √d (dot-product softmax)")
        print("   FFN: Linear → GELU → Linear (learned MLP)")
        print("   Output: Linear projection")
        print("   Learning: Backpropagation")
        print("="*70 + "\n")
        config = STANDARD_CONFIG.copy()
        ffn_mode = 'standard'

    elif mode == 'VFE_dynamic':
        print("\n" + "="*70)
        print("MODE: VFE_EM (VFE with EM-step dynamics)")
        print("="*70)
        print("   Attention: KL-divergence based (gauge-equivariant)")
        print("   FFN: VFE EM-step dynamics")
        print("   Output: Linear projection")
        print("   Learning: Backpropagation")
        print("   Position: None (emergent)")
        print("="*70 + "\n")
        config = VFE_EM_CONFIG.copy()
        ffn_mode = 'VFE_dynamic'

    elif mode == 'pure_fep':
        print("\n" + "="*70)
        print("MODE: PURE FEP (Free Energy Principle, NO backprop!)")
        print("="*70)
        print("   Attention: KL-divergence based (gauge-equivariant)")
        print("   FFN: VFE dynamics (CE inside!)")
        print("   Output: -KL(q||π_v)/τ (most principled!)")
        print("   Learning: P-flow only (priors ← beliefs)")
        print("   Position: None (emergent)")
        print(f"   Prior LR: {args.prior_lr}")
        print("="*70 + "\n")
        config = PURE_FEP_CONFIG.copy()
        ffn_mode = 'VFE_dynamic'  # Uses VFE internally but with pure_fep_mode=True

    else:
        print(f"\nError: Unknown mode '{mode}'")
        print("Valid modes: standard, VFE_dynamic, pure_fep")
        return

    config['dataset'] = args.dataset

    # Enable full geometric learning if requested (for non-standard modes)
    if args.enable_sigma_phi and mode != 'standard':
        config['evolve_sigma'] = True
        config['evolve_phi'] = True

    result = run_single_experiment(
        config=config,
        ffn_mode=ffn_mode,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.use_wandb,
        args=args,
        pure_fep=is_pure_fep,
        prior_lr=args.prior_lr,
    )

    if result is not None:
        # Save result
        result_file = checkpoint_dir / f"result_{mode}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved result: {result_file}")

    print("\n" + "="*70)
    print("SESSION COMPLETE")
    print("="*70)


if __name__ == '__main__':

    main()

