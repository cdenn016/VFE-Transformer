#!/usr/bin/env python3
"""
Training script for Pure FEP Transformer.

Minimal, clean training loop for the Free Energy Principle Transformer.
No bells and whistles - just the core VFE minimization.

Usage:
    Edit DEFAULT_CONFIG below, then run:
    python transformer/train_fep.py
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from .fep_transformer import FEPTransformer, IrrepSpec
except ImportError:
    from fep_transformer import FEPTransformer, IrrepSpec


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'vocab_size': 50257,      # GPT-2 tokenizer
    'embed_dim': 30,          # Small for testing (3 x SO(10))
    'gauge_dim': 10,          # SO(10)
    'irrep_spec': [('fund', 3, 10)],  # 3 copies of fundamental
    'n_layers': 4,            # Number of Q-flow layers (stacked)
    'n_q_iterations': 5,      # Iterations per layer
    'residual': True,         # Residual connections between layers
    'observe_during_qflow': True,   # True=two-phase FEP, False=blind prediction
    'blind_iterations': 3,          # Alignment-only iterations before observing
    'lambda_obs': 1.0,              # Observation term weight

    # VFE weights
    'alpha': 0.1,             # Self-coupling (entropy)
    'beta': 1.0,              # Belief alignment (attention)
    'gamma': 0.1,             # Prior coupling
    'bch_order': 2,           # BCH truncation order
    'temperature': 1.0,       # Attention temperature

    # Training
    'dataset': 'wikitext-103',  # 'wikitext-2', 'wikitext-103', or 'random'
    'batch_size': 8,
    'seq_len': 128,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'epochs': 10,
    'grad_clip': 1.0,
    'output_dir': './fep_checkpoints',

    # Logging
    'log_interval': 100,
    'eval_interval': 500,
    'save_interval': 1000,
    'sample_interval': 500,   # Show random generation sample every N batches
}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_tokenizer():
    """Get GPT-2 tokenizer."""
    try:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("transformers not installed. Using dummy tokenizer.")
        return None


def get_dataset(name: str, tokenizer, seq_len: int):
    """Load dataset."""
    try:
        from datasets import load_dataset

        if name == 'wikitext-2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        elif name == 'wikitext-103':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        else:
            raise ValueError(f"Unknown dataset: {name}")

        def tokenize(examples):
            return tokenizer(examples['text'], truncation=True,
                           max_length=seq_len, padding='max_length',
                           return_tensors='pt')

        train_data = dataset['train'].map(tokenize, batched=True,
                                          remove_columns=['text'])
        val_data = dataset['validation'].map(tokenize, batched=True,
                                             remove_columns=['text'])

        train_data.set_format('torch', columns=['input_ids', 'attention_mask'])
        val_data.set_format('torch', columns=['input_ids', 'attention_mask'])

        return train_data, val_data

    except ImportError:
        print("datasets not installed. Using random data.")
        return None, None


class RandomDataset(torch.utils.data.Dataset):
    """Fallback random dataset for testing."""

    def __init__(self, vocab_size: int, seq_len: int, size: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
            'attention_mask': torch.ones(self.seq_len),
        }


# =============================================================================
# TRAINING LOOP
# =============================================================================

import random

SAMPLE_PROMPTS = [
    "The dog",
    "In the morning",
    "Scientists believe",
    "The president",
    "Once upon a time",
    "The weather",
    "According to",
    "Many people",
    "The new",
    "After the",
]


@torch.no_grad()
def quick_sample(model, tokenizer, device, temperature=0.8, max_tokens=20):
    """Generate a quick sample from a random prompt."""
    model.eval()
    prompt = random.choice(SAMPLE_PROMPTS)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    model.train()
    return prompt, generated


def train_epoch(model, dataloader, optimizer, config, device, epoch, tokenizer=None, val_loader=None):
    """Train for one epoch with periodic validation."""
    model.train()
    total_loss = 0
    total_ce = 0
    n_batches = 0

    sample_interval = config.get('sample_interval', 500)
    eval_interval = config.get('eval_interval', 500)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)

        # Shift for next-token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs, targets)

        loss = outputs['loss']
        ce_loss = outputs['ce_loss']

        # Skip batch if NaN detected
        if torch.isnan(loss) or torch.isnan(ce_loss):
            tqdm.write(f"  [Warning] NaN detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        n_batches += 1

        # Logging
        if batch_idx % config['log_interval'] == 0:
            avg_loss = total_loss / n_batches
            avg_ce = total_ce / n_batches
            ppl = math.exp(min(avg_ce, 20))  # Cap to avoid overflow
            pbar.set_postfix({
                'VFE': f'{avg_loss:.4f}',
                'CE': f'{avg_ce:.4f}',
                'PPL': f'{ppl:.1f}'
            })

        # Periodic sample generation
        if tokenizer is not None and batch_idx > 0 and batch_idx % sample_interval == 0:
            prompt, generated = quick_sample(model, tokenizer, device)
            tqdm.write(f"\n  [Sample] \"{prompt}\" → {generated}\n")

        # Periodic validation
        if val_loader is not None and batch_idx > 0 and batch_idx % eval_interval == 0:
            val_loss, val_ce, val_ppl = evaluate(model, val_loader, device)
            tqdm.write(f"\n  [Val @ {batch_idx}] CE={val_ce:.4f}, PPL={val_ppl:.1f}\n")
            model.train()  # Back to training mode

    return total_loss / n_batches, total_ce / n_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set with HONEST (blind) Q-flow."""
    model.eval()
    total_loss = 0
    total_ce = 0
    n_batches = 0

    # Force blind Q-flow during validation - no peeking at targets!
    original_observe = model.observe_during_qflow
    model.observe_during_qflow = False

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        outputs = model(inputs, targets)

        total_loss += outputs['loss'].item()
        total_ce += outputs['ce_loss'].item()
        n_batches += 1

    # Restore original setting
    model.observe_during_qflow = original_observe

    avg_loss = total_loss / n_batches
    avg_ce = total_ce / n_batches
    ppl = math.exp(min(avg_ce, 20))

    return avg_loss, avg_ce, ppl


# =============================================================================
# MAIN
# =============================================================================

def main():
    # =========================================================================
    # ALL SETTINGS IN DEFAULT_CONFIG - just edit that dict above!
    # =========================================================================
    config = DEFAULT_CONFIG.copy()

    # Set seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Auto-compute irrep_spec from embed_dim and gauge_dim
    n_copies = config['embed_dim'] // config['gauge_dim']
    config['irrep_spec'] = [('fund', n_copies, config['gauge_dim'])]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and data
    tokenizer = get_tokenizer()
    if tokenizer is not None:
        config['vocab_size'] = len(tokenizer)

    dataset_name = config.get('dataset', 'wikitext-103')
    if dataset_name == 'random' or tokenizer is None:
        print("Using random data for testing...")
        train_data = RandomDataset(config['vocab_size'], config['seq_len'], 10000)
        val_data = RandomDataset(config['vocab_size'], config['seq_len'], 1000)
    else:
        train_data, val_data = get_dataset(dataset_name, tokenizer, config['seq_len'])
        if train_data is None:
            train_data = RandomDataset(config['vocab_size'], config['seq_len'], 10000)
            val_data = RandomDataset(config['vocab_size'], config['seq_len'], 1000)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                            shuffle=False, num_workers=0)

    # Create model
    print("\n" + "=" * 60)
    print("FEP TRANSFORMER")
    print("=" * 60)
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Gauge dim: {config['gauge_dim']} (SO({config['gauge_dim']}))")
    print(f"  Irrep spec: {config['irrep_spec']}")
    print(f"  N layers: {config['n_layers']} (Q-flow iterations: {config['n_q_iterations']} per layer)")
    print(f"  Residual: {config.get('residual', True)}")
    if config.get('observe_during_qflow', False):
        blind_iters = config.get('blind_iterations', 3)
        obs_iters = config['n_q_iterations'] - blind_iters
        observe_mode = f"TWO-PHASE FEP (align:{blind_iters} → obs:{obs_iters}, λ={config.get('lambda_obs', 1.0)})"
    else:
        observe_mode = "BLIND (honest LM)"
    print(f"  Q-flow mode: {observe_mode}")
    print(f"  BCH order: {config['bch_order']}")
    print(f"  VFE weights: α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")
    print("=" * 60 + "\n")

    model = FEPTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        gauge_dim=config['gauge_dim'],
        irrep_spec=config['irrep_spec'],
        n_layers=config['n_layers'],
        n_q_iterations=config['n_q_iterations'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        bch_order=config['bch_order'],
        temperature=config['temperature'],
        residual=config.get('residual', True),
        observe_during_qflow=config.get('observe_during_qflow', False),
        blind_iterations=config.get('blind_iterations', 3),
        lambda_obs=config.get('lambda_obs', 1.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Training loop
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_ppl = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        train_loss, train_ce = train_epoch(
            model, train_loader, optimizer, config, device, epoch, tokenizer, val_loader
        )

        val_loss, val_ce, val_ppl = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch} | Time: {elapsed:.1f}s")
        print(f"  Train: VFE={train_loss:.4f}, CE={train_ce:.4f}, PPL={math.exp(min(train_ce, 20)):.1f}")
        print(f"  Valid: VFE={val_loss:.4f}, CE={val_ce:.4f}, PPL={val_ppl:.1f}")

        # Save best model
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'config': config,
            }, output_dir / 'best_model.pt')
            print(f"  Saved best model (PPL={val_ppl:.1f})")

    print(f"\nTraining complete! Best validation PPL: {best_val_ppl:.1f}")

    # Run inference examples
    if tokenizer is not None:
        print("\n" + "=" * 60)
        print("GENERATION EXAMPLES")
        print("=" * 60)
        run_inference_examples(model, tokenizer, device)


def run_inference_examples(model, tokenizer, device, num_examples=5):
    """Generate text samples from the trained model."""
    model.eval()

    prompts = [
        "The meaning of life is",
        "In the beginning",
        "Scientists have discovered",
        "The president announced",
        "Once upon a time",
    ]

    for prompt in prompts[:num_examples]:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8
            )

        # Decode
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {generated}")

    # Also show attention pattern for one example
    print("\n" + "=" * 60)
    print("ATTENTION ANALYSIS")
    print("=" * 60)

    input_ids = tokenizer.encode("The cat sat on the", return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(input_ids[:, :-1], input_ids[:, 1:], return_components=True)

    if 'attention' in outputs and outputs['attention'] is not None:
        attn = outputs['attention'][0]  # First batch
        print(f"Attention shape: {attn.shape}")
        print(f"Attention (last token attending to all):")
        print(attn[-1, :].cpu().numpy().round(3))


if __name__ == '__main__':
    main()
