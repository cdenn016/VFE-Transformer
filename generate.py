#!/usr/bin/env python3
"""
Simple click-to-run text generation with attention visualization.

Instructions:
    1. Set CHECKPOINT_PATH below to your model
    2. Set INPUT_TEXT to your prompt
    3. Run this script (F5 in Spyder, or python generate.py)
"""

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================

CHECKPOINT_PATH = r"path/to/your/best_model.pt"  # <-- Set your model path here

INPUT_TEXT = "The meaning of life is"  # <-- Set your input text here

MAX_TOKENS = 50          # Number of tokens to generate
TEMPERATURE = 0.8        # Higher = more random, lower = more focused
SAVE_ATTENTION = True    # Set True to save attention visualization

# =============================================================================
# CODE - No need to edit below
# =============================================================================

import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


def plot_attention_publication(beta, kl, tokens, save_path="attention_pattern.pdf", title=None):
    """Create publication-quality attention visualization."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    n_heads = beta.shape[0]

    # Truncate long tokens
    display_tokens = []
    for t in tokens:
        t = t.replace('\n', '\\n').replace('\t', '\\t')
        if len(t) > 8:
            t = t[:6] + '..'
        display_tokens.append(t)

    # Layout: 3 plots - Head 0, Mean Attention, Mean KL
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def plot_matrix(ax, matrix, tokens, cmap, plot_title):
        im = ax.imshow(matrix, cmap=cmap, aspect='equal', interpolation='nearest')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel('Key (j)', fontsize=10)
        ax.set_ylabel('Query (i)', fontsize=10)
        ax.set_title(plot_title, fontsize=11, fontweight='medium')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        plt.colorbar(im, cax=cax)
        ax.set_xticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(tokens), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.5, alpha=0.3)

    # Head 0
    plot_matrix(axes[0], beta[0].numpy(), display_tokens, 'Blues', 'Head 0: Attention β')

    # Mean attention
    plot_matrix(axes[1], beta.mean(dim=0).numpy(), display_tokens, 'Blues', 'Mean Attention β')

    # Mean KL (log scale)
    mean_kl_log = np.log10(kl.mean(dim=0).numpy() + 1e-6)
    plot_matrix(axes[2], mean_kl_log, display_tokens, 'Oranges', 'Mean KL Divergence (log₁₀)')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")

    png_path = save_path.rsplit('.', 1)[0] + '.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {png_path}")

    plt.close(fig)


def main():
    # Check path
    if not Path(CHECKPOINT_PATH).exists():
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nPlease edit CHECKPOINT_PATH at the top of this file.")
        return

    # Load model
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model, config = load_model(CHECKPOINT_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Load tokenizer
    tokenizer = get_tokenizer(config)
    if tokenizer is None:
        print("ERROR: Could not load tokenizer. Install tiktoken: pip install tiktoken")
        return

    # Encode input
    print(f"\nInput: \"{INPUT_TEXT}\"")
    token_ids = tokenizer.encode(INPUT_TEXT)
    input_ids = torch.tensor([token_ids], device=device)

    # Generate
    print(f"Generating {MAX_TOKENS} tokens (temperature={TEMPERATURE})...")
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids=input_ids,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=40,
            top_p=0.9,
        )

    # Decode and print
    output_text = tokenizer.decode(output_ids[0].tolist())
    generated_part = output_text[len(INPUT_TEXT):]

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"{INPUT_TEXT}\033[92m{generated_part}\033[0m")
    print("=" * 60)

    # Save attention visualization
    if SAVE_ATTENTION:
        print("\nGenerating attention visualization...")
        with torch.no_grad():
            _, attn_info = model.forward_with_attention(input_ids)

        tokens = [tokenizer.decode([tid]) for tid in token_ids]

        # Save to same directory as checkpoint
        output_dir = Path(CHECKPOINT_PATH).parent
        save_path = output_dir / "attention_pattern.pdf"

        plot_attention_publication(
            attn_info['beta'][0].cpu(),
            attn_info['kl'][0].cpu(),
            tokens,
            save_path=str(save_path),
            title=f"Attention: \"{INPUT_TEXT[:40]}{'...' if len(INPUT_TEXT) > 40 else ''}\"",
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
