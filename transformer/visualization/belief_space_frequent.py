# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 15:06:23 2025

@author: chris and christine
"""

"""
Visualize FREQUENT BPE Token Embeddings in Belief Space

Instead of mapping English words to BPE tokens, this script:
1. Finds the most frequent tokens in the training data
2. Visualizes their embeddings in belief space
3. Shows what the model ACTUALLY learned

This avoids the BPE mismatch problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import sys
from collections import Counter

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model


def get_frequent_tokens(n_tokens=100, dataset='wikitext-2'):
    """
    Get the N most frequent tokens from the dataset.

    Returns token IDs and their frequencies.
    """
    try:
        from transformer.data.datasets import WikiTextDataset

        print(f"Loading {dataset} to find frequent tokens...")
        data = WikiTextDataset(split='train', max_seq_len=128, dataset_name=dataset)

        # Count token frequencies
        token_counts = Counter()

        # Sample from dataset (don't process entire thing - too slow)
        max_samples = min(1000, len(data))
        print(f"Sampling {max_samples} sequences to count token frequencies...")

        for i in range(max_samples):
            tokens, _ = data[i]
            token_counts.update(tokens.tolist())

        # Get top N most frequent
        most_common = token_counts.most_common(n_tokens)
        token_ids = [tok_id for tok_id, count in most_common]
        frequencies = [count for tok_id, count in most_common]

        # Decode to see what they are
        decoded = []
        for tok_id in token_ids:
            try:
                decoded_str = data.decode([tok_id])
                decoded.append(decoded_str)
            except (KeyError, IndexError, RuntimeError):
                decoded.append(f"<ID:{tok_id}>")

        print(f"✓ Found {len(token_ids)} frequent tokens")
        print(f"\nTop 20 most frequent tokens:")
        print(f"{'Rank':<6} {'Token ID':<10} {'Count':<10} {'Decoded':<30}")
        print(f"{'-'*60}")
        for i in range(min(20, len(token_ids))):
            print(f"{i+1:<6} {token_ids[i]:<10} {frequencies[i]:<10} {repr(decoded[i]):<30}")

        return token_ids, decoded, frequencies, data

    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to common token IDs...")

        # Fallback: Use known common GPT-2 tokens
        # These are approximately correct for English text
        common_ids = list(range(50, 150))  # Skip special tokens, get common words

        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        decoded = [enc.decode([tid]) for tid in common_ids]
        frequencies = [1] * len(common_ids)  # Unknown

        return common_ids, decoded, frequencies, enc


def get_embeddings_by_ids(model, token_ids):
    """
    Extract μ, Σ, φ embeddings for token IDs directly from embedding layer.

    Returns:
        mu_embeddings: (N, K)
        sigma_embeddings: (N, K) or (N, K, K)
        phi_embeddings: (N, 3)
    """
    mu_list = []
    sigma_list = []
    phi_list = []

    with torch.no_grad():
        for token_id in token_ids:
            token_tensor = torch.tensor([[token_id]])
            mu, sigma, phi = model.token_embed(token_tensor)

            mu_list.append(mu[0, 0].cpu().numpy())
            sigma_list.append(sigma[0, 0].cpu().numpy())
            phi_list.append(phi[0, 0].cpu().numpy())

    mu_embeddings = np.stack(mu_list, axis=0)
    sigma_embeddings = np.stack(sigma_list, axis=0)
    phi_embeddings = np.stack(phi_list, axis=0)

    return mu_embeddings, sigma_embeddings, phi_embeddings


def categorize_tokens(decoded_tokens):
    """
    Attempt to categorize tokens by type for visualization.

    Categories:
    - punctuation
    - common_words (the, a, is, etc.)
    - content_words (nouns, verbs, adjectives)
    - numbers
    - special
    """
    categories = []

    # Simple heuristics
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                   'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with', 'from',
                   'and', 'or', 'but', 'if', 'as', 'it', 'that', 'this'}

    for token in decoded_tokens:
        token_clean = token.strip().lower()

        if len(token_clean) == 0:
            categories.append('whitespace')
        elif token_clean in {'.', ',', '!', '?', ';', ':', '"', "'", '-'}:
            categories.append('punctuation')
        elif token_clean in common_words:
            categories.append('common_words')
        elif token_clean.isdigit():
            categories.append('numbers')
        elif token_clean.startswith('<') and token_clean.endswith('>'):
            categories.append('special')
        else:
            categories.append('content_words')

    return categories


def visualize_frequent_tokens(mu_embeddings, decoded_tokens, frequencies,
                              method='pca', save_path=None):
    """Visualize embeddings with size proportional to frequency."""

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "Frequent Token Embeddings (μ) - PCA"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(decoded_tokens)-1),
                      random_state=42)
        title = "Frequent Token Embeddings (μ) - t-SNE"
    else:
        raise ValueError(f"Unknown method: {method}")

    coords_2d = reducer.fit_transform(mu_embeddings)

    # Categorize tokens
    categories = categorize_tokens(decoded_tokens)

    # Color map
    category_colors = {
        'common_words': '#FF6B6B',
        'content_words': '#4ECDC4',
        'punctuation': '#FFA500',
        'numbers': '#96CEB4',
        'whitespace': '#CCCCCC',
        'special': '#DDA15E',
    }

    # Size proportional to frequency
    sizes = np.array(frequencies)
    sizes = 50 + 200 * (sizes / sizes.max())  # Scale to 50-250

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))

    for category in set(categories):
        mask = np.array([cat == category for cat in categories])
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=category_colors.get(category, '#888888'),
                  label=category, s=sizes[mask],
                  alpha=0.6, edgecolors='black', linewidth=1)

    # Annotate top 30 most frequent
    for i in range(min(30, len(decoded_tokens))):
        ax.annotate(repr(decoded_tokens[i])[:20],
                   (coords_2d[i, 0], coords_2d[i, 1]),
                   fontsize=8, alpha=0.8,
                   xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if method == 'pca':
        var_exp = reducer.explained_variance_ratio_
        ax.text(0.02, 0.98,
               f'Variance: {var_exp[0]:.1%} + {var_exp[1]:.1%} = {var_exp.sum():.1%}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig


def analyze_embedding_diversity(mu_embeddings):
    """Compute statistics about embedding space."""
    from scipy.spatial.distance import pdist

    # Pairwise distances
    distances = pdist(mu_embeddings, metric='euclidean')

    print(f"\n{'='*70}")
    print(f"EMBEDDING SPACE STATISTICS")
    print(f"{'='*70}")
    print(f"Number of tokens:          {len(mu_embeddings)}")
    print(f"Embedding dimension:       {mu_embeddings.shape[1]}")
    print(f"")
    print(f"Pairwise distance stats:")
    print(f"  Mean:   {distances.mean():.4f}")
    print(f"  Std:    {distances.std():.4f}")
    print(f"  Min:    {distances.min():.4f}")
    print(f"  Max:    {distances.max():.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    print(f"")

    if distances.mean() < 0.3:
        print(f"❌ COLLAPSED embeddings! All tokens very similar.")
        print(f"   This limits model capacity - PPL bottleneck!")
    elif distances.mean() < 0.7:
        print(f"⚠ COMPRESSED embeddings. Moderate diversity.")
    else:
        print(f"✓ DIVERSE embeddings. Good separation.")

    print(f"{'='*70}\n")


def main(checkpoint_path: str = None):
    """
    Visualize frequent token embeddings in belief space.

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses command line arg.
    """
    import argparse

    if checkpoint_path is None:
        parser = argparse.ArgumentParser(description='Visualize frequent token embeddings')
        parser.add_argument('checkpoint', type=str, nargs='?', default=None,
                          help='Path to model checkpoint (best_model.pt)')
        args = parser.parse_args()
        checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        print("ERROR: No checkpoint path provided.")
        print("Usage: python visualize_belief_space_frequent.py <path/to/best_model.pt>")
        return

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"{'='*70}")
    print(f"FREQUENT TOKEN EMBEDDING VISUALIZATION")
    print(f"{'='*70}\n")

    # Load model
    model, config = load_model(checkpoint_path)

    # Get frequent tokens from dataset
    token_ids, decoded_tokens, frequencies, tokenizer = get_frequent_tokens(
        n_tokens=100,
        dataset=config.get('dataset', 'wikitext-2')
    )

    # Extract embeddings
    print(f"\nExtracting embeddings for {len(token_ids)} tokens...")
    mu_embeddings, sigma_embeddings, phi_embeddings = \
        get_embeddings_by_ids(model, token_ids)

    print(f"✓ Embedding shapes:")
    print(f"  μ: {mu_embeddings.shape}")
    print(f"  Σ: {sigma_embeddings.shape}")
    print(f"  φ: {phi_embeddings.shape}")

    # Analyze diversity
    analyze_embedding_diversity(mu_embeddings)

    # Create output directory
    output_dir = Path(checkpoint_path).parent / "belief_space_viz"
    output_dir.mkdir(exist_ok=True)

    # PCA visualization
    print("\nGenerating PCA visualization...")
    fig_pca = visualize_frequent_tokens(
        mu_embeddings, decoded_tokens, frequencies,
        method='pca',
        save_path=output_dir / 'frequent_tokens_pca.png'
    )

    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    fig_tsne = visualize_frequent_tokens(
        mu_embeddings, decoded_tokens, frequencies,
        method='tsne',
        save_path=output_dir / 'frequent_tokens_tsne.png'
    )

    plt.show()

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"Visualizations saved to: {output_dir}/")


if __name__ == '__main__':
    main()