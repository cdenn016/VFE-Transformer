"""
Visualize Token Embeddings in Belief Space (μ, Σ, φ)

Tests the meta-agent hypothesis: Do semantically similar tokens cluster in belief space?

If yes → Evidence for emergent coarse-graining / meta-agent formation
If no  → Beliefs don't reflect semantic structure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


# Define test tokens organized by semantic category
TOKEN_SETS = {
    'animals': ['cat', 'dog', 'elephant', 'bird', 'fish', 'mouse', 'lion', 'tiger'],
    'food': ['apple', 'pizza', 'bread', 'cheese', 'cake', 'meat', 'banana', 'orange'],
    'objects': ['telephone', 'chair', 'table', 'computer', 'book', 'pen', 'car', 'door'],
    'places': ['house', 'school', 'park', 'city', 'room', 'office', 'street', 'garden'],
    'actions': ['run', 'jump', 'eat', 'sleep', 'write', 'read', 'walk', 'talk'],
    'emotions': ['happy', 'sad', 'angry', 'love', 'fear', 'joy', 'hate', 'hope'],
    'nature': ['tree', 'flower', 'mountain', 'river', 'sun', 'moon', 'star', 'cloud'],
    'medical': ['doctor', 'hospital', 'medicine', 'disease', 'pain', 'cure', 'patient', 'health'],
}

# Flat list for easy iteration
ALL_TOKENS = []
TOKEN_CATEGORIES = []
for category, tokens in TOKEN_SETS.items():
    ALL_TOKENS.extend(tokens)
    TOKEN_CATEGORIES.extend([category] * len(tokens))

# Color map for categories
CATEGORY_COLORS = {
    'animals': '#FF6B6B',    # Red
    'food': '#FFA500',       # Orange
    'objects': '#4ECDC4',    # Teal
    'places': '#45B7D1',     # Blue
    'actions': '#96CEB4',    # Green
    'emotions': '#DDA15E',   # Brown
    'nature': '#2A9D8F',     # Dark teal
    'medical': '#E76F51',    # Coral
}


def get_token_embeddings(model, tokens: list, tokenizer):
    """
    Extract μ, Σ, φ embeddings for a list of tokens.

    Returns:
        mu_embeddings: (N, K) - mean vectors
        sigma_embeddings: (N, K) or (N, K, K) - covariance
        phi_embeddings: (N, 3) - gauge frames
        token_ids: List of token IDs
        valid_tokens: List of tokens that were found
    """
    mu_list = []
    sigma_list = []
    phi_list = []
    token_ids = []
    valid_tokens = []

    for token in tokens:
        # Try with and without space prefix (GPT-2 BPE tokenization)
        for prefix in ['', ' ', 'Ġ']:
            try:
                token_str = prefix + token
                encoded = tokenizer.encode(token_str)

                # Use first token if multiple
                if len(encoded) > 0:
                    token_id = encoded[0]
                    token_ids.append(token_id)

                    # Get embeddings
                    with torch.no_grad():
                        token_tensor = torch.tensor([[token_id]])
                        mu, sigma, phi = model.token_embed(token_tensor)

                    mu_list.append(mu[0, 0].cpu().numpy())  # (K,)
                    sigma_list.append(sigma[0, 0].cpu().numpy())  # (K,) or (K, K)
                    phi_list.append(phi[0, 0].cpu().numpy())  # (3,)
                    valid_tokens.append(token)
                    break
            except (RuntimeError, IndexError) as e:
                continue

    if len(mu_list) == 0:
        raise ValueError("No valid tokens found!")

    mu_embeddings = np.stack(mu_list, axis=0)  # (N, K)
    sigma_embeddings = np.stack(sigma_list, axis=0)
    phi_embeddings = np.stack(phi_list, axis=0)  # (N, 3)

    return mu_embeddings, sigma_embeddings, phi_embeddings, token_ids, valid_tokens


def visualize_belief_space(mu_embeddings, valid_tokens, token_categories,
                           method='pca', save_path=None):
    """
    Visualize embeddings in 2D using PCA or t-SNE.

    Args:
        mu_embeddings: (N, K) array
        valid_tokens: List of token strings
        token_categories: List of category labels
        method: 'pca' or 'tsne'
        save_path: Optional path to save figure
    """
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "Belief Space (μ) - PCA Projection"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(valid_tokens)-1),
                      random_state=42)
        title = "Belief Space (μ) - t-SNE Projection"
    else:
        raise ValueError(f"Unknown method: {method}")

    coords_2d = reducer.fit_transform(mu_embeddings)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each category with different color
    for category in set(token_categories):
        mask = np.array([cat == category for cat in token_categories])
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=CATEGORY_COLORS[category], label=category,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add token labels
    for i, token in enumerate(valid_tokens):
        ax.annotate(token, (coords_2d[i, 0], coords_2d[i, 1]),
                   fontsize=9, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.7))

    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if method == 'pca':
        variance_explained = reducer.explained_variance_ratio_
        ax.text(0.02, 0.98,
               f'Variance explained: {variance_explained[0]:.2%} + {variance_explained[1]:.2%} = {variance_explained.sum():.2%}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")

    return fig


def compute_category_cohesion(mu_embeddings, token_categories):
    """
    Compute how well tokens cluster by category in belief space.

    Metric: Ratio of within-category distance to between-category distance
    Lower = better clustering
    """
    from scipy.spatial.distance import cdist

    # Pairwise distances
    dist_matrix = cdist(mu_embeddings, mu_embeddings, metric='euclidean')

    within_distances = []
    between_distances = []

    for i in range(len(token_categories)):
        for j in range(i+1, len(token_categories)):
            dist = dist_matrix[i, j]

            if token_categories[i] == token_categories[j]:
                within_distances.append(dist)
            else:
                between_distances.append(dist)

    within_mean = np.mean(within_distances)
    between_mean = np.mean(between_distances)

    cohesion_ratio = within_mean / between_mean

    print(f"\n{'='*70}")
    print(f"CATEGORY CLUSTERING ANALYSIS")
    print(f"{'='*70}")
    print(f"Within-category distance:  {within_mean:.4f}")
    print(f"Between-category distance: {between_mean:.4f}")
    print(f"Cohesion ratio:            {cohesion_ratio:.4f}")
    print(f"")
    if cohesion_ratio < 0.8:
        print(f"✓ STRONG clustering! Categories well-separated in belief space.")
        print(f"  → Evidence FOR meta-agent hypothesis!")
    elif cohesion_ratio < 1.0:
        print(f"⚠ MODERATE clustering. Some category structure exists.")
    else:
        print(f"❌ WEAK clustering. Categories NOT well-separated.")
        print(f"  → Evidence AGAINST meta-agent hypothesis.")
    print(f"{'='*70}\n")

    return cohesion_ratio


def main(checkpoint_path: str = None):
    """
    Visualize token embeddings in belief space.

    Args:
        checkpoint_path: Path to model checkpoint. If None, uses command line arg.
    """
    import argparse

    if checkpoint_path is None:
        parser = argparse.ArgumentParser(description='Visualize belief space embeddings')
        parser.add_argument('checkpoint', type=str, nargs='?', default=None,
                          help='Path to model checkpoint (best_model.pt)')
        args = parser.parse_args()
        checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        print("ERROR: No checkpoint path provided.")
        print("Usage: python visualize_belief_space.py <path/to/best_model.pt>")
        return

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please update checkpoint_path in this script.")
        return

    print(f"{'='*70}")
    print(f"BELIEF SPACE VISUALIZATION")
    print(f"{'='*70}\n")

    # Load model
    model, config = load_model(checkpoint_path)

    # Get tokenizer from model's dataset
    print("\nExtracting tokenizer...")
    tokenizer = get_tokenizer(config)

    # Extract embeddings
    print(f"\nExtracting embeddings for {len(ALL_TOKENS)} tokens...")
    mu_embeddings, sigma_embeddings, phi_embeddings, token_ids, valid_tokens = \
        get_token_embeddings(model, ALL_TOKENS, tokenizer)

    print(f"✓ Found {len(valid_tokens)}/{len(ALL_TOKENS)} tokens")
    print(f"  μ shape: {mu_embeddings.shape}")
    print(f"  Σ shape: {sigma_embeddings.shape}")
    print(f"  φ shape: {phi_embeddings.shape}")

    # CRITICAL: Show which BPE tokens we actually got
    print(f"\nBPE Token Mappings (first 20):")
    print(f"{'Word':<15} {'BPE Token ID':<15} {'Decoded':<20}")
    print(f"{'-'*50}")
    for i in range(min(20, len(valid_tokens))):
        token_str = tokenizer.decode([token_ids[i]]) if hasattr(tokenizer, 'decode') else '???'
        print(f"{valid_tokens[i]:<15} {token_ids[i]:<15} {repr(token_str):<20}")

    # Filter categories to match valid tokens
    valid_categories = [TOKEN_CATEGORIES[ALL_TOKENS.index(t)] for t in valid_tokens]

    # Compute clustering metrics
    cohesion = compute_category_cohesion(mu_embeddings, valid_categories)

    # Create output directory
    output_dir = Path(checkpoint_path).parent / "belief_space_viz"
    output_dir.mkdir(exist_ok=True)

    # PCA visualization
    print("\nGenerating PCA visualization...")
    fig_pca = visualize_belief_space(
        mu_embeddings, valid_tokens, valid_categories,
        method='pca',
        save_path=output_dir / 'belief_space_pca.png'
    )

    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    fig_tsne = visualize_belief_space(
        mu_embeddings, valid_tokens, valid_categories,
        method='tsne',
        save_path=output_dir / 'belief_space_tsne.png'
    )

    # Show plots
    plt.show()

    print(f"\n{'='*70}")
    print(f"DONE")
    print(f"{'='*70}")
    print(f"Visualizations saved to: {output_dir}/")
    print(f"  - belief_space_pca.png")
    print(f"  - belief_space_tsne.png")


if __name__ == '__main__':
    main()