#!/usr/bin/env python3
"""
Analyze whether gauge frames φ encode semantic relationships.

This module provides functions for:
1. Analyzing gauge frame semantic structure during training
2. Generating visualization plots of φ embeddings
3. Computing distance metrics between token classes

Can be used as a standalone script or imported for use during training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for training
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    PCA = None
    SKLEARN_AVAILABLE = False

# =============================================================================
# Tokenizer Setup
# =============================================================================

_tokenizer = None

def get_tokenizer():
    """Lazily load the GPT-2 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("gpt2")
        except ImportError:
            print("WARNING: tiktoken not installed. Install with: pip install tiktoken")
            return None
    return _tokenizer


# =============================================================================
# Helper Functions
# =============================================================================

def dist(t1: int, t2: int, embed: torch.Tensor) -> float:
    """Euclidean distance between embeddings."""
    if embed is None or t1 >= len(embed) or t2 >= len(embed):
        return float('nan')
    return torch.norm(embed[t1] - embed[t2]).item()


def get_token_id(word: str) -> Optional[int]:
    """Get token ID if word is a single BPE token."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return None
    tokens = tokenizer.encode(word)
    if len(tokens) == 1:
        return tokens[0]
    return None


def categorize_token(tid: int) -> str:
    """Categorize a token by type."""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return 'other'
    try:
        s = tokenizer.decode([tid])
        if len(s) == 1:
            if s.isalpha():
                return 'letter'
            elif s.isdigit():
                return 'digit'
            elif not s.isalnum() and not s.isspace():
                return 'punct'
        if s.strip() in {'the', 'a', 'an', 'is', 'are', 'was', 'of', 'to', 'in', 'for', 'and', 'or'}:
            return 'function'
        return 'content'
    except (KeyError, IndexError, TypeError, UnicodeDecodeError):
        return 'other'


def format_gauge_group_label(gauge_group: str, gauge_dim: int) -> str:
    """Format gauge group config values into a display label.

    Args:
        gauge_group: Config string ('SO3', 'SON', 'GLK')
        gauge_dim: Gauge dimension (N for SO(N), K for GL(K))

    Returns:
        Human-readable label like 'SO(3)', 'SO(10)', 'GL(30)'
    """
    if gauge_group == 'SO3':
        return "SO(3)"
    elif gauge_group == 'SON':
        return f"SO({gauge_dim})"
    elif gauge_group == 'GLK':
        return f"GL({gauge_dim})"
    else:
        return f"{gauge_group}({gauge_dim})"


def identify_gauge_group(phi_dim: int) -> str:
    """Identify gauge group from phi dimension (number of generators).

    This is a fallback heuristic used when the model config is not available.
    Prefer format_gauge_group_label() with model.gauge_group when possible.
    """
    # SO(2): 1 generator, SO(3): 3 generators, SO(N): N(N-1)/2 generators
    # GL(K): K² generators
    if phi_dim == 1:
        return "SO(2)"
    elif phi_dim == 3:
        return "SO(3)"
    else:
        # Check if it's a perfect square (GL(K) has K² generators)
        k_sqrt = int(round(np.sqrt(phi_dim)))
        if k_sqrt * k_sqrt == phi_dim:
            return f"GL({k_sqrt})"
        else:
            # Solve N(N-1)/2 = phi_dim for N (SO(N))
            n_approx = int(round((1 + np.sqrt(1 + 8 * phi_dim)) / 2))
            # Verify the solution: n*(n-1)/2 should equal phi_dim
            if n_approx * (n_approx - 1) // 2 == phi_dim:
                return f"SO({n_approx})"
            else:
                return f"Gauge({phi_dim}D)"


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_token_classes(
    mu_embed: torch.Tensor,
    phi_embed: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Analyze distances between token classes (letters, digits, punctuation).

    Returns metrics indicating whether embeddings show class structure.
    """
    tokenizer = get_tokenizer()
    if tokenizer is None:
        return {'error': 'tokenizer not available'}

    # Identify token classes
    letter_ids = []
    digit_ids = []
    punct_ids = []

    for tid in range(256):
        try:
            s = tokenizer.decode([tid])
            if len(s) == 1:
                if s.isalpha():
                    letter_ids.append(tid)
                elif s.isdigit():
                    digit_ids.append(tid)
                elif not s.isalnum() and not s.isspace():
                    punct_ids.append(tid)
        except (KeyError, IndexError, TypeError, UnicodeDecodeError):
            pass

    # Compute intra-class distances (letter-letter)
    intra_mu, intra_phi = [], []
    for i, t1 in enumerate(letter_ids[:10]):
        for t2 in letter_ids[i+1:10]:
            intra_mu.append(dist(t1, t2, mu_embed))
            if phi_embed is not None:
                intra_phi.append(dist(t1, t2, phi_embed))

    # Compute inter-class distances (letter-digit, letter-punct)
    inter_mu, inter_phi = [], []
    for t1 in letter_ids[:10]:
        for t2 in digit_ids[:5] + punct_ids[:5]:
            inter_mu.append(dist(t1, t2, mu_embed))
            if phi_embed is not None:
                inter_phi.append(dist(t1, t2, phi_embed))

    # Clean NaNs
    intra_mu = [x for x in intra_mu if not np.isnan(x)]
    intra_phi = [x for x in intra_phi if not np.isnan(x)]
    inter_mu = [x for x in inter_mu if not np.isnan(x)]
    inter_phi = [x for x in inter_phi if not np.isnan(x)]

    results = {
        'n_letters': len(letter_ids),
        'n_digits': len(digit_ids),
        'n_punct': len(punct_ids),
        'mu_intra_class_dist': np.mean(intra_mu) if intra_mu else 0,
        'mu_inter_class_dist': np.mean(inter_mu) if inter_mu else 0,
        'mu_class_ratio': (np.mean(inter_mu) / np.mean(intra_mu)) if intra_mu and inter_mu else 0,
    }

    if intra_phi and inter_phi:
        results['phi_intra_class_dist'] = np.mean(intra_phi)
        results['phi_inter_class_dist'] = np.mean(inter_phi)
        results['phi_class_ratio'] = np.mean(inter_phi) / np.mean(intra_phi)
        results['phi_shows_structure'] = results['phi_class_ratio'] > 1.2

    return results


def compute_clustering_metrics(
    embed: torch.Tensor,
    n_tokens: int = 500,
    embed_name: str = 'embed',
) -> Dict[str, Any]:
    """
    Compute quantitative clustering metrics in the full-dimensional space.

    These metrics address the concern that PCA projections can create
    illusory cluster structure. All metrics are computed in the original
    high-dimensional space, not in a PCA projection.

    Metrics computed:
        - silhouette_score: Mean silhouette coefficient [-1, 1]. Measures
          how similar each token is to its own category vs. nearest other
          category. Higher = better separation. Computed in full space.
        - calinski_harabasz: Ratio of between-cluster to within-cluster
          variance. Higher = better defined clusters. Unbounded.
        - inter_intra_ratio: Ratio of mean inter-class to mean intra-class
          distance across all categorized tokens in full space.
        - anova_f_stat: Mean F-statistic from one-way ANOVA across dimensions.
          Tests whether category means differ significantly.
        - anova_p_value: Geometric mean of per-dimension ANOVA p-values.
        - pca_variance_cumulative: Number of components for 50%, 90%, 95%
          of variance, contextualizing low per-component explained variance.
        - n_tokens_per_category: Counts per category used in analysis.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim]
        n_tokens: Number of tokens to analyze (first n_tokens)
        embed_name: Name prefix for result keys (e.g., 'mu' or 'phi')

    Returns:
        Dictionary with all computed metrics.
    """
    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]
    results = {}

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(len(embed_np))]
    unique_cats = sorted(set(categories))

    # Need at least 2 categories with 2+ members for clustering metrics
    cat_counts = {c: categories.count(c) for c in unique_cats}
    valid_cats = [c for c, cnt in cat_counts.items() if cnt >= 2]
    results[f'{embed_name}_n_tokens_per_category'] = cat_counts

    if len(valid_cats) < 2:
        results[f'{embed_name}_clustering_error'] = 'fewer than 2 categories with 2+ members'
        return results

    # Build label array for valid categories only
    cat_to_int = {c: i for i, c in enumerate(valid_cats)}
    mask = [categories[i] in cat_to_int for i in range(len(embed_np))]
    X = embed_np[mask]
    labels = np.array([cat_to_int[categories[i]] for i in range(len(embed_np)) if mask[i]])

    # --- Silhouette Score (full-dimensional) ---
    try:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X, labels, metric='euclidean', sample_size=min(len(X), 2000))
        results[f'{embed_name}_silhouette_score'] = float(sil)
    except ImportError as e:
        results[f'{embed_name}_silhouette_score'] = f'error: {e}'
    except ValueError as e:
        results[f'{embed_name}_silhouette_score'] = f'error: {e}'

    # --- Calinski-Harabasz Index ---
    try:
        from sklearn.metrics import calinski_harabasz_score
        ch = calinski_harabasz_score(X, labels)
        results[f'{embed_name}_calinski_harabasz'] = float(ch)
    except ImportError as e:
        results[f'{embed_name}_calinski_harabasz'] = f'error: {e}'
    except ValueError as e:
        results[f'{embed_name}_calinski_harabasz'] = f'error: {e}'

    # --- Inter/Intra class distance ratio (full space, all tokens) ---
    try:
        intra_dists = []
        inter_dists = []
        # Sample within and between categories for efficiency
        rng = np.random.RandomState(42)
        for ci, cat in enumerate(valid_cats):
            idx_cat = np.where(labels == ci)[0]
            if len(idx_cat) < 2:
                continue
            # Sample pairs within category
            n_pairs = min(200, len(idx_cat) * (len(idx_cat) - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.choice(len(idx_cat), 2, replace=False)
                intra_dists.append(np.linalg.norm(X[idx_cat[i]] - X[idx_cat[j]]))
            # Sample pairs between this category and others
            other_idx = np.where(labels != ci)[0]
            n_inter = min(200, len(idx_cat) * len(other_idx))
            for _ in range(n_inter):
                i = rng.choice(len(idx_cat))
                j = rng.choice(len(other_idx))
                inter_dists.append(np.linalg.norm(X[idx_cat[i]] - X[other_idx[j]]))

        if intra_dists and inter_dists:
            mean_intra = np.mean(intra_dists)
            mean_inter = np.mean(inter_dists)
            results[f'{embed_name}_mean_intra_dist'] = float(mean_intra)
            results[f'{embed_name}_mean_inter_dist'] = float(mean_inter)
            results[f'{embed_name}_inter_intra_ratio'] = float(mean_inter / mean_intra) if mean_intra > 0 else float('inf')
    except (ValueError, np.linalg.LinAlgError) as e:
        results[f'{embed_name}_inter_intra_ratio'] = f'error: {e}'

    # --- Per-dimension ANOVA F-test ---
    try:
        from scipy.stats import f_oneway
        f_stats = []
        p_vals = []
        # Test each dimension: do category means differ?
        groups_per_dim = {ci: X[labels == ci] for ci in range(len(valid_cats))}
        n_dims_to_test = min(embed_dim, 100)  # cap for efficiency
        dim_indices = rng.choice(embed_dim, n_dims_to_test, replace=False) if embed_dim > 100 else range(embed_dim)
        for d in dim_indices:
            groups = [groups_per_dim[ci][:, d] for ci in range(len(valid_cats)) if len(groups_per_dim[ci]) >= 2]
            if len(groups) >= 2:
                f, p = f_oneway(*groups)
                if np.isfinite(f):
                    f_stats.append(f)
                    p_vals.append(max(p, 1e-300))  # floor for log

        if f_stats:
            results[f'{embed_name}_anova_mean_f'] = float(np.mean(f_stats))
            results[f'{embed_name}_anova_median_f'] = float(np.median(f_stats))
            # Geometric mean of p-values (more robust than arithmetic for small p)
            results[f'{embed_name}_anova_geomean_p'] = float(np.exp(np.mean(np.log(p_vals))))
            results[f'{embed_name}_anova_frac_significant'] = float(np.mean([p < 0.05 for p in p_vals]))
            results[f'{embed_name}_anova_n_dims_tested'] = len(f_stats)
    except ImportError:
        results[f'{embed_name}_anova_mean_f'] = 'scipy not available'
    except (ValueError, FloatingPointError) as e:
        results[f'{embed_name}_anova_mean_f'] = f'error: {e}'

    # --- PCA variance profile ---
    try:
        from sklearn.decomposition import PCA as PCA_full
        n_components = min(embed_dim, len(X), 50)
        pca = PCA_full(n_components=n_components)
        pca.fit(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        results[f'{embed_name}_pca_var_3comp'] = float(cumvar[min(2, len(cumvar)-1)])
        for threshold in [0.5, 0.9, 0.95]:
            idx = np.searchsorted(cumvar, threshold)
            if idx < len(cumvar):
                results[f'{embed_name}_pca_n_components_{int(threshold*100)}pct'] = int(idx + 1)
            else:
                results[f'{embed_name}_pca_n_components_{int(threshold*100)}pct'] = f'>{n_components}'
        results[f'{embed_name}_pca_total_components'] = embed_dim
    except ImportError as e:
        results[f'{embed_name}_pca_variance_profile'] = f'error: {e}'
    except (ValueError, np.linalg.LinAlgError) as e:
        results[f'{embed_name}_pca_variance_profile'] = f'error: {e}'

    return results


def analyze_word_pairs(
    mu_embed: torch.Tensor,
    phi_embed: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Analyze distances between related vs unrelated word pairs.
    """
    pairs = [
        ("cat", "dog", "related"),
        ("cat", "the", "unrelated"),
        ("man", "day", "unrelated"),
        ("big", "new", "unrelated"),
        ("run", "see", "related"),
        ("has", "had", "related"),
    ]

    related_mu, unrelated_mu = [], []
    related_phi, unrelated_phi = [], []
    pair_results = []

    for w1, w2, rel in pairs:
        t1 = get_token_id(w1)
        t2 = get_token_id(w2)

        if t1 is None or t2 is None:
            continue

        mu_d = dist(t1, t2, mu_embed)
        phi_d = dist(t1, t2, phi_embed) if phi_embed is not None else float('nan')

        pair_results.append({
            'word1': w1,
            'word2': w2,
            'relation': rel,
            'mu_dist': mu_d,
            'phi_dist': phi_d,
        })

        if "related" in rel:
            related_mu.append(mu_d)
            if not np.isnan(phi_d):
                related_phi.append(phi_d)
        else:
            unrelated_mu.append(mu_d)
            if not np.isnan(phi_d):
                unrelated_phi.append(phi_d)

    results = {
        'pairs': pair_results,
        'mu_related_mean': np.mean(related_mu) if related_mu else 0,
        'mu_unrelated_mean': np.mean(unrelated_mu) if unrelated_mu else 0,
    }

    if related_phi and unrelated_phi:
        results['phi_related_mean'] = np.mean(related_phi)
        results['phi_unrelated_mean'] = np.mean(unrelated_phi)
        results['phi_semantic_ratio'] = np.mean(unrelated_phi) / np.mean(related_phi)

    return results


def analyze_gauge_semantics(
    model: Any = None,
    mu_embed: Optional[torch.Tensor] = None,
    phi_embed: Optional[torch.Tensor] = None,
    step: Optional[int] = None,
    save_dir: Optional[Path] = None,
    save_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive gauge frame semantic analysis.

    Can be called with either a model object or directly with embeddings.

    Args:
        model: Model with mu_embed and phi_embed attributes
        mu_embed: Alternatively, provide mu embeddings directly
        phi_embed: Alternatively, provide phi embeddings directly
        step: Training step (for labeling plots)
        save_dir: Directory to save plots
        save_plots: Whether to generate and save plots
        verbose: Print analysis results

    Returns:
        Dictionary with analysis results
    """
    # Extract embeddings from model if provided
    if model is not None:
        # Try direct attributes first
        if hasattr(model, 'mu_embed'):
            mu_embed = model.mu_embed.weight.detach().cpu()
        # Then try nested in token_embed (GaugeTransformerLM structure)
        elif hasattr(model, 'token_embed') and hasattr(model.token_embed, 'mu_embed'):
            mu_embed = model.token_embed.mu_embed.weight.detach().cpu()

        # Same for phi_embed
        if hasattr(model, 'phi_embed'):
            phi_embed = model.phi_embed.weight.detach().cpu()
        elif hasattr(model, 'token_embed') and hasattr(model.token_embed, 'phi_embed'):
            phi_embed = model.token_embed.phi_embed.weight.detach().cpu()

    if mu_embed is None:
        return {'error': 'No mu embeddings provided'}

    # Ensure CPU tensors
    if isinstance(mu_embed, torch.Tensor) and mu_embed.device.type != 'cpu':
        mu_embed = mu_embed.detach().cpu()
    if phi_embed is not None and isinstance(phi_embed, torch.Tensor) and phi_embed.device.type != 'cpu':
        phi_embed = phi_embed.detach().cpu()

    results = {
        'step': step,
        'mu_shape': list(mu_embed.shape),
        'phi_shape': list(phi_embed.shape) if phi_embed is not None else None,
    }

    # Determine gauge group label from model config (authoritative) or phi dim (fallback)
    gauge_group_label = None
    if model is not None and hasattr(model, 'gauge_group') and hasattr(model, 'gauge_dim'):
        gauge_group_label = format_gauge_group_label(model.gauge_group, model.gauge_dim)
    if phi_embed is not None:
        if gauge_group_label is None:
            gauge_group_label = identify_gauge_group(phi_embed.shape[1])
        results['gauge_group'] = gauge_group_label

    # Token class analysis
    class_results = analyze_token_classes(mu_embed, phi_embed)
    results['token_classes'] = class_results

    # Word pair analysis
    pair_results = analyze_word_pairs(mu_embed, phi_embed)
    results['word_pairs'] = pair_results

    # Quantitative clustering metrics (full-dimensional, not PCA)
    mu_cluster = compute_clustering_metrics(mu_embed, n_tokens=500, embed_name='mu')
    results['mu_clustering'] = mu_cluster
    if phi_embed is not None:
        phi_cluster = compute_clustering_metrics(phi_embed, n_tokens=500, embed_name='phi')
        results['phi_clustering'] = phi_cluster

    if verbose:
        step_str = f" (step {step})" if step is not None else ""
        print(f"\n{'='*60}")
        print(f"GAUGE FRAME SEMANTIC ANALYSIS{step_str}")
        print(f"{'='*60}")
        print(f"mu shape: {results['mu_shape']}")
        if results['phi_shape']:
            print(f"phi shape: {results['phi_shape']} ({results.get('gauge_group', 'N/A')})")

        print(f"\nToken Class Analysis:")
        print(f"  mu inter/intra ratio: {class_results.get('mu_class_ratio', 0):.2f}x")
        if 'phi_class_ratio' in class_results:
            print(f"  phi inter/intra ratio: {class_results['phi_class_ratio']:.2f}x")
            if class_results.get('phi_shows_structure'):
                print(f"  --> phi DOES show class structure!")
            else:
                print(f"  --> phi does NOT show clear class structure")

        if 'phi_semantic_ratio' in pair_results:
            print(f"\nWord Pair Analysis:")
            print(f"  phi unrelated/related ratio: {pair_results['phi_semantic_ratio']:.2f}x")

        # Report quantitative clustering metrics
        for name, metrics in [('mu', mu_cluster)] + ([('phi', results.get('phi_clustering', {}))] if phi_embed is not None else []):
            print(f"\n  {name} Clustering Metrics (full-dimensional):")
            sil = metrics.get(f'{name}_silhouette_score')
            if isinstance(sil, float):
                print(f"    Silhouette score: {sil:.3f} (range [-1,1], higher=better)")
            ch = metrics.get(f'{name}_calinski_harabasz')
            if isinstance(ch, float):
                print(f"    Calinski-Harabasz: {ch:.1f}")
            ratio = metrics.get(f'{name}_inter_intra_ratio')
            if isinstance(ratio, float):
                print(f"    Inter/intra distance ratio: {ratio:.3f}")
            f_stat = metrics.get(f'{name}_anova_mean_f')
            if isinstance(f_stat, float):
                p_val = metrics.get(f'{name}_anova_geomean_p', 'N/A')
                frac_sig = metrics.get(f'{name}_anova_frac_significant', 0)
                print(f"    ANOVA mean F: {f_stat:.1f}, geomean p: {p_val:.2e}, {frac_sig*100:.0f}% dims significant")
            pca3 = metrics.get(f'{name}_pca_var_3comp')
            if isinstance(pca3, float):
                n50 = metrics.get(f'{name}_pca_n_components_50pct', '?')
                n90 = metrics.get(f'{name}_pca_n_components_90pct', '?')
                n95 = metrics.get(f'{name}_pca_n_components_95pct', '?')
                total = metrics.get(f'{name}_pca_total_components', '?')
                print(f"    PCA: 3 comp = {pca3*100:.1f}% var; 50%@{n50}, 90%@{n90}, 95%@{n95} of {total} dims")

    # Generate plots and save CSVs
    if save_plots:
        save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot and save phi embeddings (gauge frames)
        if phi_embed is not None:
            try:
                fig = plot_embedding_clustering(
                    phi_embed,
                    embed_type='phi',
                    step=step,
                    save_path=save_dir / f"gauge_frame_clustering{'_step'+str(step) if step is not None else ''}.png",
                    gauge_group_label=gauge_group_label,
                )
                plt.close(fig)
                results['phi_plot_saved'] = True
            except (ValueError, TypeError, OSError) as e:
                if verbose:
                    print(f"  [WARN] Could not generate phi plot: {e}")
                results['phi_plot_saved'] = False

            try:
                csv_path = save_embedding_csv(phi_embed, embed_type='phi', step=step, save_dir=save_dir)
                results['phi_csv_saved'] = str(csv_path)
            except (OSError, ValueError) as e:
                if verbose:
                    print(f"  [WARN] Could not save phi CSV: {e}")

        # Plot and save mu embeddings (beliefs)
        if mu_embed is not None:
            try:
                fig = plot_embedding_clustering(
                    mu_embed,
                    embed_type='mu',
                    step=step,
                    save_path=save_dir / f"belief_clustering{'_step'+str(step) if step is not None else ''}.png"
                )
                plt.close(fig)
                results['mu_plot_saved'] = True
            except (ValueError, TypeError, OSError) as e:
                if verbose:
                    print(f"  [WARN] Could not generate mu plot: {e}")
                results['mu_plot_saved'] = False

            try:
                csv_path = save_embedding_csv(mu_embed, embed_type='mu', step=step, save_dir=save_dir)
                results['mu_csv_saved'] = str(csv_path)
            except (OSError, ValueError) as e:
                if verbose:
                    print(f"  [WARN] Could not save mu CSV: {e}")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

CATEGORY_COLORS = {
    'letter': '#E74C3C',    # red
    'digit': '#3498DB',     # blue
    'punct': '#2ECC71',     # green
    'function': '#9B59B6',  # purple
    'content': '#F39C12',   # orange
    'other': '#95A5A6',     # gray
}


def plot_embedding_clustering(
    embed: torch.Tensor,
    embed_type: str = 'phi',
    step: Optional[int] = None,
    save_path: Optional[Path] = None,
    n_tokens: int = 500,
    gauge_group_label: Optional[str] = None,
) -> "Any":
    """
    Visualize embeddings (mu or phi) colored by token category.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim]
        embed_type: 'phi' for gauge frames, 'mu' for beliefs
        step: Training step for title
        save_path: Path to save figure
        n_tokens: Number of tokens to plot
        gauge_group_label: Override gauge group label (e.g. 'GL(30)').
                          If None and embed_type='phi', inferred from dimension.

    Returns:
        matplotlib Figure
    """
    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]

    if embed_type == 'phi':
        type_str = gauge_group_label if gauge_group_label else identify_gauge_group(embed_dim)
        title_prefix = f"{type_str} Gauge Frames"
    else:
        type_str = f"{embed_dim}D"
        title_prefix = f"Belief Embeddings (μ)"

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(len(embed_np))]
    colors = [CATEGORY_COLORS.get(c, '#95A5A6') for c in categories]

    step_str = f" (Step {step})" if step is not None else ""

    if embed_dim == 1 and embed_type == 'phi':
        # SO(2): 1D gauge frames - histogram and jittered scatter
        fig = plt.figure(figsize=(14, 6))

        # Histogram
        ax1 = fig.add_subplot(121)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                vals = embed_np[idx, 0]
                ax1.hist(vals, bins=30, alpha=0.5, label=cat, color=CATEGORY_COLORS[cat])

        ax1.set_xlabel('φ (SO(2) angle)')
        ax1.set_ylabel('Count')
        ax1.set_title(f'SO(2) Gauge Frame Distribution{step_str}')
        ax1.legend(loc='upper right', fontsize=8)

        # Jittered scatter
        ax2 = fig.add_subplot(122)
        np.random.seed(42)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                x_vals = embed_np[idx, 0]
                y_jitter = np.random.uniform(-0.4, 0.4, len(idx))
                ax2.scatter(x_vals, y_jitter, c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ (SO(2) angle)')
        ax2.set_ylabel('(jittered for visibility)')
        ax2.set_title(f'SO(2) Gauge Frames by Token Type{step_str}')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.6, 0.6)

    elif embed_dim == 3 and embed_type == 'phi':
        # SO(3): Direct 3D visualization on sphere
        fig = plt.figure(figsize=(14, 6))

        # Normalize to unit sphere
        embed_norms = np.linalg.norm(embed_np, axis=1, keepdims=True)
        embed_norms = np.clip(embed_norms, 1e-8, None)
        embed_unit = embed_np / embed_norms

        # 3D sphere plot
        ax1 = fig.add_subplot(121, projection='3d')

        # Draw unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

        # Plot points
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax1.scatter(embed_unit[idx, 0], embed_unit[idx, 1], embed_unit[idx, 2],
                           c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax1.set_xlabel('φ₁')
        ax1.set_ylabel('φ₂')
        ax1.set_zlabel('φ₃')
        ax1.set_title(f'SO(3) Gauge Frames on Unit Sphere{step_str}')
        ax1.legend(loc='upper left', fontsize=8)

        # 2D projection
        ax2 = fig.add_subplot(122)
        for cat in CATEGORY_COLORS:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax2.scatter(embed_np[idx, 0], embed_np[idx, 1],
                           c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ₁')
        ax2.set_ylabel('φ₂')
        ax2.set_title(f'SO(3) Gauge Frames (φ₁ vs φ₂){step_str}')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

    else:
        # High-dimensional: Use PCA
        n_components = min(3, embed_dim)
        pca = PCA(n_components=n_components)
        embed_pca = pca.fit_transform(embed_np)

        var_explained = pca.explained_variance_ratio_
        var_str = " + ".join([f"{v:.1%}" for v in var_explained])

        fig = plt.figure(figsize=(14, 6))

        if n_components >= 3:
            # 3D PCA plot
            ax1 = fig.add_subplot(121, projection='3d')
            for cat in CATEGORY_COLORS:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax1.scatter(embed_pca[idx, 0], embed_pca[idx, 1], embed_pca[idx, 2],
                               c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

            ax1.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax1.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax1.set_zlabel(f'PC3 ({var_explained[2]:.1%})')
            ax1.set_title(f'{title_prefix} (PCA){step_str}')
            ax1.legend(loc='upper left', fontsize=8)

            # 2D PCA plot
            ax2 = fig.add_subplot(122)
        else:
            ax2 = fig.add_subplot(111)

        if n_components >= 2:
            for cat in CATEGORY_COLORS:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax2.scatter(embed_pca[idx, 0], embed_pca[idx, 1],
                               c=CATEGORY_COLORS[cat], label=cat, alpha=0.6, s=20)

            ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax2.set_title(f'{title_prefix} (PCA from {embed_dim}D){step_str}')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# CSV Export and Animation
# =============================================================================

def save_embedding_csv(
    embed: torch.Tensor,
    embed_type: str = 'phi',
    step: Optional[int] = None,
    save_dir: Optional[Path] = None,
    n_tokens: int = 500,
) -> Path:
    """
    Save embeddings (mu or phi) to CSV for later plotting/animation.

    Args:
        embed: Embedding tensor [vocab_size, embed_dim]
        embed_type: 'phi' for gauge frames, 'mu' for beliefs
        step: Training step
        save_dir: Directory to save CSV
        n_tokens: Number of tokens to save

    Returns:
        Path to saved CSV
    """
    import pandas as pd

    save_dir = Path(save_dir) if save_dir else Path("./outputs/data")
    save_dir.mkdir(parents=True, exist_ok=True)

    embed_np = embed[:n_tokens].numpy() if isinstance(embed, torch.Tensor) else embed[:n_tokens]
    embed_dim = embed_np.shape[1]

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(len(embed_np))]

    # Build dataframe
    data = {
        'token_id': list(range(len(embed_np))),
        'category': categories,
    }
    for i in range(embed_dim):
        data[f'{embed_type}_{i}'] = embed_np[:, i]

    df = pd.DataFrame(data)

    # File naming
    if embed_type == 'phi':
        filename = f"gauge_frames{'_step'+str(step) if step is not None else ''}.csv"
    else:
        filename = f"belief_embeddings{'_step'+str(step) if step is not None else ''}.csv"

    csv_path = save_dir / filename
    df.to_csv(csv_path, index=False)

    return csv_path


# Backwards compatibility aliases
def plot_gauge_frame_clustering(phi_embed, step=None, save_path=None, n_tokens=500, gauge_group_label=None):
    """Alias for plot_embedding_clustering with embed_type='phi'."""
    return plot_embedding_clustering(phi_embed, embed_type='phi', step=step, save_path=save_path, n_tokens=n_tokens, gauge_group_label=gauge_group_label)


def save_gauge_frame_csv(phi_embed, step=None, save_dir=None, n_tokens=500):
    """Alias for save_embedding_csv with embed_type='phi'."""
    return save_embedding_csv(phi_embed, embed_type='phi', step=step, save_dir=save_dir, n_tokens=n_tokens)


def create_gauge_frame_animation(
    csv_dir: Path,
    output_path: Optional[Path] = None,
    fps: int = 2,
    dpi: int = 150,
) -> Path:
    """
    Create animated GIF from saved gauge frame CSVs.

    Uses PCA fitted on the final step and applies it to all steps
    for consistent axes across frames.

    Args:
        csv_dir: Directory containing gauge_frames_step*.csv files
        output_path: Output path for GIF (default: csv_dir/gauge_frame_evolution.gif)
        fps: Frames per second
        dpi: Resolution

    Returns:
        Path to saved GIF
    """
    import pandas as pd
    import glob
    import re

    csv_dir = Path(csv_dir)
    output_path = output_path or csv_dir / "gauge_frame_evolution.gif"

    # Find all CSV files and sort by step
    csv_files = list(csv_dir.glob("gauge_frames_step*.csv"))
    if not csv_files:
        raise ValueError(f"No gauge_frames_step*.csv files found in {csv_dir}")

    def extract_step(path):
        match = re.search(r'step(\d+)', path.name)
        return int(match.group(1)) if match else 0

    csv_files = sorted(csv_files, key=extract_step)
    steps = [extract_step(f) for f in csv_files]

    print(f"Found {len(csv_files)} CSV files: steps {steps}")

    # Load all data
    all_data = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        step = extract_step(csv_path)
        df['step'] = step
        all_data.append(df)

    # Get phi columns
    phi_cols = [c for c in all_data[0].columns if c.startswith('phi_')]
    phi_dim = len(phi_cols)

    # Fit PCA on final step
    final_phi = all_data[-1][phi_cols].values
    n_components = min(3, phi_dim)
    pca = PCA(n_components=n_components)
    pca.fit(final_phi)

    # Compute global axis limits from all steps (projected)
    all_projected = []
    for df in all_data:
        phi_vals = df[phi_cols].values
        projected = pca.transform(phi_vals)
        all_projected.append(projected)

    all_projected_concat = np.vstack(all_projected)
    x_min, x_max = all_projected_concat[:, 0].min(), all_projected_concat[:, 0].max()
    y_min, y_max = all_projected_concat[:, 1].min(), all_projected_concat[:, 1].max()

    # Add 10% padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    var_explained = pca.explained_variance_ratio_

    # Generate frames
    frames = []
    for df, projected, step in zip(all_data, all_projected, steps):
        fig, ax = plt.subplots(figsize=(10, 8))

        categories = df['category'].values

        for cat in CATEGORY_COLORS:
            mask = categories == cat
            if mask.any():
                ax.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    c=CATEGORY_COLORS[cat],
                    label=cat,
                    alpha=0.6,
                    s=30
                )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
        ax.set_title(f'Gauge Frame Evolution - Step {step}')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    # Save as GIF
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved animation to: {output_path}")
    except ImportError:
        # Fallback: save individual frames
        print("imageio not installed, saving individual frames instead")
        for i, (frame, step) in enumerate(zip(frames, steps)):
            frame_path = csv_dir / f"gauge_frame_step{step}_fixed_axes.png"
            plt.imsave(frame_path, frame)
        output_path = csv_dir / "frames"

    return output_path


# =============================================================================
# Standalone Script Mode
# =============================================================================

if __name__ == "__main__":
    import sys
    import json

    print("=" * 60)
    print("Gauge Frame Semantic Analysis")
    print("=" * 60)

    # Configuration - set these paths when running as script
    EXPERIMENT_CONFIG_PATH = "runs/your_experiment/experiment_config.json"
    CHECKPOINT_PATH = "runs/your_experiment/best_model.pt"

    if len(sys.argv) > 1:
        CHECKPOINT_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        EXPERIMENT_CONFIG_PATH = sys.argv[2]

    # Load config
    config_path = Path(EXPERIMENT_CONFIG_PATH)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded config: {config_path}")
    else:
        print(f"Config not found: {config_path}")
        config = {}

    # Load checkpoint
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Extract embeddings
    mu_embed = None
    phi_embed = None

    for key, value in state_dict.items():
        if 'mu_embed' in key and 'weight' in key:
            mu_embed = value
            print(f"Found mu_embed: {value.shape}")
        if 'phi_embed' in key and 'weight' in key:
            phi_embed = value
            print(f"Found phi_embed: {value.shape}")

    if mu_embed is None:
        print("ERROR: No mu_embed found!")
        sys.exit(1)

    # Run analysis
    save_dir = ckpt_path.parent
    results = analyze_gauge_semantics(
        mu_embed=mu_embed,
        phi_embed=phi_embed,
        save_dir=save_dir,
        save_plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
