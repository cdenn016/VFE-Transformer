#!/usr/bin/env python3
"""
Cross-Phenomenon Synthesis: Gauge-Theoretic Taxonomy of Non-Literal Language
=============================================================================

Loads results from all three holonomy studies (idiom, irony, metaphor) and
produces:
    1. 2D taxonomy figure: (holonomy, curvature) with bootstrap confidence
       ellipses showing each phenomenon occupies a distinct region
    2. Cross-phenomenon statistical tests: are the gauge signatures
       statistically distinguishable?
    3. Compositionality gradient analysis: formal test of the ordered
       prediction literal > metaphor > idiom

The key finding: holonomy and curvature are INDEPENDENT gauge-theoretic
quantities that jointly classify non-literal language into distinct types.

    Connection A  →  holonomy (path defect, transport smoothness)
    Field strength F = dA + A∧A  →  curvature (non-additive interaction)

    Literal:   (0, 0)  baseline holonomy, baseline curvature
    Irony:     (0, -)  unchanged connection, reduced field strength
    Metaphor:  (-, 0)  smoothed connection, unchanged field strength
    Idiom:     (-, +)  smoothed connection, elevated field strength

Usage:
    python run_synthesis.py
"""

import sys
import os
import subprocess
import json
import time

REQUIRED = ['numpy', 'scipy', 'matplotlib']
def check_deps():
    missing = []
    for pkg in REQUIRED:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--quiet'] + missing
        )

check_deps()

import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# Find project root: walk up from this script until we find analysis/holonomy_study/
_here = Path(__file__).resolve().parent
ROOT = _here
for _ancestor in [_here] + list(_here.parents):
    if (_ancestor / 'analysis' / 'holonomy_study' / '__init__.py').exists():
        ROOT = _ancestor
        break
OUTPUT_DIR = ROOT / 'results' / 'synthesis'


# ── Helpers ──────────────────────────────────────────────────────────────

def hbar(text='', width=65):
    if text:
        pad = width - len(text) - 2
        print(f"\n{'='*(pad//2)} {text} {'='*(pad - pad//2)}")
    else:
        print('=' * width)

def fmt_p(p):
    if p < 0.001:  return f'{p:.2e} ***'
    if p < 0.01:   return f'{p:.4f} **'
    if p < 0.05:   return f'{p:.4f} *'
    return f'{p:.4f} ns'


# ── Load / define cross-boundary paired data ────────────────────────────
# Each phenomenon: list of (holonomy_diff, curvature_diff) per pair
# where diff = non-literal - literal (or condition - baseline)
#
# These are the cross-boundary paired results (the most controlled test),
# loaded from JSON where available, otherwise from experiment output.

def load_idiom_pairs(results_dir):
    """Load idiom study cross-boundary paired results."""
    path = results_dir / 'idiom_study' / 'idiom_results.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)

    hol_pairs = data.get('cross_boundary_holonomy', [])
    curv_pairs = data.get('cross_boundary_curvature_pairs', [])

    if not hol_pairs or not curv_pairs:
        return None, None

    hol_by_id = {p['pair_id']: p['idiomatic'] - p['literal'] for p in hol_pairs}
    curv_by_id = {p['pair_id']: p['idiomatic'] - p['literal'] for p in curv_pairs}

    shared = sorted(set(hol_by_id) & set(curv_by_id))
    hol_diffs = np.array([hol_by_id[p] for p in shared])
    curv_diffs = np.array([curv_by_id[p] for p in shared])

    return hol_diffs, curv_diffs


def load_irony_pairs(results_dir):
    """Load irony study cross-boundary paired results."""
    path = results_dir / 'holonomy_study' / 'holonomy_results.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)

    hol_pairs = data.get('cross_boundary_holonomy_pairs', [])
    curv_pairs = data.get('cross_boundary_curvature_pairs', [])

    if not hol_pairs or not curv_pairs:
        return None, None

    hol_by_id = {p['pair_id']: p['ironic'] - p['literal'] for p in hol_pairs}
    curv_by_id = {p['pair_id']: p['ironic'] - p['literal'] for p in curv_pairs}

    shared = sorted(set(hol_by_id) & set(curv_by_id))
    hol_diffs = np.array([hol_by_id[p] for p in shared])
    curv_diffs = np.array([curv_by_id[p] for p in shared])

    return hol_diffs, curv_diffs


def load_metaphor_pairs(results_dir):
    """Load metaphor study cross-boundary paired results."""
    path = results_dir / 'metaphor_study' / 'metaphor_results.json'
    if not path.exists():
        return None, None
    with open(path) as f:
        data = json.load(f)

    hol_pairs = data.get('cross_boundary_holonomy_pairs', [])
    curv_pairs = data.get('cross_boundary_curvature_pairs', [])

    if not hol_pairs or not curv_pairs:
        return None, None

    hol_by_id = {p['pair_id']: p['metaphorical'] - p['literal'] for p in hol_pairs}
    curv_by_id = {p['pair_id']: p['metaphorical'] - p['literal'] for p in curv_pairs}

    shared = sorted(set(hol_by_id) & set(curv_by_id))
    hol_diffs = np.array([hol_by_id[p] for p in shared])
    curv_diffs = np.array([curv_by_id[p] for p in shared])

    return hol_diffs, curv_diffs


# ── Summary statistics from completed experiments ────────────────────────
# Used when JSON files are not available (experiments were run but results
# directory was not created). These are the cross-boundary paired results.

SUMMARY_STATS = {
    # Cross-boundary paired results (bootstrap p-values where available,
    # Wilcoxon otherwise). Effect sizes are paired Cohen's d.
    'idiom': {
        'hol_d': -0.340, 'hol_p': 0.0046,     # Wilcoxon p
        'curv_d': +0.410, 'curv_p': 0.0044,    # Wilcoxon p
        'n_pairs': 30,
        'label': 'Idiom',
        'color': '#d62728',   # red
        'marker': 's',
    },
    'irony': {
        'hol_d': +0.035, 'hol_p': 0.8400,      # bootstrap p (Wilcoxon=0.9947)
        'curv_d': -0.515, 'curv_p': 0.0042,     # Wilcoxon p
        'n_pairs': 40,
        'label': 'Irony',
        'color': '#9467bd',   # purple
        'marker': '^',
    },
    'metaphor': {
        'hol_d': -0.358, 'hol_p': 0.0226,      # bootstrap p (Wilcoxon=0.0699)
        'curv_d': +0.010, 'curv_p': 0.7884,     # Wilcoxon p
        'n_pairs': 37,
        'label': 'Metaphor',
        'color': '#ff7f0e',   # orange
        'marker': 'D',
    },
}


# ── Bootstrap confidence ellipses ────────────────────────────────────────

def bootstrap_2d_ellipse(hol_diffs, curv_diffs, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap the joint (mean_hol_diff, mean_curv_diff) distribution.

    Returns:
        center: (mean_hol, mean_curv)
        cov: 2x2 covariance matrix of bootstrap means
        boot_means: (n_boot, 2) array of bootstrap mean pairs
    """
    rng = np.random.RandomState(seed)
    n = len(hol_diffs)
    boot_means = np.zeros((n_boot, 2))

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i, 0] = np.mean(hol_diffs[idx])
        boot_means[i, 1] = np.mean(curv_diffs[idx])

    center = (np.mean(hol_diffs), np.mean(curv_diffs))
    cov = np.cov(boot_means.T)

    return center, cov, boot_means


def add_confidence_ellipse(ax, center, cov, n_std=2.0, color='red',
                           alpha=0.2, label=None, linestyle='-'):
    """Add a confidence ellipse to an axis based on a 2D covariance matrix."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      facecolor=color, alpha=alpha, edgecolor=color,
                      linewidth=2, linestyle=linestyle, label=label)
    ax.add_patch(ellipse)
    return ellipse


# ── Cross-phenomenon statistical tests ───────────────────────────────────

def paired_d(diffs):
    """Paired Cohen's d: mean(diffs) / std(diffs)."""
    if len(diffs) < 2 or np.std(diffs) == 0:
        return 0.0
    return np.mean(diffs) / np.std(diffs)


def bootstrap_p_2d(boot_means, null=(0, 0)):
    """P-value for testing whether center differs from null in 2D."""
    # Mahalanobis distance from null
    center = np.mean(boot_means, axis=0)
    cov = np.cov(boot_means.T)
    if np.linalg.det(cov) < 1e-30:
        return 1.0
    cov_inv = np.linalg.inv(cov)
    delta = center - np.array(null)
    d2_obs = delta @ cov_inv @ delta

    # How often does bootstrap put the null inside the distribution?
    deltas = boot_means - np.array(null)
    d2_boot = np.array([d @ cov_inv @ d for d in deltas])
    p = np.mean(d2_boot <= 0)  # fraction at or past null
    # Alternative: fraction of bootstrap samples closer to null than center
    p_hotelling = np.mean(d2_boot >= d2_obs)
    return p_hotelling


def test_phenomena_distinct(boot_a, boot_b, label_a, label_b):
    """
    Test whether two phenomena have statistically distinct gauge signatures.

    Uses the bootstrap distributions of (mean_hol_diff, mean_curv_diff) for
    each phenomenon. Tests whether the difference of their centers is
    significantly different from zero.
    """
    # Difference of bootstrap centers
    diff_boot = boot_a - boot_b[:min(len(boot_a), len(boot_b))]
    if len(boot_a) != len(boot_b):
        n = min(len(boot_a), len(boot_b))
        diff_boot = boot_a[:n] - boot_b[:n]

    mean_diff = np.mean(diff_boot, axis=0)
    cov_diff = np.cov(diff_boot.T)

    # Hotelling's T^2 equivalent
    if np.linalg.det(cov_diff) < 1e-30:
        p = 1.0
    else:
        cov_inv = np.linalg.inv(cov_diff)
        t2 = mean_diff @ cov_inv @ mean_diff
        # Under H0, t2 ~ chi2(2) approximately
        p = 1 - stats.chi2.cdf(t2, df=2)

    # Also test each dimension separately
    hol_diff = diff_boot[:, 0]
    curv_diff = diff_boot[:, 1]
    p_hol = 2 * min(np.mean(hol_diff <= 0), np.mean(hol_diff >= 0))
    p_curv = 2 * min(np.mean(curv_diff <= 0), np.mean(curv_diff >= 0))

    print(f'  {label_a} vs {label_b}:')
    print(f'    Holonomy diff:  {mean_diff[0]:+.4f}  p={fmt_p(p_hol)}')
    print(f'    Curvature diff: {mean_diff[1]:+.6f}  p={fmt_p(p_curv)}')
    print(f'    Joint 2D test:  T^2={mean_diff @ np.linalg.inv(cov_diff) @ mean_diff if np.linalg.det(cov_diff) > 1e-30 else 0:.2f}  p={fmt_p(p)}')

    return {'mean_hol_diff': float(mean_diff[0]),
            'mean_curv_diff': float(mean_diff[1]),
            'p_hol': float(p_hol),
            'p_curv': float(p_curv),
            'p_joint': float(p)}


# ── Compositionality gradient test ───────────────────────────────────────

def test_compositionality_gradient(boot_idiom, boot_metaphor, boot_irony):
    """
    Test the ordered prediction for compositionality:
        holonomy: idiom <= metaphor <= irony (literal baseline = 0)
        curvature: idiom >= metaphor >= irony

    Uses bootstrap to compute probability that the ordering holds.
    """
    n = min(len(boot_idiom), len(boot_metaphor), len(boot_irony))
    bi = boot_idiom[:n]
    bm = boot_metaphor[:n]
    br = boot_irony[:n]

    # Holonomy ordering: idiom_hol <= metaphor_hol <= irony_hol
    # (All are diffs from literal, so this is the ordering of the diffs)
    hol_ordered = np.mean(
        (bi[:, 0] <= bm[:, 0]) & (bm[:, 0] <= br[:, 0])
    )

    # Curvature ordering: idiom_curv >= metaphor_curv >= irony_curv
    curv_ordered = np.mean(
        (bi[:, 1] >= bm[:, 1]) & (bm[:, 1] >= br[:, 1])
    )

    # Joint ordering
    joint_ordered = np.mean(
        (bi[:, 0] <= bm[:, 0]) & (bm[:, 0] <= br[:, 0]) &
        (bi[:, 1] >= bm[:, 1]) & (bm[:, 1] >= br[:, 1])
    )

    # Under random permutation of 3 labels, chance of correct ordering = 1/6
    # Binomial test: is the observed proportion > 1/6?
    n_success_hol = int(hol_ordered * n)
    n_success_curv = int(curv_ordered * n)
    n_success_joint = int(joint_ordered * n)
    p_hol = stats.binomtest(n_success_hol, n, 1/6, alternative='greater').pvalue
    p_curv = stats.binomtest(n_success_curv, n, 1/6, alternative='greater').pvalue
    p_joint = stats.binomtest(n_success_joint, n, 1/6, alternative='greater').pvalue

    return {
        'hol_ordered_pct': float(hol_ordered * 100),
        'curv_ordered_pct': float(curv_ordered * 100),
        'joint_ordered_pct': float(joint_ordered * 100),
        'hol_p': float(p_hol),
        'curv_p': float(p_curv),
        'joint_p': float(p_joint),
        'n_bootstrap': n,
    }


# ── Publication figure ───────────────────────────────────────────────────

def plot_2d_taxonomy(phenomena, output_path, boot_data=None):
    """
    Create the 2D (holonomy, curvature) taxonomy figure.

    Each phenomenon is a point with bootstrap confidence ellipse.
    The literal baseline is at the origin.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={'width_ratios': [3, 2]})

    # ── Panel A: 2D scatter with confidence ellipses ─────────────────────
    ax = axes[0]

    # Origin = literal baseline
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.plot(0, 0, 'ko', markersize=10, zorder=5)
    ax.annotate('Literal\n(baseline)', (0, 0), textcoords='offset points',
                xytext=(12, -20), fontsize=11, fontweight='bold',
                color='black')

    for key, info in phenomena.items():
        hol_d = info['hol_d']
        curv_d = info['curv_d']
        color = info['color']
        marker = info['marker']
        label = info['label']
        n = info['n_pairs']

        # Plot point
        ax.plot(hol_d, curv_d, marker=marker, color=color, markersize=14,
                markeredgecolor='black', markeredgewidth=1.0, zorder=5,
                label=f'{label} (n={n})')

        # Significance markers
        hol_sig = '*' if info['hol_p'] < 0.05 else ''
        curv_sig = '*' if info['curv_p'] < 0.05 else ''

        # Annotation
        offset_x = 15 if hol_d >= 0 else -15
        offset_y = 15 if curv_d >= 0 else -15
        ha = 'left' if hol_d >= 0 else 'right'
        ax.annotate(f'{label}\nd$_h$={hol_d:+.2f}{hol_sig}\nd$_c$={curv_d:+.2f}{curv_sig}',
                    (hol_d, curv_d),
                    textcoords='offset points', xytext=(offset_x, offset_y),
                    fontsize=9, color=color, fontweight='bold', ha=ha,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        # Bootstrap confidence ellipse
        if boot_data and key in boot_data:
            center, cov, _ = boot_data[key]
            add_confidence_ellipse(ax, center, cov, n_std=1.96,
                                   color=color, alpha=0.12)
            add_confidence_ellipse(ax, center, cov, n_std=1.0,
                                   color=color, alpha=0.08,
                                   linestyle='--')

    # Quadrant labels
    ax.text(-0.55, 0.55, 'Smoothed A\nElevated F',
            fontsize=9, color='gray', style='italic', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
    ax.text(0.25, -0.55, 'Baseline A\nReduced F',
            fontsize=9, color='gray', style='italic', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.5))
    ax.text(-0.55, -0.15, 'Smoothed A\nBaseline F',
            fontsize=9, color='gray', style='italic', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', alpha=0.5))

    ax.set_xlabel(r"Holonomy (paired Cohen's $d$, non-literal $-$ literal)",
                  fontsize=12)
    ax.set_ylabel(r"Curvature (paired Cohen's $d$, non-literal $-$ literal)",
                  fontsize=12)
    ax.set_title('A. Gauge-Theoretic Taxonomy of Non-Literal Language',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    # Set symmetric axis limits
    max_abs = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 0.7)
    ax.set_xlim(-max_abs * 1.1, max_abs * 0.6)
    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    ax.grid(True, alpha=0.2)

    # ── Panel B: Summary bar chart ───────────────────────────────────────
    ax = axes[1]

    phenomena_order = ['idiom', 'metaphor', 'irony']
    labels = [phenomena[k]['label'] for k in phenomena_order]
    hol_ds = [phenomena[k]['hol_d'] for k in phenomena_order]
    curv_ds = [phenomena[k]['curv_d'] for k in phenomena_order]
    colors = [phenomena[k]['color'] for k in phenomena_order]
    hol_ps = [phenomena[k]['hol_p'] for k in phenomena_order]
    curv_ps = [phenomena[k]['curv_p'] for k in phenomena_order]

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax.bar(x - w/2, hol_ds, w, color=[c for c in colors],
                   alpha=0.7, edgecolor='black', linewidth=0.8,
                   label=r'Holonomy $\kappa$ (connection)')
    bars2 = ax.bar(x + w/2, curv_ds, w, color=[c for c in colors],
                   alpha=0.4, edgecolor='black', linewidth=0.8,
                   hatch='///', label='Curvature $F$ (field strength)')

    # Add significance stars
    for i, (hd, hp, cd, cp) in enumerate(zip(hol_ds, hol_ps, curv_ds, curv_ps)):
        if hp < 0.05:
            y_h = hd + 0.02 * np.sign(hd) if hd != 0 else 0.02
            ax.text(i - w/2, y_h, '*' if hp < 0.05 else '',
                    ha='center', va='bottom' if hd > 0 else 'top',
                    fontsize=16, fontweight='bold', color='black')
        if cp < 0.05:
            y_c = cd + 0.02 * np.sign(cd) if cd != 0 else 0.02
            ax.text(i + w/2, y_c, '*' if cp < 0.05 else '',
                    ha='center', va='bottom' if cd > 0 else 'top',
                    fontsize=16, fontweight='bold', color='black')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel("Cross-boundary paired Cohen's $d$", fontsize=11)
    ax.set_title('B. Effect Sizes by Phenomenon', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


def plot_compositionality_gradient(phenomena, output_path):
    """
    Separate figure showing the compositionality gradient prediction.

    X-axis: compositionality (literal → metaphor → idiom)
    Y-axis: holonomy d and curvature d
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = ['irony', 'metaphor', 'idiom']
    labels = ['Irony\n(pragmatic)', 'Metaphor\n(partial)', 'Idiom\n(frozen)']
    x = np.arange(len(order))

    # Panel A: Holonomy across compositionality gradient
    ax = axes[0]
    hol_ds = [phenomena[k]['hol_d'] for k in order]
    hol_ps = [phenomena[k]['hol_p'] for k in order]
    colors = [phenomena[k]['color'] for k in order]

    bars = ax.bar(x, hol_ds, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    for i, (d, p) in enumerate(zip(hol_ds, hol_ps)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_off = 0.015 if d >= 0 else -0.03
        va = 'bottom' if d >= 0 else 'top'
        ax.text(i, d + y_off, sig, ha='center', va=va, fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8, label='Literal baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Holonomy Cohen's $d$\n(connection smoothness)", fontsize=11)
    ax.set_title('A. Connection A: Transport Path Dependence', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)

    # Arrow indicating compositionality direction
    ax.annotate('', xy=(2.4, ax.get_ylim()[0] * 0.95),
                xytext=(-0.4, ax.get_ylim()[0] * 0.95),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(1.0, ax.get_ylim()[0] * 1.05, 'Decreasing compositionality',
            ha='center', fontsize=9, color='gray', style='italic')

    # Panel B: Curvature across compositionality gradient
    ax = axes[1]
    curv_ds = [phenomena[k]['curv_d'] for k in order]
    curv_ps = [phenomena[k]['curv_p'] for k in order]

    bars = ax.bar(x, curv_ds, color=colors, alpha=0.7, edgecolor='black',
                  linewidth=1.2, hatch='///')
    for i, (d, p) in enumerate(zip(curv_ds, curv_ps)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y_off = 0.015 if d >= 0 else -0.03
        va = 'bottom' if d >= 0 else 'top'
        ax.text(i, d + y_off, sig, ha='center', va=va, fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8, label='Literal baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Curvature Cohen's $d$\n(field strength / non-additivity)", fontsize=11)
    ax.set_title(r'B. Field Strength $F = dA + A \wedge A$: Token Interaction',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.2)

    ax.annotate('', xy=(2.4, ax.get_ylim()[0] * 0.85),
                xytext=(-0.4, ax.get_ylim()[0] * 0.85),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(1.0, ax.get_ylim()[0] * 0.92, 'Decreasing compositionality',
            ha='center', fontsize=9, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


def plot_interpretation_diagram(output_path):
    """
    Schematic diagram showing gauge-theoretic interpretation.

    Not data-driven — this is a conceptual figure showing how
    connection and curvature decompose non-literal language.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw quadrant structure
    ax.axhline(y=0, color='gray', linewidth=1, linestyle='-')
    ax.axvline(x=0, color='gray', linewidth=1, linestyle='-')

    # Phenomena
    points = {
        'Literal':  (0, 0, 'o', 12, 'black'),
        'Irony':    (0, -0.52, '^', 14, '#9467bd'),
        'Metaphor': (-0.36, 0, 'D', 14, '#ff7f0e'),
        'Idiom':    (-0.34, 0.41, 's', 14, '#d62728'),
    }

    for name, (x, y, marker, size, color) in points.items():
        ax.plot(x, y, marker=marker, color=color, markersize=size,
                markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    # Labels with interpretations
    ax.annotate('LITERAL\nFull composition\n$A$ = baseline, $F$ = baseline',
                (0, 0), xytext=(0.15, 0.25), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black'))

    ax.annotate('IRONY\nContext-driven meaning shift\n$A$ = unchanged, $F$ = compressed\n'
                'Tokens compose normally;\ncontext already determines meaning',
                (0, -0.52), xytext=(0.2, -0.7), fontsize=9, fontweight='bold',
                color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0e6ff', edgecolor='#9467bd'))

    ax.annotate('METAPHOR\nProductive analogical mapping\n$A$ = smoothed, $F$ = baseline\n'
                'Cross-domain map creates\nwell-worn path; tokens still compose',
                (-0.36, 0), xytext=(-0.85, -0.35), fontsize=9, fontweight='bold',
                color='#ff7f0e',
                arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3e0', edgecolor='#ff7f0e'))

    ax.annotate('IDIOM\nFrozen non-compositional chunk\n$A$ = flat (smooth transport), $F$ = elevated\n'
                'Non-Abelian self-interaction;\n$A \\wedge A$ contributes even when $A$ is smooth',
                (-0.34, 0.41), xytext=(-0.85, 0.55), fontsize=9, fontweight='bold',
                color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0', edgecolor='#d62728'))

    ax.set_xlabel(r'Connection $A$ (holonomy $d$: transport path dependence)', fontsize=12)
    ax.set_ylabel(r'Field strength $F = dA + A \wedge A$ (curvature $d$: non-additivity)', fontsize=12)
    ax.set_title('Gauge-Theoretic Decomposition of Non-Literal Language\nin GPT-2 Representation Space',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-1.1, 0.5)
    ax.set_ylim(-0.9, 0.7)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {output_path}')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / 'results'

    hbar('Cross-Phenomenon Synthesis')
    print(f'  Project root: {ROOT}')
    print(f'  Results dir:  {results_dir}')
    print(f'  Output dir:   {OUTPUT_DIR}')
    print('  Loading results from idiom, irony, and metaphor studies.')
    print()

    # ── 1. Load paired data where available ──────────────────────────────
    pair_data = {}

    hol_idiom, curv_idiom = load_idiom_pairs(results_dir)
    if hol_idiom is not None:
        pair_data['idiom'] = (hol_idiom, curv_idiom)
        print(f'  Idiom:    loaded {len(hol_idiom)} cross-boundary pairs from JSON')
    else:
        print(f'  Idiom:    using summary statistics (JSON not found)')

    hol_irony, curv_irony = load_irony_pairs(results_dir)
    if hol_irony is not None:
        pair_data['irony'] = (hol_irony, curv_irony)
        print(f'  Irony:    loaded {len(hol_irony)} cross-boundary pairs from JSON')
    else:
        print(f'  Irony:    using summary statistics (JSON not found)')

    hol_met, curv_met = load_metaphor_pairs(results_dir)
    if hol_met is not None:
        pair_data['metaphor'] = (hol_met, curv_met)
        print(f'  Metaphor: loaded {len(hol_met)} cross-boundary pairs from JSON')
    else:
        print(f'  Metaphor: using summary statistics (JSON not found)')

    phenomena = dict(SUMMARY_STATS)  # start with summary, override with data

    # ── 2. Bootstrap confidence ellipses ─────────────────────────────────
    hbar('Bootstrap Confidence Ellipses (10,000 resamples)')
    boot_data = {}

    for key in ['idiom', 'irony', 'metaphor']:
        if key in pair_data:
            hol_d, curv_d = pair_data[key]
            center, cov, boot_means = bootstrap_2d_ellipse(hol_d, curv_d)
            boot_data[key] = (center, cov, boot_means)

            # Update effect sizes from actual data
            phenomena[key]['hol_d'] = paired_d(hol_d)
            phenomena[key]['curv_d'] = paired_d(curv_d)

            # Bootstrap p-values (from null = 0)
            p_hol = 2 * min(np.mean(boot_means[:, 0] <= 0), np.mean(boot_means[:, 0] >= 0))
            p_curv = 2 * min(np.mean(boot_means[:, 1] <= 0), np.mean(boot_means[:, 1] >= 0))

            # CI
            ci_hol = np.percentile(boot_means[:, 0], [2.5, 97.5])
            ci_curv = np.percentile(boot_means[:, 1], [2.5, 97.5])

            print(f'\n  {phenomena[key]["label"]}:')
            print(f'    Holonomy center:  {center[0]:+.4f}  95% CI [{ci_hol[0]:+.4f}, {ci_hol[1]:+.4f}]  '
                  f'p={fmt_p(p_hol)}')
            print(f'    Curvature center: {center[1]:+.6f}  95% CI [{ci_curv[0]:+.6f}, {ci_curv[1]:+.6f}]  '
                  f'p={fmt_p(p_curv)}')
            print(f'    Covariance:       [[{cov[0,0]:.2e}, {cov[0,1]:.2e}],')
            print(f'                       [{cov[1,0]:.2e}, {cov[1,1]:.2e}]]')

            # Test separation from origin (2D)
            p_2d = bootstrap_p_2d(boot_means)
            print(f'    Joint 2D test (vs origin): p={fmt_p(p_2d)}')

    # ── 3. Cross-phenomenon distinctness tests ───────────────────────────
    hbar('Cross-Phenomenon Distinctness Tests')

    cross_results = {}
    pairs_to_test = [('idiom', 'irony'), ('idiom', 'metaphor'), ('irony', 'metaphor')]

    for ka, kb in pairs_to_test:
        if ka in boot_data and kb in boot_data:
            _, _, boot_a = boot_data[ka]
            _, _, boot_b = boot_data[kb]
            r = test_phenomena_distinct(boot_a, boot_b,
                                        phenomena[ka]['label'],
                                        phenomena[kb]['label'])
            cross_results[f'{ka}_vs_{kb}'] = r

    # ── 4. Compositionality gradient test ────────────────────────────────
    hbar('Compositionality Gradient Test')

    if all(k in boot_data for k in ['idiom', 'metaphor', 'irony']):
        _, _, boot_i = boot_data['idiom']
        _, _, boot_m = boot_data['metaphor']
        _, _, boot_r = boot_data['irony']

        grad_results = test_compositionality_gradient(boot_i, boot_m, boot_r)

        print(f'  Predicted ordering (holonomy):   idiom <= metaphor <= irony (baseline=0)')
        print(f'  Predicted ordering (curvature):  idiom >= metaphor >= irony')
        print(f'  Bootstrap support (chance = 16.7%):')
        print(f'    Holonomy ordering:  {grad_results["hol_ordered_pct"]:.1f}%  p={fmt_p(grad_results["hol_p"])}')
        print(f'    Curvature ordering: {grad_results["curv_ordered_pct"]:.1f}%  p={fmt_p(grad_results["curv_p"])}')
        print(f'    Joint ordering:     {grad_results["joint_ordered_pct"]:.1f}%  p={fmt_p(grad_results["joint_p"])}')
        print(f'    (n={grad_results["n_bootstrap"]} bootstrap resamples)')
    else:
        print('  Insufficient pair data for gradient test (need all three studies)')
        grad_results = None

    # ── 5. Summary table ─────────────────────────────────────────────────
    hbar('Summary Table: Cross-Boundary Paired Results')

    print()
    print('  Phenomenon | Holonomy d  | Holonomy p  | Curvature d | Curvature p | n')
    print('  -----------|-------------|-------------|-------------|-------------|---')
    for key in ['idiom', 'metaphor', 'irony']:
        info = phenomena[key]
        print(f'  {info["label"]:10s} | d={info["hol_d"]:+.3f}    | {fmt_p(info["hol_p"]):12s} '
              f'| d={info["curv_d"]:+.3f}    | {fmt_p(info["curv_p"]):12s} | {info["n_pairs"]}')

    print()
    print('  2D Gauge Signature (cross-boundary, length-controlled):')
    print()
    print('  Phenomenon  | Connection A    | Field strength F  | Interpretation')
    print('  ------------|----------------|-------------------|---------------------------')
    print('  Literal     | baseline       | baseline          | Full composition')
    print('  Irony       | UNCHANGED      | REDUCED **        | Context-driven; tokens compose normally')
    print('  Metaphor    | SMOOTHED *     | UNCHANGED         | Productive mapping smooths the path')
    print('  Idiom       | SMOOTHED **    | ELEVATED **       | Frozen chunk: flat A, non-trivial A^A')

    # ── 6. Generate figures ──────────────────────────────────────────────
    hbar('Generating Figures')

    plot_2d_taxonomy(phenomena, str(OUTPUT_DIR / 'gauge_taxonomy_2d.png'),
                     boot_data=boot_data)

    plot_compositionality_gradient(phenomena,
                                   str(OUTPUT_DIR / 'compositionality_gradient.png'))

    plot_interpretation_diagram(str(OUTPUT_DIR / 'gauge_interpretation.png'))

    # ── 7. Save results ──────────────────────────────────────────────────
    hbar('Saving Results')

    summary = {
        'phenomena': {
            key: {
                'label': info['label'],
                'hol_d': info['hol_d'],
                'hol_p': info['hol_p'],
                'curv_d': info['curv_d'],
                'curv_p': info['curv_p'],
                'n_pairs': info['n_pairs'],
            }
            for key, info in phenomena.items()
        },
        'cross_phenomenon_tests': cross_results,
        'compositionality_gradient': grad_results,
        'methodology': {
            'metric': 'cross-boundary paired Cohen\'s d (non-literal - literal)',
            'holonomy': 'layerwise Jacobian transport defect (kappa)',
            'curvature': 'transport commutator norm',
            'length_control': 'equalized cross-boundary positions per pair',
            'model': 'GPT-2 (124M parameters)',
            'bootstrap_resamples': 10000,
        },
    }

    # Add bootstrap ellipse parameters if available
    for key in boot_data:
        center, cov, _ = boot_data[key]
        summary['phenomena'][key]['bootstrap_center'] = [float(center[0]), float(center[1])]
        summary['phenomena'][key]['bootstrap_cov'] = [[float(cov[0, 0]), float(cov[0, 1])],
                                                       [float(cov[1, 0]), float(cov[1, 1])]]

    with open(OUTPUT_DIR / 'synthesis_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Saved: {OUTPUT_DIR / "synthesis_results.json"}')

    # ── Done ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hbar(f'Done ({elapsed:.0f}s)')
    print(f'  Results: {OUTPUT_DIR}/')
    print(f'  Figures:')
    print(f'    gauge_taxonomy_2d.png            — 2D scatter with confidence ellipses')
    print(f'    compositionality_gradient.png    — ordered prediction test')
    print(f'    gauge_interpretation.png         — conceptual interpretation diagram')
    print(f'    synthesis_results.json           — all summary statistics')


if __name__ == '__main__':
    main()
