#!/usr/bin/env python3
"""
Holonomy Study: Ironic Language as Gauge Curvature
====================================================

Measures gauge curvature in pretrained GPT-2 to test whether ironic language
exhibits different holonomy and curvature than literal language.

Three measurement scales:
    1. Whole-sentence: holonomy and curvature over all tokens
    2. Phrase-localized: restrict to target phrase tokens only (length control)
    3. Cross-boundary: target phrase + nearby context tokens (interaction)

Key finding from idiom study: expect a DOUBLE DISSOCIATION
    - Holonomy (path defect) may decrease for ironic usage (smooth transport)
    - Curvature (superposition violation) may increase (non-additive interaction)

Usage:
    python run_holonomy_study.py
"""

import sys
import os
import time
import subprocess

# ── Dependencies ──────────────────────────────────────────────────────────

REQUIRED = ['torch', 'transformers', 'scipy', 'matplotlib', 'numpy']

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

import torch
import numpy as np
from pathlib import Path

# Find project root: walk up from this script until we find analysis/holonomy_study/__init__.py
_here = Path(__file__).resolve().parent
ROOT = _here
for _ancestor in [_here] + list(_here.parents):
    if (_ancestor / 'analysis' / 'holonomy_study' / '__init__.py').exists():
        ROOT = _ancestor
        break
sys.path.insert(0, str(ROOT))

from analysis.holonomy_study.transport import (
    load_model,
    attention_flow_asymmetry,
    layerwise_jacobian_holonomy,
    discrete_curvature,
)
from analysis.holonomy_study.holonomy import HolonomyResult
from analysis.holonomy_study.datasets import load_irony_pairs, by_label, get_paired_only

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json


# ── Config ────────────────────────────────────────────────────────────────

MODEL_NAME  = 'gpt2'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TRI     = 300          # triangles per sentence
MAX_PAIRS   = 150          # pairs for curvature
OUTPUT_DIR  = ROOT / 'results' / 'holonomy_study'

COLORS = {
    'ironic':  '#d62728',   # red
    'literal': '#2ca02c',   # green
    'control': '#1f77b4',   # blue
}


# ── Helpers ───────────────────────────────────────────────────────────────

def hbar(text='', width=60):
    if text:
        pad = width - len(text) - 2
        print(f"\n{'='*(pad//2)} {text} {'='*(pad - pad//2)}")
    else:
        print('=' * width)

def tokenize(tokenizer, text):
    return torch.tensor([tokenizer.encode(text)], device=DEVICE)

def fmt_p(p):
    if p < 0.001:  return f'{p:.2e} ***'
    if p < 0.01:   return f'{p:.4f} **'
    if p < 0.05:   return f'{p:.4f} *'
    return f'{p:.4f} ns'


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction. Returns array of q-values."""
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([])
    ranked = np.argsort(p_arr)
    q = np.zeros(n)
    for i, rank_idx in enumerate(ranked):
        q[rank_idx] = p_arr[rank_idx] * n / (i + 1)
    # Enforce monotonicity: q[i] = min(q[i], q[i+1])
    for i in range(n - 2, -1, -1):
        idx = ranked[i]
        idx_next = ranked[i + 1]
        q[idx] = min(q[idx], q[idx_next])
    return np.clip(q, 0, 1)


def find_phrase_positions(tokenizer, full_text, phrase):
    """
    Find the token positions of a target phrase within a tokenized sentence.

    Handles GPT-2 BPE convention where mid-sentence tokens have leading space.
    Returns list of token indices, or None if phrase not found.
    """
    full_ids = tokenizer.encode(full_text)

    # Try several BPE encodings of the phrase
    candidates = []
    try:
        candidates.append(tokenizer.encode(' ' + phrase))   # mid-sentence (Ġ prefix)
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        pass
    try:
        candidates.append(tokenizer.encode(phrase))          # start-of-text
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        pass
    # Also try with common preceding contexts
    for prefix in ['. ', ', ', '  ']:
        try:
            enc = tokenizer.encode(prefix + phrase)
            prefix_enc = tokenizer.encode(prefix)
            # The phrase tokens are everything after the prefix tokens
            if len(enc) > len(prefix_enc):
                candidates.append(enc[len(prefix_enc):])
        except Exception:
            pass

    for phrase_ids in candidates:
        n = len(phrase_ids)
        if n == 0:
            continue
        for i in range(len(full_ids) - n + 1):
            if full_ids[i:i+n] == phrase_ids:
                return list(range(i, i + n))

    # Fallback: character-level matching via incremental decode
    try:
        text_lower = full_text.lower()
        phrase_lower = phrase.lower()
        char_start = text_lower.find(phrase_lower)
        if char_start < 0:
            return None
        char_end = char_start + len(phrase)

        # Decode tokens incrementally to map token → character positions
        positions = []
        current_char = 0
        for tok_idx in range(len(full_ids)):
            tok_text = tokenizer.decode([full_ids[tok_idx]])
            tok_char_start = current_char
            tok_char_end = current_char + len(tok_text)
            if tok_char_end > char_start and tok_char_start < char_end:
                positions.append(tok_idx)
            current_char = tok_char_end
        if positions:
            return positions
    except (KeyError, TypeError, UnicodeDecodeError, ValueError):
        pass

    return None


def cross_boundary_positions(phrase_pos, n_tokens, context_window=3):
    """
    Generate positions for cross-boundary holonomy measurement.

    Returns positions that include the phrase tokens PLUS nearby context tokens
    (up to context_window tokens on each side). Triples sampled from these
    positions will naturally span the phrase-context boundary.

    Also returns the phrase_set for filtering purely-internal triples.
    """
    phrase_set = set(phrase_pos)
    min_pos = max(0, min(phrase_pos) - context_window)
    max_pos = min(n_tokens - 1, max(phrase_pos) + context_window)
    all_pos = list(range(min_pos, max_pos + 1))
    return all_pos, phrase_set


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_distributions(kappas_by_label, title, ylabel, output_path):
    """Violin + strip + histogram for any metric."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels_order = ['literal', 'ironic', 'control']

    # Left: violin + strip
    ax = axes[0]
    positions, data, colors, tick_labels = [], [], [], []
    for idx, label in enumerate(labels_order):
        if label in kappas_by_label and len(kappas_by_label[label]) > 0:
            positions.append(idx)
            data.append(kappas_by_label[label])
            colors.append(COLORS[label])
            tick_labels.append(f"{label}\n(n={len(kappas_by_label[label])})")

    if data:
        parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
        parts['cmeans'].set_color('black')
        parts['cmedians'].set_color('gray')
        for i, (pos, d) in enumerate(zip(positions, data)):
            jitter = np.random.RandomState(42).uniform(-0.1, 0.1, size=len(d))
            ax.scatter(pos + jitter, d, c=colors[i], alpha=0.5, s=15, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Right: histogram
    ax = axes[1]
    all_vals = np.concatenate([v for v in kappas_by_label.values() if len(v) > 0])
    bins = np.linspace(np.min(all_vals), np.max(all_vals), 30)
    for label in labels_order:
        if label in kappas_by_label and len(kappas_by_label[label]) > 0:
            ax.hist(kappas_by_label[label], bins=bins, alpha=0.4,
                    color=COLORS[label], label=label, density=True)
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Density')
    ax.set_title(f'{title} (Histogram)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_layer_profiles(profiles_by_label, title, output_path, n_layers=12):
    """Plot per-layer kappa or curvature as a function of depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))

    for label in ['literal', 'ironic', 'control']:
        if label not in profiles_by_label:
            continue
        all_profiles = profiles_by_label[label]
        if not all_profiles:
            continue

        # Individual traces
        for prof in all_profiles:
            ax.plot(layers, prof[:n_layers], color=COLORS[label], alpha=0.08, linewidth=0.8)

        # Mean + SEM
        stacked = np.array(all_profiles)
        mean = np.nanmean(stacked, axis=0)[:n_layers]
        sem = np.nanstd(stacked, axis=0)[:n_layers] / np.sqrt(len(all_profiles))
        ax.plot(layers, mean, color=COLORS[label], linewidth=2.5, label=f'{label} (n={len(all_profiles)})')
        ax.fill_between(layers, mean - sem, mean + sem, color=COLORS[label], alpha=0.15)

    ax.set_xlabel('Layer (VFE step)')
    ax.set_ylabel(r'Mean $\kappa$')
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_curvature_layer_profiles(profiles_by_label, output_path, n_layers=12):
    """Plot per-layer curvature (commutator) as a function of depth."""
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))

    for label in ['literal', 'ironic', 'control']:
        if label not in profiles_by_label:
            continue
        all_profiles = profiles_by_label[label]
        if not all_profiles:
            continue

        for prof in all_profiles:
            ax.plot(layers, prof[:n_layers], color=COLORS[label], alpha=0.08, linewidth=0.8)

        stacked = np.array(all_profiles)
        mean = np.nanmean(stacked, axis=0)[:n_layers]
        sem = np.nanstd(stacked, axis=0)[:n_layers] / np.sqrt(len(all_profiles))
        ax.plot(layers, mean, color=COLORS[label], linewidth=2.5, label=f'{label} (n={len(all_profiles)})')
        ax.fill_between(layers, mean - sem, mean + sem, color=COLORS[label], alpha=0.15)

    ax.set_xlabel('Layer (VFE step)')
    ax.set_ylabel('Curvature (commutator norm)')
    ax.set_title('Discrete Riemann Curvature by Layer')
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_paired_comparison(ironic_by_pair, literal_by_pair, shared, targets_by_pair, ylabel, title, output_path):
    """Paired lines connecting ironic vs literal for same target phrase."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ival = [ironic_by_pair[p] for p in shared]
    lval = [literal_by_pair[p] for p in shared]

    for iv, lv in zip(ival, lval):
        color = COLORS['ironic'] if iv > lv else COLORS['literal']
        ax.plot([0, 1], [lv, iv], color=color, alpha=0.4, linewidth=1)

    ax.scatter([0]*len(lval), lval, c=COLORS['literal'], s=40, zorder=3, label='literal')
    ax.scatter([1]*len(ival), ival, c=COLORS['ironic'], s=40, zorder=3, label='ironic')

    n_higher = sum(1 for i, l in zip(ival, lval) if i > l)
    pct = 100 * n_higher / len(shared) if shared else 0

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Literal context', 'Ironic context'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nIronic > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ── Statistical tests ─────────────────────────────────────────────────────

def run_stats(kappas, label_a, label_b, metric_name='kappa'):
    """Run Mann-Whitney U + effect sizes."""
    if label_a not in kappas or label_b not in kappas:
        return None
    a, b = kappas[label_a], kappas[label_b]
    if len(a) < 2 or len(b) < 2:
        return None
    U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled = np.sqrt((np.var(a) + np.var(b)) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0
    r = 1 - (2*U) / (len(a)*len(b))
    print(f'  {label_a} vs {label_b} ({metric_name}): U={U:.0f}  p={fmt_p(p)}  d={d:+.3f}  r={r:+.3f}')
    return {'U': float(U), 'p_value': float(p), 'cohens_d': float(d), 'rank_biserial_r': float(r)}


def run_permutation_test(a, b, label_a, label_b, metric_name='kappa', n_perm=10000, seed=42):
    """Permutation test for difference in means."""
    rng = np.random.RandomState(seed)
    obs_diff = np.mean(a) - np.mean(b)
    pooled = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(np.mean(pooled[:na]) - np.mean(pooled[na:])) >= abs(obs_diff):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    print(f'  {label_a} vs {label_b} ({metric_name}): obs_diff={obs_diff:+.4f}  p_perm={fmt_p(p_perm)}')
    return float(p_perm)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────
    hbar('Loading GPT-2')
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(model.h)
    d_model = model.config.n_embd
    print(f'  Model:      {MODEL_NAME}')
    print(f'  Parameters: {n_params:,}')
    print(f'  Layers:     {n_layers}')
    print(f'  Hidden dim: {d_model}')
    print(f'  Device:     {DEVICE}')

    # ── 2. Load irony dataset ────────────────────────────────────────────
    hbar('Irony Dataset')
    all_pairs = load_irony_pairs()
    groups = by_label(all_pairs)
    for label, items in groups.items():
        print(f'  {label:8s}: {len(items)} sentences')
    paired = get_paired_only(all_pairs)
    print(f'  paired:   {len(paired)} (strongest test: same target, different context)')

    # ── 3. Method 0: Attention path defect (fast sanity check) ────────────
    hbar('Method 0: Attention Path Defect')
    asymmetry_by_label = {}
    for label, items in groups.items():
        vals = []
        for sp in items:
            ids = tokenize(tokenizer, sp.text)
            r = attention_flow_asymmetry(model, ids)
            vals.append(r['defect_per_layer'].mean().item())
        asymmetry_by_label[label] = np.array(vals)
        print(f'  {label:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')

    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        U, p = stats.mannwhitneyu(asymmetry_by_label[la], asymmetry_by_label[lb], alternative='two-sided')
        print(f'  {la} vs {lb}: p = {fmt_p(p)}')

    # ── 4. Layer-wise Jacobian holonomy (each layer = VFE step) ──────────
    hbar('Layer-wise Jacobian Holonomy (VFE Steps)')

    results = {'ironic': [], 'literal': [], 'control': []}
    layer_profiles = {'ironic': [], 'literal': [], 'control': []}
    total = sum(len(v) for v in groups.values())
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            plh = layerwise_jacobian_holonomy(
                model, ids, max_triples=MAX_TRI,
            )

            # Store per-layer profile
            layer_profiles[label].append(plh['kappa_per_layer'])

            # Wrap for compatibility
            kappa_arr = plh['kappa_all']
            if kappa_arr.ndim == 2:
                kappa_arr = np.nanmean(kappa_arr, axis=0)

            hr = HolonomyResult(
                kappa=kappa_arr,
                triangles=plh['triples'],
                kappa_mean=plh['kappa_mean'],
                kappa_median=plh['kappa_median'],
                kappa_max=plh['kappa_max'],
                kappa_std=plh['kappa_std'],
                metadata={
                    'text': sp.text,
                    'label': label,
                    'pair_id': sp.pair_id,
                    'target': getattr(sp, 'target', ''),
                    'method': 'layerwise_jacobian',
                    'n_tokens': N,
                    'd_model': plh['d_model'],
                    'n_layers': plh['n_layers'],
                    'kappa_per_layer': plh['kappa_per_layer'],
                    'cos_sim_per_layer': plh['cos_sim_per_layer'],
                    'n_forward_passes': plh['n_forward_passes'],
                },
            )
            results[label].append(hr)

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:8s} kappa={plh["kappa_mean"]:.4f}  '
                  f'({N} tok, {plh["n_forward_passes"]} fwd, {dt:.1f}s)  '
                  f'{sp.text[:55]}...')

    # ── 5. Discrete curvature (commutator) ────────────────────────────────
    hbar('Discrete Riemann Curvature (Transport Commutator)')

    curvature_results = {'ironic': [], 'literal': [], 'control': []}
    curvature_layer_profiles = {'ironic': [], 'literal': [], 'control': []}
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            ids = tokenize(tokenizer, sp.text)
            N = ids.shape[1]
            if N < 3:
                done += 1
                continue

            cr = discrete_curvature(
                model, ids, max_triples=MAX_PAIRS,
            )

            curvature_results[label].append(cr['curvature_mean'])
            curvature_layer_profiles[label].append(cr['curvature_per_layer'])

            done += 1
            dt = time.time() - t0
            print(f'  [{done:3d}/{total}] {label:8s} curv={cr["curvature_mean"]:.6f}  '
                  f'({N} tok, {dt:.1f}s)  {sp.text[:55]}...')

    # Convert to arrays
    for label in curvature_results:
        curvature_results[label] = np.array(curvature_results[label])

    # ── 6. Statistical analysis ──────────────────────────────────────────
    hbar('Statistical Analysis: Layer-wise Holonomy')

    kappas = {
        label: np.array([hr.kappa_mean for hr in hrs])
        for label, hrs in results.items() if hrs
    }

    for label in ['ironic', 'literal', 'control']:
        if label in kappas:
            k = kappas[label]
            print(f'  {label:8s}: mean={np.mean(k):.4f}  median={np.median(k):.4f}  std={np.std(k):.4f}')

    stat_results = {}
    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        r = run_stats(kappas, la, lb, 'holonomy')
        if r:
            stat_results[f'{la}_vs_{lb}_holonomy'] = r

    hbar('Statistical Analysis: Curvature')
    for label in ['ironic', 'literal', 'control']:
        if label in curvature_results and len(curvature_results[label]) > 0:
            c = curvature_results[label]
            print(f'  {label:8s}: mean={np.mean(c):.6f}  median={np.median(c):.6f}  std={np.std(c):.6f}')

    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        r = run_stats(curvature_results, la, lb, 'curvature')
        if r:
            stat_results[f'{la}_vs_{lb}_curvature'] = r

    # ── 7. Paired comparison ──────────────────────────────────────────────
    hbar('Paired Comparison (Same Target, Different Context)')

    ironic_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results['ironic']}
    literal_by_pair = {hr.metadata['pair_id']: hr.kappa_mean for hr in results['literal']}
    targets_by_pair = {}
    for hr in results['ironic']:
        targets_by_pair[hr.metadata['pair_id']] = hr.metadata.get('target', '')
    shared = sorted(set(ironic_by_pair) & set(literal_by_pair))

    n_higher = 0
    for pid in shared:
        ik, lk = ironic_by_pair[pid], literal_by_pair[pid]
        arrow = '>' if ik > lk else '<'
        tag = '*' if ik > lk else ' '
        target = targets_by_pair.get(pid, '')
        print(f'  {tag} pair {pid:2d}: ironic={ik:.4f} {arrow} literal={lk:.4f}  "{target}"')
        if ik > lk:
            n_higher += 1

    pct = 100 * n_higher / len(shared) if shared else 0
    print(f'\n  Ironic > Literal in {n_higher}/{len(shared)} pairs ({pct:.0f}%)')

    if len(shared) >= 5:
        paired_ironic = [ironic_by_pair[p] for p in shared]
        paired_literal = [literal_by_pair[p] for p in shared]
        stat_w, p_w = stats.wilcoxon(paired_ironic, paired_literal, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')
        stat_results['paired_wilcoxon'] = {'W': float(stat_w), 'p_value': float(p_w)}

    # Also paired comparison for curvature
    curv_ironic_by_pair = {}
    curv_literal_by_pair = {}
    for idx, sp in enumerate(groups['ironic']):
        if idx < len(curvature_results['ironic']):
            curv_ironic_by_pair[sp.pair_id] = curvature_results['ironic'][idx]
    for idx, sp in enumerate(groups['literal']):
        if idx < len(curvature_results['literal']):
            curv_literal_by_pair[sp.pair_id] = curvature_results['literal'][idx]

    shared_curv = sorted(set(curv_ironic_by_pair) & set(curv_literal_by_pair))
    if len(shared_curv) >= 5:
        pc_ironic = [curv_ironic_by_pair[p] for p in shared_curv]
        pc_literal = [curv_literal_by_pair[p] for p in shared_curv]
        n_higher_c = sum(1 for i, l in zip(pc_ironic, pc_literal) if i > l)
        pct_c = 100 * n_higher_c / len(shared_curv)
        print(f'\n  Curvature: Ironic > Literal in {n_higher_c}/{len(shared_curv)} pairs ({pct_c:.0f}%)')
        stat_w, p_w = stats.wilcoxon(pc_ironic, pc_literal, alternative='two-sided')
        print(f'  Curvature Wilcoxon: W={stat_w:.0f}, p={fmt_p(p_w)}')

    # ── 8. Permutation tests ─────────────────────────────────────────────
    hbar('Permutation Tests (10,000 shuffles)')
    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        if la in kappas and lb in kappas:
            run_permutation_test(kappas[la], kappas[lb], la, lb, 'holonomy')
        if la in curvature_results and lb in curvature_results:
            if len(curvature_results[la]) > 0 and len(curvature_results[lb]) > 0:
                run_permutation_test(curvature_results[la], curvature_results[lb], la, lb, 'curvature')

    # Paired sign-flip
    if len(shared) >= 5:
        paired_diffs = np.array(paired_ironic) - np.array(paired_literal)
        obs_mean_diff = np.mean(paired_diffs)
        rng = np.random.RandomState(42)
        count = 0
        for _ in range(10000):
            signs = rng.choice([-1, 1], size=len(paired_diffs))
            if abs(np.mean(paired_diffs * signs)) >= abs(obs_mean_diff):
                count += 1
        p_pp = (count + 1) / 10001
        print(f'  paired sign-flip (holonomy): obs_diff={obs_mean_diff:+.4f}  p_perm={fmt_p(p_pp)}')

    # ── 9. Length-controlled analysis ─────────────────────────────────────
    hbar('Length-Controlled Analysis')

    all_k, all_len, all_lab = [], [], []
    for label, hrs in results.items():
        for hr in hrs:
            all_k.append(hr.kappa_mean)
            all_len.append(hr.metadata['n_tokens'])
            all_lab.append(label)

    all_k = np.array(all_k)
    all_len = np.array(all_len, dtype=float)
    all_lab = np.array(all_lab)

    for label in ['ironic', 'literal', 'control']:
        mask = all_lab == label
        if mask.any():
            print(f'  {label:8s}: mean_len={np.mean(all_len[mask]):.1f}  mean_kappa={np.mean(all_k[mask]):.4f}')

    r_len, p_len = stats.pearsonr(all_len, all_k)
    print(f'  kappa ~ length: r={r_len:+.3f}  p={fmt_p(p_len)}')

    slope, intercept = np.polyfit(all_len, all_k, 1)
    residuals = all_k - (slope * all_len + intercept)
    print(f'  Regression: kappa = {slope:+.5f} * n_tokens + {intercept:.4f}')

    resid_by_label = {}
    for label in ['ironic', 'literal', 'control']:
        mask = all_lab == label
        if mask.any():
            resid_by_label[label] = residuals[mask]
            print(f'  {label:8s} residual: mean={np.mean(residuals[mask]):+.4f}  std={np.std(residuals[mask]):.4f}')

    print('\n  Length-controlled Mann-Whitney (on residuals):')
    for la, lb in [('ironic','literal'), ('ironic','control'), ('literal','control')]:
        if la in resid_by_label and lb in resid_by_label:
            a, b = resid_by_label[la], resid_by_label[lb]
            U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
            d_val = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2) if (np.var(a) + np.var(b)) > 0 else 0
            print(f'    {la} vs {lb}: U={U:.0f}  p={fmt_p(p)}  d={d_val:+.3f}')

    # ── 10. Bootstrap CIs ────────────────────────────────────────────────
    hbar('Bootstrap CIs (Paired Ironic - Literal)')
    N_BOOT = 10_000
    rng = np.random.RandomState(42)

    if len(shared) >= 5:
        paired_diffs = np.array(paired_ironic) - np.array(paired_literal)
        boot_means = np.zeros(N_BOOT)
        for i in range(N_BOOT):
            sample = rng.choice(paired_diffs, size=len(paired_diffs), replace=True)
            boot_means[i] = np.mean(sample)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        boot_p = 2 * min(np.mean(boot_means <= 0), np.mean(boot_means >= 0))
        print(f'  Mean paired diff: {np.mean(paired_diffs):+.4f}')
        print(f'  95% CI:           [{ci_lo:+.4f}, {ci_hi:+.4f}]')
        print(f'  Bootstrap p:      {fmt_p(boot_p)}')
        print(f'  CI excludes 0:    {"YES" if ci_lo > 0 or ci_hi < 0 else "NO"}')

    # ── 11. Per-layer statistical tests ──────────────────────────────────
    hbar('Per-Layer Analysis (Where Does the Effect Emerge?)')

    layer_pvals = []
    layer_rows = []
    for l in range(n_layers):
        kl_ironic = [prof[l] for prof in layer_profiles['ironic'] if len(prof) > l]
        kl_literal = [prof[l] for prof in layer_profiles['literal'] if len(prof) > l]
        if len(kl_ironic) >= 2 and len(kl_literal) >= 2:
            U, p = stats.mannwhitneyu(kl_ironic, kl_literal, alternative='two-sided')
            d_mean = np.mean(kl_ironic) - np.mean(kl_literal)
            layer_pvals.append(p)
            layer_rows.append((l, np.mean(kl_ironic), np.mean(kl_literal), d_mean, p))
        else:
            layer_pvals.append(1.0)
            layer_rows.append((l, 0, 0, 0, 1.0))

    layer_qvals = benjamini_hochberg(layer_pvals)
    for (l, mi, ml, d_mean, p), q in zip(layer_rows, layer_qvals):
        sig = '*' if q < 0.05 else ' '
        print(f'  Layer {l:2d}: ironic={mi:.4f}  '
              f'literal={ml:.4f}  diff={d_mean:+.4f}  p={p:.4f}  q={q:.4f} {sig}')

    # ── 12. Phrase-localized holonomy (length confound control) ─────────
    hbar('Phrase-Localized Holonomy (Length Control)')
    print('  Measuring holonomy ONLY on target-phrase tokens.')
    print('  Same target in both contexts → perfect length control.')
    print('  Full sentence context preserved (model sees everything).')
    print()

    loc_ironic_by_pair = {}
    loc_literal_by_pair = {}
    loc_curv_ironic_by_pair = {}
    loc_curv_literal_by_pair = {}
    loc_layer_profiles = {'ironic': [], 'literal': []}
    loc_curv_layer_profiles = {'ironic': [], 'literal': []}

    # Build lookup: pair_id → (ironic_item, literal_item) for paired examples with target
    pair_map = {}
    for sp in all_pairs:
        if sp.label in ('ironic', 'literal') and sp.target:
            pair_map.setdefault(sp.pair_id, {})[sp.label] = sp

    n_found = 0
    n_skip = 0
    for pid in sorted(pair_map.keys()):
        if 'ironic' not in pair_map[pid] or 'literal' not in pair_map[pid]:
            continue
        sp_i = pair_map[pid]['ironic']
        sp_l = pair_map[pid]['literal']
        target_phrase = sp_i.target

        ids_i = tokenize(tokenizer, sp_i.text)
        ids_l = tokenize(tokenizer, sp_l.text)

        pos_i = find_phrase_positions(tokenizer, sp_i.text, target_phrase)
        pos_l = find_phrase_positions(tokenizer, sp_l.text, target_phrase)

        if pos_i is None or pos_l is None or len(pos_i) < 3 or len(pos_l) < 3:
            n_skip += 1
            print(f'  SKIP pair {pid}: target "{target_phrase}" — '
                  f'pos_i={pos_i}, pos_l={pos_l}')
            continue

        n_found += 1
        t0 = time.time()

        # Holonomy: ironic context, target tokens only
        plh_i = layerwise_jacobian_holonomy(
            model, ids_i, max_triples=MAX_TRI, positions=pos_i,
        )
        # Holonomy: literal context, target tokens only
        plh_l = layerwise_jacobian_holonomy(
            model, ids_l, max_triples=MAX_TRI, positions=pos_l,
        )

        loc_ironic_by_pair[pid] = plh_i['kappa_mean']
        loc_literal_by_pair[pid] = plh_l['kappa_mean']
        loc_layer_profiles['ironic'].append(plh_i['kappa_per_layer'])
        loc_layer_profiles['literal'].append(plh_l['kappa_per_layer'])

        # Curvature: phrase-localized
        cr_i = discrete_curvature(
            model, ids_i, max_triples=MAX_PAIRS, positions=pos_i,
        )
        cr_l = discrete_curvature(
            model, ids_l, max_triples=MAX_PAIRS, positions=pos_l,
        )

        loc_curv_ironic_by_pair[pid] = cr_i['curvature_mean']
        loc_curv_literal_by_pair[pid] = cr_l['curvature_mean']
        loc_curv_layer_profiles['ironic'].append(cr_i['curvature_per_layer'])
        loc_curv_layer_profiles['literal'].append(cr_l['curvature_per_layer'])

        dt = time.time() - t0
        arrow_h = '>' if plh_i['kappa_mean'] > plh_l['kappa_mean'] else '<'
        arrow_c = '>' if cr_i['curvature_mean'] > cr_l['curvature_mean'] else '<'
        print(f'  pair {pid:2d} "{target_phrase}": '
              f'hol {plh_i["kappa_mean"]:.4f} {arrow_h} {plh_l["kappa_mean"]:.4f}  '
              f'curv {cr_i["curvature_mean"]:.5f} {arrow_c} {cr_l["curvature_mean"]:.5f}  '
              f'({len(pos_i)}/{len(pos_l)} tok, {dt:.1f}s)')

    print(f'\n  Found phrase positions: {n_found} pairs  (skipped: {n_skip})')

    # ── 12a. Phrase-localized statistics ──────────────────────────────────
    loc_shared = sorted(set(loc_ironic_by_pair) & set(loc_literal_by_pair))
    if len(loc_shared) >= 5:
        hbar('Phrase-Localized Statistics')

        loc_pi = np.array([loc_ironic_by_pair[p] for p in loc_shared])
        loc_pl = np.array([loc_literal_by_pair[p] for p in loc_shared])
        loc_diffs = loc_pi - loc_pl

        n_higher_loc = int(np.sum(loc_pi > loc_pl))
        pct_loc = 100 * n_higher_loc / len(loc_shared)
        print(f'  Holonomy: Ironic > Literal in {n_higher_loc}/{len(loc_shared)} pairs ({pct_loc:.0f}%)')
        print(f'  Mean ironic: {np.mean(loc_pi):.4f}  literal: {np.mean(loc_pl):.4f}  '
              f'diff: {np.mean(loc_diffs):+.4f}')

        # Wilcoxon signed-rank
        stat_w, p_w = stats.wilcoxon(loc_pi, loc_pl, alternative='two-sided')
        print(f'  Wilcoxon signed-rank: W={stat_w:.0f}, p={fmt_p(p_w)}')

        # Effect size (paired Cohen's d)
        d_paired = np.mean(loc_diffs) / np.std(loc_diffs) if np.std(loc_diffs) > 0 else 0
        print(f'  Paired d: {d_paired:+.3f}')

        # Mann-Whitney (unpaired)
        U_loc, p_loc = stats.mannwhitneyu(loc_pi, loc_pl, alternative='two-sided')
        pooled_std = np.sqrt((np.var(loc_pi) + np.var(loc_pl)) / 2)
        d_loc = (np.mean(loc_pi) - np.mean(loc_pl)) / pooled_std if pooled_std > 0 else 0
        print(f'  Mann-Whitney U={U_loc:.0f}, p={fmt_p(p_loc)}, d={d_loc:+.3f}')

        # Bootstrap CI
        rng_loc = np.random.RandomState(42)
        boot_loc = np.zeros(10000)
        for bi in range(10000):
            sample = rng_loc.choice(loc_diffs, size=len(loc_diffs), replace=True)
            boot_loc[bi] = np.mean(sample)
        ci_lo_loc, ci_hi_loc = np.percentile(boot_loc, [2.5, 97.5])
        boot_p_loc = 2 * min(np.mean(boot_loc <= 0), np.mean(boot_loc >= 0))
        print(f'  Bootstrap 95% CI: [{ci_lo_loc:+.4f}, {ci_hi_loc:+.4f}]')
        print(f'  Bootstrap p: {fmt_p(boot_p_loc)}')
        print(f'  CI excludes 0: {"YES" if ci_lo_loc > 0 or ci_hi_loc < 0 else "NO"}')

        # Sign-flip permutation
        obs_mean = np.mean(loc_diffs)
        rng_perm = np.random.RandomState(42)
        count_perm = 0
        for _ in range(10000):
            signs = rng_perm.choice([-1, 1], size=len(loc_diffs))
            if abs(np.mean(loc_diffs * signs)) >= abs(obs_mean):
                count_perm += 1
        p_perm_loc = (count_perm + 1) / 10001
        print(f'  Paired sign-flip permutation: obs_diff={obs_mean:+.4f}  p={fmt_p(p_perm_loc)}')

        # Verify no length confound
        tok_counts_i = [len(find_phrase_positions(tokenizer, pair_map[p]['ironic'].text,
                            pair_map[p]['ironic'].target) or []) for p in loc_shared]
        tok_counts_l = [len(find_phrase_positions(tokenizer, pair_map[p]['literal'].text,
                            pair_map[p]['literal'].target) or []) for p in loc_shared]
        n_exact_match = sum(1 for ti, tl in zip(tok_counts_i, tok_counts_l) if ti == tl)
        print(f'\n  Length verification: {n_exact_match}/{len(loc_shared)} pairs have identical target token count')
        print(f'  Mean target tokens: ironic={np.mean(tok_counts_i):.1f}  literal={np.mean(tok_counts_l):.1f}')

        stat_results['phrase_localized_holonomy'] = {
            'n_pairs': len(loc_shared),
            'pct_higher': float(pct_loc),
            'wilcoxon_W': float(stat_w),
            'wilcoxon_p': float(p_w),
            'paired_d': float(d_paired),
            'mann_whitney_U': float(U_loc),
            'mann_whitney_p': float(p_loc),
            'cohens_d': float(d_loc),
            'bootstrap_ci': [float(ci_lo_loc), float(ci_hi_loc)],
            'bootstrap_p': float(boot_p_loc),
            'permutation_p': float(p_perm_loc),
        }

    # Phrase-localized curvature
    loc_shared_c = sorted(set(loc_curv_ironic_by_pair) & set(loc_curv_literal_by_pair))
    if len(loc_shared_c) >= 5:
        loc_ci = np.array([loc_curv_ironic_by_pair[p] for p in loc_shared_c])
        loc_cl = np.array([loc_curv_literal_by_pair[p] for p in loc_shared_c])
        finite_mask = np.isfinite(loc_ci) & np.isfinite(loc_cl)
        loc_ci_f = loc_ci[finite_mask]
        loc_cl_f = loc_cl[finite_mask]
        if len(loc_ci_f) >= 5:
            n_hc = int(np.sum(loc_ci_f > loc_cl_f))
            pct_hc = 100 * n_hc / len(loc_ci_f)
            print(f'\n  Phrase-localized curvature: Ironic > Literal in {n_hc}/{len(loc_ci_f)} ({pct_hc:.0f}%)')
            sw_c, pw_c = stats.wilcoxon(loc_ci_f, loc_cl_f, alternative='two-sided')
            d_c = np.mean(loc_ci_f - loc_cl_f) / np.std(loc_ci_f - loc_cl_f) if np.std(loc_ci_f - loc_cl_f) > 0 else 0
            print(f'  Wilcoxon: W={sw_c:.0f}, p={fmt_p(pw_c)}, paired d={d_c:+.3f}')

            stat_results['phrase_localized_curvature'] = {
                'n_pairs': int(len(loc_ci_f)),
                'pct_higher': float(pct_hc),
                'wilcoxon_W': float(sw_c),
                'wilcoxon_p': float(pw_c),
                'paired_d': float(d_c),
            }

    # ── 12b. Phrase-localized per-layer analysis ─────────────────────────
    if loc_layer_profiles['ironic'] and loc_layer_profiles['literal']:
        hbar('Phrase-Localized Per-Layer Analysis')
        pl_pvals = []
        pl_rows = []
        for l in range(n_layers):
            kl_i = [prof[l] for prof in loc_layer_profiles['ironic'] if len(prof) > l]
            kl_l = [prof[l] for prof in loc_layer_profiles['literal'] if len(prof) > l]
            if len(kl_i) >= 2 and len(kl_l) >= 2:
                U_l, p_l = stats.mannwhitneyu(kl_i, kl_l, alternative='two-sided')
                d_mean_l = np.mean(kl_i) - np.mean(kl_l)
                pl_pvals.append(p_l)
                pl_rows.append((l, np.mean(kl_i), np.mean(kl_l), d_mean_l, p_l))
            else:
                pl_pvals.append(1.0)
                pl_rows.append((l, 0, 0, 0, 1.0))

        pl_qvals = benjamini_hochberg(pl_pvals)
        for (l, mi, ml, d_mean_l, p_l), q_l in zip(pl_rows, pl_qvals):
            sig = '*' if q_l < 0.05 else ' '
            print(f'  Layer {l:2d}: ironic={mi:.4f}  '
                  f'literal={ml:.4f}  diff={d_mean_l:+.4f}  p={p_l:.4f}  q={q_l:.4f} {sig}')

    # ── 12c. Cross-boundary holonomy (phrase ↔ context interaction) ──────
    hbar('Cross-Boundary Holonomy (Target-Context Interaction)')
    print('  Measuring holonomy on target phrase + nearby context tokens.')
    print('  Triples span the target-context boundary, capturing how the')
    print('  target phrase interacts with its surrounding context.')
    print()

    CONTEXT_WINDOW = 3  # tokens on each side of phrase

    xb_ironic_by_pair = {}
    xb_literal_by_pair = {}
    xb_curv_ironic_by_pair = {}
    xb_curv_literal_by_pair = {}
    xb_layer_profiles = {'ironic': [], 'literal': []}
    xb_pos_counts = {'ironic': [], 'literal': []}  # track equalized position counts

    n_xb_found = 0
    for pid in sorted(pair_map.keys()):
        if 'ironic' not in pair_map[pid] or 'literal' not in pair_map[pid]:
            continue
        sp_i = pair_map[pid]['ironic']
        sp_l = pair_map[pid]['literal']
        target_phrase = sp_i.target

        ids_i = tokenize(tokenizer, sp_i.text)
        ids_l = tokenize(tokenizer, sp_l.text)
        N_i = ids_i.shape[1]
        N_l = ids_l.shape[1]

        pos_i = find_phrase_positions(tokenizer, sp_i.text, target_phrase)
        pos_l = find_phrase_positions(tokenizer, sp_l.text, target_phrase)

        if pos_i is None or pos_l is None or len(pos_i) < 2 or len(pos_l) < 2:
            continue

        xb_pos_i, _ = cross_boundary_positions(pos_i, N_i, CONTEXT_WINDOW)
        xb_pos_l, _ = cross_boundary_positions(pos_l, N_l, CONTEXT_WINDOW)

        # Equalize window sizes: trim the longer window to match the shorter
        # This eliminates any residual length confound from asymmetric context
        if len(xb_pos_i) != len(xb_pos_l):
            target_len = min(len(xb_pos_i), len(xb_pos_l))
            if len(xb_pos_i) > target_len:
                # Trim symmetrically around phrase center
                center_i = (min(pos_i) + max(pos_i)) / 2
                xb_pos_i = sorted(xb_pos_i, key=lambda p: abs(p - center_i))[:target_len]
                xb_pos_i = sorted(xb_pos_i)
            if len(xb_pos_l) > target_len:
                center_l = (min(pos_l) + max(pos_l)) / 2
                xb_pos_l = sorted(xb_pos_l, key=lambda p: abs(p - center_l))[:target_len]
                xb_pos_l = sorted(xb_pos_l)

        if len(xb_pos_i) < 3 or len(xb_pos_l) < 3:
            continue

        n_xb_found += 1
        xb_pos_counts['ironic'].append(len(xb_pos_i))
        xb_pos_counts['literal'].append(len(xb_pos_l))
        t0 = time.time()

        plh_i = layerwise_jacobian_holonomy(
            model, ids_i, max_triples=MAX_TRI, positions=xb_pos_i,
        )
        plh_l = layerwise_jacobian_holonomy(
            model, ids_l, max_triples=MAX_TRI, positions=xb_pos_l,
        )

        xb_ironic_by_pair[pid] = plh_i['kappa_mean']
        xb_literal_by_pair[pid] = plh_l['kappa_mean']
        xb_layer_profiles['ironic'].append(plh_i['kappa_per_layer'])
        xb_layer_profiles['literal'].append(plh_l['kappa_per_layer'])

        cr_i = discrete_curvature(
            model, ids_i, max_triples=MAX_PAIRS, positions=xb_pos_i,
        )
        cr_l = discrete_curvature(
            model, ids_l, max_triples=MAX_PAIRS, positions=xb_pos_l,
        )

        xb_curv_ironic_by_pair[pid] = cr_i['curvature_mean']
        xb_curv_literal_by_pair[pid] = cr_l['curvature_mean']

        dt = time.time() - t0
        arrow_h = '>' if plh_i['kappa_mean'] > plh_l['kappa_mean'] else '<'
        arrow_c = '>' if cr_i['curvature_mean'] > cr_l['curvature_mean'] else '<'
        print(f'  pair {pid:2d}: hol {plh_i["kappa_mean"]:.4f} {arrow_h} {plh_l["kappa_mean"]:.4f}  '
              f'curv {cr_i["curvature_mean"]:.5f} {arrow_c} {cr_l["curvature_mean"]:.5f}  '
              f'({len(xb_pos_i)}/{len(xb_pos_l)} pos, {dt:.1f}s)  "{target_phrase}"')

    print(f'\n  Cross-boundary pairs: {n_xb_found}')

    # Statistics
    xb_shared = sorted(set(xb_ironic_by_pair) & set(xb_literal_by_pair))
    if len(xb_shared) >= 5:
        hbar('Cross-Boundary Statistics')

        xb_pi = np.array([xb_ironic_by_pair[p] for p in xb_shared])
        xb_pl = np.array([xb_literal_by_pair[p] for p in xb_shared])
        xb_diffs = xb_pi - xb_pl

        n_higher_xb = int(np.sum(xb_pi > xb_pl))
        pct_xb = 100 * n_higher_xb / len(xb_shared)
        print(f'  Holonomy: Ironic > Literal in {n_higher_xb}/{len(xb_shared)} pairs ({pct_xb:.0f}%)')
        print(f'  Mean ironic: {np.mean(xb_pi):.4f}  literal: {np.mean(xb_pl):.4f}  '
              f'diff: {np.mean(xb_diffs):+.4f}')

        stat_w_xb, p_w_xb = stats.wilcoxon(xb_pi, xb_pl, alternative='two-sided')
        d_xb = np.mean(xb_diffs) / np.std(xb_diffs) if np.std(xb_diffs) > 0 else 0
        print(f'  Wilcoxon: W={stat_w_xb:.0f}, p={fmt_p(p_w_xb)}, paired d={d_xb:+.3f}')

        # Bootstrap CI
        rng_xb = np.random.RandomState(42)
        boot_xb = np.zeros(10000)
        for bi in range(10000):
            sample = rng_xb.choice(xb_diffs, size=len(xb_diffs), replace=True)
            boot_xb[bi] = np.mean(sample)
        ci_lo_xb, ci_hi_xb = np.percentile(boot_xb, [2.5, 97.5])
        boot_p_xb = 2 * min(np.mean(boot_xb <= 0), np.mean(boot_xb >= 0))
        print(f'  Bootstrap 95% CI: [{ci_lo_xb:+.4f}, {ci_hi_xb:+.4f}]')
        print(f'  Bootstrap p: {fmt_p(boot_p_xb)}')
        print(f'  CI excludes 0: {"YES" if ci_lo_xb > 0 or ci_hi_xb < 0 else "NO"}')

        # Mean equalized positions per pair (length check)
        n_exact_eq = sum(1 for a, b in zip(xb_pos_counts['ironic'], xb_pos_counts['literal']) if a == b)
        print(f'  Equalized positions: {n_exact_eq}/{len(xb_pos_counts["ironic"])} pairs identical  '
              f'mean ironic={np.mean(xb_pos_counts["ironic"]):.1f}  '
              f'literal={np.mean(xb_pos_counts["literal"]):.1f}')

        stat_results['cross_boundary_holonomy'] = {
            'n_pairs': len(xb_shared),
            'pct_higher': float(pct_xb),
            'wilcoxon_W': float(stat_w_xb),
            'wilcoxon_p': float(p_w_xb),
            'paired_d': float(d_xb),
            'bootstrap_ci': [float(ci_lo_xb), float(ci_hi_xb)],
            'bootstrap_p': float(boot_p_xb),
            'context_window': CONTEXT_WINDOW,
        }

    # Cross-boundary curvature
    xb_shared_c = sorted(set(xb_curv_ironic_by_pair) & set(xb_curv_literal_by_pair))
    if len(xb_shared_c) >= 5:
        xb_ci = np.array([xb_curv_ironic_by_pair[p] for p in xb_shared_c])
        xb_cl = np.array([xb_curv_literal_by_pair[p] for p in xb_shared_c])
        fm = np.isfinite(xb_ci) & np.isfinite(xb_cl)
        if np.sum(fm) >= 5:
            xb_ci_f, xb_cl_f = xb_ci[fm], xb_cl[fm]
            n_hc_xb = int(np.sum(xb_ci_f > xb_cl_f))
            pct_hc_xb = 100 * n_hc_xb / len(xb_ci_f)
            sw_xb, pw_xb = stats.wilcoxon(xb_ci_f, xb_cl_f, alternative='two-sided')
            d_c_xb = np.mean(xb_ci_f - xb_cl_f) / np.std(xb_ci_f - xb_cl_f) if np.std(xb_ci_f - xb_cl_f) > 0 else 0
            print(f'\n  Cross-boundary curvature: Ironic > Literal in {n_hc_xb}/{len(xb_ci_f)} ({pct_hc_xb:.0f}%)')
            print(f'  Wilcoxon: W={sw_xb:.0f}, p={fmt_p(pw_xb)}, paired d={d_c_xb:+.3f}')

            stat_results['cross_boundary_curvature'] = {
                'n_pairs': int(len(xb_ci_f)),
                'pct_higher': float(pct_hc_xb),
                'wilcoxon_W': float(sw_xb),
                'wilcoxon_p': float(pw_xb),
                'paired_d': float(d_c_xb),
            }

    # Cross-boundary per-layer
    xb_layer_pvals = []
    xb_layer_rows = []
    xb_layer_qvals = None
    if xb_layer_profiles['ironic'] and xb_layer_profiles['literal']:
        hbar('Cross-Boundary Per-Layer Analysis')
        for l in range(n_layers):
            kl_i = [prof[l] for prof in xb_layer_profiles['ironic'] if len(prof) > l]
            kl_l = [prof[l] for prof in xb_layer_profiles['literal'] if len(prof) > l]
            if len(kl_i) >= 2 and len(kl_l) >= 2:
                U_l, p_l = stats.mannwhitneyu(kl_i, kl_l, alternative='two-sided')
                d_mean_l = np.mean(kl_i) - np.mean(kl_l)
                xb_layer_pvals.append(p_l)
                xb_layer_rows.append((l, np.mean(kl_i), np.mean(kl_l), d_mean_l, p_l))
            else:
                xb_layer_pvals.append(1.0)
                xb_layer_rows.append((l, 0, 0, 0, 1.0))

        xb_layer_qvals = benjamini_hochberg(xb_layer_pvals)
        for (l, mi, ml, d_mean_l, p_l), q_l in zip(xb_layer_rows, xb_layer_qvals):
            sig = '*' if q_l < 0.05 else ' '
            print(f'  Layer {l:2d}: ironic={mi:.4f}  '
                  f'literal={ml:.4f}  diff={d_mean_l:+.4f}  p={p_l:.4f}  q={q_l:.4f} {sig}')

    # ── 13. Plots ─────────────────────────────────────────────────────────
    hbar('Generating Plots')

    plot_distributions(kappas, 'Holonomy Distribution by Condition',
                       r'Mean holonomy $\kappa$',
                       str(OUTPUT_DIR / 'holonomy_distributions.png'))

    plot_distributions(curvature_results, 'Curvature Distribution by Condition',
                       'Curvature (commutator norm)',
                       str(OUTPUT_DIR / 'curvature_distributions.png'))

    plot_layer_profiles(layer_profiles,
                        'Layer-wise Holonomy (VFE Steps)',
                        str(OUTPUT_DIR / 'layer_holonomy_profiles.png'),
                        n_layers=n_layers)

    plot_curvature_layer_profiles(curvature_layer_profiles,
                                  str(OUTPUT_DIR / 'layer_curvature_profiles.png'),
                                  n_layers=n_layers)

    if shared:
        plot_paired_comparison(ironic_by_pair, literal_by_pair, shared, targets_by_pair,
                               r'Mean holonomy $\kappa$',
                               'Paired Comparison: Same Target, Different Context',
                               str(OUTPUT_DIR / 'paired_holonomy.png'))

    if shared_curv:
        plot_paired_comparison(curv_ironic_by_pair, curv_literal_by_pair, shared_curv, targets_by_pair,
                               'Curvature (commutator norm)',
                               'Paired Curvature: Same Target, Different Context',
                               str(OUTPUT_DIR / 'paired_curvature.png'))

    # Phrase-localized plots
    if loc_shared:
        plot_paired_comparison(loc_ironic_by_pair, loc_literal_by_pair, loc_shared, targets_by_pair,
                               r'Phrase-localized $\kappa$',
                               'Phrase-Localized Holonomy (Length Controlled)',
                               str(OUTPUT_DIR / 'phrase_localized_holonomy.png'))

    if loc_shared_c:
        plot_paired_comparison(loc_curv_ironic_by_pair, loc_curv_literal_by_pair, loc_shared_c, targets_by_pair,
                               'Phrase-localized curvature',
                               'Phrase-Localized Curvature (Length Controlled)',
                               str(OUTPUT_DIR / 'phrase_localized_curvature.png'))

    if loc_layer_profiles['ironic'] and loc_layer_profiles['literal']:
        plot_layer_profiles(loc_layer_profiles,
                            'Phrase-Localized Layer Holonomy (Length Controlled)',
                            str(OUTPUT_DIR / 'phrase_localized_layer_profiles.png'),
                            n_layers=n_layers)

    if loc_curv_layer_profiles['ironic'] and loc_curv_layer_profiles['literal']:
        plot_curvature_layer_profiles(loc_curv_layer_profiles,
                                      str(OUTPUT_DIR / 'phrase_localized_layer_curvature.png'),
                                      n_layers=n_layers)

    # Cross-boundary plots
    if xb_shared:
        plot_paired_comparison(xb_ironic_by_pair, xb_literal_by_pair, xb_shared, targets_by_pair,
                               r'Cross-boundary $\kappa$',
                               'Cross-Boundary Holonomy (Target+Context)',
                               str(OUTPUT_DIR / 'cross_boundary_holonomy.png'))

    if xb_shared_c:
        plot_paired_comparison(xb_curv_ironic_by_pair, xb_curv_literal_by_pair, xb_shared_c, targets_by_pair,
                               'Cross-boundary curvature',
                               'Cross-Boundary Curvature (Target+Context)',
                               str(OUTPUT_DIR / 'cross_boundary_curvature.png'))

    if xb_layer_profiles['ironic'] and xb_layer_profiles['literal']:
        plot_layer_profiles(xb_layer_profiles,
                            'Cross-Boundary Layer Holonomy (Target+Context)',
                            str(OUTPUT_DIR / 'cross_boundary_layer_profiles.png'),
                            n_layers=n_layers)

    # ── 13b. Double dissociation plot ────────────────────────────────────
    # Combined figure showing holonomy vs curvature for ironic language
    if xb_shared and xb_shared_c:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Effect sizes across three scales
        ax = axes[0]
        scales = []
        hol_ds = []
        curv_ds = []

        # Whole-sentence
        if 'paired_wilcoxon' in stat_results:
            scales.append('Whole\nsentence')
            ws_pi = np.array([ironic_by_pair[p] for p in shared])
            ws_pl = np.array([literal_by_pair[p] for p in shared])
            ws_diff = ws_pi - ws_pl
            hol_ds.append(np.mean(ws_diff) / np.std(ws_diff) if np.std(ws_diff) > 0 else 0)
            # curvature paired d for whole sentence
            if shared_curv:
                c_pi = np.array([curv_ironic_by_pair.get(p, float('nan')) for p in shared_curv])
                c_pl = np.array([curv_literal_by_pair.get(p, float('nan')) for p in shared_curv])
                c_diff = c_pi - c_pl
                fm = np.isfinite(c_diff)
                curv_ds.append(np.mean(c_diff[fm]) / np.std(c_diff[fm]) if np.std(c_diff[fm]) > 0 else 0)
            else:
                curv_ds.append(0)

        # Phrase-localized
        if loc_shared:
            scales.append('Target\nonly')
            hol_ds.append(float(stat_results.get('phrase_localized_holonomy', {}).get('paired_d', 0)))
            curv_ds.append(float(stat_results.get('phrase_localized_curvature', {}).get('paired_d', 0)))

        # Cross-boundary
        scales.append('Cross-\nboundary')
        hol_ds.append(float(stat_results.get('cross_boundary_holonomy', {}).get('paired_d', 0)))
        curv_ds.append(float(stat_results.get('cross_boundary_curvature', {}).get('paired_d', 0)))

        x = np.arange(len(scales))
        w = 0.35
        ax.bar(x - w/2, hol_ds, w, label='Holonomy (path defect)',
               color='#1f77b4', alpha=0.8)
        ax.bar(x + w/2, curv_ds, w, label='Curvature (superposition)',
               color='#d62728', alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.set_ylabel("Paired Cohen's d (ironic - literal)")
        ax.set_title('Effect Size by Scale')
        ax.legend(loc='upper left', fontsize=9)

        # Panel 2: Cross-boundary paired differences (holonomy vs curvature)
        ax = axes[1]
        xb_hol_diffs = np.array([xb_ironic_by_pair[p] - xb_literal_by_pair[p] for p in xb_shared])
        xb_curv_diffs_paired = []
        xb_shared_both = sorted(set(xb_shared) & set(xb_shared_c))
        for p in xb_shared_both:
            xb_curv_diffs_paired.append(xb_curv_ironic_by_pair[p] - xb_curv_literal_by_pair[p])
        xb_curv_diffs_arr = np.array(xb_curv_diffs_paired)

        ax.hist(xb_hol_diffs, bins=20, alpha=0.5, color='#1f77b4', label='Holonomy diff', density=True)
        ax.hist(xb_curv_diffs_arr, bins=20, alpha=0.5, color='#d62728', label='Curvature diff', density=True)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(x=np.mean(xb_hol_diffs), color='#1f77b4', linewidth=2, linestyle='-',
                   label=f'Hol mean={np.mean(xb_hol_diffs):+.4f}')
        if len(xb_curv_diffs_arr) > 0:
            ax.axvline(x=np.mean(xb_curv_diffs_arr), color='#d62728', linewidth=2, linestyle='-',
                       label=f'Curv mean={np.mean(xb_curv_diffs_arr):+.4f}')
        ax.set_xlabel('Ironic - Literal')
        ax.set_ylabel('Density')
        ax.set_title('Cross-Boundary: Holonomy vs Curvature')
        ax.legend(fontsize=8)

        # Panel 3: Per-layer cross-boundary effect
        ax = axes[2]
        if xb_layer_profiles['ironic'] and xb_layer_profiles['literal']:
            layers = list(range(n_layers))
            stack_i = np.array(xb_layer_profiles['ironic'])
            stack_l = np.array(xb_layer_profiles['literal'])
            diff_per_layer = np.mean(stack_i, axis=0)[:n_layers] - np.mean(stack_l, axis=0)[:n_layers]
            sem_diff = np.sqrt(
                (np.var(stack_i, axis=0)[:n_layers] / len(stack_i)) +
                (np.var(stack_l, axis=0)[:n_layers] / len(stack_l))
            )
            colors_layer = ['#d62728' if d > 0 else '#1f77b4' for d in diff_per_layer]
            ax.bar(layers, diff_per_layer, color=colors_layer, alpha=0.7)
            ax.errorbar(layers, diff_per_layer, yerr=sem_diff, fmt='none', ecolor='black',
                       capsize=3, linewidth=1)
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
            ax.set_xlabel('Layer')
            ax.set_ylabel(r'$\Delta\kappa$ (ironic - literal)')
            ax.set_title('Cross-Boundary Holonomy Difference by Layer')
            ax.set_xticks(layers)

            # Mark FDR-significant layers
            if xb_layer_qvals is not None and len(xb_layer_qvals) == n_layers:
                for l in layers:
                    if xb_layer_qvals[l] < 0.05:
                        ax.text(l, diff_per_layer[l] - 0.002 * np.sign(diff_per_layer[l]),
                               '*', ha='center', va='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / 'double_dissociation.png'), dpi=150, bbox_inches='tight')
        print(f'  Saved: {OUTPUT_DIR / "double_dissociation.png"}')
        plt.close()

    # ── 13c. Synthesis summary ──────────────────────────────────────────
    hbar('SYNTHESIS: Gauge-Theoretic Interpretation')
    print()
    print('  Ironic language gauge structure analysis:')
    print()
    print('  Metric               | Direction for irony  | Interpretation')
    print('  ---------------------|---------------------|---------------------------')

    if 'cross_boundary_holonomy' in stat_results:
        xb_h = stat_results['cross_boundary_holonomy']
        direction_h = 'HIGHER' if xb_h['paired_d'] > 0 else 'LOWER'
        print(f'  Holonomy (path defect)| {direction_h:6s} d={xb_h["paired_d"]:+.3f} p={xb_h["wilcoxon_p"]:.4f} | '
              f'Path dependence of transport')
    if 'cross_boundary_curvature' in stat_results:
        xb_c = stat_results['cross_boundary_curvature']
        direction_c = 'HIGHER' if xb_c['paired_d'] > 0 else 'LOWER'
        print(f'  Curvature (F=dA+A^A) | {direction_c:6s} d={xb_c["paired_d"]:+.3f} p={xb_c["wilcoxon_p"]:.4f} | '
              f'Non-additive token interaction')

    print()
    print('  Comparison with idiom study:')
    print('    Idioms: holonomy DOWN, curvature UP (double dissociation)')
    print('    Irony:  see results above — context-dependent meaning shift')
    print()
    print('  Key difference: idioms have frozen phrases (non-compositional);')
    print('  irony has the SAME words with context-shifted meaning.')
    print('  The gauge signature may differ accordingly.')

    # ── 14. Save results ──────────────────────────────────────────────────
    hbar('Saving Results')
    summary = {
        'model': MODEL_NAME,
        'device': DEVICE,
        'n_layers': n_layers,
        'd_model': d_model,
        'dataset': 'irony_pairs',
        'n_ironic': len(results['ironic']),
        'n_literal': len(results['literal']),
        'n_control': len(results['control']),
        'n_paired': len(shared),
        'stats': stat_results,
        'holonomy': {},
        'curvature': {},
    }

    for label in ['ironic', 'literal', 'control']:
        if label in kappas:
            summary['holonomy'][label] = {
                'mean': float(np.mean(kappas[label])),
                'median': float(np.median(kappas[label])),
                'std': float(np.std(kappas[label])),
                'values': kappas[label].tolist(),
            }
        if label in curvature_results and len(curvature_results[label]) > 0:
            summary['curvature'][label] = {
                'mean': float(np.mean(curvature_results[label])),
                'median': float(np.median(curvature_results[label])),
                'std': float(np.std(curvature_results[label])),
                'values': curvature_results[label].tolist(),
            }

    # Per-layer profiles
    summary['layer_holonomy_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in layer_profiles.items() if profiles
    }
    summary['layer_curvature_profiles'] = {
        label: [p for p in profiles]
        for label, profiles in curvature_layer_profiles.items() if profiles
    }

    # Paired results
    if shared:
        summary['paired_holonomy'] = [
            {'pair_id': p, 'target': targets_by_pair.get(p, ''),
             'ironic': ironic_by_pair[p], 'literal': literal_by_pair[p]}
            for p in shared
        ]

    # Asymmetry
    for label, a in asymmetry_by_label.items():
        summary[f'{label}_asymmetry'] = a.tolist()

    # Phrase-localized results
    if loc_shared:
        summary['phrase_localized_holonomy'] = [
            {'pair_id': p, 'target': targets_by_pair.get(p, ''),
             'ironic': float(loc_ironic_by_pair[p]),
             'literal': float(loc_literal_by_pair[p])}
            for p in loc_shared
        ]
    if loc_shared_c:
        summary['phrase_localized_curvature'] = [
            {'pair_id': p, 'target': targets_by_pair.get(p, ''),
             'ironic': float(loc_curv_ironic_by_pair[p]),
             'literal': float(loc_curv_literal_by_pair[p])}
            for p in loc_shared_c
        ]

    # Cross-boundary results
    if xb_shared:
        summary['cross_boundary_holonomy_pairs'] = [
            {'pair_id': p, 'target': targets_by_pair.get(p, ''),
             'ironic': float(xb_ironic_by_pair[p]),
             'literal': float(xb_literal_by_pair[p])}
            for p in xb_shared
        ]
    if xb_shared_c:
        summary['cross_boundary_curvature_pairs'] = [
            {'pair_id': p, 'target': targets_by_pair.get(p, ''),
             'ironic': float(xb_curv_ironic_by_pair[p]),
             'literal': float(xb_curv_literal_by_pair[p])}
            for p in xb_shared_c
        ]

    with open(OUTPUT_DIR / 'holonomy_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'  Saved: {OUTPUT_DIR / "holonomy_results.json"}')

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    hbar(f'Done ({elapsed:.0f}s)')
    print(f'  Results: {OUTPUT_DIR}/')
    print(f'  Plots:')
    print(f'    holonomy_distributions.png          — violin + histogram of kappa')
    print(f'    curvature_distributions.png         — violin + histogram of curvature')
    print(f'    layer_holonomy_profiles.png         — per-layer kappa (VFE steps)')
    print(f'    layer_curvature_profiles.png        — per-layer curvature')
    print(f'    paired_holonomy.png                 — paired comparison')
    print(f'    paired_curvature.png                — paired curvature comparison')
    print(f'    phrase_localized_holonomy.png       — target-only holonomy (length controlled)')
    print(f'    phrase_localized_curvature.png      — target-only curvature (length controlled)')
    print(f'    phrase_localized_layer_profiles.png — target-only layer profiles')
    print(f'    phrase_localized_layer_curvature.png— target-only layer curvature')
    print(f'    cross_boundary_holonomy.png         — target+context holonomy')
    print(f'    cross_boundary_curvature.png        — target+context curvature')
    print(f'    cross_boundary_layer_profiles.png   — target+context layer profiles')
    print(f'    double_dissociation.png             — holonomy vs curvature synthesis')


if __name__ == '__main__':
    main()
