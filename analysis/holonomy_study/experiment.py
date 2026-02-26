"""
Holonomy Study: Main Experiment Pipeline
=========================================

Runs the full holonomy measurement pipeline:
    1. Load pretrained transformer
    2. Load irony/literal/control dataset
    3. Extract transport operators for each sentence
    4. Compute holonomy statistics
    5. Statistical comparison between conditions
    6. Generate plots

Usage:
    python -m analysis.holonomy_study.experiment [--method 0|1|2] [--model gpt2]
"""

import torch
import numpy as np
from scipy import stats
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import time
import json
from pathlib import Path

from .transport import (
    load_model,
    attention_flow_asymmetry,
    attention_decomposed_transport,
    jacobian_transport,
)
from .holonomy import sentence_holonomy, layer_resolved_holonomy, HolonomyResult
from .datasets import load_irony_pairs, by_label, get_paired_only, SentencePair


@dataclass
class ExperimentResult:
    """Full experiment results."""
    # Per-sentence holonomy results, keyed by label
    results_by_label: Dict[str, List[HolonomyResult]]

    # Statistical tests
    ironic_vs_literal: Optional[dict] = None
    ironic_vs_control: Optional[dict] = None
    literal_vs_control: Optional[dict] = None

    # Method 0 results (if run)
    asymmetry_by_label: Optional[Dict[str, np.ndarray]] = None

    # Layer-resolved results (subset)
    layer_resolved: Optional[dict] = None

    metadata: dict = field(default_factory=dict)


def run_holonomy_study(
    model_name: str = 'gpt2',
    method: int = 1,
    device: str = 'cpu',
    max_triangles: int = 300,
    n_probes: int = 50,
    run_layer_resolved: bool = False,
    n_layer_resolved_samples: int = 5,
    paired_only: bool = False,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run the full holonomy study.

    Args:
        model_name: HuggingFace model name
        method: 0 (attention asymmetry), 1 (attention-decomposed), 2 (Jacobian)
        device: 'cpu' or 'cuda'
        max_triangles: max triangles per sentence
        n_probes: probe directions for method 2
        run_layer_resolved: also compute depth-resolved holonomy
        n_layer_resolved_samples: how many sentences per label for layer analysis
        paired_only: use only paired ironic/literal examples
        output_dir: save results here (if provided)
        verbose: print progress

    Returns:
        ExperimentResult
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    if verbose:
        print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print(f"  Device: {device}")

    # --- Load dataset ---
    all_pairs = load_irony_pairs()
    if paired_only:
        all_pairs = get_paired_only(all_pairs)
    groups = by_label(all_pairs)

    if verbose:
        for label, items in groups.items():
            print(f"  {label}: {len(items)} sentences")

    # --- Run Method 0 (always, it's free) ---
    if verbose:
        print("\n--- Method 0: Attention flow asymmetry ---")

    asymmetry_by_label = {}
    for label, items in groups.items():
        asymmetries = []
        for sp in items:
            input_ids = _tokenize(tokenizer, sp.text, device)
            result = attention_flow_asymmetry(model, input_ids)
            # Mean asymmetry across all layers and triangles
            mean_asym = result['asymmetry_per_layer'].mean().item()
            asymmetries.append(mean_asym)
        asymmetry_by_label[label] = np.array(asymmetries)
        if verbose:
            print(f"  {label}: mean asymmetry = {np.mean(asymmetries):.4f} "
                  f"+/- {np.std(asymmetries):.4f}")

    _print_comparison(asymmetry_by_label, "Asymmetry", verbose)

    if method == 0:
        return ExperimentResult(
            results_by_label={},
            asymmetry_by_label=asymmetry_by_label,
            metadata={'model': model_name, 'method': 0},
        )

    # --- Run Method 1 or 2: full transport + holonomy ---
    if verbose:
        method_name = "Attention-decomposed" if method == 1 else "Jacobian probing"
        print(f"\n--- Method {method}: {method_name} transport ---")

    results_by_label: Dict[str, List[HolonomyResult]] = {
        'ironic': [], 'literal': [], 'control': []
    }

    total = sum(len(v) for v in groups.values())
    done = 0

    for label, items in groups.items():
        for sp in items:
            t0 = time.time()
            input_ids = _tokenize(tokenizer, sp.text, device)
            N = input_ids.shape[1]

            if N < 3:
                done += 1
                continue

            # Extract transport
            if method == 1:
                tr = attention_decomposed_transport(model, input_ids)
            else:
                wte = model.wte if hasattr(model, 'wte') else model.transformer.wte
                tr = jacobian_transport(
                    model, input_ids, wte, n_probes=n_probes
                )

            # Compute holonomy
            hr = sentence_holonomy(tr, max_triangles=max_triangles)
            hr.metadata['text'] = sp.text
            hr.metadata['label'] = label
            hr.metadata['pair_id'] = sp.pair_id
            results_by_label[label].append(hr)

            done += 1
            elapsed = time.time() - t0
            if verbose:
                print(f"  [{done}/{total}] {label}: kappa={hr.kappa_mean:.4f} "
                      f"({N} tokens, {elapsed:.1f}s)")

    # --- Statistical tests ---
    if verbose:
        print("\n--- Statistical Analysis ---")

    kappas = {
        label: np.array([hr.kappa_mean for hr in results])
        for label, results in results_by_label.items()
        if len(results) > 0
    }

    ironic_vs_literal = _stat_test(kappas, 'ironic', 'literal', verbose)
    ironic_vs_control = _stat_test(kappas, 'ironic', 'control', verbose)
    literal_vs_control = _stat_test(kappas, 'literal', 'control', verbose)

    # --- Layer-resolved (optional) ---
    layer_results = None
    if run_layer_resolved and method == 1:
        if verbose:
            print("\n--- Layer-Resolved Holonomy ---")

        layer_results = {}
        for label in ['ironic', 'literal']:
            items = groups[label][:n_layer_resolved_samples]
            for sp in items:
                input_ids = _tokenize(tokenizer, sp.text, device)
                if input_ids.shape[1] < 3:
                    continue
                lr = layer_resolved_holonomy(
                    model, input_ids,
                    attention_decomposed_transport,
                    max_triangles=100,
                )
                key = f"{label}_{sp.pair_id}"
                layer_results[key] = {
                    depth: hr.kappa_mean for depth, hr in lr.items()
                }
                if verbose:
                    depths = sorted(lr.keys())
                    profile = [f"{d}:{lr[d].kappa_mean:.3f}" for d in depths]
                    print(f"  {label} (pair {sp.pair_id}): {', '.join(profile)}")

    # --- Build result ---
    result = ExperimentResult(
        results_by_label=results_by_label,
        ironic_vs_literal=ironic_vs_literal,
        ironic_vs_control=ironic_vs_control,
        literal_vs_control=literal_vs_control,
        asymmetry_by_label=asymmetry_by_label,
        layer_resolved=layer_results,
        metadata={
            'model': model_name,
            'method': method,
            'max_triangles': max_triangles,
            'n_probes': n_probes if method == 2 else None,
            'paired_only': paired_only,
        },
    )

    # --- Save ---
    if output_dir:
        _save_results(result, output_dir, verbose)

    return result


# =========================================================================
# Helpers
# =========================================================================

def _tokenize(tokenizer, text: str, device: str) -> torch.Tensor:
    """Tokenize text and return input_ids tensor."""
    tokens = tokenizer.encode(text)
    return torch.tensor([tokens], device=device)


def _stat_test(
    kappas: dict, label_a: str, label_b: str, verbose: bool
) -> Optional[dict]:
    """Run Mann-Whitney U test comparing two conditions."""
    if label_a not in kappas or label_b not in kappas:
        return None

    a = kappas[label_a]
    b = kappas[label_b]

    if len(a) < 2 or len(b) < 2:
        return None

    U, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n1, n2 = len(a), len(b)
    r = 1 - (2 * U) / (n1 * n2)

    # Cohen's d
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    result = {
        'U': float(U),
        'p_value': float(p_value),
        'rank_biserial_r': float(r),
        'cohens_d': float(cohens_d),
        f'mean_{label_a}': float(np.mean(a)),
        f'mean_{label_b}': float(np.mean(b)),
        f'std_{label_a}': float(np.std(a)),
        f'std_{label_b}': float(np.std(b)),
        f'n_{label_a}': n1,
        f'n_{label_b}': n2,
    }

    if verbose:
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  {label_a} vs {label_b}: U={U:.0f}, p={p_value:.4f} {sig}, "
              f"d={cohens_d:.3f}, "
              f"mean {np.mean(a):.4f} vs {np.mean(b):.4f}")

    return result


def _print_comparison(data: dict, metric_name: str, verbose: bool):
    """Quick comparison printout for Method 0."""
    if not verbose:
        return

    labels = [l for l in ['ironic', 'literal', 'control'] if l in data]
    if len(labels) < 2:
        return

    for i, la in enumerate(labels):
        for lb in labels[i+1:]:
            a, b = data[la], data[lb]
            if len(a) >= 2 and len(b) >= 2:
                U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                print(f"  {metric_name} {la} vs {lb}: p={p:.4f}")


def _save_results(result: ExperimentResult, output_dir: str, verbose: bool):
    """Save results to JSON."""
    out = Path(output_dir)

    # Save summary statistics (JSON-serializable)
    summary = {
        'metadata': result.metadata,
        'ironic_vs_literal': result.ironic_vs_literal,
        'ironic_vs_control': result.ironic_vs_control,
        'literal_vs_control': result.literal_vs_control,
    }

    # Per-sentence kappas
    for label, hrs in result.results_by_label.items():
        summary[f'{label}_kappas'] = [hr.kappa_mean for hr in hrs]
        summary[f'{label}_kappa_medians'] = [hr.kappa_median for hr in hrs]

    if result.asymmetry_by_label:
        for label, arr in result.asymmetry_by_label.items():
            summary[f'{label}_asymmetry'] = arr.tolist()

    if result.layer_resolved:
        summary['layer_resolved'] = result.layer_resolved

    path = out / 'holonomy_results.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nResults saved to {path}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Holonomy study')
    parser.add_argument('--method', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max-triangles', type=int, default=300)
    parser.add_argument('--n-probes', type=int, default=50)
    parser.add_argument('--paired-only', action='store_true')
    parser.add_argument('--layer-resolved', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results/holonomy_study')
    args = parser.parse_args()

    result = run_holonomy_study(
        model_name=args.model,
        method=args.method,
        device=args.device,
        max_triangles=args.max_triangles,
        n_probes=args.n_probes,
        run_layer_resolved=args.layer_resolved,
        paired_only=args.paired_only,
        output_dir=args.output_dir,
    )
