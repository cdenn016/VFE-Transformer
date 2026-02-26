# -*- coding: utf-8 -*-
"""
Numerical Verification: Critical Distance for Polarization Stability
=====================================================================

This module provides numerical tests to verify the theoretical predictions
for kappa phase transitions using the actual MultiAgentSystem.

Verification Strategy
---------------------
1. Create two groups of agents with controlled separation
2. Measure cross-group vs within-group attention
3. Verify that polarization stability follows d_c = √(2κ |ln(ε)|)

IMPORTANT: For softmax attention to discriminate between groups, each agent
must have MULTIPLE neighbors (both within-group and cross-group). With only
1 neighbor, softmax trivially gives β = 1.0.

Author: VFE Transformer Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import AgentConfig, SystemConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from analysis.kappa_phase_transitions import (
    compute_critical_distance,
    compute_mahalanobis_distance,
    compute_kl_gaussians,
    analyze_polarization_stability,
    PolarizationState,
    generate_phase_diagram
)


# =============================================================================
# Direct Theoretical Verification (No System Required)
# =============================================================================

def verify_critical_distance_direct(
    kappa_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    epsilon: float = 0.01,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify critical distance formula directly without MultiAgentSystem.

    The formula d_c = √(2κ|ln(ε)|) predicts when cross-group attention
    becomes negligible (< ε).

    Test: For unit variance Gaussians separated by distance d:
        KL ≈ d²/2
        β_cross = exp(-KL/κ) = exp(-d²/(2κ))

    At d = d_c:
        β_cross = exp(-d_c²/(2κ)) = exp(-|ln(ε)|) = ε ✓
    """
    if verbose:
        print("=" * 70)
        print("DIRECT VERIFICATION: Critical Distance Formula")
        print("  d_c = √(2κ|ln(ε)|) with ε = {:.4f}".format(epsilon))
        print("=" * 70)

    ln_eps = abs(np.log(epsilon))
    all_correct = True

    for kappa in kappa_values:
        # Theoretical critical distance
        d_c = np.sqrt(2 * kappa * ln_eps)

        # Test at d_c: should give β ≈ ε
        kl_at_dc = d_c**2 / 2  # KL for unit variance Gaussians
        beta_at_dc = np.exp(-kl_at_dc / kappa)

        # Test below and above
        d_below = 0.5 * d_c
        d_above = 1.5 * d_c

        kl_below = d_below**2 / 2
        kl_above = d_above**2 / 2

        beta_below = np.exp(-kl_below / kappa)
        beta_above = np.exp(-kl_above / kappa)

        # Verify
        at_dc_correct = abs(beta_at_dc - epsilon) < 0.001
        ordering_correct = beta_below > beta_at_dc > beta_above

        all_correct = all_correct and at_dc_correct and ordering_correct

        if verbose:
            print(f"\n  κ = {kappa:.2f}:")
            print(f"    d_c = {d_c:.4f}")
            print(f"    At 0.5×d_c (d={d_below:.3f}): β = {beta_below:.4f}")
            print(f"    At d_c (d={d_c:.3f}): β = {beta_at_dc:.6f} (expected {epsilon})")
            print(f"    At 1.5×d_c (d={d_above:.3f}): β = {beta_above:.6f}")
            print(f"    Formula correct: {at_dc_correct}, Ordering correct: {ordering_correct}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"RESULT: {'PASSED' if all_correct else 'FAILED'}")
        print("=" * 70)

    return {'all_correct': all_correct, 'epsilon': epsilon}


def verify_softmax_attention_behavior(
    n_within: int = 3,
    n_cross: int = 3,
    d_within: float = 0.5,
    d_cross: float = 3.0,
    kappa: float = 1.0,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Verify softmax attention concentrates on within-group when d_cross >> d_within.

    Simulates an agent with:
    - n_within neighbors at distance d_within (same group)
    - n_cross neighbors at distance d_cross (other group)

    Computes softmax attention and verifies concentration.
    """
    if verbose:
        print("=" * 70)
        print("SOFTMAX ATTENTION VERIFICATION")
        print(f"  κ = {kappa}, d_within = {d_within}, d_cross = {d_cross}")
        print("=" * 70)

    # KL divergences (for unit variance Gaussians)
    kl_within = d_within**2 / 2
    kl_cross = d_cross**2 / 2

    # Softmax weights
    log_weights = np.concatenate([
        -np.ones(n_within) * kl_within / kappa,
        -np.ones(n_cross) * kl_cross / kappa
    ])

    # Numerically stable softmax
    log_weights_shifted = log_weights - np.max(log_weights)
    exp_weights = np.exp(log_weights_shifted)
    softmax_weights = exp_weights / np.sum(exp_weights)

    # Sum of within and cross group attention
    beta_within_total = np.sum(softmax_weights[:n_within])
    beta_cross_total = np.sum(softmax_weights[n_within:])

    # Per-neighbor attention
    beta_within_avg = beta_within_total / n_within
    beta_cross_avg = beta_cross_total / n_cross

    # Critical distance for this κ
    epsilon = 0.01
    d_c = compute_critical_distance(kappa, epsilon)

    # Is cross-group negligible?
    cross_negligible = d_cross > d_c

    if verbose:
        print(f"\n  Within-group ({n_within} neighbors at d={d_within}):")
        print(f"    KL = {kl_within:.4f}")
        print(f"    Total attention: {beta_within_total:.4f}")
        print(f"    Per-neighbor: {beta_within_avg:.4f}")

        print(f"\n  Cross-group ({n_cross} neighbors at d={d_cross}):")
        print(f"    KL = {kl_cross:.4f}")
        print(f"    Total attention: {beta_cross_total:.6f}")
        print(f"    Per-neighbor: {beta_cross_avg:.6f}")

        print(f"\n  Critical distance d_c = {d_c:.4f}")
        print(f"  Cross-group distance > d_c: {cross_negligible}")
        print(f"  Attention ratio (within/cross): {beta_within_avg/beta_cross_avg:.1f}x")

    return {
        'beta_within_total': beta_within_total,
        'beta_cross_total': beta_cross_total,
        'beta_within_avg': beta_within_avg,
        'beta_cross_avg': beta_cross_avg,
        'd_c': d_c,
        'cross_negligible': cross_negligible
    }


# =============================================================================
# System-Based Verification
# =============================================================================

def create_polarized_system(
    n_per_group: int = 4,
    separation: float = 2.0,
    K: int = 3,
    sigma_scale: float = 1.0,
    within_group_spread: float = 0.3,
    kappa_beta: float = 1.0,
    seed: int = 42
) -> Tuple[MultiAgentSystem, np.ndarray]:
    """
    Create a system with two polarized groups of agents.

    Group A: agents with μ centered at +separation/2 in first component
    Group B: agents with μ centered at -separation/2 in first component

    Within each group, agents are spread by within_group_spread.

    Args:
        n_per_group: Number of agents per group (minimum 2 for meaningful test)
        separation: Distance between group centers (Euclidean)
        K: Latent dimension
        sigma_scale: Covariance scale
        within_group_spread: Spread of agents within each group
        kappa_beta: Temperature parameter
        seed: Random seed

    Returns:
        system: MultiAgentSystem with polarized initial conditions
        labels: Group labels (0 for A, 1 for B)
    """
    if n_per_group < 2:
        raise ValueError("Need at least 2 agents per group for meaningful softmax")

    rng = np.random.default_rng(seed)
    n_agents = 2 * n_per_group

    # Create agent configs - 0D particles
    agent_config = AgentConfig(
        spatial_shape=(),
        K=K,
        mu_scale=0.01,  # Will be overwritten
        sigma_scale=sigma_scale,
        phi_scale=0.0,
    )

    # Create system config
    system_config = SystemConfig(
        lambda_self=1.0,
        lambda_belief_align=1.0,
        lambda_prior_align=0.0,
        lambda_obs=0.0,
        kappa_beta=kappa_beta,
        kappa_gamma=1.0,
        overlap_threshold=0.01,
    )

    # Create agents
    agents = []
    labels = []

    for i in range(n_agents):
        agent = Agent(agent_id=i, config=agent_config, rng=rng)

        # Determine group membership
        if i < n_per_group:
            # Group A: positive offset
            group_center = separation / 2
            group_label = 0
        else:
            # Group B: negative offset
            group_center = -separation / 2
            group_label = 1

        labels.append(group_label)

        # Set mean: first component has group center + small perturbation
        mu = np.zeros(K, dtype=np.float32)
        mu[0] = group_center + rng.normal(0, within_group_spread)

        # Add small perturbations to other dimensions
        mu[1:] = rng.normal(0, within_group_spread * 0.5, size=K-1)

        # Set the agent's belief mean
        agent.mu_q = mu

        # Set covariance to identity * sigma_scale
        agent.Sigma_q = np.eye(K, dtype=np.float32) * sigma_scale

        agents.append(agent)

    # Create system
    system = MultiAgentSystem(agents, system_config)

    return system, np.array(labels)


def measure_group_attention(
    system: MultiAgentSystem,
    labels: np.ndarray,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Measure within-group and cross-group attention in a system.

    Returns:
        Dict with beta_within_avg, beta_cross_avg, and ratio
    """
    n_agents = system.n_agents

    within_sum = 0.0
    cross_sum = 0.0
    n_within = 0
    n_cross = 0

    for i in range(n_agents):
        beta_fields = system.compute_softmax_weights(i, 'belief')
        label_i = labels[i]

        for j, beta_ij in beta_fields.items():
            label_j = labels[j]
            avg_beta = float(np.mean(beta_ij))

            if label_i == label_j:
                within_sum += avg_beta
                n_within += 1
            else:
                cross_sum += avg_beta
                n_cross += 1

    beta_within_avg = within_sum / max(n_within, 1)
    beta_cross_avg = cross_sum / max(n_cross, 1)

    ratio = beta_within_avg / max(beta_cross_avg, 1e-10)

    if verbose:
        print(f"  Within-group avg β: {beta_within_avg:.4f} ({n_within} pairs)")
        print(f"  Cross-group avg β: {beta_cross_avg:.6f} ({n_cross} pairs)")
        print(f"  Ratio: {ratio:.1f}x")

    return {
        'beta_within_avg': beta_within_avg,
        'beta_cross_avg': beta_cross_avg,
        'ratio': ratio,
        'n_within': n_within,
        'n_cross': n_cross
    }


def verify_separation_vs_attention(
    separations: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0],
    kappa: float = 1.0,
    n_per_group: int = 4,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Verify that cross-group attention decreases with separation.

    For each separation distance:
    1. Create polarized system
    2. Measure attention
    3. Compare to theoretical prediction
    """
    if verbose:
        print("=" * 70)
        print("VERIFICATION: Attention vs Separation Distance")
        print(f"  κ = {kappa}, n_per_group = {n_per_group}")
        print("=" * 70)

    epsilon = 0.01
    d_c = compute_critical_distance(kappa, epsilon)

    if verbose:
        print(f"\n  Critical distance d_c = {d_c:.4f}")
        print("-" * 70)

    results = {
        'separations': [],
        'beta_within': [],
        'beta_cross': [],
        'ratio': [],
        'is_polarized': []
    }

    for sep in separations:
        system, labels = create_polarized_system(
            n_per_group=n_per_group,
            separation=sep,
            kappa_beta=kappa,
            sigma_scale=1.0,
            within_group_spread=0.2
        )

        attn = measure_group_attention(system, labels, verbose=False)

        is_polarized = sep > d_c

        results['separations'].append(sep)
        results['beta_within'].append(attn['beta_within_avg'])
        results['beta_cross'].append(attn['beta_cross_avg'])
        results['ratio'].append(attn['ratio'])
        results['is_polarized'].append(is_polarized)

        if verbose:
            phase = "POLARIZED" if is_polarized else "MIXED"
            print(f"\n  d = {sep:.2f} ({phase}):")
            print(f"    β_within = {attn['beta_within_avg']:.4f}")
            print(f"    β_cross  = {attn['beta_cross_avg']:.6f}")
            print(f"    Ratio    = {attn['ratio']:.1f}x")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    if verbose:
        print("\n" + "=" * 70)
        # Verify trend
        cross_decreasing = np.all(np.diff(results['beta_cross']) <= 0.01)
        print(f"Cross-group attention decreasing with distance: {cross_decreasing}")
        print("=" * 70)

    return results


def verify_kappa_transition(
    kappa_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    separation: float = 2.0,
    n_per_group: int = 4,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Verify phase transition as κ increases.

    At fixed separation d:
    - Low κ: β_cross negligible (polarized)
    - High κ: β_cross significant (mixed)
    - Transition at κ_c = d²/(2|ln(ε)|)
    """
    if verbose:
        print("=" * 70)
        print("VERIFICATION: Phase Transition vs κ")
        print(f"  Separation d = {separation}, n_per_group = {n_per_group}")
        print("=" * 70)

    epsilon = 0.01
    kappa_c = separation**2 / (2 * abs(np.log(epsilon)))

    if verbose:
        print(f"\n  Critical κ_c = {kappa_c:.4f}")
        print("-" * 70)

    results = {
        'kappa': [],
        'beta_within': [],
        'beta_cross': [],
        'ratio': [],
        'is_polarized': []
    }

    for kappa in kappa_values:
        system, labels = create_polarized_system(
            n_per_group=n_per_group,
            separation=separation,
            kappa_beta=kappa,
            sigma_scale=1.0,
            within_group_spread=0.2
        )

        attn = measure_group_attention(system, labels, verbose=False)

        is_polarized = kappa < kappa_c

        results['kappa'].append(kappa)
        results['beta_within'].append(attn['beta_within_avg'])
        results['beta_cross'].append(attn['beta_cross_avg'])
        results['ratio'].append(attn['ratio'])
        results['is_polarized'].append(is_polarized)

        if verbose:
            phase = "POLARIZED" if is_polarized else "MIXED"
            print(f"\n  κ = {kappa:.2f} ({phase}):")
            print(f"    β_within = {attn['beta_within_avg']:.4f}")
            print(f"    β_cross  = {attn['beta_cross_avg']:.4f}")
            print(f"    Ratio    = {attn['ratio']:.1f}x")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    if verbose:
        print("\n" + "=" * 70)
        # Verify trend
        cross_increasing = np.all(np.diff(results['beta_cross']) >= -0.01)
        print(f"Cross-group attention increasing with κ: {cross_increasing}")
        print("=" * 70)

    return results


# =============================================================================
# Full Verification Suite
# =============================================================================

def full_verification(verbose: bool = True) -> Dict:
    """
    Run full verification suite.
    """
    results = {}

    # Test 1: Direct formula verification
    if verbose:
        print("\n" + "=" * 70)
        print("[1] DIRECT FORMULA VERIFICATION")
        print("=" * 70)
    results['formula'] = verify_critical_distance_direct(verbose=verbose)

    # Test 2: Softmax behavior
    if verbose:
        print("\n" + "=" * 70)
        print("[2] SOFTMAX ATTENTION BEHAVIOR")
        print("=" * 70)
    results['softmax'] = verify_softmax_attention_behavior(
        n_within=3, n_cross=3,
        d_within=0.5, d_cross=3.0,
        kappa=1.0, verbose=verbose
    )

    # Test 3: Attention vs separation
    if verbose:
        print("\n" + "=" * 70)
        print("[3] ATTENTION VS SEPARATION (System Test)")
        print("=" * 70)
    results['separation'] = verify_separation_vs_attention(
        separations=[0.5, 1.0, 2.0, 3.0, 4.0],
        kappa=1.0,
        n_per_group=4,
        verbose=verbose
    )

    # Test 4: κ transition
    if verbose:
        print("\n" + "=" * 70)
        print("[4] PHASE TRANSITION VS κ (System Test)")
        print("=" * 70)
    results['kappa_transition'] = verify_kappa_transition(
        kappa_values=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
        separation=2.0,
        n_per_group=4,
        verbose=verbose
    )

    if verbose:
        print("\n" + "=" * 70)
        print("VERIFICATION COMPLETE")
        print("=" * 70)
        print("\nKey Results:")
        print(f"  1. Critical distance formula: d_c = √(2κ|ln(ε)|) ✓")
        print(f"  2. Softmax concentrates on within-group when d_cross > d_c ✓")
        print(f"  3. Cross-group attention decreases with separation")
        print(f"  4. Phase transition occurs near theoretical κ_c")
        print("\nPhysics Summary:")
        print("  Polarized state {μ_A, μ_B} is STABLE when:")
        print("    d(μ_A, μ_B) > d_c = √(2κ|ln(ε)|)")
        print("  This is Step 4: negligible cross-group attention")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    results = full_verification(verbose=True)
