# -*- coding: utf-8 -*-
"""
Kappa Phase Transitions: Critical Distance for Polarization Stability
======================================================================

Step 4 Analysis: Polarized state {μ_A, μ_B} is stable when cross-group
attention is negligible.

Mathematical Framework
----------------------
For a system with two groups A and B with beliefs:
    q_A = N(μ_A, Σ_A)
    q_B = N(μ_B, Σ_B)

Cross-group attention weights:
    β_AB = exp(-KL(q_A || Ω_AB[q_B]) / κ) / Z

where Z is the softmax normalization.

Polarization Stability Condition
--------------------------------
The polarized state is STABLE when cross-group attention is negligible:
    β_AB << ε    (ε ~ threshold for "negligible")

This occurs when:
    KL(q_A || q_B) >> κ

For Gaussians with similar covariances Σ:
    KL ≈ (1/2) (μ_A - μ_B)^T Σ^{-1} (μ_A - μ_B) = d²_M / 2

where d_M is the Mahalanobis distance.

Critical Distance
-----------------
Polarization becomes stable when:
    exp(-d²_M / (2κ)) << ε
    d²_M >> -2κ ln(ε)

Critical distance:
    d_c = √(2κ · |ln(ε)|)

Below d_c: Cross-group coupling dominates → consensus/mixing
Above d_c: Groups decouple → stable polarization

Phase Diagram
-------------
In (κ, d) space:
- Region I  (d < d_c(κ)): Mixed/consensus phase
- Region II (d > d_c(κ)): Polarized phase
- Critical line: d = d_c(κ) ∝ √κ

Author: VFE Transformer Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
from dataclasses import dataclass, field
import torch


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PolarizationState:
    """
    Represents a polarized state with two groups.

    Attributes:
        mu_A: Mean of group A (K,) or (*S, K)
        mu_B: Mean of group B (K,) or (*S, K)
        Sigma_A: Covariance of group A (K, K) or (*S, K, K)
        Sigma_B: Covariance of group B (K, K) or (*S, K, K)
        n_A: Number of agents in group A
        n_B: Number of agents in group B
    """
    mu_A: np.ndarray
    mu_B: np.ndarray
    Sigma_A: np.ndarray
    Sigma_B: np.ndarray
    n_A: int = 1
    n_B: int = 1


@dataclass
class StabilityAnalysis:
    """
    Results of polarization stability analysis.

    Attributes:
        kappa: Temperature parameter
        distance_mahalanobis: Mahalanobis distance between group means
        distance_euclidean: Euclidean distance between group means
        critical_distance: Critical distance for stability
        cross_group_attention: β_AB attention weight
        within_group_attention_A: β_AA attention weight
        within_group_attention_B: β_BB attention weight
        is_stable: True if polarized state is stable
        stability_margin: How far above critical (d - d_c)
    """
    kappa: float
    distance_mahalanobis: float
    distance_euclidean: float
    critical_distance: float
    cross_group_attention: float
    within_group_attention_A: float
    within_group_attention_B: float
    is_stable: bool
    stability_margin: float


@dataclass
class PhaseTransitionPoint:
    """
    A point on the phase transition boundary.

    Attributes:
        kappa: Temperature at transition
        critical_distance: Distance at transition
        order_parameter: Value of order parameter (0 at transition)
    """
    kappa: float
    critical_distance: float
    order_parameter: float = 0.0


@dataclass
class PhaseDiagram:
    """
    Phase diagram in (κ, d) space.

    Attributes:
        kappa_range: Array of κ values
        critical_curve: d_c(κ) for each κ
        stability_threshold: ε used for stability criterion
        phase_labels: Labels for each region
    """
    kappa_range: np.ndarray
    critical_curve: np.ndarray
    stability_threshold: float

    def get_phase(self, kappa: float, distance: float) -> str:
        """Determine phase for given (κ, d) point."""
        d_c = self.critical_distance(kappa)
        if distance < d_c:
            return "mixed"
        else:
            return "polarized"

    def critical_distance(self, kappa: float) -> float:
        """Interpolate critical distance for given κ."""
        return np.interp(kappa, self.kappa_range, self.critical_curve)


# =============================================================================
# Core Calculations: KL Divergence and Attention
# =============================================================================

def compute_kl_gaussians(
    mu_1: np.ndarray,
    Sigma_1: np.ndarray,
    mu_2: np.ndarray,
    Sigma_2: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Compute KL(N(μ_1, Σ_1) || N(μ_2, Σ_2)).

    KL = (1/2) [tr(Σ_2^{-1} Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1)
                - K + ln(det Σ_2 / det Σ_1)]

    Args:
        mu_1: Mean of distribution 1, shape (..., K)
        Sigma_1: Covariance of distribution 1, shape (..., K, K)
        mu_2: Mean of distribution 2, shape (..., K)
        Sigma_2: Covariance of distribution 2, shape (..., K, K)
        eps: Regularization for numerical stability

    Returns:
        kl: KL divergence, shape (...)
    """
    # Handle dimensionality
    if mu_1.ndim == 1:
        K = mu_1.shape[0]
        mu_1 = mu_1.reshape(1, K)
        mu_2 = mu_2.reshape(1, K)
        Sigma_1 = Sigma_1.reshape(1, K, K)
        Sigma_2 = Sigma_2.reshape(1, K, K)
        squeeze_output = True
    else:
        K = mu_1.shape[-1]
        squeeze_output = False

    # Add regularization
    Sigma_1_reg = Sigma_1 + eps * np.eye(K)
    Sigma_2_reg = Sigma_2 + eps * np.eye(K)

    # Compute Σ_2^{-1}
    Sigma_2_inv = np.linalg.inv(Sigma_2_reg)

    # Mean difference
    delta_mu = mu_2 - mu_1  # (..., K)

    # Trace term: tr(Σ_2^{-1} Σ_1)
    trace_term = np.trace(Sigma_2_inv @ Sigma_1_reg, axis1=-2, axis2=-1)

    # Quadratic term: (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1)
    # Shape: (..., K) @ (..., K, K) @ (..., K) -> (...)
    quad_term = np.einsum('...i,...ij,...j->...', delta_mu, Sigma_2_inv, delta_mu)

    # Log determinant term
    _, logdet_1 = np.linalg.slogdet(Sigma_1_reg)
    _, logdet_2 = np.linalg.slogdet(Sigma_2_reg)
    logdet_term = logdet_2 - logdet_1

    # KL divergence
    kl = 0.5 * (trace_term + quad_term - K + logdet_term)

    if squeeze_output:
        kl = kl.squeeze()

    return kl


def compute_mahalanobis_distance(
    mu_1: np.ndarray,
    mu_2: np.ndarray,
    Sigma: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute Mahalanobis distance: d_M = √[(μ_1 - μ_2)^T Σ^{-1} (μ_1 - μ_2)]

    This is the key geometric quantity for polarization stability.
    For identical covariances, KL ≈ d²_M / 2.

    Args:
        mu_1: First mean (K,)
        mu_2: Second mean (K,)
        Sigma: Common covariance (K, K)
        eps: Regularization

    Returns:
        d_M: Mahalanobis distance (scalar)
    """
    K = mu_1.shape[-1]
    Sigma_reg = Sigma + eps * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)

    delta_mu = mu_1 - mu_2
    d_squared = delta_mu @ Sigma_inv @ delta_mu

    return np.sqrt(max(0, d_squared))


def compute_cross_group_attention(
    kl_AB: float,
    kl_AA: float,
    kl_BB: float,
    kappa: float
) -> Tuple[float, float, float]:
    """
    Compute softmax attention weights for two-group system.

    β_AB = exp(-KL_AB/κ) / Z
    β_AA = exp(-KL_AA/κ) / Z
    β_BB = exp(-KL_BB/κ) / Z

    where Z = exp(-KL_AA/κ) + exp(-KL_AB/κ) + exp(-KL_BB/κ)

    For polarization stability, we want β_AB << β_AA, β_BB.

    Args:
        kl_AB: KL divergence between groups
        kl_AA: KL divergence within group A (typically 0 for identical agents)
        kl_BB: KL divergence within group B
        kappa: Temperature parameter

    Returns:
        (β_AB, β_AA, β_BB): Attention weights
    """
    # Use log-sum-exp for numerical stability
    log_weights = np.array([-kl_AA/kappa, -kl_AB/kappa, -kl_BB/kappa])
    log_weights_max = np.max(log_weights)
    log_weights_shifted = log_weights - log_weights_max

    exp_weights = np.exp(log_weights_shifted)
    Z = np.sum(exp_weights)

    beta_AA = exp_weights[0] / Z
    beta_AB = exp_weights[1] / Z
    beta_BB = exp_weights[2] / Z

    return beta_AB, beta_AA, beta_BB


# =============================================================================
# Critical Distance Calculation
# =============================================================================

def compute_critical_distance(
    kappa: float,
    epsilon: float = 0.01,
    Sigma: Optional[np.ndarray] = None
) -> float:
    """
    Compute critical distance d_c for polarization stability.

    Polarization is stable when:
        exp(-d²_M / (2κ)) << ε

    Solving for d_c:
        d²_M = -2κ ln(ε)
        d_c = √(2κ |ln(ε)|)

    This is the Mahalanobis distance. For Euclidean distance with
    covariance Σ, multiply by average eigenvalue scale.

    Args:
        kappa: Temperature parameter (softmax κ)
        epsilon: Threshold for "negligible" attention
        Sigma: Optional covariance for Euclidean conversion

    Returns:
        d_c: Critical Mahalanobis distance
    """
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

    # Critical Mahalanobis distance
    d_c_mahal = np.sqrt(2 * kappa * abs(np.log(epsilon)))

    # If Sigma provided, convert to approximate Euclidean distance
    if Sigma is not None:
        # Average eigenvalue gives typical scale
        eigvals = np.linalg.eigvalsh(Sigma)
        avg_scale = np.sqrt(np.mean(eigvals))
        d_c_euclidean = d_c_mahal * avg_scale
        return d_c_euclidean

    return d_c_mahal


def compute_critical_kappa(
    distance: float,
    epsilon: float = 0.01
) -> float:
    """
    Compute critical κ for given separation distance.

    Given d, find κ_c such that d = d_c(κ_c):
        d² = 2κ_c |ln(ε)|
        κ_c = d² / (2 |ln(ε)|)

    For κ < κ_c: Polarization is stable (cold regime)
    For κ > κ_c: Groups mix (hot regime)

    Args:
        distance: Mahalanobis distance between groups
        epsilon: Stability threshold

    Returns:
        kappa_c: Critical temperature
    """
    if distance <= 0:
        return 0.0

    kappa_c = distance**2 / (2 * abs(np.log(epsilon)))
    return kappa_c


# =============================================================================
# Stability Analysis
# =============================================================================

def analyze_polarization_stability(
    state: PolarizationState,
    kappa: float,
    epsilon: float = 0.01,
    transport_matrix: Optional[np.ndarray] = None
) -> StabilityAnalysis:
    """
    Analyze stability of a polarized state at given temperature.

    This is the core Step 4 analysis: determining when cross-group
    attention is negligible enough for polarization to persist.

    Args:
        state: PolarizationState with group means and covariances
        kappa: Temperature parameter
        epsilon: Threshold for stability
        transport_matrix: Optional Ω_AB transport (identity if None)

    Returns:
        StabilityAnalysis with detailed metrics
    """
    # Use average covariance for distance computation
    Sigma_avg = 0.5 * (state.Sigma_A + state.Sigma_B)

    # Compute distances
    d_euclidean = np.linalg.norm(state.mu_A - state.mu_B)
    d_mahal = compute_mahalanobis_distance(
        state.mu_A, state.mu_B, Sigma_avg
    )

    # Critical distance at this κ
    d_c = compute_critical_distance(kappa, epsilon)

    # Compute KL divergences
    # If transport provided, apply it to group B
    if transport_matrix is not None:
        mu_B_transported = transport_matrix @ state.mu_B
        Sigma_B_transported = transport_matrix @ state.Sigma_B @ transport_matrix.T
    else:
        mu_B_transported = state.mu_B
        Sigma_B_transported = state.Sigma_B

    # KL(A || B): Cross-group
    kl_AB = compute_kl_gaussians(
        state.mu_A, state.Sigma_A,
        mu_B_transported, Sigma_B_transported
    )

    # Within-group KL (0 for identical agents, but can have variance)
    kl_AA = 0.0  # Simplified: identical agents within group
    kl_BB = 0.0

    # Compute attention weights
    beta_AB, beta_AA, beta_BB = compute_cross_group_attention(
        kl_AB, kl_AA, kl_BB, kappa
    )

    # Stability criterion
    is_stable = d_mahal > d_c
    stability_margin = d_mahal - d_c

    return StabilityAnalysis(
        kappa=kappa,
        distance_mahalanobis=d_mahal,
        distance_euclidean=d_euclidean,
        critical_distance=d_c,
        cross_group_attention=beta_AB,
        within_group_attention_A=beta_AA,
        within_group_attention_B=beta_BB,
        is_stable=is_stable,
        stability_margin=stability_margin
    )


def find_critical_kappa_for_state(
    state: PolarizationState,
    epsilon: float = 0.01,
    kappa_range: Tuple[float, float] = (0.01, 10.0),
    n_points: int = 100
) -> PhaseTransitionPoint:
    """
    Find critical κ where polarization becomes unstable.

    Scans κ range to find transition point where:
        d_M(state) = d_c(κ_c)

    Args:
        state: Current polarization state
        epsilon: Stability threshold
        kappa_range: Range to search
        n_points: Resolution of search

    Returns:
        PhaseTransitionPoint with critical values
    """
    # Compute Mahalanobis distance of current state
    Sigma_avg = 0.5 * (state.Sigma_A + state.Sigma_B)
    d_state = compute_mahalanobis_distance(
        state.mu_A, state.mu_B, Sigma_avg
    )

    # Analytical solution for critical κ
    kappa_c = compute_critical_kappa(d_state, epsilon)

    # Verify it's in range
    if kappa_c < kappa_range[0]:
        kappa_c = kappa_range[0]
    elif kappa_c > kappa_range[1]:
        kappa_c = kappa_range[1]

    return PhaseTransitionPoint(
        kappa=kappa_c,
        critical_distance=d_state,
        order_parameter=0.0
    )


# =============================================================================
# Phase Diagram Generation
# =============================================================================

def generate_phase_diagram(
    kappa_range: Tuple[float, float] = (0.1, 10.0),
    n_kappa: int = 100,
    epsilon: float = 0.01
) -> PhaseDiagram:
    """
    Generate phase diagram showing critical curve d_c(κ).

    The critical curve separates:
    - Mixed phase (d < d_c): Cross-group attention significant
    - Polarized phase (d > d_c): Groups decoupled

    The curve follows: d_c = √(2κ |ln(ε)|)

    Args:
        kappa_range: (kappa_min, kappa_max)
        n_kappa: Number of points
        epsilon: Stability threshold

    Returns:
        PhaseDiagram object
    """
    kappa_values = np.linspace(kappa_range[0], kappa_range[1], n_kappa)
    critical_distances = np.array([
        compute_critical_distance(k, epsilon) for k in kappa_values
    ])

    return PhaseDiagram(
        kappa_range=kappa_values,
        critical_curve=critical_distances,
        stability_threshold=epsilon
    )


def compute_order_parameter(
    system,
    cluster_labels: np.ndarray
) -> float:
    """
    Compute order parameter for polarization.

    Order parameter = 1 - 2 * <β_cross> / (<β_within_A> + <β_within_B>)

    - η = 1: Complete polarization (no cross-group coupling)
    - η = 0: Critical point
    - η < 0: Mixed phase (cross-group dominates)

    Args:
        system: MultiAgentSystem
        cluster_labels: Array of group assignments (0 or 1)

    Returns:
        order_parameter: Value in (-∞, 1]
    """
    n_agents = system.n_agents

    # Compute average within-group and cross-group attention
    beta_within_sum = 0.0
    beta_cross_sum = 0.0
    n_within = 0
    n_cross = 0

    for i in range(n_agents):
        beta_fields = system.compute_softmax_weights(i, 'belief')
        label_i = cluster_labels[i]

        for j, beta_ij in beta_fields.items():
            label_j = cluster_labels[j]
            avg_beta = float(np.mean(beta_ij))

            if label_i == label_j:
                beta_within_sum += avg_beta
                n_within += 1
            else:
                beta_cross_sum += avg_beta
                n_cross += 1

    # Compute average attention weights
    beta_within_avg = beta_within_sum / max(n_within, 1)
    beta_cross_avg = beta_cross_sum / max(n_cross, 1)

    # Order parameter
    if beta_within_avg < 1e-10:
        return 0.0

    order_param = 1 - 2 * beta_cross_avg / beta_within_avg

    return order_param


# =============================================================================
# Temperature Sweep Analysis
# =============================================================================

@dataclass
class KappaSweepResults:
    """
    Results from sweeping κ to find phase transitions.

    Attributes:
        kappa_values: Array of κ values tested
        order_parameters: Order parameter at each κ
        cross_group_attention: β_cross at each κ
        within_group_attention: β_within at each κ
        modularity: Network modularity at each κ
        kappa_critical: Estimated critical κ
    """
    kappa_values: np.ndarray
    order_parameters: np.ndarray
    cross_group_attention: np.ndarray
    within_group_attention: np.ndarray
    modularity: np.ndarray
    kappa_critical: float


def sweep_kappa_for_system(
    system,
    cluster_labels: np.ndarray,
    kappa_range: Tuple[float, float] = (0.1, 10.0),
    n_kappa: int = 50,
    mode: str = 'belief'
) -> KappaSweepResults:
    """
    Sweep κ to find phase transition in existing system.

    For each κ value:
    1. Recompute softmax weights with new κ
    2. Compute order parameter
    3. Compute modularity

    The critical κ is where order parameter crosses 0.

    Args:
        system: MultiAgentSystem
        cluster_labels: Group assignments
        kappa_range: Range to sweep
        n_kappa: Number of points
        mode: 'belief' or 'prior'

    Returns:
        KappaSweepResults with phase transition analysis
    """
    from transformer.rg_metrics import compute_modularity

    kappa_values = np.linspace(kappa_range[0], kappa_range[1], n_kappa)
    order_params = []
    cross_attns = []
    within_attns = []
    modularities = []

    for kappa in kappa_values:
        # Compute attention matrix at this κ
        n_agents = system.n_agents
        beta_matrix = np.zeros((n_agents, n_agents))

        for i in range(n_agents):
            beta_fields = system.compute_softmax_weights(i, mode, kappa=kappa)
            for j, beta_ij in beta_fields.items():
                beta_matrix[i, j] = float(np.mean(beta_ij))

        # Normalize rows
        row_sums = beta_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        beta_matrix = beta_matrix / row_sums

        # Compute metrics
        # Cross vs within attention
        cross_sum = 0.0
        within_sum = 0.0
        n_cross = 0
        n_within = 0

        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                if cluster_labels[i] == cluster_labels[j]:
                    within_sum += beta_matrix[i, j]
                    n_within += 1
                else:
                    cross_sum += beta_matrix[i, j]
                    n_cross += 1

        cross_avg = cross_sum / max(n_cross, 1)
        within_avg = within_sum / max(n_within, 1)

        cross_attns.append(cross_avg)
        within_attns.append(within_avg)

        # Order parameter
        if within_avg > 1e-10:
            order_param = 1 - 2 * cross_avg / within_avg
        else:
            order_param = 0.0
        order_params.append(order_param)

        # Modularity
        mod = compute_modularity(
            torch.from_numpy(beta_matrix).float(),
            torch.from_numpy(cluster_labels).long()
        )
        modularities.append(mod)

    # Convert to arrays
    order_params = np.array(order_params)
    cross_attns = np.array(cross_attns)
    within_attns = np.array(within_attns)
    modularities = np.array(modularities)

    # Find critical κ (where order parameter crosses 0)
    # Use linear interpolation
    sign_changes = np.where(np.diff(np.sign(order_params)) != 0)[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation
        kappa_c = kappa_values[idx] + (
            -order_params[idx] * (kappa_values[idx+1] - kappa_values[idx]) /
            (order_params[idx+1] - order_params[idx] + 1e-10)
        )
    else:
        # No crossing found
        if order_params[0] > 0:
            kappa_c = kappa_range[1]  # Always polarized
        else:
            kappa_c = kappa_range[0]  # Always mixed

    return KappaSweepResults(
        kappa_values=kappa_values,
        order_parameters=order_params,
        cross_group_attention=cross_attns,
        within_group_attention=within_attns,
        modularity=modularities,
        kappa_critical=kappa_c
    )


# =============================================================================
# Analytical Predictions
# =============================================================================

def predict_critical_exponents() -> Dict[str, float]:
    """
    Return theoretical critical exponents for κ phase transition.

    Near the critical point κ_c, we expect mean-field scaling:

    Order parameter: η ~ (κ_c - κ)^β  for κ < κ_c
    Susceptibility:  χ ~ |κ - κ_c|^{-γ}
    Correlation:     ξ ~ |κ - κ_c|^{-ν}

    For softmax attention with Gaussian beliefs (mean-field):
        β = 1/2 (order parameter)
        γ = 1   (susceptibility)
        ν = 1/2 (correlation length)

    Returns:
        Dict with critical exponent names and values
    """
    return {
        'beta': 0.5,      # Order parameter exponent
        'gamma': 1.0,     # Susceptibility exponent
        'nu': 0.5,        # Correlation length exponent
        'delta': 3.0,     # Critical isotherm
        'alpha': 0.0,     # Specific heat (log divergence)
    }


def scaling_prediction(
    kappa: float,
    kappa_c: float,
    exponent: float = 0.5
) -> float:
    """
    Compute scaling prediction near critical point.

    For κ < κ_c (polarized phase):
        η ~ (κ_c - κ)^β

    For κ > κ_c (mixed phase):
        η = 0

    Args:
        kappa: Current temperature
        kappa_c: Critical temperature
        exponent: Critical exponent β

    Returns:
        Predicted order parameter
    """
    if kappa >= kappa_c:
        return 0.0

    return np.power(kappa_c - kappa, exponent)


# =============================================================================
# Visualization Helpers
# =============================================================================

def plot_phase_diagram(
    diagram: PhaseDiagram,
    ax=None,
    show_regions: bool = True,
    **kwargs
):
    """
    Plot phase diagram with critical curve.

    Args:
        diagram: PhaseDiagram object
        ax: Matplotlib axis (creates new if None)
        show_regions: Shade phase regions
        **kwargs: Additional plot arguments
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot critical curve
    ax.plot(
        diagram.kappa_range,
        diagram.critical_curve,
        'k-',
        linewidth=2,
        label=r'Critical: $d_c = \sqrt{2\kappa |\ln\epsilon|}$'
    )

    if show_regions:
        # Shade polarized region (above curve)
        ax.fill_between(
            diagram.kappa_range,
            diagram.critical_curve,
            diagram.critical_curve.max() * 1.5,
            alpha=0.3,
            color='blue',
            label='Polarized'
        )

        # Shade mixed region (below curve)
        ax.fill_between(
            diagram.kappa_range,
            0,
            diagram.critical_curve,
            alpha=0.3,
            color='red',
            label='Mixed'
        )

    ax.set_xlabel(r'Temperature $\kappa$', fontsize=12)
    ax.set_ylabel(r'Mahalanobis Distance $d_M$', fontsize=12)
    ax.set_title(r'Phase Diagram: $\kappa$ vs Distance', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(diagram.kappa_range[0], diagram.kappa_range[-1])
    ax.set_ylim(0, diagram.critical_curve.max() * 1.2)
    ax.grid(True, alpha=0.3)

    return ax


def plot_kappa_sweep(
    results: KappaSweepResults,
    ax=None,
    **kwargs
):
    """
    Plot results of κ sweep analysis.

    Args:
        results: KappaSweepResults object
        ax: Matplotlib axis (creates new if None)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax_order, ax_attn, ax_mod, ax_ratio = axes.flat
    else:
        ax_order = ax
        ax_attn = ax_mod = ax_ratio = None

    # Order parameter
    ax_order.plot(results.kappa_values, results.order_parameters, 'b-', linewidth=2)
    ax_order.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_order.axvline(results.kappa_critical, color='r', linestyle='--',
                      label=f'$\\kappa_c = {results.kappa_critical:.2f}$')
    ax_order.set_xlabel(r'$\kappa$')
    ax_order.set_ylabel(r'Order Parameter $\eta$')
    ax_order.set_title('Order Parameter vs Temperature')
    ax_order.legend()
    ax_order.grid(True, alpha=0.3)

    if ax_attn is not None:
        # Attention weights
        ax_attn.plot(results.kappa_values, results.within_group_attention,
                     'b-', linewidth=2, label='Within-group')
        ax_attn.plot(results.kappa_values, results.cross_group_attention,
                     'r-', linewidth=2, label='Cross-group')
        ax_attn.axvline(results.kappa_critical, color='k', linestyle='--', alpha=0.5)
        ax_attn.set_xlabel(r'$\kappa$')
        ax_attn.set_ylabel('Average Attention')
        ax_attn.set_title('Attention Weights vs Temperature')
        ax_attn.legend()
        ax_attn.grid(True, alpha=0.3)

        # Modularity
        ax_mod.plot(results.kappa_values, results.modularity, 'g-', linewidth=2)
        ax_mod.axvline(results.kappa_critical, color='r', linestyle='--')
        ax_mod.set_xlabel(r'$\kappa$')
        ax_mod.set_ylabel('Modularity Q')
        ax_mod.set_title('Network Modularity vs Temperature')
        ax_mod.grid(True, alpha=0.3)

        # Attention ratio
        ratio = results.cross_group_attention / (results.within_group_attention + 1e-10)
        ax_ratio.semilogy(results.kappa_values, ratio, 'm-', linewidth=2)
        ax_ratio.axhline(1, color='k', linestyle='--', alpha=0.5, label='Equal coupling')
        ax_ratio.axvline(results.kappa_critical, color='r', linestyle='--')
        ax_ratio.set_xlabel(r'$\kappa$')
        ax_ratio.set_ylabel(r'$\beta_{cross} / \beta_{within}$')
        ax_ratio.set_title('Cross/Within Attention Ratio')
        ax_ratio.legend()
        ax_ratio.grid(True, alpha=0.3)

        plt.tight_layout()

    return ax_order


# =============================================================================
# Testing and Demonstration
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("KAPPA PHASE TRANSITIONS: Critical Distance for Polarization")
    print("=" * 70)

    # Example 1: Analytical critical distance
    print("\n[1] Critical Distance Formula")
    print("-" * 40)

    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        d_c = compute_critical_distance(kappa, epsilon=0.01)
        print(f"  κ = {kappa:5.1f}  →  d_c = {d_c:.4f}")

    print("\n  Formula: d_c = √(2κ |ln(ε)|)")
    print(f"  With ε = 0.01: d_c = √(2κ × {abs(np.log(0.01)):.2f}) = √({2*abs(np.log(0.01)):.2f}κ)")

    # Example 2: Stability analysis of a state
    print("\n[2] Polarization Stability Analysis")
    print("-" * 40)

    # Create example polarized state
    K = 3
    mu_A = np.array([1.0, 0.0, 0.0])
    mu_B = np.array([-1.0, 0.0, 0.0])
    Sigma = np.eye(K) * 0.5

    state = PolarizationState(
        mu_A=mu_A, mu_B=mu_B,
        Sigma_A=Sigma, Sigma_B=Sigma
    )

    d_eucl = np.linalg.norm(mu_A - mu_B)
    print(f"  Group means: μ_A = {mu_A}, μ_B = {mu_B}")
    print(f"  Euclidean distance: |μ_A - μ_B| = {d_eucl:.2f}")

    print("\n  Stability at different κ:")
    for kappa in [0.1, 0.5, 1.0, 2.0, 5.0]:
        analysis = analyze_polarization_stability(state, kappa)
        status = "STABLE" if analysis.is_stable else "UNSTABLE"
        print(f"    κ = {kappa:4.1f}: d_c = {analysis.critical_distance:.3f}, "
              f"β_cross = {analysis.cross_group_attention:.4f} → {status}")

    # Example 3: Phase diagram
    print("\n[3] Phase Diagram Generation")
    print("-" * 40)

    diagram = generate_phase_diagram(
        kappa_range=(0.1, 10.0),
        n_kappa=50,
        epsilon=0.01
    )

    print(f"  κ range: [{diagram.kappa_range[0]:.1f}, {diagram.kappa_range[-1]:.1f}]")
    print(f"  ε threshold: {diagram.stability_threshold}")
    print(f"  d_c at κ=1.0: {diagram.critical_distance(1.0):.3f}")
    print(f"  d_c at κ=5.0: {diagram.critical_distance(5.0):.3f}")

    # Example 4: Critical exponents
    print("\n[4] Theoretical Critical Exponents (Mean-Field)")
    print("-" * 40)

    exponents = predict_critical_exponents()
    for name, value in exponents.items():
        print(f"  {name}: {value}")

    print("\n" + "=" * 70)
    print("Key Insight: Polarization stable when d > d_c = √(2κ |ln(ε)|)")
    print("  - Low κ (cold): Small d_c → easy to polarize")
    print("  - High κ (hot): Large d_c → requires greater separation")
    print("=" * 70)


# =============================================================================
# Gauge Transport Effects
# =============================================================================

def compute_kl_with_transport(
    mu_A: np.ndarray,
    Sigma_A: np.ndarray,
    mu_B: np.ndarray,
    Sigma_B: np.ndarray,
    Omega_AB: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute KL(q_A || Ω_AB[q_B]) with gauge transport.

    The transport operator Ω_AB ∈ SO(K) rotates the distribution:
        Ω_AB[q_B] = N(Ω_AB @ μ_B, Ω_AB @ Σ_B @ Ω_AB^T)

    This changes the effective distance between groups.

    Args:
        mu_A, Sigma_A: Group A belief
        mu_B, Sigma_B: Group B belief
        Omega_AB: Transport operator (K, K) rotation matrix
        eps: Regularization

    Returns:
        kl: KL divergence with transport
    """
    K = mu_A.shape[-1]

    # Transport B → A frame
    mu_B_transported = Omega_AB @ mu_B
    Sigma_B_transported = Omega_AB @ Sigma_B @ Omega_AB.T

    # Compute KL
    Sigma_B_inv = np.linalg.inv(Sigma_B_transported + eps * np.eye(K))
    delta_mu = mu_B_transported - mu_A

    trace_term = np.trace(Sigma_B_inv @ Sigma_A)
    quad_term = delta_mu @ Sigma_B_inv @ delta_mu
    _, logdet_A = np.linalg.slogdet(Sigma_A + eps * np.eye(K))
    _, logdet_B = np.linalg.slogdet(Sigma_B_transported + eps * np.eye(K))

    kl = 0.5 * (trace_term + quad_term - K + logdet_B - logdet_A)

    return max(0, kl)


def effective_distance_with_transport(
    mu_A: np.ndarray,
    mu_B: np.ndarray,
    Sigma: np.ndarray,
    Omega_AB: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Compute effective Mahalanobis distance with gauge transport.

    d_eff = ||μ_A - Ω_AB @ μ_B||_{Σ^{-1}}

    The transport can either align or misalign the groups:
    - Ω_AB = I: standard distance
    - Ω_AB aligning: reduces effective distance
    - Ω_AB misaligning: increases effective distance
    """
    K = mu_A.shape[-1]
    Sigma_inv = np.linalg.inv(Sigma + eps * np.eye(K))

    mu_B_transported = Omega_AB @ mu_B
    delta = mu_A - mu_B_transported

    d_squared = delta @ Sigma_inv @ delta

    return np.sqrt(max(0, d_squared))


def critical_distance_with_transport(
    kappa: float,
    epsilon: float,
    Omega_AB: np.ndarray,
    Sigma: np.ndarray
) -> Tuple[float, float]:
    """
    Compute critical distance accounting for gauge transport.

    The transport operator affects the effective KL:
        KL_eff ≈ d²_eff / 2

    where d_eff depends on the rotation.

    Returns:
        (d_c_intrinsic, transport_factor)

    d_c_intrinsic is the base critical distance (without transport)
    transport_factor describes how transport modifies it
    """
    d_c_base = compute_critical_distance(kappa, epsilon)

    # Estimate transport effect on distance
    # For identity transport, factor = 1
    # For general transport, need to account for misalignment

    K = Omega_AB.shape[0]

    # Compute how much transport deviates from identity
    I = np.eye(K)
    transport_deviation = np.linalg.norm(Omega_AB - I, 'fro') / np.sqrt(K)

    # Transport can increase or decrease effective distance
    # As rough estimate: factor ≈ 1 ± deviation
    transport_factor = 1.0  # First-order: transport is a rotation, preserves distances

    return d_c_base, transport_factor


def analyze_gauge_effect_on_polarization(
    state: PolarizationState,
    kappa: float,
    phi_A: np.ndarray,
    phi_B: np.ndarray,
    generators: np.ndarray,
    epsilon: float = 0.01
) -> Dict[str, float]:
    """
    Analyze how gauge fields affect polarization stability.

    Args:
        state: Polarization state
        kappa: Temperature
        phi_A: Gauge field for group A (3,) for SO(3)
        phi_B: Gauge field for group B
        generators: Lie algebra generators (3, K, K)
        epsilon: Stability threshold

    Returns:
        Dict with stability metrics including gauge effects
    """
    from math_utils.transport import compute_transport

    # Compute transport operator Ω_AB = exp(φ_A) exp(-φ_B)
    Omega_AB = compute_transport(phi_A, phi_B, generators)

    # Compute KL with and without transport
    kl_no_transport = compute_kl_gaussians(
        state.mu_A, state.Sigma_A,
        state.mu_B, state.Sigma_B
    )

    kl_with_transport = compute_kl_with_transport(
        state.mu_A, state.Sigma_A,
        state.mu_B, state.Sigma_B,
        Omega_AB
    )

    # Effective distances
    Sigma_avg = 0.5 * (state.Sigma_A + state.Sigma_B)
    d_no_transport = compute_mahalanobis_distance(
        state.mu_A, state.mu_B, Sigma_avg
    )
    d_with_transport = effective_distance_with_transport(
        state.mu_A, state.mu_B, Sigma_avg, Omega_AB
    )

    # Critical distances
    d_c = compute_critical_distance(kappa, epsilon)

    # Attention with transport
    beta_cross_no_transport = np.exp(-kl_no_transport / kappa)
    beta_cross_with_transport = np.exp(-kl_with_transport / kappa)

    return {
        'kl_no_transport': kl_no_transport,
        'kl_with_transport': kl_with_transport,
        'd_no_transport': d_no_transport,
        'd_with_transport': d_with_transport,
        'd_critical': d_c,
        'beta_cross_no_transport': beta_cross_no_transport,
        'beta_cross_with_transport': beta_cross_with_transport,
        'stable_no_transport': d_no_transport > d_c,
        'stable_with_transport': d_with_transport > d_c,
        'gauge_alignment': np.trace(Omega_AB) / Omega_AB.shape[0]
    }


# =============================================================================
# Multi-Group Extension
# =============================================================================

@dataclass
class MultiGroupState:
    """
    State with N > 2 groups.

    Attributes:
        means: List of group means [μ_1, ..., μ_N]
        covariances: List of group covariances [Σ_1, ..., Σ_N]
        sizes: Number of agents per group
        labels: Group label names
    """
    means: List[np.ndarray]
    covariances: List[np.ndarray]
    sizes: List[int]
    labels: Optional[List[str]] = None

    @property
    def n_groups(self) -> int:
        return len(self.means)


def compute_pairwise_distances(
    state: MultiGroupState,
    use_mahalanobis: bool = True
) -> np.ndarray:
    """
    Compute pairwise distances between all group pairs.

    Returns:
        D: (N, N) distance matrix where D[i,j] = d(group_i, group_j)
    """
    N = state.n_groups
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(i+1, N):
            if use_mahalanobis:
                Sigma_avg = 0.5 * (state.covariances[i] + state.covariances[j])
                d = compute_mahalanobis_distance(
                    state.means[i], state.means[j], Sigma_avg
                )
            else:
                d = np.linalg.norm(state.means[i] - state.means[j])

            D[i, j] = d
            D[j, i] = d

    return D


def compute_attention_matrix(
    state: MultiGroupState,
    kappa: float
) -> np.ndarray:
    """
    Compute attention matrix between groups.

    β[i,j] = exp(-KL(q_i || q_j) / κ) / Z_i

    where Z_i = Σ_k exp(-KL(q_i || q_k) / κ)
    """
    N = state.n_groups

    # Compute pairwise KL
    KL = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                KL[i, j] = compute_kl_gaussians(
                    state.means[i], state.covariances[i],
                    state.means[j], state.covariances[j]
                )

    # Softmax to get attention
    log_weights = -KL / kappa
    # Mask diagonal
    np.fill_diagonal(log_weights, -np.inf)

    # Numerically stable softmax
    beta = np.zeros((N, N))
    for i in range(N):
        row = log_weights[i, :]
        row_shifted = row - np.max(row[row > -np.inf])
        exp_row = np.exp(row_shifted)
        exp_row[row == -np.inf] = 0
        beta[i, :] = exp_row / (np.sum(exp_row) + 1e-10)

    return beta


def analyze_multigroup_stability(
    state: MultiGroupState,
    kappa: float,
    epsilon: float = 0.01
) -> Dict[str, any]:
    """
    Analyze stability of multi-group polarization.

    A multi-group polarized state is stable when:
    - Within-group attention dominates for all groups
    - Cross-group attention is negligible between distant groups

    Returns:
        Dict with stability metrics for each group pair
    """
    N = state.n_groups

    # Pairwise distances
    D = compute_pairwise_distances(state)

    # Attention matrix
    beta = compute_attention_matrix(state, kappa)

    # Critical distance
    d_c = compute_critical_distance(kappa, epsilon)

    # Analyze each pair
    pair_analysis = {}
    for i in range(N):
        for j in range(i+1, N):
            pair_key = f"({i},{j})"
            d_ij = D[i, j]
            beta_ij = beta[i, j]
            is_stable = d_ij > d_c

            pair_analysis[pair_key] = {
                'distance': d_ij,
                'attention': beta_ij,
                'stable': is_stable,
                'margin': d_ij - d_c
            }

    # Overall stability: all pairs must be stable
    all_stable = all(p['stable'] for p in pair_analysis.values())

    # Minimum margin (most vulnerable pair)
    min_margin = min(p['margin'] for p in pair_analysis.values())
    vulnerable_pair = min(pair_analysis.keys(),
                          key=lambda k: pair_analysis[k]['margin'])

    return {
        'n_groups': N,
        'kappa': kappa,
        'd_critical': d_c,
        'distance_matrix': D,
        'attention_matrix': beta,
        'pair_analysis': pair_analysis,
        'all_stable': all_stable,
        'min_margin': min_margin,
        'vulnerable_pair': vulnerable_pair
    }


def find_multigroup_critical_kappa(
    state: MultiGroupState,
    epsilon: float = 0.01
) -> Tuple[float, str]:
    """
    Find critical κ for multi-group system.

    The critical κ is determined by the closest pair of groups:
        κ_c = d²_min / (2|ln(ε)|)

    Returns:
        (kappa_c, limiting_pair): Critical temperature and limiting pair
    """
    D = compute_pairwise_distances(state)
    N = state.n_groups

    # Find minimum distance (closest groups)
    d_min = np.inf
    limiting_pair = None

    for i in range(N):
        for j in range(i+1, N):
            if D[i, j] < d_min:
                d_min = D[i, j]
                limiting_pair = f"({i},{j})"

    # Critical kappa from minimum distance
    kappa_c = d_min**2 / (2 * abs(np.log(epsilon)))

    return kappa_c, limiting_pair
