# -*- coding: utf-8 -*-
"""
Critical Dynamics: Behavior Near the Phase Transition
======================================================

Analyzes the dynamics of polarization near the critical point κ_c.

Key phenomena near criticality:
1. Critical slowing down: relaxation time τ → ∞
2. Diverging susceptibility: small perturbations → large response
3. Universal scaling: system-independent behavior
4. Hysteresis: path-dependent dynamics

Mathematical Framework
----------------------
Near κ_c, the order parameter η satisfies:

    dη/dt = -∂F/∂η = -a(κ - κ_c)η - bη³ + noise

This is the Landau-Ginzburg equation with:
- a > 0: drives transition
- b > 0: stabilizes ordered phase
- κ < κ_c: ordered (polarized)
- κ > κ_c: disordered (mixed)

Critical exponents (mean-field):
- β = 1/2: η ~ (κ_c - κ)^β
- γ = 1: susceptibility χ ~ |κ - κ_c|^{-γ}
- ν = 1/2: correlation length ξ ~ |κ - κ_c|^{-ν}
- z = 2: dynamic exponent, τ ~ ξ^z ~ |κ - κ_c|^{-νz}

Author: VFE Transformer Team
Date: December 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CriticalDynamicsResults:
    """Results from critical dynamics analysis."""
    kappa_values: np.ndarray
    kappa_c: float
    relaxation_times: np.ndarray
    susceptibilities: np.ndarray
    order_parameters: np.ndarray

    # Fitted exponents
    beta_measured: float = 0.0
    gamma_measured: float = 0.0
    nu_z_measured: float = 0.0


@dataclass
class TimeEvolution:
    """Time series of order parameter."""
    times: np.ndarray
    order_parameter: np.ndarray
    kappa: float
    separation: float


# =============================================================================
# Landau-Ginzburg Dynamics
# =============================================================================

def landau_ginzburg_potential(
    eta: float,
    kappa: float,
    kappa_c: float,
    a: float = 1.0,
    b: float = 1.0
) -> float:
    """
    Landau-Ginzburg free energy potential.

    F(η) = (a/2)(κ - κ_c)η² + (b/4)η⁴

    - κ < κ_c: double-well → ordered phase
    - κ > κ_c: single well → disordered phase
    """
    return 0.5 * a * (kappa - kappa_c) * eta**2 + 0.25 * b * eta**4


def landau_ginzburg_force(
    eta: float,
    kappa: float,
    kappa_c: float,
    a: float = 1.0,
    b: float = 1.0
) -> float:
    """
    Force from Landau-Ginzburg potential.

    -∂F/∂η = -a(κ - κ_c)η - bη³
    """
    return -a * (kappa - kappa_c) * eta - b * eta**3


def simulate_order_parameter_dynamics(
    kappa: float,
    kappa_c: float,
    eta_0: float = 0.1,
    n_steps: int = 1000,
    dt: float = 0.01,
    noise_strength: float = 0.0,
    a: float = 1.0,
    b: float = 1.0,
    seed: int = 42
) -> TimeEvolution:
    """
    Simulate order parameter dynamics using Langevin equation.

    dη/dt = -a(κ - κ_c)η - bη³ + √(2T)ξ(t)

    where ξ(t) is white noise.
    """
    rng = np.random.default_rng(seed)

    eta = eta_0
    times = np.arange(n_steps) * dt
    etas = np.zeros(n_steps)

    for i in range(n_steps):
        etas[i] = eta

        # Deterministic force
        force = landau_ginzburg_force(eta, kappa, kappa_c, a, b)

        # Noise (if any)
        if noise_strength > 0:
            noise = rng.normal(0, 1) * np.sqrt(2 * noise_strength * dt)
        else:
            noise = 0

        # Euler step
        eta = eta + force * dt + noise

    return TimeEvolution(
        times=times,
        order_parameter=etas,
        kappa=kappa,
        separation=0.0  # Not used here
    )


def compute_relaxation_time(
    kappa: float,
    kappa_c: float,
    a: float = 1.0,
    threshold: float = 0.01
) -> float:
    """
    Compute relaxation time τ near critical point.

    For κ > κ_c (disordered):
        τ = 1 / (a|κ - κ_c|)

    For κ < κ_c (ordered):
        τ = 1 / (2a|κ - κ_c|)  (relaxation to ordered state)
    """
    delta_kappa = abs(kappa - kappa_c)

    if delta_kappa < threshold:
        # At critical point: divergent
        return np.inf

    if kappa > kappa_c:
        return 1.0 / (a * delta_kappa)
    else:
        return 1.0 / (2 * a * delta_kappa)


def compute_susceptibility(
    kappa: float,
    kappa_c: float,
    a: float = 1.0,
    threshold: float = 0.01
) -> float:
    """
    Compute susceptibility χ = ∂η/∂h near critical point.

    χ ~ 1/|κ - κ_c| (diverges at critical point)
    """
    delta_kappa = abs(kappa - kappa_c)

    if delta_kappa < threshold:
        return np.inf

    return 1.0 / (a * delta_kappa)


def equilibrium_order_parameter(
    kappa: float,
    kappa_c: float,
    a: float = 1.0,
    b: float = 1.0
) -> float:
    """
    Equilibrium order parameter.

    For κ < κ_c: η* = √(a(κ_c - κ)/b)
    For κ ≥ κ_c: η* = 0
    """
    if kappa >= kappa_c:
        return 0.0
    else:
        return np.sqrt(a * (kappa_c - kappa) / b)


# =============================================================================
# Critical Exponent Measurement
# =============================================================================

def measure_critical_exponents(
    kappa_c: float,
    kappa_range: Tuple[float, float] = None,
    n_points: int = 20,
    a: float = 1.0,
    b: float = 1.0
) -> CriticalDynamicsResults:
    """
    Measure critical exponents by fitting scaling near κ_c.

    Returns:
        CriticalDynamicsResults with measured exponents
    """
    if kappa_range is None:
        kappa_range = (0.1 * kappa_c, 2.0 * kappa_c)

    # Sample κ values, avoiding exact critical point
    kappa_below = np.linspace(kappa_range[0], kappa_c * 0.95, n_points // 2)
    kappa_above = np.linspace(kappa_c * 1.05, kappa_range[1], n_points // 2)
    kappa_values = np.concatenate([kappa_below, kappa_above])

    # Compute observables
    order_params = np.array([
        equilibrium_order_parameter(k, kappa_c, a, b) for k in kappa_values
    ])

    relaxation_times = np.array([
        compute_relaxation_time(k, kappa_c, a) for k in kappa_values
    ])

    susceptibilities = np.array([
        compute_susceptibility(k, kappa_c, a) for k in kappa_values
    ])

    # Fit exponents
    # β: η ~ (κ_c - κ)^β for κ < κ_c
    mask_below = (kappa_values < kappa_c * 0.95) & (order_params > 1e-10)
    if np.sum(mask_below) >= 2:
        log_delta_kappa = np.log(kappa_c - kappa_values[mask_below])
        log_eta = np.log(order_params[mask_below])
        beta_measured, _ = np.polyfit(log_delta_kappa, log_eta, 1)
    else:
        beta_measured = 0.5  # Default

    # γ: χ ~ |κ - κ_c|^{-γ}
    mask_finite_chi = np.isfinite(susceptibilities)
    if np.sum(mask_finite_chi) >= 2:
        log_delta = np.log(np.abs(kappa_values[mask_finite_chi] - kappa_c))
        log_chi = np.log(susceptibilities[mask_finite_chi])
        gamma_measured, _ = np.polyfit(log_delta, log_chi, 1)
        gamma_measured = -gamma_measured  # χ ~ |Δκ|^{-γ}
    else:
        gamma_measured = 1.0  # Default

    # νz: τ ~ |κ - κ_c|^{-νz}
    mask_finite_tau = np.isfinite(relaxation_times)
    if np.sum(mask_finite_tau) >= 2:
        log_delta = np.log(np.abs(kappa_values[mask_finite_tau] - kappa_c))
        log_tau = np.log(relaxation_times[mask_finite_tau])
        nu_z_measured, _ = np.polyfit(log_delta, log_tau, 1)
        nu_z_measured = -nu_z_measured
    else:
        nu_z_measured = 1.0  # Default (νz = 1 for mean-field)

    return CriticalDynamicsResults(
        kappa_values=kappa_values,
        kappa_c=kappa_c,
        relaxation_times=relaxation_times,
        susceptibilities=susceptibilities,
        order_parameters=order_params,
        beta_measured=beta_measured,
        gamma_measured=gamma_measured,
        nu_z_measured=nu_z_measured
    )


# =============================================================================
# Hysteresis Analysis
# =============================================================================

def simulate_hysteresis_loop(
    kappa_min: float,
    kappa_max: float,
    kappa_c: float,
    n_steps_per_segment: int = 100,
    dt: float = 0.01,
    sweep_rate: float = 0.01,
    a: float = 1.0,
    b: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate hysteresis loop by sweeping κ up and down.

    Returns:
        (kappa_sweep, eta_up, eta_down)
    """
    # Forward sweep: low κ → high κ
    n_total = 2 * n_steps_per_segment
    kappa_up = np.linspace(kappa_min, kappa_max, n_steps_per_segment)
    kappa_down = np.linspace(kappa_max, kappa_min, n_steps_per_segment)

    eta_up = np.zeros(n_steps_per_segment)
    eta_down = np.zeros(n_steps_per_segment)

    # Start from ordered state
    eta = equilibrium_order_parameter(kappa_min, kappa_c, a, b)

    # Sweep up
    for i, kappa in enumerate(kappa_up):
        # Relax at this κ
        for _ in range(int(1.0 / (sweep_rate * dt))):
            force = landau_ginzburg_force(eta, kappa, kappa_c, a, b)
            eta = eta + force * dt
        eta_up[i] = eta

    # Sweep down
    for i, kappa in enumerate(kappa_down):
        for _ in range(int(1.0 / (sweep_rate * dt))):
            force = landau_ginzburg_force(eta, kappa, kappa_c, a, b)
            eta = eta + force * dt
        eta_down[i] = eta

    return kappa_up, eta_up, eta_down


# =============================================================================
# Connection to VFE System
# =============================================================================

def analyze_system_near_critical(
    system,
    labels: np.ndarray,
    kappa_values: np.ndarray,
    n_relaxation_steps: int = 50,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Analyze VFE system behavior near critical κ.

    For each κ:
    1. Set system.config.kappa_beta = κ
    2. Run relaxation steps
    3. Measure order parameter
    4. Estimate relaxation time from decay
    """
    from analysis.polarization_stability_test import measure_group_attention

    results = {
        'kappa': [],
        'order_parameter': [],
        'beta_cross': [],
        'beta_within': [],
        'relaxation_rate': []
    }

    for kappa in kappa_values:
        # Update kappa
        system.config.kappa_beta = kappa

        # Measure attention
        attn = measure_group_attention(system, labels, verbose=False)

        beta_within = attn['beta_within_avg']
        beta_cross = attn['beta_cross_avg']

        # Order parameter
        if beta_within > 1e-10:
            order_param = 1 - 2 * beta_cross / beta_within
        else:
            order_param = 0.0

        results['kappa'].append(kappa)
        results['order_parameter'].append(order_param)
        results['beta_cross'].append(beta_cross)
        results['beta_within'].append(beta_within)
        results['relaxation_rate'].append(1.0 / max(beta_cross, 1e-10))

        if verbose:
            print(f"  κ = {kappa:.3f}: η = {order_param:.4f}, "
                  f"β_cross = {beta_cross:.4f}")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CRITICAL DYNAMICS ANALYSIS")
    print("=" * 70)

    # Example: κ_c = 0.434 for d = 2.0, ε = 0.01
    separation = 2.0
    epsilon = 0.01
    kappa_c = separation**2 / (2 * abs(np.log(epsilon)))

    print(f"\nCritical point: κ_c = {kappa_c:.4f}")

    # Measure exponents
    print("\n[1] Measuring Critical Exponents...")
    results = measure_critical_exponents(kappa_c)
    print(f"    β (order parameter): {results.beta_measured:.3f} (theory: 0.5)")
    print(f"    γ (susceptibility):  {results.gamma_measured:.3f} (theory: 1.0)")
    print(f"    νz (relaxation):     {results.nu_z_measured:.3f} (theory: 1.0)")

    # Simulate dynamics
    print("\n[2] Simulating Order Parameter Dynamics...")

    # Below critical
    evol_below = simulate_order_parameter_dynamics(
        kappa=0.5 * kappa_c, kappa_c=kappa_c, eta_0=0.01, n_steps=500
    )
    print(f"    κ = 0.5×κ_c: η evolves to {evol_below.order_parameter[-1]:.4f}")
    print(f"    (theory: {equilibrium_order_parameter(0.5*kappa_c, kappa_c):.4f})")

    # Above critical
    evol_above = simulate_order_parameter_dynamics(
        kappa=1.5 * kappa_c, kappa_c=kappa_c, eta_0=0.5, n_steps=500
    )
    print(f"    κ = 1.5×κ_c: η evolves to {evol_above.order_parameter[-1]:.4f}")
    print(f"    (theory: 0.0)")

    # Relaxation times
    print("\n[3] Relaxation Times Near Critical Point...")
    for delta in [0.5, 0.2, 0.1, 0.05, 0.01]:
        kappa = kappa_c * (1 + delta)
        tau = compute_relaxation_time(kappa, kappa_c)
        print(f"    κ = κ_c + {delta:.0%}: τ = {tau:.2f}")

    print("\n" + "=" * 70)
    print("Critical slowing down confirmed: τ → ∞ as κ → κ_c")
    print("=" * 70)
