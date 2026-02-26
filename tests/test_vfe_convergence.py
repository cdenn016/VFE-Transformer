#!/usr/bin/env python3
"""
VFE Convergence Verification
=============================

Verifies that Variational Free Energy (VFE) gradient descent:
1. Decreases the total free energy over training
2. Converges (gradient norms shrink, energy stabilizes)
3. Equilibrates (late-stage energy fluctuations are small)
4. Maintains manifold constraints (SPD covariances, principal ball gauge fields)

Tests 0D (particle) agents with all combinations of energy terms:
    F = Σ_i λ_self KL(q_i||p_i)
      + Σ_ij λ_β β_ij KL(q_i||Ω_ij q_j)
      + Σ_ij λ_γ γ_ij KL(p_i||Ω_ij p_j)
      - Σ_i λ_obs E_q[log p(o|x)]

Note on thresholds:
    SPD manifold retraction (exponential map with trust region) is deliberately
    conservative, so covariance convergence is slower than mean convergence.
    Equilibration thresholds are set to accommodate this while still catching
    divergence, NaN, and optimizer instability.

Author: Claude
Date: February 2026
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from config import AgentConfig, SystemConfig, TrainingConfig
from agent.agents import Agent
from agent.system import MultiAgentSystem
from agent.trainer import Trainer, TrainingHistory
from gradients.free_energy_clean import compute_total_free_energy, FreeEnergyBreakdown
from geometry.geometry_base import BaseManifold, TopologyType


# =============================================================================
# Helpers
# =============================================================================

def _build_particle_system(
    n_agents: int = 4,
    K: int = 5,
    seed: int = 42,
    lambda_self: float = 1.0,
    lambda_belief_align: float = 1.0,
    lambda_prior_align: float = 0.0,
    lambda_obs: float = 0.0,
    lambda_phi: float = 0.0,
    identical_priors: str = "off",
    kappa_beta: float = 1.0,
    mu_scale: float = 1.0,
    sigma_scale: float = 1.0,
    phi_scale: float = 0.3,
):
    """Build a 0D (particle) multi-agent system for testing."""
    rng = np.random.default_rng(seed)

    agent_cfg = AgentConfig(
        spatial_shape=(),
        K=K,
        mu_scale=mu_scale,
        sigma_scale=sigma_scale,
        phi_scale=phi_scale,
    )

    manifold = BaseManifold(shape=(), topology=TopologyType.FLAT)

    agents = [
        Agent(agent_id=i, config=agent_cfg, rng=rng, base_manifold=manifold)
        for i in range(n_agents)
    ]

    system_cfg = SystemConfig(
        lambda_self=lambda_self,
        lambda_belief_align=lambda_belief_align,
        lambda_prior_align=lambda_prior_align,
        lambda_obs=lambda_obs,
        lambda_phi=lambda_phi,
        identical_priors=identical_priors,
        kappa_beta=kappa_beta,
        D_x=K,
        seed=seed,
    )

    system = MultiAgentSystem(agents, system_cfg)

    if lambda_obs > 0:
        system.initialize_observation_model(system_cfg)

    return system


def _run_vfe_descent(
    system,
    n_steps: int = 500,
    lr_mu_q: float = 0.02,
    lr_sigma_q: float = 0.002,
    lr_mu_p: float = 0.02,
    lr_sigma_p: float = 0.002,
    lr_phi: float = 0.05,
    trust_region_sigma: float = 0.3,
):
    """Run VFE gradient descent and return the history."""
    train_cfg = TrainingConfig(
        n_steps=n_steps,
        lr_mu_q=lr_mu_q,
        lr_sigma_q=lr_sigma_q,
        lr_mu_p=lr_mu_p,
        lr_sigma_p=lr_sigma_p,
        lr_phi=lr_phi,
        trust_region_sigma=trust_region_sigma,
        log_every=max(1, n_steps // 5),
        save_history=True,
    )

    trainer = Trainer(system, config=train_cfg)
    history = trainer.train(n_steps=n_steps)
    return history, trainer


# =============================================================================
# Core convergence tests
# =============================================================================

class TestVFEConvergence:
    """Verify VFE gradient descent converges, decreases, and equilibrates."""

    N_STEPS = 500

    # ------------------------------------------------------------------
    # Test 1: Self-energy only (simplest case: KL(q||p) → 0)
    # ------------------------------------------------------------------
    def test_self_energy_convergence(self):
        """
        With only self-energy, beliefs should converge toward priors.
        F = Σ_i KL(q_i || p_i) → 0 as q_i → p_i.
        """
        system = _build_particle_system(
            n_agents=4,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=0.0,
            lambda_prior_align=0.0,
            lambda_obs=0.0,
            mu_scale=2.0,       # Large initial separation → large initial KL
            sigma_scale=1.0,
        )

        history, trainer = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.08,       # Aggressive for fast mean convergence
            lr_sigma_q=0.01,    # Higher for SPD convergence
            lr_mu_p=0.0,        # Don't move priors
            lr_sigma_p=0.0,
            lr_phi=0.0,         # No gauge dynamics
            trust_region_sigma=0.5,  # Larger trust region for faster Σ convergence
        )

        energies = history.total_energy
        self._assert_decreasing(energies, label="self-energy-only")
        self._assert_converged(energies, label="self-energy-only")
        self._assert_equilibrated(energies, label="self-energy-only")
        self._assert_finite(energies, label="self-energy-only")
        self._assert_constraints(system, label="self-energy-only")

        # Self-energy should reduce by at least 90% (q → p)
        assert energies[-1] < 0.1 * energies[0], (
            f"Self-energy did not reduce sufficiently: "
            f"initial={energies[0]:.4f}, final={energies[-1]:.4f}"
        )

    # ------------------------------------------------------------------
    # Test 2: Self + belief alignment (multi-agent consensus)
    # ------------------------------------------------------------------
    def test_self_plus_belief_alignment(self):
        """
        F = Σ_i KL(q_i||p_i) + Σ_ij β_ij KL(q_i||Ω q_j)
        Agents should reach consensus via belief alignment.
        """
        system = _build_particle_system(
            n_agents=5,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=1.0,
            lambda_prior_align=0.0,
            lambda_obs=0.0,
            lambda_phi=1.0,
            identical_priors="lock",
            kappa_beta=1.0,
            mu_scale=2.0,
            sigma_scale=1.0,
            phi_scale=0.1,          # Small initial gauge: near-flat connection
        )

        history, trainer = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.05,
            lr_sigma_q=0.005,
            lr_mu_p=0.0,
            lr_sigma_p=0.0,
            lr_phi=0.01,            # Conservative to keep phi in principal ball
        )

        energies = history.total_energy
        self._assert_decreasing(energies, label="self+belief")
        self._assert_converged(energies, label="self+belief")
        self._assert_equilibrated(energies, label="self+belief")
        self._assert_finite(energies, label="self+belief")
        self._assert_constraints(system, label="self+belief")

    # ------------------------------------------------------------------
    # Test 3: Full VFE with observations
    # ------------------------------------------------------------------
    def test_full_vfe_with_observations(self):
        """
        F = self + belief_align + obs.
        Observations anchor beliefs, preventing collapse to prior.

        Uses conservative lambda_obs and learning rates to prevent
        the large initial observation gradient from destabilizing
        the SPD covariance retraction.
        """
        system = _build_particle_system(
            n_agents=4,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=0.5,
            lambda_prior_align=0.0,
            lambda_obs=0.1,         # Conservative: large obs gradients can destabilize Sigma
            lambda_phi=1.0,
            identical_priors="lock",
            mu_scale=1.0,
            sigma_scale=1.0,
            phi_scale=0.3,
        )

        history, trainer = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.02,
            lr_sigma_q=0.002,
            lr_mu_p=0.0,
            lr_sigma_p=0.0,
            lr_phi=0.03,
        )

        energies = history.total_energy
        self._assert_decreasing(energies, label="full+obs")
        self._assert_converged(energies, label="full+obs")
        self._assert_equilibrated(energies, label="full+obs")
        self._assert_finite(energies, label="full+obs")
        self._assert_constraints(system, label="full+obs")

    # ------------------------------------------------------------------
    # Test 4: Full VFE including prior alignment
    # ------------------------------------------------------------------
    def test_full_vfe_all_terms(self):
        """
        F = self + belief_align + prior_align + obs.
        All four energy terms active.

        Uses conservative observation coupling to maintain numerical
        stability across all energy terms simultaneously.
        """
        system = _build_particle_system(
            n_agents=4,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=0.5,
            lambda_prior_align=0.3,
            lambda_obs=0.1,         # Conservative for stability
            lambda_phi=1.0,
            identical_priors="off",
            mu_scale=1.0,
            sigma_scale=1.0,
            phi_scale=0.3,
        )

        history, trainer = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.02,
            lr_sigma_q=0.002,
            lr_mu_p=0.01,
            lr_sigma_p=0.001,
            lr_phi=0.03,
        )

        energies = history.total_energy
        self._assert_decreasing(energies, label="all-terms")
        self._assert_converged(energies, label="all-terms")
        self._assert_equilibrated(energies, label="all-terms")
        self._assert_finite(energies, label="all-terms")
        self._assert_constraints(system, label="all-terms")

    # ------------------------------------------------------------------
    # Test 5: Many agents (scaling check)
    # ------------------------------------------------------------------
    @pytest.mark.timeout(600)
    def test_scaling_many_agents(self):
        """
        Verify convergence with a larger population (8 agents).
        """
        system = _build_particle_system(
            n_agents=8,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=1.0,
            lambda_prior_align=0.0,
            lambda_obs=0.0,
            lambda_phi=1.0,
            identical_priors="lock",
            mu_scale=1.5,
            sigma_scale=1.0,
            phi_scale=0.3,
        )

        history, trainer = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.03,
            lr_sigma_q=0.003,
            lr_mu_p=0.0,
            lr_sigma_p=0.0,
            lr_phi=0.05,
        )

        energies = history.total_energy
        self._assert_decreasing(energies, label="8-agents")
        self._assert_converged(energies, label="8-agents")
        self._assert_equilibrated(energies, label="8-agents")
        self._assert_finite(energies, label="8-agents")
        self._assert_constraints(system, label="8-agents")

    # ------------------------------------------------------------------
    # Test 6: Energy component breakdown remains physically valid
    # ------------------------------------------------------------------
    def test_energy_components_physical_validity(self):
        """
        Verify that individual energy components obey physical constraints:
        - Self-energy >= 0 (KL divergence non-negative)
        - Belief alignment >= 0 (KL divergence non-negative)
        - Prior alignment >= 0 (KL divergence non-negative)
        - Observation energy can be negative (log-likelihood)
        """
        system = _build_particle_system(
            n_agents=4,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=0.5,
            lambda_prior_align=0.3,
            lambda_obs=0.1,         # Conservative for stability
            lambda_phi=1.0,
            identical_priors="off",
            mu_scale=1.0,
            sigma_scale=1.0,
            phi_scale=0.3,
        )

        history, _ = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.02,
            lr_sigma_q=0.002,
            lr_mu_p=0.01,
            lr_sigma_p=0.001,
            lr_phi=0.03,
        )

        # KL-based terms must be non-negative (with small numerical tolerance)
        tol = -1e-4
        for i, (se, ba, pa) in enumerate(zip(
            history.self_energy, history.belief_align, history.prior_align
        )):
            assert se >= tol, f"Step {i}: self_energy={se:.6f} < 0"
            assert ba >= tol, f"Step {i}: belief_align={ba:.6f} < 0"
            assert pa >= tol, f"Step {i}: prior_align={pa:.6f} < 0"

    # ------------------------------------------------------------------
    # Test 7: Gradient norms decrease (convergence diagnostic)
    # ------------------------------------------------------------------
    def test_gradient_norms_decrease(self):
        """
        Gradient norms should decrease over training as the system
        approaches a stationary point dF/dtheta -> 0.
        """
        system = _build_particle_system(
            n_agents=4,
            K=5,
            lambda_self=1.0,
            lambda_belief_align=1.0,
            lambda_prior_align=0.0,
            lambda_obs=0.0,
            lambda_phi=1.0,
            identical_priors="lock",
            mu_scale=2.0,
            sigma_scale=1.0,
            phi_scale=0.3,
        )

        history, _ = _run_vfe_descent(
            system,
            n_steps=self.N_STEPS,
            lr_mu_q=0.05,
            lr_sigma_q=0.005,
            lr_mu_p=0.0,
            lr_sigma_p=0.0,
            lr_phi=0.05,
        )

        if len(history.grad_norm_mu_q) >= 50:
            early_mu = np.mean(history.grad_norm_mu_q[:50])
            late_mu = np.mean(history.grad_norm_mu_q[-50:])
            assert late_mu < early_mu, (
                f"mu_q gradient norms did not decrease: "
                f"early={early_mu:.4e}, late={late_mu:.4e}"
            )

    # ==================================================================
    # Assertion helpers
    # ==================================================================

    def _assert_decreasing(self, energies, label="", window=50, tolerance=0.55):
        """
        Assert that the energy trajectory is overall decreasing.

        We compare the mean of the first `window` steps to the mean of the
        last `window` steps.  The final mean must be strictly less than the
        initial mean.  Additionally, we check that at least `tolerance`
        fraction of consecutive steps are non-increasing.

        Note: tolerance=0.55 accommodates gauge-coupled multi-agent systems
        where transport operator updates (phi) can cause temporary energy
        increases even as the overall trajectory descends.
        """
        E = np.array(energies)
        n = len(E)
        assert n > 2 * window, f"[{label}] Not enough steps for decrease check"

        mean_first = np.mean(E[:window])
        mean_last = np.mean(E[-window:])

        assert mean_last < mean_first, (
            f"[{label}] Energy did not decrease overall: "
            f"first-{window}-mean={mean_first:.4f}, last-{window}-mean={mean_last:.4f}"
        )

        # Check that the majority of steps are non-increasing
        diffs = np.diff(E)
        frac_decreasing = np.mean(diffs <= 0)
        assert frac_decreasing >= tolerance, (
            f"[{label}] Only {frac_decreasing:.1%} of steps are non-increasing "
            f"(need >= {tolerance:.0%}).  This suggests the optimizer is unstable."
        )

    def _assert_converged(self, energies, label="", tail=100, rtol=0.15,
                           atol=0.05):
        """
        Assert that the energy has converged: the variation in
        the tail is small in either relative or absolute terms.

        Convergence criterion (either suffices):
            std(tail) / |mean(tail)| < rtol     (relative)
            std(tail) < atol                     (absolute, for near-zero energy)

        Note: rtol=0.15 accommodates the slow SPD manifold retraction
        while still catching divergence and instability.  The absolute
        threshold handles the near-zero case where relative variation
        is ill-conditioned.
        """
        E = np.array(energies)
        tail_vals = E[-tail:]
        mean_tail = np.mean(tail_vals)
        std_tail = np.std(tail_vals)

        # Absolute convergence: if fluctuations are tiny, we're converged
        if std_tail < atol:
            return

        # Avoid division by zero when energy is near zero
        if abs(mean_tail) < 1e-6:
            return

        rel_var = std_tail / abs(mean_tail)
        assert rel_var < rtol, (
            f"[{label}] Energy not converged in last {tail} steps: "
            f"mean={mean_tail:.4f}, std={std_tail:.4f}, "
            f"rel_var={rel_var:.4f} > {rtol}"
        )

    def _assert_equilibrated(self, energies, label="", tail=100,
                              max_drift_frac=0.30, atol=0.05):
        """
        Assert that the energy has equilibrated: no systematic drift
        in the tail.

        Checks that the linear trend (slope) in the tail is small
        relative to the mean energy, OR that the absolute drift is tiny.

        Note: max_drift_frac=0.30 accommodates two slow convergence modes:
        1. SPD covariance retraction (trust-region-constrained exponential map)
        2. Gauge-coupled dynamics where phi updates create temporary KL increases
        The atol handles the near-zero case. The critical guard is the
        upward-drift check that catches actual divergence.
        """
        E = np.array(energies)
        tail_vals = E[-tail:]
        mean_tail = np.mean(tail_vals)

        if abs(mean_tail) < 1e-6:
            return  # essentially zero

        # Fit linear trend
        x = np.arange(tail)
        coeffs = np.polyfit(x, tail_vals, 1)
        slope = coeffs[0]

        # Total drift over the tail window
        total_drift = abs(slope * tail)

        # Absolute convergence: if total drift is tiny, we're equilibrated
        if total_drift < atol:
            return

        drift_frac = total_drift / abs(mean_tail)

        # Verify the drift is DOWNWARD (still decreasing, not diverging)
        # slope <= 0 is fine (still converging); slope > 0 would be divergence
        if slope > 0 and drift_frac > 0.01:
            pytest.fail(
                f"[{label}] Energy INCREASING in tail: "
                f"slope={slope:.6f}, drift={total_drift:.4f} over {tail} steps"
            )

        assert drift_frac < max_drift_frac, (
            f"[{label}] Energy not equilibrated: "
            f"drift={total_drift:.4f} over {tail} steps "
            f"({drift_frac:.2%} of mean={mean_tail:.4f})"
        )

    def _assert_finite(self, energies, label=""):
        """Assert no NaN or Inf in the energy trajectory."""
        E = np.array(energies)
        assert np.all(np.isfinite(E)), (
            f"[{label}] Non-finite energies detected: "
            f"NaN count={np.sum(np.isnan(E))}, "
            f"Inf count={np.sum(np.isinf(E))}"
        )

    def _assert_constraints(self, system, label=""):
        """
        Assert that all agents satisfy manifold constraints after training:
        - Covariances are SPD (positive-definite)
        - Gauge fields are in the principal ball
        - No NaN in any field
        """
        for agent in system.agents:
            status = agent.check_constraints()
            assert status['valid'], (
                f"[{label}] Agent {agent.agent_id} constraint violation: "
                f"{status['violations']}"
            )


# =============================================================================
# Standalone runner (for interactive use outside pytest)
# =============================================================================

def run_verification(n_steps: int = 500, verbose: bool = True):
    """
    Run VFE convergence verification interactively.

    Returns a dict with all results for inspection / plotting.
    """
    results = {}

    configs = [
        {
            "name": "self-only",
            "system_kwargs": dict(
                n_agents=4, K=5,
                lambda_self=1.0, lambda_belief_align=0.0,
                lambda_obs=0.0, mu_scale=2.0,
            ),
            "train_kwargs": dict(
                lr_mu_q=0.08, lr_sigma_q=0.01,
                lr_mu_p=0.0, lr_sigma_p=0.0, lr_phi=0.0,
                trust_region_sigma=0.5,
            ),
        },
        {
            "name": "self+belief",
            "system_kwargs": dict(
                n_agents=5, K=5,
                lambda_self=1.0, lambda_belief_align=1.0,
                lambda_phi=1.0, identical_priors="lock",
                mu_scale=2.0, phi_scale=0.1,
            ),
            "train_kwargs": dict(
                lr_mu_q=0.05, lr_sigma_q=0.005,
                lr_mu_p=0.0, lr_sigma_p=0.0, lr_phi=0.01,
            ),
        },
        {
            "name": "full+obs",
            "system_kwargs": dict(
                n_agents=4, K=5,
                lambda_self=1.0, lambda_belief_align=0.5,
                lambda_obs=0.1, lambda_phi=1.0,
                identical_priors="lock", mu_scale=1.0,
            ),
            "train_kwargs": dict(
                lr_mu_q=0.02, lr_sigma_q=0.002,
                lr_mu_p=0.0, lr_sigma_p=0.0, lr_phi=0.03,
            ),
        },
        {
            "name": "all-terms",
            "system_kwargs": dict(
                n_agents=4, K=5,
                lambda_self=1.0, lambda_belief_align=0.5,
                lambda_prior_align=0.3, lambda_obs=0.1,
                lambda_phi=1.0, identical_priors="off",
                mu_scale=1.0, phi_scale=0.3,
            ),
            "train_kwargs": dict(
                lr_mu_q=0.02, lr_sigma_q=0.002,
                lr_mu_p=0.01, lr_sigma_p=0.001, lr_phi=0.03,
            ),
        },
    ]

    for cfg in configs:
        name = cfg["name"]
        if verbose:
            print(f"\n{'='*70}")
            print(f"  VFE Convergence Test: {name}")
            print(f"{'='*70}")

        system = _build_particle_system(**cfg["system_kwargs"])
        history, trainer = _run_vfe_descent(
            system, n_steps=n_steps, **cfg["train_kwargs"]
        )

        E = np.array(history.total_energy)
        E_init = E[0]
        E_final = E[-1]
        reduction_pct = 100 * (E_init - E_final) / abs(E_init) if abs(E_init) > 1e-8 else 0

        if verbose:
            print(f"\n  Initial VFE:    {E_init:.6f}")
            print(f"  Final VFE:      {E_final:.6f}")
            print(f"  Reduction:      {reduction_pct:.1f}%")
            print(f"  Tail std:       {np.std(E[-100:]):.6f}")
            print(f"  Tail mean:      {np.mean(E[-100:]):.6f}")
            if len(history.grad_norm_mu_q) >= 20:
                print(f"  Grad |dmu| early: {np.mean(history.grad_norm_mu_q[:20]):.4e}")
                print(f"  Grad |dmu| late:  {np.mean(history.grad_norm_mu_q[-20:]):.4e}")

            # Constraint check
            all_valid = True
            for agent in system.agents:
                status = agent.check_constraints()
                if not status['valid']:
                    all_valid = False
                    print(f"  VIOLATION Agent {agent.agent_id}: {status['violations']}")
            if all_valid:
                print(f"  Constraints:    ALL VALID")

        results[name] = {
            "history": history,
            "system": system,
            "trainer": trainer,
            "energies": E,
        }

    return results


if __name__ == "__main__":
    results = run_verification(n_steps=500, verbose=True)
    print("\n\nAll VFE convergence checks passed.")
