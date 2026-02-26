# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:44:05 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
TensorTrainer: GPU Training with PyTorch Autograd
=================================================

GPU-accelerated training using automatic differentiation.
No manual gradient derivation needed - just compute energy and call .backward()!

Key advantages over NumPy trainer:
- Automatic gradient computation via autograd
- GPU parallel execution
- Built-in optimizers (Adam, SGD, etc.)
- Mixed precision training support

Author: Claude (refactoring)
Date: December 2024
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import time
import numpy as np
from agent.tensor_system import TensorSystem
from agent.system import MultiAgentSystem


@dataclass
class TensorTrainingConfig:
    """Configuration for tensor-based training."""

    # Training steps
    n_steps: int = 1000
    log_every: int = 100

    # Learning rates (per parameter group)
    lr_mu: float = 0.01
    lr_sigma: float = 0.001
    lr_phi: float = 0.001

    # Optimizer settings
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9

    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0

    # Learning rate scheduler
    use_scheduler: bool = False
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    scheduler_step_size: int = 100
    scheduler_gamma: float = 0.5

    # Constraint enforcement
    project_spd_every: int = 10  # Project covariances to SPD every N steps
    project_phi_every: int = 10  # Clip phi to principal ball every N steps
    min_eigenvalue: float = 1e-4  # Minimum eigenvalue for SPD

    # Early stopping
    early_stop_patience: int = 50
    early_stop_threshold: float = 1e-6

    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision


@dataclass
class TensorTrainingHistory:
    """Container for training metrics."""

    steps: List[int] = field(default_factory=list)
    total_energy: List[float] = field(default_factory=list)
    self_energy: List[float] = field(default_factory=list)
    belief_energy: List[float] = field(default_factory=list)
    prior_energy: List[float] = field(default_factory=list)

    # Gradient norms (from autograd)
    grad_norm_mu_q: List[float] = field(default_factory=list)
    grad_norm_mu_p: List[float] = field(default_factory=list)
    grad_norm_sigma_q: List[float] = field(default_factory=list)
    grad_norm_sigma_p: List[float] = field(default_factory=list)
    grad_norm_phi: List[float] = field(default_factory=list)

    # Training stats
    step_times: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    def record(
        self,
        step: int,
        energies: Dict[str, torch.Tensor],
        system: 'TensorSystem',
        lr: float,
        step_time: float
    ):
        """Record metrics for current step."""
        self.steps.append(step)

        # Energy terms
        self.total_energy.append(energies['total'].item())
        self.self_energy.append(energies['self'].item())
        self.belief_energy.append(energies['belief'].item())
        self.prior_energy.append(energies['prior'].item())

        # Gradient norms (from autograd)
        if system.mu_q.grad is not None:
            self.grad_norm_mu_q.append(system.mu_q.grad.norm().item())
        else:
            self.grad_norm_mu_q.append(0.0)

        if system.mu_p.grad is not None:
            self.grad_norm_mu_p.append(system.mu_p.grad.norm().item())
        else:
            self.grad_norm_mu_p.append(0.0)

        if system.phi.grad is not None:
            self.grad_norm_phi.append(system.phi.grad.norm().item())
        else:
            self.grad_norm_phi.append(0.0)

        # Covariance gradients
        if system.use_cholesky_param:
            if system._L_q.grad is not None:
                self.grad_norm_sigma_q.append(system._L_q.grad.norm().item())
            else:
                self.grad_norm_sigma_q.append(0.0)
            if system._L_p.grad is not None:
                self.grad_norm_sigma_p.append(system._L_p.grad.norm().item())
            else:
                self.grad_norm_sigma_p.append(0.0)
        else:
            if system._Sigma_q.grad is not None:
                self.grad_norm_sigma_q.append(system._Sigma_q.grad.norm().item())
            else:
                self.grad_norm_sigma_q.append(0.0)
            if system._Sigma_p.grad is not None:
                self.grad_norm_sigma_p.append(system._Sigma_p.grad.norm().item())
            else:
                self.grad_norm_sigma_p.append(0.0)

        self.learning_rates.append(lr)
        self.step_times.append(step_time)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for saving."""
        return {
            'steps': self.steps,
            'total_energy': self.total_energy,
            'self_energy': self.self_energy,
            'belief_energy': self.belief_energy,
            'prior_energy': self.prior_energy,
            'grad_norm_mu_q': self.grad_norm_mu_q,
            'grad_norm_mu_p': self.grad_norm_mu_p,
            'grad_norm_sigma_q': self.grad_norm_sigma_q,
            'grad_norm_sigma_p': self.grad_norm_sigma_p,
            'grad_norm_phi': self.grad_norm_phi,
            'step_times': self.step_times,
            'learning_rates': self.learning_rates,
        }


class TensorTrainer:
    """
    GPU training using PyTorch autograd.

    The big advantage: NO MANUAL GRADIENT DERIVATION!
    Just compute energy, call .backward(), and the gradients appear.

    Example:
        system = TensorSystem(N=4, K=3)
        trainer = TensorTrainer(system, config)
        history = trainer.train()
    """

    def __init__(
        self,
        system: 'TensorSystem',
        config: Optional[TensorTrainingConfig] = None,
    ):
        """
        Initialize tensor trainer.

        Args:
            system: TensorSystem to train
            config: Training configuration
        """
        self.system = system
        self.config = config or TensorTrainingConfig()

        # Create optimizer with parameter groups
        self.optimizer = self._create_optimizer()

        # Create scheduler if requested
        self.scheduler = self._create_scheduler() if self.config.use_scheduler else None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None

        # Training state
        self.history = TensorTrainingHistory()
        self.current_step = 0
        self.best_energy = float('inf')
        self.patience_counter = 0

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with per-parameter learning rates."""
        # Group parameters by type
        param_groups = []

        # Means (mu_q, mu_p)
        mu_params = [self.system.mu_q, self.system.mu_p]
        param_groups.append({
            'params': mu_params,
            'lr': self.config.lr_mu,
            'name': 'mu'
        })

        # Covariances (Sigma or L)
        if self.system.use_cholesky_param:
            sigma_params = [self.system._L_q, self.system._L_p]
        else:
            sigma_params = [self.system._Sigma_q, self.system._Sigma_p]
        param_groups.append({
            'params': sigma_params,
            'lr': self.config.lr_sigma,
            'name': 'sigma'
        })

        # Gauge field (phi)
        param_groups.append({
            'params': [self.system.phi],
            'lr': self.config.lr_phi,
            'name': 'phi'
        })

        # Create optimizer
        if self.config.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                param_groups,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.n_steps,
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_gamma,
                patience=20,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler_type}")

    def step(self) -> Dict[str, torch.Tensor]:
        """
        Perform one optimization step using autograd.

        The magic of autograd:
        1. Forward pass: compute energy
        2. Backward pass: autograd computes ALL gradients
        3. Optimizer step: update parameters

        Returns:
            energies: Dictionary of energy components
        """
        step_start = time.perf_counter()

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with optional mixed precision
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                energies = self.system.compute_energy()
                loss = energies['total']
        else:
            energies = self.system.compute_energy()
            loss = energies['total']

        # Backward pass - AUTOGRAD MAGIC!
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient clipping
        if self.config.clip_grad_norm is not None:
            if self.config.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.system.parameters(),
                self.config.clip_grad_norm
            )

        # Optimizer step
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()

        # Constraint projection
        if self.current_step % self.config.project_spd_every == 0:
            self._project_spd()

        if self.current_step % self.config.project_phi_every == 0:
            self._project_phi()

        # Record history
        step_time = time.perf_counter() - step_start
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history.record(
            self.current_step, energies, self.system, current_lr, step_time
        )

        self.current_step += 1
        return energies

    def _project_spd(self):
        """Project covariances to SPD manifold."""
        with torch.no_grad():
            if self.system.use_cholesky_param:
                # Cholesky already guarantees SPD, but ensure positive diagonal
                diag_q = torch.diagonal(self.system._L_q, dim1=-2, dim2=-1)
                diag_q.clamp_(min=self.config.min_eigenvalue ** 0.5)

                diag_p = torch.diagonal(self.system._L_p, dim1=-2, dim2=-1)
                diag_p.clamp_(min=self.config.min_eigenvalue ** 0.5)
            else:
                # Direct Sigma: project to nearest SPD
                self._project_sigma_spd(self.system._Sigma_q)
                self._project_sigma_spd(self.system._Sigma_p)

    def _project_sigma_spd(self, Sigma: torch.Tensor):
        """Project Sigma to nearest SPD matrix via eigendecomposition."""
        # Symmetrize
        Sigma.data = 0.5 * (Sigma.data + Sigma.data.transpose(-1, -2))

        # Eigendecomposition
        try:
            eigvals, eigvecs = torch.linalg.eigh(Sigma.data)

            # Clamp eigenvalues
            eigvals_clamped = eigvals.clamp(min=self.config.min_eigenvalue)

            # Reconstruct
            Sigma.data = eigvecs @ torch.diag_embed(eigvals_clamped) @ eigvecs.transpose(-1, -2)
        except RuntimeError:
            # Fallback: add regularization
            eye = torch.eye(
                Sigma.shape[-1],
                device=Sigma.device,
                dtype=Sigma.dtype
            )
            Sigma.data += 0.01 * eye

    def _project_phi(self):
        """Project phi to principal ball |phi| < pi."""
        with torch.no_grad():
            phi_norms = torch.linalg.norm(self.system.phi, dim=-1, keepdim=True)
            max_norm = torch.pi - 0.01

            # Scale phi where norm exceeds limit
            scale = torch.where(
                phi_norms > max_norm,
                max_norm / (phi_norms + 1e-8),
                torch.ones_like(phi_norms)
            )
            self.system.phi.data *= scale

    def train(self, n_steps: Optional[int] = None) -> TensorTrainingHistory:
        """
        Run full training loop.

        Args:
            n_steps: Override config.n_steps if provided

        Returns:
            history: Training history
        """
        n_steps = n_steps or self.config.n_steps

        print("=" * 70)
        print("TENSOR TRAINING (GPU + AUTOGRAD)")
        print("=" * 70)
        print(f"System: {self.system.N} agents, K={self.system.K}")
        print(f"Device: {self.system.device}")
        print(f"Steps: {n_steps}")
        print(f"Optimizer: {self.config.optimizer}")
        print(f"Learning rates: mu={self.config.lr_mu}, "
              f"sigma={self.config.lr_sigma}, phi={self.config.lr_phi}")
        if self.config.use_amp:
            print("Mixed precision: ENABLED")
        print("=" * 70)

        # Initial energy
        with torch.no_grad():
            initial = self.system.compute_energy()
        print(f"\nInitial energy: {initial['total'].item():.6f}")
        print(f"  Self: {initial['self'].item():.6f}")
        print(f"  Belief: {initial['belief'].item():.6f}")
        print(f"  Prior: {initial['prior'].item():.6f}")
        print()

        # Training loop
        try:
            for step in range(n_steps):
                energies = self.step()

                # Logging
                if step % self.config.log_every == 0:
                    self._log_step(step, energies)

                # Early stopping
                if self._check_early_stop(energies['total'].item()):
                    print(f"\nEarly stopping at step {step}")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        # Final summary
        with torch.no_grad():
            final = self.system.compute_energy()

        avg_time = np.mean(self.history.step_times[-100:])

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final energy: {final['total'].item():.6f}")
        print(f"Energy reduction: "
              f"{initial['total'].item() - final['total'].item():.6f}")
        print(f"Avg step time: {avg_time * 1000:.2f} ms")
        print(f"Steps/second: {1.0 / avg_time:.1f}")
        print("=" * 70)

        return self.history

    def _log_step(self, step: int, energies: Dict[str, torch.Tensor]):
        """Print progress."""
        msg = f"Step {step:5d}: E = {energies['total'].item():8.4f}"

        components = []
        if energies['self'].item() > 1e-6:
            components.append(f"self={energies['self'].item():.3f}")
        if energies['belief'].item() > 1e-6:
            components.append(f"belief={energies['belief'].item():.3f}")
        if energies['prior'].item() > 1e-6:
            components.append(f"prior={energies['prior'].item():.3f}")

        if components:
            msg += f"  [{', '.join(components)}]"

        # Timing
        if self.history.step_times:
            recent = np.mean(self.history.step_times[-10:])
            msg += f"  ({recent * 1000:.1f} ms/step)"

        # Gradient norms
        if self.history.grad_norm_mu_q:
            msg += f"  |grad|: mu={self.history.grad_norm_mu_q[-1]:.2e}"
            msg += f", phi={self.history.grad_norm_phi[-1]:.2e}"

        print(msg)

    def _check_early_stop(self, current_energy: float) -> bool:
        """Check early stopping criterion."""
        improvement = self.best_energy - current_energy

        if improvement > self.config.early_stop_threshold:
            self.best_energy = current_energy
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stop_patience


# =============================================================================
# Convenience functions
# =============================================================================

def train_tensor_system(
    system: 'TensorSystem',
    n_steps: int = 1000,
    lr: float = 0.01,
    **kwargs
) -> TensorTrainingHistory:
    """
    Simple interface to train a TensorSystem.

    Args:
        system: TensorSystem to train
        n_steps: Number of training steps
        lr: Learning rate (same for all parameters)
        **kwargs: Additional config options

    Returns:
        Training history
    """
    config = TensorTrainingConfig(
        n_steps=n_steps,
        lr_mu=lr,
        lr_sigma=lr * 0.1,
        lr_phi=lr * 0.1,
        **kwargs
    )
    trainer = TensorTrainer(system, config)
    return trainer.train()


def compare_to_numpy_trainer(
    tensor_system: 'TensorSystem',
    numpy_system: 'MultiAgentSystem',
    n_steps: int = 100,
) -> Dict[str, Any]:
    """
    Compare tensor trainer to numpy trainer.

    Useful for validation that autograd gives same gradients as
    hand-derived gradient_terms.py.

    Returns:
        Comparison metrics
    """
    from agent.trainer import Trainer
    from config import TrainingConfig

    # Train tensor system
    config_t = TensorTrainingConfig(n_steps=n_steps, log_every=n_steps)
    trainer_t = TensorTrainer(tensor_system, config_t)

    t0 = time.perf_counter()
    history_t = trainer_t.train()
    time_tensor = time.perf_counter() - t0

    # Train numpy system
    config_n = TrainingConfig(n_steps=n_steps, log_every=n_steps)
    trainer_n = Trainer(numpy_system, config_n)

    t0 = time.perf_counter()
    history_n = trainer_n.train()
    time_numpy = time.perf_counter() - t0

    return {
        'tensor_final_energy': history_t.total_energy[-1],
        'numpy_final_energy': history_n.total_energy[-1],
        'tensor_time': time_tensor,
        'numpy_time': time_numpy,
        'speedup': time_numpy / time_tensor,
    }