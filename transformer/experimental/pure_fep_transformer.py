# -*- coding: utf-8 -*-
"""
Pure Free Energy Principle Transformer
========================================

A transformer architecture that learns ENTIRELY through VFE minimization,
WITHOUT backpropagation or external optimizers (Adam, SGD, etc.).

Key Principles:
---------------
1. FULL VARIATIONAL FREE ENERGY:

   F = α·Σ_i KL(q_i||p_i)                       [Self-coupling: belief-to-prior]
     + λ_β·Σ_i Σ_j β_ij·KL(q_i||Ω_ij·q_j)     [Belief alignment: social inference]
     + Σ_i E_{q_i}[-log p(y_i|z_i)]            [Observation likelihood]
     + λ_γ·Σ_i Σ_j KL(p_i||Ω_ij·p_j)          [Prior coupling: world model coherence]
     + Σ_i Σ_d decay^d·KL(p_i||h_i^d)         [Ouroboros Tower: non-Markovian memory]

   Where i,j index positions (tokens), d indexes ancestor depth in the hierarchy.

   The first three terms are always active. Prior coupling (λ_γ) encourages
   priors to form a coherent world model. Ouroboros Tower adds hyperpriors
   from ALL ancestors (grandparent, great-grandparent, ...) for long-range memory.

2. BELIEF UPDATE (fast timescale - perception):
   μ_q ← μ_q - η_μ · Σ_q · ∂F/∂μ_q    (natural gradient descent)

3. PRIOR BANK (unified embedding & output):
   Each token v has a prior belief: π_v = N(μ_v, Σ_v)
   - ENCODING: Initialize belief from token prior: q ← π_{y_t}
   - DECODING: p(y=v|q) ∝ exp(-KL(q||π_v)/τ)    [KL to token priors!]

   This creates beautiful symmetry - the same prior bank serves both purposes.

4. POSITION VIA GAUGE FRAMES:
   Position is encoded in the gauge frame φ_i ∈ so(3), NOT in μ!
   - Transport Ω_ij = exp(φ_i)·exp(-φ_j) encodes RELATIVE position
   - Same tokens at different positions have different φ, same μ_prior
   - This gives shift-invariant attention with position awareness

5. TWO-TIMESCALE DYNAMICS:
   - Fast: VFE gradient descent on beliefs (perception)
   - Slow: Prior evolution (learning)

6. GAUGE-EQUIVARIANT COVARIANCE TRANSPORT:
   When computing KL(q_i || Ω_ij·q_j), covariance must also be transported:
   - Full transport: Σ_transported = Ω @ Σ @ Ω^T
   - Efficient diagonal transport: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² · σ[l]

   The efficient diagonal formula computes the correct diagonal of the
   transported covariance without materializing full (B,N,N,K,K) tensors.
   This maintains gauge equivariance while being memory efficient.

   Note: Simply cloning untransported covariance (isotropic assumption)
   breaks gauge equivariance unless covariance is truly scalar (σ·I).

Theory:
-------
This implements "predictive coding" in the FEP sense:
- Each layer maintains beliefs about the layer below
- Prediction errors (KL divergences) drive belief updates
- Top-down priors constrain bottom-up inference
- Learning = adjusting priors to minimize long-term VFE

NO AD HOC STRUCTURES:
- No separate embedding lookup (use prior bank)
- No linear output projection (use KL to priors)
- No sinusoidal position encoding (use gauge frames)
- Attention emerges from information geometry: β_ij = softmax(-KL_ij/κ)
- The softmax coupling gradient IS the nonlinearity (replaces GELU/ReLU)

Author: Chris & Claude
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
import math
import numpy as np

# Import existing VFE components
from transformer.core.attention import (
    compute_attention_weights,
    aggregate_messages,
    compute_transport_operators,
    IrrepMultiHeadAttention,
)
from transformer.core.variational_ffn import (
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)
from transformer.core.embeddings import GaugeTokenEmbedding  # noqa: F401


def so3_compose_bch(X: torch.Tensor, Y: torch.Tensor, order: int = 1) -> torch.Tensor:
    """BCH composition in so(3): X ⊕ Y ≈ X + Y + [X,Y]/2 + ..."""
    result = X + Y
    if order >= 2 and X.shape[-1] == 3:
        # [X, Y] for so(3) is the cross product
        bracket = torch.cross(X, Y, dim=-1)
        result = result + 0.5 * bracket
    return result
from math_utils.generators import (
    generate_so3_generators,
    generate_multi_irrep_generators,
    generate_soN_generators,
    generate_multi_irrep_soN_generators,
    generate_glK_generators,
)
from transformer.core.prior_bank import PriorBank

# NOTE: PriorBank is now imported from transformer.core.prior_bank
# This consolidates the implementation and avoids duplication.


# =============================================================================
# GAUGE POSITIONAL ENCODING (position in φ, not μ!)
# =============================================================================

class GaugePositionEncoder(nn.Module):
    """
    Encode position in the gauge frame φ ∈ so(N), NOT in the belief mean μ!

    Key insight: Position information should be in the TRANSPORT, not the content.
    - Same token at different positions: SAME μ (semantic content)
    - Different positions: DIFFERENT φ (affects how beliefs interact)

    The transport operator Ω_ij = exp(φ_i)·exp(-φ_j) naturally encodes
    RELATIVE position through the gauge connection.

    This gives shift-invariant attention with position awareness:
    - Tokens 3 apart always have same relative transport
    - Attention pattern is translation-invariant

    Supports both SO(3) (phi_dim=3) and SO(N) (phi_dim = N(N-1)/2).
    """

    def __init__(
        self,
        max_seq_len: int,
        mode: str = 'learned',  # 'learned' or 'sinusoidal'
        scale: float = 0.1,
        composition: str = 'bch1',  # How to compose with token φ
        phi_dim: int = 3,  # Dimension of Lie algebra: 3 for SO(3), N(N-1)/2 for SO(N)
    ):
        """
        Initialize gauge position encoder.

        Args:
            max_seq_len: Maximum sequence length
            mode: 'learned' or 'sinusoidal'
            scale: Scale factor for position angles
            composition: 'add', 'bch1', 'bch2', or 'exact' for SO(3) composition
            phi_dim: Dimension of φ (3 for SO(3), N(N-1)/2 for SO(N))
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.scale = scale
        self.composition = composition
        self.phi_dim = phi_dim

        if mode == 'learned':
            # Learnable position-specific gauge frames
            self.pos_phi = nn.Parameter(torch.randn(max_seq_len, phi_dim) * scale)
        elif mode == 'sinusoidal':
            # Fixed sinusoidal position encoding
            self.register_buffer('pos_phi', self._make_sinusoidal(max_seq_len, phi_dim, scale))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _make_sinusoidal(self, max_len: int, phi_dim: int, scale: float) -> torch.Tensor:
        """Create sinusoidal positional encoding in so(N)."""
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # Create frequency terms for each dimension
        div_term = torch.exp(
            torch.arange(0, phi_dim, dtype=torch.float32) * -(math.log(10000.0) / max(phi_dim, 1))
        )

        phi = torch.zeros(max_len, phi_dim)
        for d in range(phi_dim):
            if d % 2 == 0:
                phi[:, d] = torch.sin(position.squeeze() * div_term[d])
            else:
                phi[:, d] = torch.cos(position.squeeze() * div_term[d])

        return phi * scale

    def forward(
        self,
        phi_token: torch.Tensor,  # (B, N, phi_dim) token gauge frames
        seq_len: int,
    ) -> torch.Tensor:
        """
        Compose token gauge frames with positional gauge frames.

        Args:
            phi_token: (B, N, phi_dim) token-specific gauge frames (or zeros)
            seq_len: Actual sequence length (may be < max_seq_len)

        Returns:
            phi: (B, N, phi_dim) combined gauge frames encoding position
        """
        B = phi_token.shape[0]
        device = phi_token.device

        # Get positional frames for this sequence length
        pos_phi = self.pos_phi[:seq_len].to(device)  # (N, phi_dim)
        pos_phi = pos_phi.unsqueeze(0).expand(B, -1, -1)  # (B, N, phi_dim)

        # Compose using appropriate method
        if self.composition == 'add':
            # Simple addition (valid for small angles)
            return phi_token + pos_phi
        elif self.composition in ('bch1', 'bch2') and self.phi_dim == 3:
            # BCH composition only works for SO(3) with 3-dimensional phi
            order = 1 if self.composition == 'bch1' else 2
            return so3_compose_bch(phi_token, pos_phi, order=order)
        else:
            # For SO(N) or unknown composition, use simple addition
            # (BCH for general Lie algebras requires structure constants)
            return phi_token + pos_phi

    def extra_repr(self) -> str:
        return f"max_seq_len={self.max_seq_len}, mode={self.mode}, phi_dim={self.phi_dim}"


@dataclass
class PureFEPConfig:
    """Configuration for Pure FEP Transformer."""

    # Architecture
    embed_dim: int = 127          # K - embedding dimension (MUST be odd for SO(3))
    num_layers: int = 2           # Number of hierarchical scales
    seq_length: int = 128         # N - sequence length (agents)
    vocab_size: int = 10000       # For language modeling

    # Irrep structure for SO(3) decomposition (DOCUMENTATION ONLY)
    # Each tuple: (label, multiplicity, dim) where dim must be odd (1,3,5,7,...)
    # Total dims must equal embed_dim
    # Example for K=127: 32×1 + 15×3 + 10×5 = 32 + 45 + 50 = 127
    #
    # NOTE: Currently this is only validated and displayed, NOT used for generator
    # construction. The actual generators use a single spin-ℓ irrep where ℓ=(K-1)/2.
    # Multi-irrep block-diagonal structure is a TODO for future implementation.
    irrep_spec: List[Tuple[str, int, int]] = None  # Will be auto-generated if None

    # =========================================================================
    # EMBEDDING MODE: Choose how to convert tokens → beliefs
    # =========================================================================
    # 'learned':     Standard nn.Embedding (ad hoc but fast convergence)
    # 'prior_bank':  PriorBank - unified embedding/output (principled FEP!)
    #                - Encode: q ← π_token (prior belief for token)
    #                - Decode: p(y|q) ∝ exp(-KL(q||π_token)/τ)
    # 'hybrid':      nn.Embedding + PriorBank output (compromise)
    embedding_mode: str = 'prior_bank'

    # =========================================================================
    # OUTPUT MODE: Choose how to compute observation likelihood
    # =========================================================================
    # 'linear':      Standard W·μ (ad hoc but standard)
    # 'kl_to_prior': p(y|q) ∝ exp(-KL(q||π_y)/τ) (principled FEP!)
    # 'both':        Blend of linear and KL (for comparison)
    output_mode: str = 'kl_to_prior'
    output_tau: float = 1.0       # Temperature for KL-based output

    # =========================================================================
    # POSITION MODE: Where to encode position information
    # =========================================================================
    # 'sinusoidal_mu':   Add sinusoidal to μ (standard Transformer, ad hoc)
    # 'gauge_frame':     Encode in φ ∈ so(3) (principled - affects transport!)
    # 'both':            Both for comparison
    position_mode: str = 'gauge_frame'
    position_encoding: str = 'learned'  # 'learned' or 'sinusoidal'
    position_scale: float = 0.1         # Scale for gauge position angles
    position_composition: str = 'bch1'  # How to compose: 'add', 'bch1', 'bch2'

    # VFE parameters
    alpha: float = 0.1            # Self-coupling: KL(q||p) - increased for stronger prior influence
    lambda_belief: float = 1.0    # Belief alignment weight
    lambda_obs: float = 1.0       # Observation likelihood weight (CE in VFE)
    kappa: float = 0.1            # Attention temperature (lower = sharper attention)

    # Learning rates (natural gradient allows larger steps)
    mu_lr: float = 0.1            # Belief mean learning rate
    sigma_lr: float = 0.025       # Belief variance learning rate
    prior_lr: float = 0.01        # Prior update rate (SLOWER for stability)
    phi_lr: float = 0.05          # Gauge frame learning rate

    # Timescale separation - CRITICAL for FEP!
    # Fast timescale: VFE steps (perception)
    # Slow timescale: Prior updates (learning)
    belief_steps: int = 20        # VFE steps per forward (MORE for convergence)
    prior_update_interval: int = 1   # Update priors every batch (learning happens via VFE)

    # Covariance mode
    diagonal_covariance: bool = True  # Use diagonal Σ (faster, less memory)

    # Exact covariance transport: compute Σ_transported = Ω @ Σ @ Ω^T exactly
    # When False: approximate as Σ_transported ≈ Σ (faster, less memory)
    # When True: exact gauge-equivariant transport (for theoretical validation)
    # WARNING: exact mode uses O(N²×K²) memory for full covariance!
    exact_covariance_transport: bool = False

    # Use multi-irrep block structure from IrrepMultiHeadAttention
    # When True: uses block-diagonal generators with proper per-irrep structure
    # When False: uses single spin-ℓ irrep where ℓ = (K-1)/2
    use_multi_irrep: bool = False

    # =========================================================================
    # GAUGE GROUP: Choose the symmetry group for transport operators
    # =========================================================================
    # 'SO3':  Standard SO(3) with 3 generators (φ ∈ ℝ³)
    #         - Irreps indexed by spin ℓ: dim = 2ℓ+1 (odd only)
    #         - For embed_dim K, uses spin ℓ = (K-1)/2
    # 'SON':  General SO(N) with N(N-1)/2 generators (φ ∈ ℝ^{N(N-1)/2})
    #         - Uses N-dimensional fundamental representation
    #         - More generators = richer gauge structure
    #         - Requires gauge_dim parameter
    # 'GLK':  General linear group GL(K) with K² generators (φ ∈ ℝ^{K²})
    #         - Full matrix algebra gl(K) = all KxK matrices
    #         - KL divergence is invariant under GL(K), not just SO(K)!
    #         - No orthogonality constraint needed (faster training)
    #         - Uses embed_dim as K
    gauge_group: str = 'SO3'
    gauge_dim: int = 3  # N for SO(N) when gauge_group='SON' (ignored for SO3/GLK)

    # Numerical stability
    eps: float = 1e-6             # General numerical stability floor
    variance_floor: float = 1e-4  # Minimum variance for KL computation (larger to prevent NaN)
    grad_clip: float = 1.0

    # PURE FEP MODE: No backprop, all learning via prior evolution
    pure_fep_mode: bool = True    # When True: NO backprop, ONLY VFE dynamics

    # VFE differentiability mode (only used when pure_fep_mode=False)
    differentiable_vfe: bool = True  # Compute VFE via autograd (for gradient flow)

    # =========================================================================
    # ADVANCED FEP FEATURES (toggled off by default)
    # =========================================================================

    # Prior coupling: λ_γ term for priors learning from each other
    # Implements F_prior = λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)
    # This allows priors to form consistent world model across positions
    # CRITICAL: Must be True for dφ/dt! (∂F/∂φ includes prior coupling term)
    prior_coupling_enabled: bool = True  # Was False - MUST be True for gauge evolution!
    lambda_prior: float = 0.1            # Weight for model coupling γ_ij·KL(s_i||Ω_ij·s_j)

    # Hyper-prior: KL(s_i || h) regularizes position models toward shared centroid.
    # h = mean of all position models (the "typical" model).
    # This prevents position models from memorizing per-position training statistics.
    # In the FEP framework: models (s) are coupled via γ_ij and regularized by h.
    # The prior (p) is what beliefs are measured against; the model (s) generates it.
    lambda_hyperprior: float = 0.1       # Weight for hyper-prior KL(s||h)

    # Prior-model separation: p_i = w·π_token + (1-w)·s_i
    # The PRIOR p is what beliefs are measured against in KL(q||p).
    # The MODEL s is a separate level that generates the prior.
    # token_prior_weight controls the blend:
    #   w=1.0 → prior = pure token identity (model has no influence)
    #   w=0.0 → prior = pure position model (current behavior, conflates p and s)
    #   w=0.5 → prior = equal blend (recommended)
    token_prior_weight: float = 0.5

    # Gradient-based prior updates: use VFE gradient to update priors
    # Instead of simple EMA, update priors via: p ← p - η_p · ∂F/∂p
    gradient_prior_updates: bool = False
    prior_grad_lr: float = 0.01       # Learning rate for gradient-based prior updates

    # Gauge field evolution: evolve gauge frames φ over time
    # Updates φ via: φ ← φ - η_φ · ∂F/∂φ
    # CRITICAL: Pure VFE requires this! φ is a dynamical field, not encoded
    gauge_evolution_enabled: bool = True  # Was False - TRUE for pure VFE!
    gauge_lr: float = 0.01               # Learning rate for gauge frame evolution
    phi_max_norm: float = 3.14159        # Max norm for φ (π radians = 180° rotation)

    # Dynamic layer emergence: allow layers to spawn/merge based on VFE
    # When enabled, monitors VFE gradients to detect when new layers needed
    dynamic_layers_enabled: bool = False
    layer_spawn_threshold: float = 0.5   # VFE gradient threshold for spawning
    max_layers: int = 8                   # Maximum allowed layers

    # =========================================================================
    # OUROBOROS TOWER: Multi-level hyperpriors from ALL ancestors
    # =========================================================================
    # Instead of just parent → child prior flow (Markovian), collect priors
    # from grandparent, great-grandparent, etc. (non-Markovian memory).
    #
    # Ouroboros Tower:
    #     p_i^(ζ)     ← q_M^(ζ+1)      (parent - immediate prior)
    #     h_i^(ζ,0)   ← q_M^(ζ+2)      (grandparent - 1st hyperprior)
    #     h_i^(ζ,1)   ← q_M^(ζ+3)      (great-grandparent - 2nd hyperprior)
    #
    # Each hyperprior contributes to the VFE with decaying weight:
    #     F += Σ_d decay^d · KL(p_i || h_i^d)
    #
    # This creates long-range memory: top layers influence bottom layers
    # beyond just their immediate parent, enabling deeper abstraction.
    enable_ouroboros_tower: bool = False
    tower_max_depth: int = 3              # How many ancestor levels to collect
    tower_decay: float = 0.3              # Weight decay per level (0.3^d)

    # =========================================================================
    # PERFORMANCE OPTIMIZATIONS
    # =========================================================================
    # Transport caching: cache Ω_ij when gauge frames don't evolve
    cache_transport: bool = True          # Cache transport operators across VFE steps

    # Local attention: O(N×W) instead of O(N²)
    use_local_attention: bool = False     # Use local window attention
    attention_window: int = 16            # Window size for local attention

    # VFE acceleration: momentum for faster convergence
    use_vfe_momentum: bool = True         # Use momentum in VFE updates
    vfe_momentum: float = 0.9             # Momentum coefficient

    # Fast matrix exponential: Taylor approximation for small angles
    use_fast_matrix_exp: bool = True      # Use Taylor approximation
    matrix_exp_order: int = 4             # Order of Taylor expansion

    # torch.compile: JIT compilation for kernel fusion
    use_torch_compile: bool = False       # Enable torch.compile (experimental)
    compile_mode: str = "reduce-overhead" # "default", "reduce-overhead", or "max-autotune"

    # Debugging
    debug_gradient_logging: bool = False  # Log gradient magnitudes for debugging

    def __post_init__(self):
        """Validate configuration."""
        # Validate gauge group
        if self.gauge_group not in ('SO3', 'SON', 'GLK'):
            raise ValueError(
                f"gauge_group must be 'SO3', 'SON', or 'GLK', got '{self.gauge_group}'"
            )

        if self.gauge_group == 'SON' and self.gauge_dim < 2:
            raise ValueError(
                f"gauge_dim must be >= 2 for SO(N), got {self.gauge_dim}"
            )

        # Compute phi dimension based on gauge group
        if self.gauge_group == 'SO3':
            self._phi_dim = 3  # so(3) has 3 generators
            self._n_generators = 3
        elif self.gauge_group == 'GLK':
            # GL(K) has K² generators (full matrix algebra)
            K = self.embed_dim
            self._phi_dim = K * K
            self._n_generators = self._phi_dim
        else:  # SON
            N = self.gauge_dim
            self._phi_dim = N * (N - 1) // 2  # so(N) has N(N-1)/2 generators
            self._n_generators = self._phi_dim

        # Auto-generate irrep_spec if not provided
        if self.irrep_spec is None:
            if self.gauge_group == 'SO3':
                # SO(3): Single-irrep mode requires odd embed_dim
                if self.embed_dim % 2 == 0:
                    raise ValueError(
                        f"embed_dim must be ODD for single SO(3) irrep (got {self.embed_dim}). "
                        f"Either use {self.embed_dim - 1} or {self.embed_dim + 1}, "
                        f"OR provide an irrep_spec for multi-irrep mode."
                    )
                self.irrep_spec = self._generate_irrep_spec(self.embed_dim)
            elif self.gauge_group == 'GLK':
                # GL(K): Single block of full embed_dim (no irrep structure)
                # For GL(K), the entire space transforms as one representation
                self.irrep_spec = [('full', 1, self.embed_dim)]
            else:
                # SO(N): Auto-generate as copies of fundamental rep
                N = self.gauge_dim
                n_copies = self.embed_dim // N
                remainder = self.embed_dim % N
                if remainder != 0:
                    # Add scalars to fill the gap
                    self.irrep_spec = [
                        ('scalar', remainder, 1),
                        ('fund', n_copies, N),
                    ]
                else:
                    self.irrep_spec = [('fund', n_copies, N)]
        else:
            # Validate provided irrep_spec
            if self.gauge_group == 'SO3':
                # SO(3): Each irrep dimension must be odd (2ℓ+1)
                for label, mult, dim in self.irrep_spec:
                    if dim % 2 == 0:
                        raise ValueError(
                            f"Irrep '{label}' has even dimension {dim}. "
                            f"Each SO(3) irrep must have odd dimension (2ℓ+1)."
                        )
                    if mult < 0:
                        raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")
            else:
                # SO(N): Dims must be 1 (scalar) or N (fundamental)
                N = self.gauge_dim
                for label, mult, dim in self.irrep_spec:
                    if dim != 1 and dim != N:
                        raise ValueError(
                            f"Irrep '{label}' has dimension {dim}, but SO({N}) "
                            f"only supports dim=1 (scalar) or dim={N} (fundamental)."
                        )
                    if mult < 0:
                        raise ValueError(f"Irrep '{label}' has negative multiplicity {mult}.")

        # Validate irrep_spec sums to embed_dim
        total_dim = sum(mult * dim for _, mult, dim in self.irrep_spec)
        if total_dim != self.embed_dim:
            raise ValueError(
                f"irrep_spec dimensions ({total_dim}) must equal embed_dim ({self.embed_dim}). "
                f"Current spec: {self.irrep_spec}"
            )

    @property
    def phi_dim(self) -> int:
        """Dimension of φ (Lie algebra dimension): 3 for SO(3), N(N-1)/2 for SO(N)."""
        return getattr(self, '_phi_dim', 3)

    @property
    def n_generators(self) -> int:
        """Number of generators in the gauge group."""
        return getattr(self, '_n_generators', 3)

    @staticmethod
    def _generate_irrep_spec(K: int) -> List[Tuple[str, int, int]]:
        """
        Auto-generate a reasonable irrep decomposition for dimension K.

        Strategy: Mix of scalars (ℓ=0), vectors (ℓ=1), and rank-2 tensors (ℓ=2)
        with roughly equal representation of each type.
        """
        # Target: ~40% scalars, ~35% vectors, ~25% rank-2 tensors
        n_ℓ2 = K // 15         # Each ℓ=2 irrep is 5-dim
        n_ℓ1 = K // 9          # Each ℓ=1 irrep is 3-dim
        remaining = K - (n_ℓ2 * 5 + n_ℓ1 * 3)
        n_ℓ0 = remaining       # Rest as scalars

        # Adjust to hit exact K
        current = n_ℓ0 * 1 + n_ℓ1 * 3 + n_ℓ2 * 5
        while current < K:
            n_ℓ0 += 1
            current += 1
        while current > K:
            if n_ℓ0 > 0:
                n_ℓ0 -= 1
                current -= 1
            elif n_ℓ1 > 0:
                n_ℓ1 -= 1
                current -= 3
            elif n_ℓ2 > 0:
                n_ℓ2 -= 1
                current -= 5

        spec = []
        if n_ℓ0 > 0:
            spec.append(('ℓ0', n_ℓ0, 1))
        if n_ℓ1 > 0:
            spec.append(('ℓ1', n_ℓ1, 3))
        if n_ℓ2 > 0:
            spec.append(('ℓ2', n_ℓ2, 5))

        return spec


class PureFEPLayer(nn.Module):
    """
    Single layer/scale in the Pure FEP hierarchy.

    Each layer maintains:
    - Beliefs q_i = N(μ_q, Σ_q) for each token/agent
    - Priors p_i = N(μ_p, Σ_p) constraining beliefs - NOW POSITION-DEPENDENT!
    - Gauge frames φ_i for parallel transport

    The layer performs VFE minimization:
    1. Compute attention β_ij from KL divergences (no W_Q, W_K!)
    2. Compute VFE gradients ∂F/∂μ, ∂F/∂σ INCLUDING observation term
    3. Update beliefs via natural gradient descent
    4. Optionally receive priors from parent layer

    CRITICAL FIX: Priors are now POSITION-DEPENDENT (N, K) not global (K,)
    This allows the model to learn position-specific patterns.

    OBSERVATION GRADIENT: Uses the appropriate output mode:
    - 'linear': W·μ (uses self.output_proj)
    - 'kl_to_prior': -KL(q||π_v)/τ (uses prior_bank)
    """

    def __init__(
        self,
        embed_dim: int,
        scale: int,  # Hierarchical scale ζ
        config: PureFEPConfig,
        prior_bank: Optional['PriorBank'] = None,  # Reference to prior bank for KL output
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = embed_dim
        self.N = config.seq_length  # Store sequence length for priors
        self.scale = scale
        self.config = config

        # Reference to prior bank (for KL-to-prior output mode)
        self.prior_bank = prior_bank

        # =====================================================================
        # GENERATORS: Build SO(3) or SO(N) generators based on gauge_group
        # =====================================================================
        self.gauge_group = config.gauge_group
        if config.gauge_group == 'SON':
            self.gauge_dim = config.gauge_dim
        elif config.gauge_group == 'GLK':
            self.gauge_dim = embed_dim  # GL(K) acts on K-dim space
        else:
            self.gauge_dim = 3  # SO3 default

        if config.gauge_group == 'SO3':
            # SO(3) mode: 3 generators, phi ∈ ℝ³
            if config.use_multi_irrep:
                # Multi-irrep: block-diagonal with spin-ℓ blocks
                self.irrep_attention = IrrepMultiHeadAttention(
                    embed_dim=embed_dim,
                    irrep_spec=config.irrep_spec,
                    kappa_beta=config.kappa,
                    epsilon=config.eps,
                    aggregate_mode='mean_only',
                    diagonal_covariance=config.diagonal_covariance,
                    attention_pattern='local' if config.use_local_attention else 'full',
                    attention_window=config.attention_window,
                )
                gen_np = generate_multi_irrep_generators(config.irrep_spec, validate=True)
            else:
                # Single irrep: spin-ℓ where ℓ = (K-1)/2
                self.irrep_attention = None
                gen_np = generate_so3_generators(embed_dim)
        elif config.gauge_group == 'SON':
            # SO(N) mode: N(N-1)/2 generators, phi ∈ ℝ^{N(N-1)/2}
            N = config.gauge_dim
            self.irrep_attention = None  # IrrepMultiHeadAttention is SO(3)-specific

            if config.use_multi_irrep or True:  # Always use multi-irrep for SO(N)
                # Multi-irrep: block-diagonal with N-dim fundamental blocks
                gen_np = generate_multi_irrep_soN_generators(
                    config.irrep_spec, N, validate=True
                )
            else:
                # Single fundamental rep (only if embed_dim == N)
                gen_np = generate_soN_generators(N)
        elif config.gauge_group == 'GLK':
            # GL(K) mode: K² generators spanning gl(K) = all KxK matrices
            # KL divergence is invariant under full GL(K), not just SO(K)!
            K = embed_dim
            self.irrep_attention = None  # IrrepMultiHeadAttention is SO(3)-specific
            gen_np = generate_glK_generators(K)
            print(f"[INFO] GL(K) gauge group: {K}² = {K**2} generators")
        else:
            raise ValueError(f"Unknown gauge_group: {config.gauge_group}")

        self.register_buffer('generators', torch.from_numpy(gen_np).float())

        # Store phi dimension for later use
        self._phi_dim = config.phi_dim

        # Output projection - maps beliefs to logits for observation likelihood
        # Only used when output_mode='linear' or as fallback
        # In pure FEP mode, this is updated via VFE pressure on priors, not backprop
        if config.output_mode == 'linear' or prior_bank is None:
            self.output_proj = nn.Linear(embed_dim, config.vocab_size, bias=False)
        else:
            self.output_proj = None  # Use prior_bank instead

        # =====================================================================
        # POSITION-DEPENDENT PRIORS - the key to learning structure!
        # =====================================================================
        # Shape (N, K) - each position has its own prior
        # This allows the model to learn:
        #   - Position-specific patterns (e.g., "The" often starts sentences)
        #   - Sequential dependencies via prior evolution
        #   - Context-dependent expectations
        #
        # PRINCIPLED INITIALIZATION: Start from token prior statistics
        # This avoids artificial mismatch between position priors (at origin)
        # and beliefs (initialized from token priors at non-zero locations).
        if prior_bank is not None:
            with torch.no_grad():
                if prior_bank.gauge_fixed_priors:
                    # Use base prior (all tokens are rotations of this)
                    init_mu = prior_bank.base_prior_mu.detach().clone()
                    init_sigma = prior_bank.base_prior_sigma.detach().clone()
                else:
                    # Use mean of all token priors
                    init_mu = prior_bank.prior_mu.detach().mean(dim=0)
                    init_sigma = prior_bank.prior_sigma.detach().mean(dim=0)
            # Broadcast to all positions
            self.register_buffer('prior_mu', init_mu.unsqueeze(0).expand(config.seq_length, -1).clone())
            self.register_buffer('prior_sigma', init_sigma.unsqueeze(0).expand(config.seq_length, -1).clone())
        else:
            # Fallback: zeros (ad-hoc, but prior_bank=None is already ad-hoc)
            self.register_buffer('prior_mu', torch.zeros(config.seq_length, embed_dim))
            self.register_buffer('prior_sigma', torch.ones(config.seq_length, embed_dim))

        # =====================================================================
        # GAUGE FRAMES φ: Persistent connection on principal bundle
        # =====================================================================
        # CRITICAL: φ is NOT "position encoded" - it's a dynamical field!
        # Each position has φ_i ∈ so(N) that evolves via dφ/dt = -∂F/∂φ
        # Position structure EMERGES from minimizing VFE, not imposed by encoding
        phi_dim = getattr(config, '_phi_dim', config.phi_dim)
        self.phi = nn.Parameter(torch.randn(config.seq_length, phi_dim) * 0.1)

        # Prior update statistics (for adaptive learning rate)
        self.register_buffer('prior_update_count', torch.zeros(config.seq_length))
        self.register_buffer('prior_prediction_error', torch.zeros(config.seq_length))

        # Timescale tracking
        self.info_accumulator = 0.0
        self.timescale_threshold = 10.0 ** scale

        # Statistics tracking
        self.total_vfe_steps = 0
        self.total_prior_updates = 0

        # VFE momentum buffers - registered so they save/load with model
        # Initialized to None, will be created on first use with correct shape
        self.register_buffer('_momentum_mu', None, persistent=False)
        self.register_buffer('_momentum_sigma', None, persistent=False)

        # =====================================================================
        # OUROBOROS TOWER: Hyperpriors from ancestors (grandparent, great-grandparent, ...)
        # =====================================================================
        # These store transported beliefs from ancestor layers beyond the immediate parent.
        # Each entry is a hyperprior at a different depth in the hierarchy.
        # h_i^(d) = Ω_i·q_ancestor^(ζ+d+1) where d=0 is grandparent, d=1 is great-grandparent, etc.
        self.hyperprior_mus: List[torch.Tensor] = []      # List of (N, K) tensors
        self.hyperprior_sigmas: List[torch.Tensor] = []   # List of (N, K) tensors

        # Optional torch.compile for kernel fusion
        self._compiled = False

    def maybe_compile(self):
        """Optionally compile VFE step with torch.compile for faster execution."""
        if self.config.use_torch_compile and not self._compiled:
            try:
                # Compile the VFE step for kernel fusion
                # Note: torch.compile works best with static shapes
                self.vfe_step = torch.compile(
                    self.vfe_step,
                    mode=self.config.compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
                self._compiled = True
            except RuntimeError as e:
                print(f"Warning: torch.compile failed for layer {self.scale}: {e}")
                self._compiled = True  # Don't retry

    def init_beliefs(
        self,
        x: torch.Tensor,  # (B, N, K) input embeddings
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize beliefs, priors, and gauge frames from input.

        Beliefs are initialized from input embeddings.
        Priors are loaded from PERSISTENT POSITION-DEPENDENT state.
        Gauge frames start at identity (zero angle).

        CRITICAL: Priors are now (N, K) - position-dependent!

        Returns:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances (diagonal)
            mu_p: (B, N, K) prior means (position-dependent!)
            sigma_p: (B, N, K) prior variances (position-dependent!)
            phi: (B, N, phi_dim) gauge frames in so(N) where phi_dim = N(N-1)/2
        """
        B, N, K = x.shape

        # Beliefs initialized from input
        # In pure FEP mode, we DON'T need gradient connection - learning is via priors
        if self.config.pure_fep_mode:
            mu_q = x.detach().clone()  # Detach in pure FEP mode
        else:
            mu_q = x.clone()
            if self.config.differentiable_vfe and not mu_q.requires_grad:
                mu_q.requires_grad_(True)

        # =====================================================================
        # PRIOR-MODEL SEPARATION: p = f(s, π_token)
        # =====================================================================
        # In the FEP hierarchy:
        #   s_i = position model (self.prior_mu) — learned, slow timescale
        #   π_token = token prior (x, from PriorBank) — token identity
        #   p_i = w·π_token + (1-w)·s_i — the PRIOR beliefs are measured against
        #
        # KL(q||p) pulls beliefs toward the prior (which blends token + position).
        # KL(s||Ω·s) and KL(s||h) act on the MODELS, not the priors.
        # This properly separates the three levels: q (fast) ← p (derived) ← s (slow).
        N_model = min(N, self.prior_mu.shape[0])

        # Position models s_i (expand across batch)
        model_mu = self.prior_mu[:N_model, :].unsqueeze(0).expand(B, -1, -1).clone()
        model_sigma = self.prior_sigma[:N_model, :].unsqueeze(0).expand(B, -1, -1).clone()

        # Pad if input sequence is longer than stored models
        if N > N_model:
            model_mu_mean = self.prior_mu[:N_model, :].mean(dim=0, keepdim=True)
            model_sigma_mean = self.prior_sigma[:N_model, :].mean(dim=0, keepdim=True)
            model_mu = torch.cat([model_mu, model_mu_mean.expand(B, N - N_model, K).clone()], dim=1)
            model_sigma = torch.cat([model_sigma, model_sigma_mean.expand(B, N - N_model, K).clone()], dim=1)

        # Token priors π_token = x (the embedding of input tokens)
        token_mu = x.detach().clone()  # (B, N, K) — always detach, this is the fixed token identity

        # PRIOR = blend of token prior and position model
        # p = w·π_token + (1-w)·s
        w = self.config.token_prior_weight
        mu_p = w * token_mu + (1 - w) * model_mu
        sigma_p = model_sigma  # Variance from position model (token variance not tracked separately)

        # Initialize belief variance from prior variance
        sigma_q = sigma_p.clone()

        # =====================================================================
        # GAUGE FRAMES φ: Use persistent learned field (not zeros!)
        # =====================================================================
        # φ is a dynamical parameter that evolves via dφ/dt = -∂F/∂φ
        # NOT initialized to zeros - use learned persistent state
        phi_dim = getattr(self, '_phi_dim', 3)
        N_phi = min(N, self.phi.shape[0])

        # Get persistent phi (slice to sequence length, expand across batch)
        phi = self.phi[:N_phi].unsqueeze(0).expand(B, -1, -1).clone()

        # Pad if input sequence longer than stored phi
        if N > N_phi:
            # Use mean of existing phi for new positions (principled)
            phi_mean = self.phi[:N_phi].mean(dim=0, keepdim=True)
            phi_pad = phi_mean.expand(B, N - N_phi, -1).clone()
            phi = torch.cat([phi, phi_pad], dim=1)

        # Enable gradients for VFE descent (always! not just if evolution enabled)
        # Gradient flows through VFE, then we update persistent self.phi
        phi = phi.detach().requires_grad_(True)

        return mu_q, sigma_q, mu_p, sigma_p, phi

    def vfe_step(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,
        sigma_p: torch.Tensor,
        phi: torch.Tensor,
        targets: Optional[torch.Tensor] = None,  # (B, N) target tokens
        mask: Optional[torch.Tensor] = None,     # (B, N, N) causal mask
        is_final_step: bool = False,             # Only create_graph on final step!
        cached_transport: Optional[dict] = None, # Precomputed transport operators
        step_idx: int = 0,                       # Current VFE step index (for momentum)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Single VFE gradient descent step on beliefs.

        This is the PERCEPTION step - updating beliefs to minimize VFE
        given current priors and observations.

        CRITICAL FIX: Observation term (CE) is now INSIDE the VFE!
        F = α·KL(q||p) + λ·alignment + λ_obs·E_q[-log p(y|x)]

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K) belief variances
            mu_p: (B, N, K) prior means
            sigma_p: (B, N, K) prior variances
            phi: (B, N, 3) gauge frames
            targets: (B, N) target tokens for cross-entropy
            mask: (B, N, N) causal attention mask

        Returns:
            mu_q_new: Updated belief means
            sigma_q_new: Updated belief variances
            phi_new: Updated gauge frames
            metrics: Dict of loss components
        """
        B, N, K = mu_q.shape
        device = mu_q.device

        # ==================================================================
        # 1. Compute dynamic attention from KL divergences
        # ==================================================================
        if self.config.use_multi_irrep and self.irrep_attention is not None:
            # MULTI-IRREP MODE: Use IrrepMultiHeadAttention
            # This properly handles block-diagonal structure with per-irrep generators
            mu_agg, sigma_agg, beta_heads, kl_heads = self.irrep_attention(
                mu_q, sigma_q, phi,
                generators=self.generators,  # For device/dtype reference
                mask=mask,
                return_attention=True,
            )
            # Average attention across heads for metrics/VFE computation
            beta = beta_heads.mean(dim=1)  # (B, N, N)
            kl_matrix = kl_heads.mean(dim=1)  # (B, N, N)
        else:
            # SINGLE IRREP: Full O(N²) attention with optional transport caching
            beta, kl_matrix = compute_attention_weights(
                mu_q, sigma_q, phi, self.generators,
                kappa=self.config.kappa,
                mask=mask,
                return_kl=True,
                diagonal_covariance=True,
                cached_transport=cached_transport,  # Use cached transport!
            )

        # ==================================================================
        # 2. Compute FULL VFE with proper hierarchy
        # ==================================================================
        # F = α·KL(q||p) + λ·alignment + λ_obs·CE + γ·KL(s||Ω·s) + λ_h·KL(s||h)
        #
        # where p = w·π_token + (1-w)·s  (prior derived from model + token identity)
        #       s = position model (self.prior_mu, slow timescale)
        #       h = centroid of position models (hyper-prior)
        #
        # Three levels: q (beliefs, fast) ← p (prior, derived) ← s (model, slow) ← h

        # Self-coupling: α·KL(q||p)
        # Use variance_floor (larger than eps) to prevent numerical issues in KL division
        variance_floor = getattr(self.config, 'variance_floor', 1e-4)
        sigma_q_safe = sigma_q.clamp(min=variance_floor)
        sigma_p_safe = sigma_p.clamp(min=variance_floor)
        kl_self = 0.5 * (
            sigma_q_safe / sigma_p_safe
            + (mu_q - mu_p)**2 / sigma_p_safe
            - 1.0
            + torch.log(sigma_p_safe / sigma_q_safe)
        ).sum(dim=-1).mean()

        # Alignment: λ·Σ β_ij·KL_ij (use precomputed)
        alignment = (beta * kl_matrix).sum(dim=-1).mean()

        # MODEL coupling: γ_ij · KL(s_i || Ω_ij · s_j)
        # Acts on MODELS (self.prior_mu), NOT the blended priors (mu_p).
        # The prior p = blend(s, π_token) is for KL(q||p) only.
        N_model = min(N, self.prior_mu.shape[0])
        model_mu_for_coupling = self.prior_mu[:N_model].unsqueeze(0).expand(B, -1, -1)
        model_sigma_for_coupling = self.prior_sigma[:N_model].unsqueeze(0).expand(B, -1, -1)
        if N > N_model:
            pad_mu = self.prior_mu[:N_model].mean(dim=0, keepdim=True).unsqueeze(0).expand(B, N - N_model, -1)
            pad_sigma = self.prior_sigma[:N_model].mean(dim=0, keepdim=True).unsqueeze(0).expand(B, N - N_model, -1)
            model_mu_for_coupling = torch.cat([model_mu_for_coupling, pad_mu], dim=1)
            model_sigma_for_coupling = torch.cat([model_sigma_for_coupling, pad_sigma], dim=1)
        prior_coupling = self.compute_prior_coupling_loss(model_mu_for_coupling, model_sigma_for_coupling, phi, mask)

        # ==================================================================
        # OBSERVATION TERM: E_q[-log p(y|x)]
        # ==================================================================
        # Uses the appropriate output mode:
        # - 'linear': CE(W·μ_q, targets)
        # - 'kl_to_prior': CE(-KL(q||π_v)/τ, targets)
        if targets is not None:
            if self.config.output_mode == 'kl_to_prior' and self.prior_bank is not None:
                # PRINCIPLED: KL to token priors
                logits = self.prior_bank.decode(
                    mu_q, sigma_q,
                    tau=getattr(self.config, 'output_tau', 1.0)
                )
            elif self.output_proj is not None:
                # AD HOC: Linear projection
                logits = self.output_proj(mu_q)
            else:
                # Fallback: use prior_bank if available
                logits = self.prior_bank.decode(mu_q, sigma_q, tau=1.0)

            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                reduction='mean'
            )
        else:
            ce_loss = torch.tensor(0.0, device=device)

        # ==================================================================
        # HYPER-PRIOR: KL(s_i || h) — regularizes position MODELS
        # ==================================================================
        # Acts on MODELS (self.prior_mu), NOT the blended priors (mu_p).
        # h = centroid of position models (the "typical" model).
        # Prevents position models from memorizing per-position statistics.
        model_sigma_safe = model_sigma_for_coupling.clamp(min=variance_floor)
        h_mu = model_mu_for_coupling.mean(dim=1, keepdim=True).detach()  # (B, 1, K)
        h_sigma = model_sigma_safe.mean(dim=1, keepdim=True).detach().clamp(min=variance_floor)
        # Broaden hyper-prior variance (2x) to allow model diversity
        h_sigma_broad = h_sigma * 2.0

        hyperprior_kl = 0.5 * (
            model_sigma_safe / h_sigma_broad
            + (model_mu_for_coupling - h_mu)**2 / h_sigma_broad
            - 1.0
            + torch.log(h_sigma_broad / model_sigma_safe)
        ).sum(dim=-1).mean()

        # ==================================================================
        # FULL VFE = KL terms + observation + hyper-prior
        # ==================================================================
        lambda_obs = getattr(self.config, 'lambda_obs', 1.0)
        lambda_h = getattr(self.config, 'lambda_hyperprior', 0.1)
        vfe_loss = (self.config.alpha * kl_self +
                   self.config.lambda_belief * alignment +
                   prior_coupling +
                   lambda_obs * ce_loss +
                   lambda_h * hyperprior_kl)

        # ==================================================================
        # 3. Compute gradients (analytical mode for pure FEP)
        # ==================================================================
        # Track whether gradients are enabled (needed for gauge evolution and metrics)
        grad_enabled = torch.is_grad_enabled() and (not self.config.pure_fep_mode)

        if self.config.pure_fep_mode:
            # PURE FEP MODE: Use analytical gradients, no autograd graph
            grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                mu_q, sigma_q, mu_p, sigma_p,
                beta, phi, self.generators,
                alpha=self.config.alpha,
                lambda_belief=self.config.lambda_belief,
                kappa=self.config.kappa,
                eps=self.config.eps,
                cached_transport=cached_transport,  # Use cached transport!
            )

            # Add observation gradient (CE) analytically
            # CRITICAL: Use the SAME output mode as the forward pass!
            if targets is not None:
                with torch.enable_grad():
                    mu_q_grad = mu_q.detach().requires_grad_(True)
                    sigma_q_grad = sigma_q.detach().requires_grad_(True)

                    if self.config.output_mode == 'kl_to_prior' and self.prior_bank is not None:
                        # PRINCIPLED: KL to token priors
                        logits_grad = self.prior_bank.decode(
                            mu_q_grad, sigma_q_grad,
                            tau=getattr(self.config, 'output_tau', 1.0)
                        )
                    elif self.output_proj is not None:
                        # AD HOC: Linear projection
                        logits_grad = self.output_proj(mu_q_grad)
                    else:
                        # Fallback
                        logits_grad = self.prior_bank.decode(mu_q_grad, sigma_q_grad, tau=1.0)

                    ce_for_grad = F.cross_entropy(
                        logits_grad.view(-1, self.config.vocab_size),
                        targets.view(-1),
                        reduction='sum'  # Sum - normalize per token below
                    )
                    grad_mu_ce = torch.autograd.grad(ce_for_grad, mu_q_grad, retain_graph=True)[0]

                    # Also get sigma gradient for KL-based output (affects logits!)
                    if self.config.output_mode == 'kl_to_prior' and self.prior_bank is not None:
                        grad_sigma_ce = torch.autograd.grad(ce_for_grad, sigma_q_grad)[0]
                        grad_sigma = grad_sigma + lambda_obs * grad_sigma_ce / (B * N)

                # Add CE gradient with lambda_obs weight, normalized per token
                grad_mu = grad_mu + lambda_obs * grad_mu_ce / (B * N)

                # Debug logging: Monitor gradient magnitudes
                if self.config.debug_gradient_logging and step_idx == 0:
                    # Compute gradient component magnitudes (CE is available, total is grad_mu)
                    grad_mu_ce_mag = grad_mu_ce.abs().mean().item()
                    grad_mu_total_mag = grad_mu.abs().mean().item()

                    # VFE gradient components (computed above at line 1219)
                    # grad_mu already includes VFE + CE terms
                    grad_mu_vfe_mag = grad_mu_total_mag - (lambda_obs * grad_mu_ce_mag / (B * N))

                    print(f"  [Gradient Debug] Step {step_idx}")
                    print(f"    VFE grad (KL terms): {grad_mu_vfe_mag:.6f}")
                    print(f"    Observation grad:    {grad_mu_ce_mag:.6f}")
                    print(f"    Total grad:          {grad_mu_total_mag:.6f}")
                    if grad_mu_vfe_mag > 1e-8:
                        print(f"    Ratio CE/VFE:        {grad_mu_ce_mag / grad_mu_vfe_mag:.3f}")
        else:
            # Differentiable mode (for hybrid training)
            grad_enabled = torch.is_grad_enabled() and mu_q.requires_grad
            if grad_enabled and mu_q.requires_grad:
                if is_final_step:
                    grad_mu = torch.autograd.grad(
                        vfe_loss, mu_q,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                else:
                    grad_mu = torch.autograd.grad(
                        vfe_loss, mu_q,
                        create_graph=False,
                        retain_graph=True,
                    )[0]
            else:
                grad_mu = torch.zeros_like(mu_q)
            grad_sigma = torch.zeros_like(sigma_q)

        # ==================================================================
        # 4. Project to natural gradients (Fisher-Rao metric)
        # ==================================================================
        nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
            grad_mu, grad_sigma, sigma_q, eps=self.config.eps
        )

        # ==================================================================
        # 5. Gradient clipping for stability
        # ==================================================================
        grad_norm = torch.norm(nat_grad_mu)
        if grad_norm > self.config.grad_clip:
            nat_grad_mu = nat_grad_mu * self.config.grad_clip / grad_norm

        sigma_grad_norm = torch.norm(nat_grad_sigma)
        if sigma_grad_norm > self.config.grad_clip:
            nat_grad_sigma = nat_grad_sigma * self.config.grad_clip / sigma_grad_norm

        # ==================================================================
        # 6. Update beliefs (gradient DESCENT with optional momentum)
        # ==================================================================
        if self.config.use_vfe_momentum:
            # Initialize momentum buffers if needed
            if self._momentum_mu is None or self._momentum_mu.shape != nat_grad_mu.shape:
                self._momentum_mu = torch.zeros_like(nat_grad_mu)
                self._momentum_sigma = torch.zeros_like(nat_grad_sigma)

            # Reset momentum at start of each forward pass (step_idx == 0)
            if step_idx == 0:
                self._momentum_mu = torch.zeros_like(nat_grad_mu)
                self._momentum_sigma = torch.zeros_like(nat_grad_sigma)

            # Momentum update: v = β*v + g, then θ = θ - lr*v
            self._momentum_mu = self.config.vfe_momentum * self._momentum_mu + nat_grad_mu
            self._momentum_sigma = self.config.vfe_momentum * self._momentum_sigma + nat_grad_sigma

            mu_q_new = mu_q - self.config.mu_lr * self._momentum_mu
            sigma_q_new = sigma_q - self.config.sigma_lr * self._momentum_sigma
        else:
            # Standard gradient descent
            mu_q_new = mu_q - self.config.mu_lr * nat_grad_mu
            sigma_q_new = sigma_q - self.config.sigma_lr * nat_grad_sigma

        # Ensure sigma stays positive
        sigma_q_new = sigma_q_new.clamp(min=self.config.eps)

        # ==================================================================
        # 7. Update gauge frames (if gauge evolution enabled)
        # ==================================================================
        # Only update gauge frames during training with gradients enabled
        if self.config.gauge_evolution_enabled and phi.requires_grad and grad_enabled:
            # Compute gradient of VFE w.r.t. gauge frames
            # The gauge frames affect attention via transport operators
            # Use vfe_loss if we computed it in differentiable mode
            loss_for_phi = vfe_loss if (self.config.differentiable_vfe and grad_enabled) else (kl_self + alignment)
            try:
                grad_phi = torch.autograd.grad(
                    loss_for_phi,
                    phi,
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad_phi is not None:
                    # Gradient descent on gauge frames
                    phi_new = phi - self.config.gauge_lr * grad_phi
                else:
                    phi_new = phi
            except RuntimeError:
                # If grad computation fails, keep phi unchanged
                phi_new = phi
        else:
            phi_new = phi

        # ==================================================================
        # 8. Compute metrics (reuse values computed in step 2)
        # ==================================================================
        # kl_self, alignment, prior_coupling were already computed above
        # No need to recompute - just detach for metrics
        prior_coupling_val = prior_coupling.detach().item() if isinstance(prior_coupling, torch.Tensor) else 0.0
        hyperprior_val = hyperprior_kl.detach().item() if isinstance(hyperprior_kl, torch.Tensor) else 0.0
        ce_loss_val = ce_loss.detach().item() if isinstance(ce_loss, torch.Tensor) else ce_loss
        metrics = {
            'vfe_total': (self.config.alpha * kl_self.detach() +
                         self.config.lambda_belief * alignment.detach() +
                         prior_coupling_val +
                         ce_loss_val +
                         lambda_h * hyperprior_val),
            'kl_self': kl_self.detach().item(),
            'alignment': alignment.detach().item(),
            'model_coupling': prior_coupling_val,
            'hyperprior_kl': hyperprior_val,
            'ce_loss': ce_loss_val,
            'grad_norm_mu': grad_norm.item(),
            'grad_norm_sigma': sigma_grad_norm.item(),
        }

        self.total_vfe_steps += 1

        return mu_q_new, sigma_q_new, phi_new, metrics

    def update_prior_from_parent(
        self,
        mu_q: torch.Tensor,      # Child beliefs
        sigma_q: torch.Tensor,
        mu_p: torch.Tensor,      # Child priors (to be updated)
        sigma_p: torch.Tensor,
        phi: torch.Tensor,       # Child gauge frames
        parent_mu_q: torch.Tensor,    # Parent beliefs
        parent_sigma_q: torch.Tensor,
        parent_phi: torch.Tensor,     # Parent gauge frames
        parent_generators: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical prior update: p_child ← Ω · q_parent

        This is the LEARNING mechanism in pure FEP:
        - Parent layer forms beliefs about what children should believe
        - These flow down as priors, constraining child inference
        - Over time, this shapes the "knowledge" of the network

        Args:
            mu_q, sigma_q: Child belief distribution
            mu_p, sigma_p: Child prior (to be updated)
            phi: Child gauge frames
            parent_mu_q, parent_sigma_q: Parent beliefs
            parent_phi: Parent gauge frames
            parent_generators: Parent's SO(3) generators

        Returns:
            mu_p_new: Updated prior means
            sigma_p_new: Updated prior variances
        """
        B, N, K = mu_q.shape
        device = mu_q.device

        # Compute transport from parent frame to child frame
        # Ω_ij = exp(φ_child) @ exp(-φ_parent)
        phi_child_matrix = torch.einsum('bna,aij->bnij', phi, self.generators)
        phi_parent_matrix = torch.einsum('bna,aij->bnij', parent_phi, parent_generators)

        exp_child = torch.matrix_exp(phi_child_matrix)
        exp_neg_parent = torch.matrix_exp(-phi_parent_matrix)

        # Transport operator (per token)
        Omega = torch.einsum('bnik,bnjk->bnij', exp_child, exp_neg_parent)

        # Transport parent beliefs to child frame
        # μ_p_new = Ω @ μ_parent
        parent_mu_transported = torch.einsum('bnij,bnj->bni', Omega, parent_mu_q)

        # Transport covariance: Σ_transported = Ω @ Σ @ Ω^T
        if self.config.exact_covariance_transport:
            # EXACT: Full gauge-equivariant transport
            if self.config.diagonal_covariance:
                # Diagonal input → Full output → Extract diagonal
                # Σ_transported = Ω @ diag(σ) @ Ω^T
                sigma_diag = torch.diag_embed(parent_sigma_q)  # (B, N, K, K)
                sigma_transported_full = torch.einsum(
                    'bnij,bnjk,bnlk->bnil', Omega, sigma_diag, Omega
                )  # (B, N, K, K)
                # Extract diagonal for storage (full transport still computed)
                parent_sigma_transported = torch.diagonal(
                    sigma_transported_full, dim1=-2, dim2=-1
                )  # (B, N, K)
            else:
                # Full covariance: Σ_transported = Ω @ Σ @ Ω^T
                parent_sigma_transported = torch.einsum(
                    'bnij,bnjk,bnlk->bnil', Omega, parent_sigma_q, Omega
                )
        else:
            # EFFICIENT DIAGONAL TRANSPORT: compute diagonal of Ω @ diag(σ) @ Ω^T
            # Formula: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² * σ[l]
            # This maintains gauge equivariance for diagonal elements without
            # materializing the full (B, N, K, K) covariance tensor.
            Omega_sq = Omega ** 2  # (B, N, K, K)
            parent_sigma_transported = torch.einsum('bnkl,bnl->bnk', Omega_sq, parent_sigma_q)

        # Soft update: blend old prior with new (for stability)
        blend_factor = self.config.prior_lr
        mu_p_new = (1 - blend_factor) * mu_p + blend_factor * parent_mu_transported
        sigma_p_new = (1 - blend_factor) * sigma_p + blend_factor * parent_sigma_transported

        # Ensure sigma stays positive
        sigma_p_new = sigma_p_new.clamp(min=self.config.eps)

        self.total_prior_updates += 1

        return mu_p_new, sigma_p_new

    def compute_prior_coupling_loss(
        self,
        mu_p: torch.Tensor,      # (B, N, K) prior means
        sigma_p: torch.Tensor,   # (B, N, K) prior variances
        phi: torch.Tensor,       # (B, N, 3) gauge frames
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute prior-prior coupling loss: F_prior = λ_γ · Σ_ij KL(p_i || Ω_ij · p_j)

        This encourages priors at different positions to form a consistent
        world model, where information is coherent across the sequence.

        Args:
            mu_p: Prior means (B, N, K)
            sigma_p: Prior variances (B, N, K)
            phi: Gauge frames (B, N, 3)
            mask: Optional causal mask (B, N, N)

        Returns:
            prior_coupling_loss: Scalar loss value
        """
        if not self.config.prior_coupling_enabled:
            return torch.tensor(0.0, device=mu_p.device)

        B, N, K = mu_p.shape
        device = mu_p.device

        # Compute transport operators between positions
        transport_cache = compute_transport_operators(phi, self.generators)
        Omega = transport_cache['Omega']  # (B, N, N, K, K)

        # Transported prior mean: Ω_ij · μ_j
        mu_p_transported = torch.einsum('bnmij,bmj->bnmi', Omega, mu_p)  # (B, N, N, K)

        # Use variance_floor for numerical stability in KL computation
        variance_floor = getattr(self.config, 'variance_floor', 1e-4)
        sigma_p_safe = sigma_p.clamp(min=variance_floor)

        # Transport covariance - compute diagonal of Ω @ diag(σ_j) @ Ω^T
        # Even in approximate mode, we should transport covariance for gauge consistency
        if self.config.exact_covariance_transport:
            # EXACT: Full Σ_transported = Ω @ diag(σ_j) @ Ω^T, then extract diagonal
            if self.config.diagonal_covariance:
                # Expand sigma to (B, N, K, K) diagonal matrices
                sigma_j_diag = torch.diag_embed(sigma_p_safe)  # (B, N, K, K)
                # Transport: Ω_ij @ Σ_j @ Ω_ij^T -> (B, N, N, K, K)
                sigma_j_transported_full = torch.einsum(
                    'bnmij,bmjk,bnmlk->bnmil', Omega, sigma_j_diag, Omega
                )  # (B, N, N, K, K)
                # For KL computation with diagonal storage, extract diagonal
                sigma_j_transported = torch.diagonal(
                    sigma_j_transported_full, dim1=-2, dim2=-1
                )  # (B, N, N, K)
            else:
                # Full covariance transport
                sigma_j_transported = torch.einsum(
                    'bnmij,bmjk,bnmlk->bnmil', Omega, sigma_p_safe, Omega
                )
        else:
            # EFFICIENT DIAGONAL TRANSPORT: compute diagonal of Ω @ diag(σ_j) @ Ω^T
            # Formula: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² * σ[l]
            # This avoids materializing the full (B, N, N, K, K) covariance tensor
            # while maintaining gauge equivariance for the diagonal elements.
            #
            # sigma_j_expanded[b, i, j, l] = sigma_p_safe[b, j, l] (j-th position's variance)
            sigma_j_expanded = sigma_p_safe[:, None, :, :].expand(-1, N, -1, -1)  # (B, N, N, K)
            Omega_sq = Omega ** 2  # (B, N, N, K, K)
            # result[b,i,j,k] = Σ_l Omega_sq[b,i,j,k,l] * sigma_j_expanded[b,i,j,l]
            sigma_j_transported = torch.einsum('bijkl,bijl->bijk', Omega_sq, sigma_j_expanded)  # (B, N, N, K)

        # KL divergence: KL(p_i || Ω_ij·p_j)
        # For diagonal Gaussians: KL = 0.5 * (σ_i/σ_j_t + (μ_i-μ_j_t)²/σ_j_t - 1 + log(σ_j_t/σ_i))
        mu_i = mu_p.unsqueeze(2)  # (B, N, 1, K)
        sigma_i = sigma_p_safe.unsqueeze(2)  # (B, N, 1, K)
        sigma_j_t_safe = sigma_j_transported.clamp(min=variance_floor)

        kl_prior = 0.5 * (
            sigma_i / sigma_j_t_safe
            + (mu_i - mu_p_transported)**2 / sigma_j_t_safe
            - 1.0
            + torch.log(sigma_j_t_safe / sigma_i)
        ).sum(dim=-1)  # (B, N, N)

        # Apply causal mask if provided
        if mask is not None:
            kl_prior = kl_prior * mask

        # Average over all pairs
        prior_coupling_loss = self.config.lambda_prior * kl_prior.mean()

        return prior_coupling_loss

    def _compute_prior_alignment_gradient(
        self,
        mu_p: torch.Tensor,      # (N, K) prior means
        sigma_p: torch.Tensor,   # (N, K) prior variances
    ) -> torch.Tensor:
        """
        Compute prior alignment gradient: ∂/∂μ_p [λ_γ · Σ_j KL(p_i || Ω_ij p_j)]

        This gradient encourages priors to form a coherent world model
        by aligning with transported priors from other positions.

        For diagonal Gaussians:
            ∂KL(p_i || Ω_ij p_j)/∂μ_p[i] = Σ_j^{-1} @ (μ_p[i] - Ω_ij @ μ_p[j])

        Args:
            mu_p: Prior means (N, K)
            sigma_p: Prior variances (N, K)

        Returns:
            grad_mu_p: Gradient w.r.t. μ_p, shape (N, K)
        """
        N, K = mu_p.shape
        device = mu_p.device

        # Need batch dimension for transport operators
        mu_p_batch = mu_p.unsqueeze(0)  # (1, N, K)
        sigma_p_batch = sigma_p.unsqueeze(0)  # (1, N, K)

        # Create phi for prior (use zero gauge frames for priors)
        # Priors exist in a "reference frame" at φ=0
        # phi_dim = 3 for SO(3), N(N-1)/2 for SO(N)
        phi_dim = self.generators.shape[0]  # Number of generators = phi dimension
        phi = torch.zeros(1, N, phi_dim, device=device)

        # Compute transport operators
        transport_cache = compute_transport_operators(phi, self.generators)
        Omega = transport_cache['Omega']  # (1, N, N, K, K)

        # Transport all priors: Ω_ij @ μ_p[j]
        mu_p_transported = torch.einsum('bnmij,bmj->bnmi', Omega, mu_p_batch)  # (1, N, N, K)

        # Transport covariance - compute diagonal of Ω @ diag(σ_j) @ Ω^T
        # Use variance_floor for numerical stability
        variance_floor = getattr(self.config, 'variance_floor', 1e-4)
        sigma_p_safe = sigma_p_batch.clamp(min=variance_floor)
        if self.config.exact_covariance_transport:
            # EXACT: Full transport then extract diagonal
            sigma_j_diag = torch.diag_embed(sigma_p_safe)  # (1, N, K, K)
            sigma_j_transported_full = torch.einsum(
                'bnmij,bmjk,bnmlk->bnmil', Omega, sigma_j_diag, Omega
            )
            sigma_j_transported = torch.diagonal(
                sigma_j_transported_full, dim1=-2, dim2=-1
            )  # (1, N, N, K)
        else:
            # EFFICIENT DIAGONAL TRANSPORT: compute diagonal of Ω @ diag(σ_j) @ Ω^T
            # Formula: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² * σ[l]
            sigma_j_expanded = sigma_p_safe[:, None, :, :].expand(-1, N, -1, -1)  # (1, N, N, K)
            Omega_sq = Omega ** 2  # (1, N, N, K, K)
            sigma_j_transported = torch.einsum('bijkl,bijl->bijk', Omega_sq, sigma_j_expanded)  # (1, N, N, K)

        # Difference: μ_p[i] - Ω_ij @ μ_p[j]
        mu_i = mu_p_batch.unsqueeze(2)  # (1, N, 1, K)
        delta_mu = mu_i - mu_p_transported  # (1, N, N, K)

        # Gradient per pair: Σ_j^{-1} @ δμ (for diagonal: δμ / σ_j)
        grad_per_pair = delta_mu / sigma_j_transported.clamp(min=variance_floor)  # (1, N, N, K)

        # Sum over j (all positions contribute to gradient at i)
        # Weight by attention-like factor (uniform for priors)
        grad_mu_p = self.config.lambda_prior * grad_per_pair.sum(dim=2).squeeze(0) / N  # (N, K)

        return grad_mu_p

    def update_hyperpriors_from_ancestors(
        self,
        ancestor_beliefs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ancestor_generators: List[torch.Tensor],
        phi: torch.Tensor,  # (B, N, 3) current layer gauge frames
    ):
        """
        Ouroboros Tower: Collect hyperpriors from ALL ancestors beyond parent.

        This creates non-Markovian memory in the hierarchy:
        - h^(0) ← grandparent beliefs (layer ζ+2)
        - h^(1) ← great-grandparent beliefs (layer ζ+3)
        - h^(d) ← ancestor at depth d+2 above current layer

        Each ancestor's beliefs are transported to the current layer's frame.

        Args:
            ancestor_beliefs: List of (mu_q, sigma_q, phi) from ancestors
                              Index 0 = grandparent, 1 = great-grandparent, etc.
            ancestor_generators: SO(3) generators for each ancestor layer
            phi: Current layer's gauge frames (B, N, 3)

        Updates:
            self.hyperprior_mus: List of transported ancestor means (N, K)
            self.hyperprior_sigmas: List of transported ancestor variances (N, K)
        """
        if not self.config.enable_ouroboros_tower or len(ancestor_beliefs) == 0:
            self.hyperprior_mus = []
            self.hyperprior_sigmas = []
            return

        device = phi.device
        B, N, K = phi.shape[0], phi.shape[1], self.K

        # Clear previous hyperpriors
        self.hyperprior_mus = []
        self.hyperprior_sigmas = []

        # Limit to configured max depth
        max_ancestors = min(len(ancestor_beliefs), self.config.tower_max_depth)

        for depth in range(max_ancestors):
            ancestor_mu_q, ancestor_sigma_q, ancestor_phi = ancestor_beliefs[depth]
            ancestor_gen = ancestor_generators[depth]

            # Compute transport from ancestor frame to current frame
            # Ω = exp(φ_current) @ exp(-φ_ancestor)
            phi_current_matrix = torch.einsum('bna,aij->bnij', phi, self.generators)
            phi_ancestor_matrix = torch.einsum('bna,aij->bnij', ancestor_phi, ancestor_gen)

            exp_current = torch.matrix_exp(phi_current_matrix)
            exp_neg_ancestor = torch.matrix_exp(-phi_ancestor_matrix)

            # Transport operator
            Omega = torch.einsum('bnik,bnjk->bnij', exp_current, exp_neg_ancestor)

            # Transport ancestor beliefs to current frame
            mu_h = torch.einsum('bnij,bnj->bni', Omega, ancestor_mu_q)  # (B, N, K)

            # Transport covariance
            if self.config.exact_covariance_transport:
                if self.config.diagonal_covariance:
                    sigma_diag = torch.diag_embed(ancestor_sigma_q)  # (B, N, K, K)
                    sigma_transported_full = torch.einsum(
                        'bnij,bnjk,bnlk->bnil', Omega, sigma_diag, Omega
                    )  # (B, N, K, K)
                    sigma_h = torch.diagonal(
                        sigma_transported_full, dim1=-2, dim2=-1
                    )  # (B, N, K)
                else:
                    sigma_h = torch.einsum(
                        'bnij,bnjk,bnlk->bnil', Omega, ancestor_sigma_q, Omega
                    )
            else:
                # EFFICIENT DIAGONAL TRANSPORT: compute diagonal of Ω @ diag(σ) @ Ω^T
                # Formula: (Ω @ diag(σ) @ Ω^T)_kk = Σ_l Ω_kl² * σ[l]
                Omega_sq = Omega ** 2  # (B, N, K, K)
                sigma_h = torch.einsum('bnkl,bnl->bnk', Omega_sq, ancestor_sigma_q)

            # Average across batch and store as hyperprior
            # Store as (N, K) position-dependent hyperprior
            mu_h_mean = mu_h.mean(dim=0)  # (N, K)
            sigma_h_mean = sigma_h.mean(dim=0).clamp(min=self.config.eps)  # (N, K)

            self.hyperprior_mus.append(mu_h_mean)
            self.hyperprior_sigmas.append(sigma_h_mean)

    def compute_hyperprior_kl_loss(
        self,
        mu_p: torch.Tensor,      # (N, K) prior means
        sigma_p: torch.Tensor,   # (N, K) prior variances
    ) -> torch.Tensor:
        """
        Compute KL loss from hyperpriors: Σ_d decay^d · KL(p || h^d)

        This encourages priors to align with beliefs from ALL ancestors,
        not just the immediate parent.

        Args:
            mu_p: Prior means (N, K)
            sigma_p: Prior variances (N, K)

        Returns:
            Scalar loss value
        """
        if not self.config.enable_ouroboros_tower or len(self.hyperprior_mus) == 0:
            return torch.tensor(0.0, device=mu_p.device)

        total_kl = torch.tensor(0.0, device=mu_p.device)
        decay = self.config.tower_decay
        variance_floor = getattr(self.config, 'variance_floor', 1e-4)

        for depth, (mu_h, sigma_h) in enumerate(zip(self.hyperprior_mus, self.hyperprior_sigmas)):
            # KL(p || h^d) for diagonal Gaussians
            # = 0.5 * (σ_p/σ_h + (μ_p - μ_h)²/σ_h - 1 + log(σ_h/σ_p))
            sigma_p_safe = sigma_p.clamp(min=variance_floor)
            sigma_h_safe = sigma_h.clamp(min=variance_floor)

            kl = 0.5 * (
                sigma_p_safe / sigma_h_safe
                + (mu_p - mu_h) ** 2 / sigma_h_safe
                - 1.0
                + torch.log(sigma_h_safe / sigma_p_safe)
            ).sum(dim=-1).mean()  # Average over positions

            # Weight by decay^d (grandparent has weight decay^0, great-grandparent has decay^1, etc.)
            weight = decay ** depth
            total_kl = total_kl + weight * kl

        return total_kl

    def _compute_hyperprior_gradient(
        self,
        mu_p: torch.Tensor,      # (N, K) prior means
        sigma_p: torch.Tensor,   # (N, K) prior variances
    ) -> torch.Tensor:
        """
        Compute gradient from hyperpriors: ∂/∂μ_p [Σ_d decay^d · KL(p || h^d)]

        For diagonal Gaussians:
            ∂KL(p || h^d)/∂μ_p = (μ_p - μ_h^d) / σ_h^d

        Args:
            mu_p: Prior means (N, K)
            sigma_p: Prior variances (N, K)

        Returns:
            Gradient w.r.t. μ_p, shape (N, K)
        """
        if not self.config.enable_ouroboros_tower or len(self.hyperprior_mus) == 0:
            return torch.zeros_like(mu_p)

        grad = torch.zeros_like(mu_p)
        decay = self.config.tower_decay
        variance_floor = getattr(self.config, 'variance_floor', 1e-4)

        for depth, (mu_h, sigma_h) in enumerate(zip(self.hyperprior_mus, self.hyperprior_sigmas)):
            sigma_h_safe = sigma_h.clamp(min=variance_floor)

            # ∂KL/∂μ_p = (μ_p - μ_h) / σ_h
            grad_d = (mu_p - mu_h) / sigma_h_safe

            # Weight by decay^d
            weight = decay ** depth
            grad = grad + weight * grad_d

        return grad

    def update_persistent_prior(
        self,
        mu_q_batch: torch.Tensor,     # (B, N, K) batch belief means (after VFE)
        sigma_q_batch: torch.Tensor,  # (B, N, K) batch belief variances
        prediction_error: Optional[torch.Tensor] = None,  # (B,) or (B, N) CE loss
        per_position_error: Optional[torch.Tensor] = None,  # (B, N) per-position CE
    ):
        """
        Update POSITION-DEPENDENT persistent priors using prediction-error-weighted learning.

        This is where LEARNING is stored! Each position's prior moves towards beliefs
        that successfully minimize prediction error at THAT position.

        CRITICAL FIX: Priors are now (N, K) - position-specific!
        Each position learns its own prior based on:
        1. Beliefs at that position across the batch
        2. Prediction error weighting (beliefs with lower error have more influence)

        Args:
            mu_q_batch: Beliefs after VFE convergence (B, N, K)
            sigma_q_batch: Belief variances (B, N, K)
            prediction_error: Per-sample CE loss (B,) - for batch-level weighting
            per_position_error: Per-position CE loss (B, N) - for position-level weighting
        """
        B, N, K = mu_q_batch.shape

        # Detach beliefs to prevent graph accumulation
        mu_q_batch = mu_q_batch.detach()
        sigma_q_batch = sigma_q_batch.detach()

        # =====================================================================
        # POSITION-SPECIFIC PRIOR UPDATE
        # =====================================================================
        # Update each position's prior based on beliefs at that position

        # Handle sequence length mismatch
        N_prior = min(N, self.prior_mu.shape[0])

        # =====================================================================
        # UNIFORM WEIGHTING for stable position-dependent prior learning
        # =====================================================================
        # Uniform averaging computes stable TARGET beliefs.
        # Prediction error scales LEARNING RATE (how much to update).
        #
        # This is the FEP principle: prediction error IS the learning signal.
        # High error = surprise = model wrong = learn more
        # Low error = expected = model good = learn less
        # =====================================================================
        mu_p_new = mu_q_batch[:, :N_prior, :].mean(dim=0)  # (N, K)
        sigma_p_new = sigma_q_batch[:, :N_prior, :].mean(dim=0)  # (N, K)

        # =====================================================================
        # ERROR-SCALED LEARNING RATE (per-position)
        # =====================================================================
        # FEP principle: prediction error DRIVES learning magnitude.
        # High error = more surprise = MORE total learning (not just redistribution)
        #
        # Two-level scaling:
        # 1. Mean error scales BASE learning rate (more error → more total learning)
        # 2. Per-position error redistributes within batch (optional refinement)
        # =====================================================================
        if per_position_error is not None and per_position_error.numel() > 0:
            # Mean error across all positions and batch
            mean_error = per_position_error[:, :N_prior].mean()

            # Scale base learning rate by error magnitude
            # Use sqrt to prevent extreme scaling while maintaining signal
            # Typical CE for random = ln(vocab) ≈ 10, good model ≈ 2-3
            # sqrt(10) ≈ 3.2, sqrt(2) ≈ 1.4, so ~2x difference
            error_magnitude_scale = torch.sqrt(mean_error.clamp(min=0.1))

            # Per-position redistribution (optional: relative to mean)
            pos_error = per_position_error[:, :N_prior].mean(dim=0)  # (N,)
            # Relative error: how much more/less than mean
            relative_error = pos_error / mean_error.clamp(min=0.1)  # (N,)
            # Soft scaling: sqrt for stability, clamp for bounds
            lr_scale = torch.sqrt(relative_error.clamp(min=0.25, max=4.0))  # Range [0.5, 2.0]

            # Combine: base_lr * error_magnitude * per_position_redistribution
            lr_scale = error_magnitude_scale * lr_scale
        else:
            lr_scale = torch.ones(N_prior, device=mu_q_batch.device)

        # =====================================================================
        # Apply update to persistent models (position-dependent generative models)
        # =====================================================================
        # Warmup: prevent aggressive early updates that memorize training data.
        # Early in training, model updates are near-zero; they ramp up over time.
        warmup_steps = 100
        warmup_factor = min(1.0, self.total_prior_updates / max(warmup_steps, 1))
        base_blend = self.config.prior_lr * warmup_factor

        if self.config.gradient_prior_updates:
            # GRADIENT-BASED PRIOR UPDATE
            # Full VFE gradient w.r.t. prior:
            #   ∂F/∂μ_p = α·Σ_p^{-1}·(μ_p - μ_q)           (self-coupling)
            #           + λ_γ·Σ_j [prior alignment]         (inter-position)
            #           + Σ_d decay^d · [hyperprior grad]   (Ouroboros Tower)

            # Use larger minimum (1e-4) to avoid explosion when σ² → 0
            sigma_squared = self.prior_sigma[:N_prior, :].clamp(min=1e-4) ** 2

            # Term 1: Self-coupling gradient ∂/∂μ_p [KL(q||p)]
            # This pulls priors toward beliefs
            grad_self = (self.prior_mu[:N_prior, :] - mu_p_new) / sigma_squared

            # Term 2: Prior alignment gradient ∂/∂μ_p [λ_γ · Σ_j KL(p_i || Ω_ij p_j)]
            # This makes priors consistent across positions
            if self.config.prior_coupling_enabled:
                grad_align = self._compute_prior_alignment_gradient(
                    self.prior_mu[:N_prior, :],
                    self.prior_sigma[:N_prior, :],
                )
            else:
                grad_align = torch.zeros_like(grad_self)

            # Term 3: Ouroboros Tower gradient ∂/∂μ_p [Σ_d decay^d · KL(p || h^d)]
            # This pulls priors toward hyperpriors from ALL ancestors
            if self.config.enable_ouroboros_tower:
                grad_tower = self._compute_hyperprior_gradient(
                    self.prior_mu[:N_prior, :],
                    self.prior_sigma[:N_prior, :],
                )
            else:
                grad_tower = torch.zeros_like(grad_self)

            # Term 4: Hyper-prior gradient ∂/∂s [λ_h · KL(s || h)]
            # h = centroid of all position models. Pulls each model toward
            # the shared mean, preventing memorization of per-position statistics.
            lambda_h = getattr(self.config, 'lambda_hyperprior', 0.1)
            h_mu = self.prior_mu[:N_prior, :].mean(dim=0, keepdim=True)  # (1, K) centroid
            h_sigma = self.prior_sigma[:N_prior, :].mean(dim=0, keepdim=True).clamp(min=1e-4)
            # ∂KL(s||h)/∂s = (s - h) / σ_h, with broadened σ_h
            grad_hyperprior = lambda_h * (self.prior_mu[:N_prior, :] - h_mu) / (h_sigma * 2.0)

            # Total gradient
            grad_mu_p = self.config.alpha * grad_self + grad_align + grad_tower + grad_hyperprior

            # Clip gradient to prevent explosion
            grad_norm = grad_mu_p.norm()
            if grad_norm > self.config.grad_clip:
                grad_mu_p = grad_mu_p * (self.config.grad_clip / grad_norm)

            # Gradient descent on prior
            self.prior_mu[:N_prior, :].sub_(self.config.prior_grad_lr * grad_mu_p)

            # EMA for sigma (use base blend, not error-scaled)
            self.prior_sigma[:N_prior, :].lerp_(sigma_p_new, base_blend)
        else:
            # EXPONENTIAL MOVING AVERAGE with per-position error-scaled learning rate
            # blend_per_pos: (N,) -> (N, 1) for broadcasting with (N, K)
            blend_per_pos = (base_blend * lr_scale).unsqueeze(-1)  # (N, 1)
            # Manual lerp: prior = (1 - blend) * prior + blend * target
            self.prior_mu[:N_prior, :] = (
                (1 - blend_per_pos) * self.prior_mu[:N_prior, :] +
                blend_per_pos * mu_p_new
            )
            # Sigma uses base blend (uncertainty shouldn't scale with error)
            self.prior_sigma[:N_prior, :].lerp_(sigma_p_new, base_blend)

            # Hyper-prior regularization: pull models toward centroid
            # This is the EMA-path equivalent of the hyper-prior gradient.
            # Prevents position models from drifting to memorize training data.
            lambda_h = getattr(self.config, 'lambda_hyperprior', 0.1)
            h_mu = self.prior_mu[:N_prior, :].mean(dim=0, keepdim=True)  # (1, K)
            reg_strength = base_blend * lambda_h
            self.prior_mu[:N_prior, :] -= reg_strength * (self.prior_mu[:N_prior, :] - h_mu)

        # Track update statistics
        self.prior_update_count[:N_prior] += 1

        # Ensure sigma stays positive
        self.prior_sigma.clamp_(min=self.config.eps)

        self.total_prior_updates += 1

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        parent_beliefs: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        n_vfe_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through layer: initialize beliefs and run VFE steps.

        Args:
            x: (B, N, K) input embeddings
            targets: (B, N) target tokens
            mask: (B, N, N) causal mask
            parent_beliefs: Optional (mu_q, sigma_q, phi) from parent layer
            n_vfe_steps: Number of VFE gradient descent steps

        Returns:
            mu_q: Final belief means (B, N, K)
            info: Dict with metrics and intermediate states
        """
        # Try to compile VFE step on first forward (if enabled)
        self.maybe_compile()

        B, N, K = x.shape
        device = x.device

        # KEEP gradient connection to embeddings!
        # VFE dynamics should influence embedding learning through the belief updates.
        # Previously this was detached, breaking the learning signal.

        # Initialize beliefs from input
        mu_q, sigma_q, mu_p, sigma_p, phi = self.init_beliefs(x, device)

        # Update priors from parent if available
        if parent_beliefs is not None:
            parent_mu_q, parent_sigma_q, parent_phi = parent_beliefs
            mu_p, sigma_p = self.update_prior_from_parent(
                mu_q, sigma_q, mu_p, sigma_p, phi,
                parent_mu_q, parent_sigma_q, parent_phi,
                self.generators,
            )

        # ==================================================================
        # PERFORMANCE: Cache transport operators if gauge frames don't evolve
        # ==================================================================
        # This is the BIGGEST optimization: avoid recomputing matrix_exp
        # every VFE step when φ is constant.
        if self.config.cache_transport and not self.config.gauge_evolution_enabled:
            cached_transport = compute_transport_operators(phi, self.generators)
        else:
            cached_transport = None

        # Run VFE gradient descent steps
        # CRITICAL: Only use create_graph=True on FINAL step to avoid
        # "backward through graph twice" errors from nested autograd.grad calls
        all_metrics = []
        for step in range(n_vfe_steps):
            is_final = (step == n_vfe_steps - 1)
            mu_q, sigma_q, phi, metrics = self.vfe_step(
                mu_q, sigma_q, mu_p, sigma_p, phi,
                targets=targets, mask=mask,
                is_final_step=is_final,
                cached_transport=cached_transport,  # Pass cached transport!
                step_idx=step,                      # For momentum reset
            )
            all_metrics.append(metrics)

            # If gauge evolution is enabled, recompute transport after φ update
            if self.config.gauge_evolution_enabled and cached_transport is not None:
                cached_transport = compute_transport_operators(phi, self.generators)

        # Aggregate metrics
        final_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                        for k in all_metrics[0].keys()}

        # ==================================================================
        # UPDATE PERSISTENT φ via dφ/dt = -∂F/∂φ
        # ==================================================================
        # CRITICAL: φ is a learned parameter that evolves via VFE gradients!
        # After VFE steps, update the persistent self.phi (per-position)
        if self.config.gauge_evolution_enabled:
            # Compute gauge frame gradients for evolution: dφ/dt = -∂F/∂φ
            # In pure FEP mode, phi.grad is NOT populated by autograd (analytical
            # gradients are used for beliefs only). We must compute ∂F/∂φ explicitly
            # via a small autograd computation on the alignment term.
            if phi.grad is not None:
                grad_phi_batch = phi.grad.mean(dim=0)  # (N, phi_dim)
            else:
                # Explicit gradient computation for phi
                # The alignment term β_ij·KL(q_i||Ω_ij·q_j) depends on φ through
                # the transport operators Ω_ij = exp(φ_i·G)·exp(-φ_j·G)
                with torch.enable_grad():
                    phi_for_grad = phi.detach().requires_grad_(True)
                    beta_g, kl_g = compute_attention_weights(
                        mu_q.detach(), sigma_q.detach(), phi_for_grad, self.generators,
                        kappa=self.config.kappa, mask=mask, return_kl=True,
                        diagonal_covariance=True,
                    )
                    alignment_for_phi = (beta_g * kl_g).sum(dim=-1).mean()
                    grad_phi_batch = torch.autograd.grad(
                        self.config.lambda_belief * alignment_for_phi,
                        phi_for_grad,
                    )[0].mean(dim=0)  # Average across batch -> (N, phi_dim)

            with torch.no_grad():
                N_phi = min(N, self.phi.shape[0])
                self.phi.data[:N_phi] -= self.config.gauge_lr * grad_phi_batch[:N_phi]

                # Normalize to prevent explosion
                if hasattr(self.config, 'phi_max_norm') and self.config.phi_max_norm > 0:
                    phi_norm = self.phi.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    scale = torch.clamp(self.config.phi_max_norm / phi_norm, max=1.0)
                    self.phi.data = self.phi.data * scale

        info = {
            'metrics': final_metrics,
            'beliefs': (mu_q, sigma_q),
            'priors': (mu_p, sigma_p),
            'phi': phi,
        }

        return mu_q, info


class PureFEPTransformer(nn.Module):
    """
    Complete Pure FEP Transformer for language modeling.

    Architecture (with toggleable components):
    =========================================

    EMBEDDING MODE (config.embedding_mode):
    - 'learned':     Standard nn.Embedding → μ (ad hoc)
    - 'prior_bank':  PriorBank → (μ, Σ) from token priors (principled!)
    - 'hybrid':      nn.Embedding with PriorBank output

    POSITION MODE (config.position_mode):
    - 'sinusoidal_mu':  Add sinusoidal to μ (standard Transformer)
    - 'gauge_frame':    Encode in φ ∈ so(3) (principled - affects transport!)
    - 'both':           Both for comparison

    OUTPUT MODE (config.output_mode):
    - 'linear':       W·μ → logits (ad hoc)
    - 'kl_to_prior':  -KL(q||π_v)/τ → logits (principled FEP!)
    - 'both':         Blend for comparison

    The PUREST FEP configuration:
    - embedding_mode='prior_bank'
    - position_mode='gauge_frame'
    - output_mode='kl_to_prior'

    Training:
    - In pure_fep_mode: NO backprop, all learning via prior evolution
    - Otherwise: Hybrid with backprop on embeddings/output
    """

    def __init__(self, config: PureFEPConfig):
        super().__init__()
        self.config = config

        # =====================================================================
        # GENERATORS: Create before PriorBank (needed for gauge-fixed priors)
        # =====================================================================
        if config.gauge_group == 'SO3':
            if config.use_multi_irrep:
                gen_np = generate_multi_irrep_generators(config.irrep_spec, validate=True)
            else:
                gen_np = generate_so3_generators(config.embed_dim)
        elif config.gauge_group == 'SON':
            N = config.gauge_dim
            gen_np = generate_multi_irrep_soN_generators(config.irrep_spec, N, validate=True)
        elif config.gauge_group == 'GLK':
            # GL(K) mode: K² generators spanning full gl(K)
            K = config.embed_dim
            gen_np = generate_glK_generators(K)
        else:
            raise ValueError(f"Unknown gauge_group: {config.gauge_group}")
        self.register_buffer('generators', torch.from_numpy(gen_np).float())

        # =====================================================================
        # EMBEDDING: Token → Belief Initialization
        # =====================================================================
        if config.embedding_mode == 'prior_bank':
            # PRINCIPLED: PriorBank serves as both embedding AND output
            # NOTE: gauge_fixed_priors=False for now because P-flow doesn't
            # update phi_embed yet. Each token gets its own prior_mu that
            # can be updated via scatter_add in P-flow.
            self.prior_bank = PriorBank(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                init_std=None,  # Use default 1/sqrt(embed_dim) for O(1) KL
                init_sigma_scale=1.0,  # Scaled to match init_std for O(1) KL
                learnable_sigma=True,
                eps=config.eps,
                gauge_fixed_priors=False,  # Per-token priors (P-flow updates each)
                generators=self.generators,
                phi_dim=config.phi_dim,
            )
            self.embedding = None  # Not used
        elif config.embedding_mode == 'hybrid':
            # HYBRID: Learned embedding + PriorBank output
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.prior_bank = PriorBank(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                init_std=None,  # Use default 1/sqrt(embed_dim) for O(1) KL
                init_sigma_scale=1.0,  # Scaled to match init_std for O(1) KL
                learnable_sigma=True,
                eps=config.eps,
                gauge_fixed_priors=False,  # Per-token priors (P-flow updates each)
                generators=self.generators,
                phi_dim=config.phi_dim,
            )
        else:  # 'learned'
            # AD HOC: Standard nn.Embedding
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
            self.prior_bank = None

        # =====================================================================
        # POSITION ENCODING: Where to encode position
        # =====================================================================
        if config.position_mode in ['gauge_frame', 'both']:
            # PRINCIPLED: Position in gauge frame φ
            self.gauge_position = GaugePositionEncoder(
                max_seq_len=config.seq_length,
                mode=config.position_encoding,
                scale=config.position_scale,
                composition=config.position_composition,
                phi_dim=config.phi_dim,  # 3 for SO(3), N(N-1)/2 for SO(N)
            )
        else:
            self.gauge_position = None

        if config.position_mode in ['sinusoidal_mu', 'both']:
            # AD HOC: Sinusoidal added to μ
            self.register_buffer(
                'pos_encoding_mu',
                self._create_pos_encoding(config.seq_length, config.embed_dim)
            )
        else:
            self.pos_encoding_mu = None

        # =====================================================================
        # OUTPUT: Belief → Logits
        # =====================================================================
        if config.output_mode == 'linear':
            # AD HOC: Linear projection
            self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        elif config.output_mode == 'kl_to_prior':
            # PRINCIPLED: KL to prior bank (uses self.prior_bank)
            self.output_proj = None
            # Ensure prior_bank exists for KL-based output
            if self.prior_bank is None:
                self.prior_bank = PriorBank(
                    vocab_size=config.vocab_size,
                    embed_dim=config.embed_dim,
                    init_std=None,  # Use default 1/sqrt(embed_dim) for O(1) KL
                    init_sigma_scale=1.0,  # Scaled to match init_std for O(1) KL
                    learnable_sigma=True,
                    eps=config.eps,
                    gauge_fixed_priors=False,  # Per-token priors (P-flow updates each)
                    generators=self.generators,
                    phi_dim=config.phi_dim,
                )
        else:  # 'both'
            # Blend of linear and KL
            self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
            if self.prior_bank is None:
                self.prior_bank = PriorBank(
                    vocab_size=config.vocab_size,
                    embed_dim=config.embed_dim,
                    gauge_fixed_priors=False,  # Per-token priors (P-flow updates each)
                    generators=self.generators,
                    phi_dim=config.phi_dim,
                )

        # =====================================================================
        # HIERARCHICAL LAYERS
        # =====================================================================
        # Pass prior_bank reference so layers can use KL-to-prior output
        self.layers = nn.ModuleList([
            PureFEPLayer(
                config.embed_dim,
                scale=i,
                config=config,
                prior_bank=self.prior_bank,  # Share prior_bank for KL output
            )
            for i in range(config.num_layers)
        ])

        # Backward compat: also store pos_encoding for layers that might use it
        self.register_buffer(
            'pos_encoding',
            self._create_pos_encoding(config.seq_length, config.embed_dim)
        )

        # Training state
        self.step_count = 0

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal position encoding (handles odd dimensions)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Handle odd d_model: sin gets ceil(d/2), cos gets floor(d/2)
        n_sin = (d_model + 1) // 2  # ceil
        n_cos = d_model // 2        # floor

        div_term_sin = torch.exp(torch.arange(0, n_sin).float() * (-math.log(10000.0) / d_model) * 2)
        div_term_cos = torch.exp(torch.arange(0, n_cos).float() * (-math.log(10000.0) / d_model) * 2)

        pe[:, 0::2] = torch.sin(position * div_term_sin)
        pe[:, 1::2] = torch.cos(position * div_term_cos)

        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, N) token IDs
        targets: Optional[torch.Tensor] = None,  # (B, N) target token IDs
        n_vfe_steps: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with pure VFE learning.

        Uses toggleable components based on config:
        - embedding_mode: 'learned', 'prior_bank', or 'hybrid'
        - position_mode: 'sinusoidal_mu', 'gauge_frame', or 'both'
        - output_mode: 'linear', 'kl_to_prior', or 'both'

        Args:
            input_ids: (B, N) input token IDs
            targets: (B, N) target token IDs (shifted by 1 for LM)
            n_vfe_steps: VFE steps per layer

        Returns:
            logits: (B, N, vocab_size) output logits
            info: Dict with all metrics and states
        """
        B, N = input_ids.shape
        device = input_ids.device
        K = self.config.embed_dim

        # =====================================================================
        # EMBEDDING: Token → Initial Belief
        # =====================================================================
        # phi_dim = 3 for SO(3), N(N-1)/2 for SO(N)
        phi_dim = self.config.phi_dim

        if self.config.embedding_mode == 'prior_bank':
            # PRINCIPLED: Initialize beliefs from token priors
            # Returns (mu, sigma, phi) - token phi from gauge-fixed priors
            x, sigma_init, phi_token = self.prior_bank.encode(input_ids)
        elif self.config.embedding_mode == 'hybrid':
            # HYBRID: Learned embedding for μ, but sigma from prior_bank
            x = self.embedding(input_ids)  # (B, N, K)
            # Use prior_bank sigma if available (principled), else use config
            if self.prior_bank is not None:
                _, sigma_init, phi_token = self.prior_bank.encode(input_ids)
            else:
                sigma_init = torch.ones(B, N, K, device=device)
                phi_token = torch.zeros(B, N, phi_dim, device=device)
        else:  # 'learned'
            # AD HOC: Standard embedding (but at least use prior_bank sigma if available)
            x = self.embedding(input_ids)  # (B, N, K)
            if self.prior_bank is not None:
                _, sigma_init, phi_token = self.prior_bank.encode(input_ids)
            else:
                sigma_init = torch.ones(B, N, K, device=device)
                phi_token = torch.zeros(B, N, phi_dim, device=device)

        # =====================================================================
        # POSITION ENCODING: Add to μ and/or φ
        # =====================================================================
        # Start with token phi (from gauge-fixed priors or zeros)
        phi = phi_token

        if self.config.position_mode in ['sinusoidal_mu', 'both']:
            # AD HOC: Add sinusoidal to μ
            x = x + self.pos_encoding_mu[:, :N, :]

        if self.config.position_mode in ['gauge_frame', 'both']:
            # PRINCIPLED: Encode position in gauge frame φ
            phi = self.gauge_position(phi, N)

        # =====================================================================
        # CAUSAL MASK
        # =====================================================================
        mask = torch.tril(torch.ones(N, N, device=device)).unsqueeze(0).expand(B, -1, -1)

        # =====================================================================
        # PROCESS THROUGH HIERARCHICAL LAYERS
        # =====================================================================
        layer_infos = []
        parent_beliefs = None

        # BOTTOM-UP: process layers 0 → L-1
        for layer_idx, layer in enumerate(self.layers):
            x, info = layer(
                x, targets=targets, mask=mask,
                parent_beliefs=parent_beliefs,
                n_vfe_steps=n_vfe_steps,
            )
            layer_infos.append(info)

            # Update phi with layer's gauge frames
            phi = info['phi']

            # Current layer's beliefs become parent for next layer
            parent_beliefs = (info['beliefs'][0], info['beliefs'][1], phi)

        # =====================================================================
        # OUTPUT: Belief → Logits
        # =====================================================================
        # Get final beliefs (μ_q, Σ_q) from last layer
        mu_final = x
        sigma_final = layer_infos[-1]['beliefs'][1] if layer_infos else sigma_init

        if self.config.output_mode == 'linear':
            # AD HOC: Linear projection
            logits = self.output_proj(mu_final)  # (B, N, vocab_size)

        elif self.config.output_mode == 'kl_to_prior':
            # PRINCIPLED: KL to token priors
            # p(y=v|q) ∝ exp(-KL(q||π_v)/τ)
            logits = self.prior_bank.decode(
                mu_final, sigma_final,
                tau=self.config.output_tau
            )  # (B, N, vocab_size)

        else:  # 'both'
            # Blend: average of linear and KL-based logits
            logits_linear = self.output_proj(mu_final)
            logits_kl = self.prior_bank.decode(
                mu_final, sigma_final,
                tau=self.config.output_tau
            )
            logits = 0.5 * (logits_linear + logits_kl)

        # =====================================================================
        # BOOKKEEPING
        # =====================================================================
        self.step_count += 1

        # Aggregate info
        all_metrics = {}
        for i, info in enumerate(layer_infos):
            for k, v in info['metrics'].items():
                all_metrics[f'layer_{i}/{k}'] = v

        # Store sigma_final for observation gradient in VFE
        info = {
            'metrics': all_metrics,
            'layer_infos': layer_infos,
            'sigma_final': sigma_final,
        }

        return logits, info

    def _hierarchical_prior_update(
        self,
        layer_infos: List[Dict],
        prediction_errors: Optional[torch.Tensor] = None,
        per_position_errors: Optional[torch.Tensor] = None,
    ):
        """
        Top-down prior update: propagate beliefs down the hierarchy.

        This is where LEARNING happens in pure FEP:
        - Top layer forms beliefs about the world
        - These flow down as priors to lower layers
        - Lower layers must now explain observations under these priors
        - This shapes what the network "knows"

        CRITICAL: Updates are persisted to layer.prior_mu/prior_sigma buffers!

        When Ouroboros Tower is enabled, also collects hyperpriors from ALL
        ancestors (grandparent, great-grandparent, etc.) for non-Markovian memory.

        Args:
            layer_infos: List of layer outputs from forward pass
            prediction_errors: (B,) per-sample CE loss for weighted prior updates
            per_position_errors: (B, N) per-position CE for fine-grained learning
        """
        # =====================================================================
        # OUROBOROS TOWER: Collect hyperpriors from ALL ancestors
        # =====================================================================
        if self.config.enable_ouroboros_tower:
            self._update_ouroboros_tower(layer_infos)

        # Process top-down: layer L-1 → layer 0
        for i in range(len(self.layers) - 1, 0, -1):
            parent_info = layer_infos[i]
            child_layer = self.layers[i - 1]
            child_info = layer_infos[i - 1]

            # Get parent beliefs and child priors
            parent_mu_q, parent_sigma_q = parent_info['beliefs']
            parent_phi = parent_info['phi']

            child_mu_q, child_sigma_q = child_info['beliefs']
            child_mu_p, child_sigma_p = child_info['priors']
            child_phi = child_info['phi']

            # Update child priors from parent beliefs
            new_mu_p, new_sigma_p = child_layer.update_prior_from_parent(
                child_mu_q, child_sigma_q,
                child_mu_p, child_sigma_p,
                child_phi,
                parent_mu_q, parent_sigma_q,
                parent_phi,
                self.layers[i].generators,
            )

            # PERSIST the learning! Update the layer's stored priors.
            # Use prediction-error-weighted beliefs for the update
            child_layer.update_persistent_prior(
                child_mu_q, child_sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )

            # Store updated priors in info dict too (for current pass)
            child_info['priors'] = (new_mu_p, new_sigma_p)

        # Also update top layer's prior from its own beliefs (self-supervision)
        if len(self.layers) > 0:
            top_layer = self.layers[-1]
            top_info = layer_infos[-1]
            top_mu_q, top_sigma_q = top_info['beliefs']
            # Top layer learns from its own beliefs (no parent)
            top_layer.update_persistent_prior(
                top_mu_q, top_sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )

    def _update_ouroboros_tower(self, layer_infos: List[Dict]):
        """
        Ouroboros Tower: Collect hyperpriors from ALL ancestors for each layer.

        For layer at index i, we collect beliefs from layers i+2, i+3, ... as hyperpriors.
        - Layer 0: hyperpriors from layers 2, 3, 4, ... (grandparent, great-grandparent, ...)
        - Layer 1: hyperpriors from layers 3, 4, 5, ...
        - etc.

        This creates non-Markovian memory where information from the top of the
        hierarchy can directly influence the bottom, bypassing intermediate layers.

        Args:
            layer_infos: List of layer outputs from forward pass
        """
        num_layers = len(self.layers)

        for i in range(num_layers):
            child_layer = self.layers[i]
            child_phi = layer_infos[i]['phi']

            # Collect beliefs from ancestors beyond the immediate parent
            # Grandparent is at i+2, great-grandparent at i+3, etc.
            ancestor_beliefs = []
            ancestor_generators = []

            for ancestor_idx in range(i + 2, num_layers):
                ancestor_info = layer_infos[ancestor_idx]
                ancestor_mu_q, ancestor_sigma_q = ancestor_info['beliefs']
                ancestor_phi = ancestor_info['phi']

                ancestor_beliefs.append((ancestor_mu_q, ancestor_sigma_q, ancestor_phi))
                ancestor_generators.append(self.layers[ancestor_idx].generators)

            # Update hyperpriors for this layer
            child_layer.update_hyperpriors_from_ancestors(
                ancestor_beliefs,
                ancestor_generators,
                child_phi,
            )

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        n_vfe_steps: int = 5,
    ) -> Dict[str, float]:
        """
        Single training step using pure VFE learning.

        PURE FEP MODE (config.pure_fep_mode=True):
        - NO backprop on embeddings or output projections
        - ALL learning happens via prior evolution under VFE pressure
        - Perception: VFE gradient descent on beliefs
        - Learning: Position-dependent prior updates weighted by prediction error

        HYBRID MODE (config.pure_fep_mode=False):
        - Backprop updates embeddings and projections
        - Prior updates provide additional learning signal

        Args:
            input_ids: (B, N) input tokens
            targets: (B, N) target tokens
            n_vfe_steps: VFE steps per layer

        Returns:
            Dict of training metrics
        """
        self.train()
        B, N = input_ids.shape

        if self.config.pure_fep_mode:
            # ==================================================================
            # PURE FEP MODE: No backprop, all learning via prior evolution
            # ==================================================================
            with torch.no_grad():
                # Forward pass with VFE
                logits, info = self(input_ids, targets=targets, n_vfe_steps=n_vfe_steps)

                # Compute overall CE loss for metrics
                ce_loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    targets.view(-1),
                )

                # Compute PER-POSITION prediction error for prior learning
                # This is CRITICAL: each position learns from its own errors
                logits_reshaped = logits.view(B, N, -1)
                targets_reshaped = targets.view(B, N)
                per_position_loss = F.cross_entropy(
                    logits_reshaped.permute(0, 2, 1),  # (B, vocab, N)
                    targets_reshaped,                   # (B, N)
                    reduction='none'
                )  # (B, N) - loss at each position

                # Per-sample loss (for batch-level weighting)
                per_sample_loss = per_position_loss.mean(dim=1)  # (B,)

                # Store both for prior updates
                info['prediction_errors'] = per_sample_loss
                info['per_position_errors'] = per_position_loss
                info['targets'] = targets  # Need targets for token prior updates

            # Update persistent priors - THIS IS WHERE LEARNING HAPPENS!
            self._update_priors_with_prediction_error(info)

        else:
            # ==================================================================
            # HYBRID MODE: Backprop + prior evolution
            # ==================================================================
            # Zero gradients
            self.zero_grad()

            # Forward pass with VFE - gradients flow through
            logits, info = self(input_ids, targets=targets, n_vfe_steps=n_vfe_steps)

            # CE loss
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

            # Compute per-position errors
            with torch.no_grad():
                logits_reshaped = logits.view(B, N, -1)
                targets_reshaped = targets.view(B, N)
                per_position_loss = F.cross_entropy(
                    logits_reshaped.permute(0, 2, 1),
                    targets_reshaped,
                    reduction='none'
                )
                per_sample_loss = per_position_loss.mean(dim=1)

            info['prediction_errors'] = per_sample_loss
            info['per_position_errors'] = per_position_loss

            # Backprop
            ce_loss.backward()

            # Apply gradients (manual SGD)
            with torch.no_grad():
                # Collect parameters that exist and have gradients
                params_to_update = []
                if self.embedding is not None:
                    params_to_update.append(self.embedding.weight)
                if self.output_proj is not None:
                    params_to_update.append(self.output_proj.weight)
                if self.prior_bank is not None:
                    params_to_update.append(self.prior_bank.prior_mu)
                    if self.prior_bank.learnable_sigma:
                        params_to_update.append(self.prior_bank.log_prior_sigma)

                for param in params_to_update:
                    if param.grad is not None:
                        grad_norm = param.grad.norm()
                        if grad_norm > self.config.grad_clip:
                            param.grad.mul_(self.config.grad_clip / grad_norm)
                        param.sub_(self.config.mu_lr * param.grad)
                        param.grad.zero_()

                for layer in self.layers:
                    if layer.output_proj is not None and layer.output_proj.weight.grad is not None:
                        grad_norm = layer.output_proj.weight.grad.norm()
                        if grad_norm > self.config.grad_clip:
                            layer.output_proj.weight.grad.mul_(self.config.grad_clip / grad_norm)
                        layer.output_proj.weight.sub_(self.config.mu_lr * layer.output_proj.weight.grad)
                        layer.output_proj.weight.grad.zero_()

            # Update priors
            self._update_priors_with_prediction_error(info)

        # =====================================================================
        # DYNAMIC LAYER EMERGENCE: Check if spawning/merging is needed
        # =====================================================================
        if self.config.dynamic_layers_enabled:
            layer_infos = info.get('layer_infos', [])
            self._apply_dynamic_emergence(layer_infos, ce_loss.item())

        # Increment step counter
        self.step_count += 1

        # Perplexity
        ppl = torch.exp(ce_loss.detach()).item()

        metrics = {
            'ce_loss': ce_loss.item(),
            'perplexity': ppl,
            'num_layers': len(self.layers),  # Track dynamic layer count
            **info['metrics'],
        }

        return metrics

    def _update_priors_with_prediction_error(self, info: Dict):
        """
        Update priors using prediction-error-weighted beliefs.

        This is the LEARNING step in pure FEP:
        - Each position's prior evolves based on beliefs at that position
        - Beliefs with lower prediction error have more influence
        - Per-position errors allow fine-grained learning
        - TOKEN PRIORS (prior_bank) are updated based on which tokens were predicted

        Called from train_step() where prediction errors are available.
        """
        # Only update priors at specified intervals
        if self.step_count % self.config.prior_update_interval != 0:
            return

        prediction_errors = info.get('prediction_errors')  # (B,)
        per_position_errors = info.get('per_position_errors')  # (B, N)
        layer_infos = info.get('layer_infos', [])
        targets = info.get('targets')  # (B, N) - needed for token prior updates

        # Update each layer's persistent prior with position-specific errors
        for i, layer_info in enumerate(layer_infos):
            mu_q, sigma_q = layer_info['beliefs']
            self.layers[i].update_persistent_prior(
                mu_q, sigma_q,
                prediction_error=prediction_errors,
                per_position_error=per_position_errors,
            )

        # =====================================================================
        # CRITICAL FIX: Update TOKEN PRIORS in prior_bank
        # This is what allows pure FEP to actually learn predictions!
        # =====================================================================
        if (self.config.pure_fep_mode and
            self.prior_bank is not None and
            targets is not None and
            layer_infos):

            # Get final layer beliefs (these determine output)
            final_mu_q, final_sigma_q = layer_infos[-1]['beliefs']  # (B, N, K)
            B, N, K = final_mu_q.shape

            # Compute per-position weights for prior updates
            # =================================================================
            # UNIFORM WEIGHTING + ERROR-SCALED LEARNING RATE
            # =================================================================
            # Uniform weighting computes stable TARGET beliefs.
            # Prediction error scales the LEARNING RATE per token.
            #
            # This is the FEP principle: prediction error IS the learning signal.
            # High error = surprise = model wrong = learn more
            # Low error = expected = model good = learn less
            # =================================================================
            weights = torch.ones(B, N, device=final_mu_q.device) / (B * N)

            # Check if using gauge-fixed priors or standard per-token priors
            if self.prior_bank.gauge_fixed_priors:
                # ===============================================================
                # GAUGE-FIXED MODE: Update phi_embed (gauge frames) per token
                # The base_prior_mu is shared; individual tokens differ by rotation
                # ===============================================================
                # For now, update base_prior_mu with global average (all tokens share it)
                # This is a simplification - could also update phi_embed per token
                global_avg_belief = (weights.view(-1, 1) * final_mu_q.view(-1, K)).sum(dim=0)
                blend_rate = self.config.prior_lr
                self.prior_bank.base_prior_mu.data.lerp_(global_avg_belief, blend_rate)

                if self.prior_bank.learnable_sigma:
                    global_avg_sigma = (weights.view(-1, 1) * final_sigma_q.view(-1, K)).sum(dim=0)
                    current_sigma = torch.exp(self.prior_bank.base_log_prior_sigma.data)
                    new_sigma = (1 - blend_rate) * current_sigma + blend_rate * global_avg_sigma
                    self.prior_bank.base_log_prior_sigma.data = torch.log(new_sigma.clamp(min=1e-6))
            else:
                # ===============================================================
                # PURE FEP: Token priors updated via VFE gradient descent
                # ===============================================================
                # d(π_v)/dt = -∂F/∂π_v
                # where F includes observation term: E_q[-log p(y|z, π)]
                #
                # This replaces ALL hand-crafted updates (EMA, discriminative, etc.)
                # Contrastive behavior emerges naturally from VFE minimization!
                # ===============================================================
                if self.config.gradient_prior_updates:
                    # Gradient-based update: ∂F/∂π_v from observation term
                    with torch.enable_grad():
                        # Detach beliefs (already converged via VFE)
                        # But keep token priors differentiable
                        mu_q_detached = final_mu_q.detach()
                        sigma_q_detached = final_sigma_q.detach()

                        # Ensure token priors require gradients
                        if not self.prior_bank.prior_mu.requires_grad:
                            self.prior_bank.prior_mu.requires_grad_(True)

                        # Compute observation loss (differentiable w.r.t. token priors)
                        logits = self.prior_bank.decode(
                            mu_q_detached,
                            sigma_q_detached,
                            tau=getattr(self.config, 'output_tau', 1.0)
                        )

                        ce_loss = F.cross_entropy(
                            logits.view(-1, self.config.vocab_size),
                            targets.view(-1),
                            reduction='sum'
                        ) / (B * N)

                        # Backprop to token priors
                        grad_token_mu = torch.autograd.grad(
                            ce_loss,
                            self.prior_bank.prior_mu,
                            retain_graph=False,
                            create_graph=False
                        )[0]

                        # Clip gradient for stability
                        grad_norm = grad_token_mu.norm()
                        if grad_norm > self.config.grad_clip:
                            grad_token_mu = grad_token_mu * (self.config.grad_clip / grad_norm)

                    # Gradient descent: π ← π - lr·∂F/∂π
                    with torch.no_grad():
                        self.prior_bank.prior_mu.sub_(
                            self.config.prior_grad_lr * grad_token_mu
                        )

                else:
                    # Fallback: Simple EMA (if gradient updates disabled)
                    # This is NOT pure FEP - just for backward compatibility
                    targets_flat = targets.view(-1)
                    mu_flat = final_mu_q.view(-1, K)
                    weights_flat = weights.view(-1, 1)

                    weighted_beliefs = weights_flat * mu_flat
                    token_counts = torch.zeros(self.config.vocab_size, device=final_mu_q.device)
                    token_updates = torch.zeros_like(self.prior_bank.prior_mu.data)

                    token_counts.scatter_add_(0, targets_flat, weights_flat.squeeze(-1))
                    token_updates.scatter_add_(0, targets_flat.unsqueeze(-1).expand(-1, K), weighted_beliefs)

                    mask = token_counts > 0
                    if mask.any():
                        token_counts_safe = token_counts.clamp(min=1e-8).unsqueeze(-1)
                        avg_beliefs = token_updates / token_counts_safe

                        # Simple EMA update
                        base_blend = self.config.prior_lr
                        self.prior_bank.prior_mu.data[mask] = (
                            (1 - base_blend) * self.prior_bank.prior_mu.data[mask] +
                            base_blend * avg_beliefs[mask]
                        )

                # =============================================================
                # DIVERSITY REGULARIZATION: prevent token prior collapse
                # =============================================================
                # Without this, token priors converge to the same mean embedding,
                # losing discriminative power. Push priors of different tokens
                # apart by repelling each prior from the batch centroid.
                # This is the token-level analogue of the position hyper-prior:
                # it ensures token models SPREAD OUT in embedding space.
                with torch.no_grad():
                    unique_tokens = torch.unique(targets)
                    if len(unique_tokens) > 1:
                        batch_priors = self.prior_bank.prior_mu.data[unique_tokens]  # (n, K)
                        centroid = batch_priors.mean(dim=0)  # (K,)
                        # Repulsion: push each token prior away from centroid
                        repulsion = batch_priors - centroid  # (n, K)
                        # Scale repulsion by prior_lr (small, stabilizing force)
                        repulsion_lr = self.config.prior_lr * 0.1
                        self.prior_bank.prior_mu.data[unique_tokens] += repulsion_lr * repulsion

    # =========================================================================
    # DYNAMIC LAYER EMERGENCE: Spawn/merge layers based on VFE pressure
    # =========================================================================

    def _check_emergence_conditions(
        self,
        layer_infos: List[Dict],
        ce_loss: float,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Check if conditions warrant spawning or merging layers.

        Emergence Logic:
        ----------------
        1. SPAWN: When VFE gradients are high AND prediction error is high
           - Signals that current hierarchy can't capture the structure
           - New layer inserted to add representational capacity

        2. MERGE: When adjacent layers have very similar beliefs
           - Signals redundant representation
           - Layers combined to reduce computational cost

        Args:
            layer_infos: List of layer outputs from forward pass
            ce_loss: Cross-entropy loss for this step

        Returns:
            should_act: Whether to spawn or merge
            layer_idx: Which layer to act on (spawn after / merge with next)
            action: 'spawn', 'merge', or 'none'
        """
        if not self.config.dynamic_layers_enabled:
            return False, None, 'none'

        num_layers = len(self.layers)

        # Check layer cap before spawning
        if num_layers >= self.config.max_layers:
            return False, None, 'none'

        # Compute VFE gradient norms for each layer
        vfe_grad_norms = []
        for layer_info in layer_infos:
            metrics = layer_info.get('metrics', {})
            grad_norm = metrics.get('grad_norm_mu', 0.0) + metrics.get('grad_norm_sigma', 0.0)
            vfe_grad_norms.append(grad_norm)

        # Compute belief similarity between adjacent layers
        belief_similarities = []
        for i in range(num_layers - 1):
            mu_i, sigma_i = layer_infos[i]['beliefs']
            mu_j, sigma_j = layer_infos[i + 1]['beliefs']

            # Simplified similarity: cosine similarity of means
            similarity = F.cosine_similarity(
                mu_i.view(-1), mu_j.view(-1), dim=0
            ).item()
            belief_similarities.append(similarity)

        # =====================================================================
        # SPAWN CONDITION: High VFE gradient + high loss
        # =====================================================================
        # Find layer with highest VFE gradient
        if vfe_grad_norms:
            max_grad_idx = int(np.argmax(vfe_grad_norms))
            max_grad = vfe_grad_norms[max_grad_idx]

            # Spawn if gradient exceeds threshold and we have capacity
            if max_grad > self.config.layer_spawn_threshold and num_layers < self.config.max_layers:
                # Insert new layer after the one with highest gradient
                return True, max_grad_idx, 'spawn'

        # =====================================================================
        # MERGE CONDITION: Very high belief similarity between adjacent layers
        # =====================================================================
        merge_threshold = 0.99  # Very high similarity threshold
        for i, sim in enumerate(belief_similarities):
            if sim > merge_threshold and num_layers > 1:
                # Merge layers i and i+1
                return True, i, 'merge'

        return False, None, 'none'

    def _spawn_layer(self, after_idx: int):
        """
        Spawn a new layer after the specified index.

        The new layer is initialized with:
        - Priors interpolated from neighbors
        - Fresh generators (same structure as existing layers)
        - Identity gauge frames

        Args:
            after_idx: Insert new layer after this index
        """
        if len(self.layers) >= self.config.max_layers:
            print(f"[Dynamic] Cannot spawn: max_layers ({self.config.max_layers}) reached")
            return

        print(f"[Dynamic] Spawning new layer after layer {after_idx}")

        # Create new layer
        new_scale = after_idx + 1  # Will be inserted at this position
        new_layer = PureFEPLayer(
            embed_dim=self.config.embed_dim,
            scale=new_scale,
            config=self.config,
            prior_bank=self.prior_bank,
        )

        # Move to same device as existing layers
        device = next(self.layers[0].parameters()).device
        new_layer = new_layer.to(device)

        # Initialize priors from neighbors (interpolation)
        if after_idx < len(self.layers) - 1:
            # Interpolate between layer[after_idx] and layer[after_idx+1]
            layer_before = self.layers[after_idx]
            layer_after = self.layers[after_idx + 1]

            with torch.no_grad():
                # Average the priors from neighboring layers
                N_prior = min(
                    layer_before.prior_mu.shape[0],
                    layer_after.prior_mu.shape[0],
                    new_layer.prior_mu.shape[0]
                )
                new_layer.prior_mu[:N_prior] = 0.5 * (
                    layer_before.prior_mu[:N_prior] + layer_after.prior_mu[:N_prior]
                )
                new_layer.prior_sigma[:N_prior] = 0.5 * (
                    layer_before.prior_sigma[:N_prior] + layer_after.prior_sigma[:N_prior]
                )
        else:
            # Last layer - copy from the layer before
            layer_before = self.layers[after_idx]
            with torch.no_grad():
                N_prior = min(layer_before.prior_mu.shape[0], new_layer.prior_mu.shape[0])
                new_layer.prior_mu[:N_prior] = layer_before.prior_mu[:N_prior].clone()
                new_layer.prior_sigma[:N_prior] = layer_before.prior_sigma[:N_prior].clone()

        # Insert into layer list
        # Convert ModuleList to list, insert, convert back
        layers_list = list(self.layers)
        layers_list.insert(after_idx + 1, new_layer)

        # Update scale indices for all layers after the insertion
        for i, layer in enumerate(layers_list):
            layer.scale = i

        # Replace ModuleList
        self.layers = nn.ModuleList(layers_list)

        print(f"[Dynamic] New layer count: {len(self.layers)}")

    def _merge_layers(self, layer_idx: int):
        """
        Merge layer at layer_idx with layer at layer_idx+1.

        The merged layer inherits:
        - Average of priors from both layers
        - Combined statistics

        Args:
            layer_idx: Merge this layer with the next one
        """
        if len(self.layers) <= 1:
            print("[Dynamic] Cannot merge: only one layer remaining")
            return

        if layer_idx >= len(self.layers) - 1:
            print(f"[Dynamic] Cannot merge: layer {layer_idx} has no next layer")
            return

        print(f"[Dynamic] Merging layers {layer_idx} and {layer_idx + 1}")

        layer_keep = self.layers[layer_idx]
        layer_remove = self.layers[layer_idx + 1]

        # Average the priors
        with torch.no_grad():
            N_prior = min(layer_keep.prior_mu.shape[0], layer_remove.prior_mu.shape[0])
            layer_keep.prior_mu[:N_prior] = 0.5 * (
                layer_keep.prior_mu[:N_prior] + layer_remove.prior_mu[:N_prior]
            )
            layer_keep.prior_sigma[:N_prior] = 0.5 * (
                layer_keep.prior_sigma[:N_prior] + layer_remove.prior_sigma[:N_prior]
            )

            # Average statistics
            layer_keep.prior_update_count[:N_prior] = (
                layer_keep.prior_update_count[:N_prior] + layer_remove.prior_update_count[:N_prior]
            ) / 2

        # Remove the merged layer
        layers_list = list(self.layers)
        del layers_list[layer_idx + 1]

        # Update scale indices
        for i, layer in enumerate(layers_list):
            layer.scale = i

        # Replace ModuleList
        self.layers = nn.ModuleList(layers_list)

        print(f"[Dynamic] New layer count: {len(self.layers)}")

    def _apply_dynamic_emergence(
        self,
        layer_infos: List[Dict],
        ce_loss: float,
    ):
        """
        Apply dynamic layer emergence based on VFE analysis.

        Called at the end of train_step when dynamic_layers_enabled is True.

        Args:
            layer_infos: Layer outputs from forward pass
            ce_loss: Cross-entropy loss for this step
        """
        should_act, layer_idx, action = self._check_emergence_conditions(layer_infos, ce_loss)

        if not should_act:
            return

        if action == 'spawn':
            self._spawn_layer(layer_idx)
        elif action == 'merge':
            self._merge_layers(layer_idx)


class PureFEPTrainer:
    """
    Trainer for Pure FEP Transformer.

    Implements the training loop without external optimizers.
    """

    def __init__(self, model: PureFEPTransformer, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)

        # Training statistics
        self.total_steps = 0
        self.best_ppl = float('inf')
        self.history = []

    def train_epoch(
        self,
        dataloader,
        n_vfe_steps: int = 1,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ppl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            metrics = self.model.train_step(input_ids, targets, n_vfe_steps)

            total_loss += metrics['ce_loss']
            total_ppl += metrics['perplexity']
            n_batches += 1
            self.total_steps += 1

            if batch_idx % log_interval == 0:
                avg_ppl = total_ppl / n_batches
                print(f"Step {self.total_steps} | Loss: {metrics['ce_loss']:.4f} | PPL: {avg_ppl:.2f}")

        epoch_metrics = {
            'loss': total_loss / n_batches,
            'perplexity': total_ppl / n_batches,
        }

        self.history.append(epoch_metrics)

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, self.model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20))  # Clamp to prevent overflow

        return {'loss': avg_loss, 'perplexity': ppl}


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PURE FEP TRANSFORMER TEST")
    print("=" * 70)
    print("\nThis tests the principled FEP transformer with:")
    print("  - PriorBank: unified embedding/output via token priors")
    print("  - Gauge position: position in φ, not μ")
    print("  - KL-to-prior output: p(y|q) ∝ exp(-KL(q||π_y)/τ)")
    print("=" * 70)

    # =========================================================================
    # Test 1: PriorBank (standalone with gauge-fixed priors)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] PriorBank: Unified Embedding & Output (Gauge-Fixed)")
    print("=" * 70)

    # Create generators for testing (using odd dim for SO(3))
    test_embed_dim = 33  # Must be odd for SO(3)
    test_generators = torch.from_numpy(generate_so3_generators(test_embed_dim)).float()
    prior_bank = PriorBank(
        vocab_size=100,
        embed_dim=test_embed_dim,
        gauge_fixed_priors=True,
        generators=test_generators,
        phi_dim=3,
    )
    print(f"    Created PriorBank: {prior_bank.vocab_size} tokens × {prior_bank.embed_dim} dims (gauge-fixed)")

    # Test encoding - now returns (mu, sigma, phi)
    test_tokens = torch.randint(0, 100, (2, 10))  # (B=2, N=10)
    mu_q, sigma_q, phi_q = prior_bank.encode(test_tokens)
    print(f"    Encode: tokens {test_tokens.shape} → μ {mu_q.shape}, σ {sigma_q.shape}, φ {phi_q.shape}")

    # Test decoding
    logits = prior_bank.decode(mu_q, sigma_q, tau=1.0)
    print(f"    Decode: (μ, σ) → logits {logits.shape}")
    print(f"    Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Verify: encoding then decoding should give highest prob to original token
    probs = F.softmax(logits, dim=-1)
    predicted = probs.argmax(dim=-1)
    accuracy = (predicted == test_tokens).float().mean().item()
    print(f"    Self-consistency: accuracy = {accuracy:.1%}")
    print(f"    ✓ PriorBank working correctly!" if accuracy > 0.5 else "    ⚠ Low accuracy (expected for random init)")

    # =========================================================================
    # Test 2: GaugePositionEncoder
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] GaugePositionEncoder: Position in φ, not μ")
    print("=" * 70)

    pos_encoder = GaugePositionEncoder(max_seq_len=64, mode='learned', scale=0.1)
    print(f"    Created GaugePositionEncoder: max_len=64, mode=learned")

    phi_token = torch.zeros(2, 10, 3)  # (B, N, 3)
    phi_with_pos = pos_encoder(phi_token, seq_len=10)
    print(f"    Input φ: {phi_token.shape}, Output φ: {phi_with_pos.shape}")
    print(f"    φ[0,0]: {phi_with_pos[0,0].tolist()}")
    print(f"    φ[0,9]: {phi_with_pos[0,9].tolist()}")

    # Check that different positions have different φ
    pos_diff = (phi_with_pos[0, 0] - phi_with_pos[0, 5]).norm().item()
    print(f"    Position difference (0 vs 5): {pos_diff:.4f}")
    print(f"    ✓ Position encoded in gauge frame!" if pos_diff > 0 else "    ⚠ No position encoding")

    # =========================================================================
    # Test 3: Full PureFEPTransformer (PUREST config)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 3] PureFEPTransformer (PUREST FEP config)")
    print("=" * 70)

    # PUREST FEP configuration
    config_pure = PureFEPConfig(
        embed_dim=63,  # Must be odd for SO(3)!
        num_layers=2,
        seq_length=32,
        vocab_size=1000,
        # PUREST settings:
        embedding_mode='prior_bank',    # Beliefs from token priors
        position_mode='gauge_frame',    # Position in φ
        output_mode='kl_to_prior',      # Output via KL to priors
        output_tau=1.0,
        # VFE parameters
        mu_lr=0.1,
        sigma_lr=0.025,
        prior_lr=0.05,
        pure_fep_mode=True,             # No backprop!
    )

    print(f"\n    Config:")
    print(f"      embed_dim: {config_pure.embed_dim}")
    print(f"      embedding_mode: {config_pure.embedding_mode}")
    print(f"      position_mode: {config_pure.position_mode}")
    print(f"      output_mode: {config_pure.output_mode}")
    print(f"      pure_fep_mode: {config_pure.pure_fep_mode}")

    model_pure = PureFEPTransformer(config_pure)

    # Count parameters
    total_params = sum(p.numel() for p in model_pure.parameters())
    print(f"\n    Total parameters: {total_params:,}")

    # Test forward pass
    B, N = 2, 32
    input_ids = torch.randint(0, config_pure.vocab_size, (B, N))
    targets = torch.randint(0, config_pure.vocab_size, (B, N))

    print(f"\n    Forward pass...")
    logits, info = model_pure(input_ids, targets=targets, n_vfe_steps=1)
    print(f"    Input: {input_ids.shape}")
    print(f"    Output logits: {logits.shape}")

    # Verify output is valid probability distribution
    probs = F.softmax(logits, dim=-1)
    prob_sum = probs.sum(dim=-1).mean().item()
    print(f"    Prob sum (should be ~1.0): {prob_sum:.4f}")

    # Test training step
    print(f"\n    Training step (pure FEP - no backprop)...")
    metrics = model_pure.train_step(input_ids, targets, n_vfe_steps=1)
    print(f"    CE Loss: {metrics['ce_loss']:.4f}")
    print(f"    Perplexity: {metrics['perplexity']:.2f}")

    # =========================================================================
    # Test 4: Comparison with AD HOC config
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 4] Comparison: PURE vs AD HOC")
    print("=" * 70)

    # AD HOC configuration (standard transformer style)
    config_adhoc = PureFEPConfig(
        embed_dim=63,
        num_layers=2,
        seq_length=32,
        vocab_size=1000,
        # AD HOC settings:
        embedding_mode='learned',       # Standard nn.Embedding
        position_mode='sinusoidal_mu',  # Sinusoidal added to μ
        output_mode='linear',           # Linear projection
        pure_fep_mode=False,            # Use backprop
        mu_lr=0.1,
    )

    model_adhoc = PureFEPTransformer(config_adhoc)

    print(f"\n    AD HOC model:")
    print(f"      embedding_mode: {config_adhoc.embedding_mode}")
    print(f"      position_mode: {config_adhoc.position_mode}")
    print(f"      output_mode: {config_adhoc.output_mode}")

    # Compare forward passes
    logits_pure, _ = model_pure(input_ids, targets=targets, n_vfe_steps=1)
    logits_adhoc, _ = model_adhoc(input_ids, targets=targets, n_vfe_steps=1)

    print(f"\n    PURE logits range: [{logits_pure.min():.2f}, {logits_pure.max():.2f}]")
    print(f"    AD HOC logits range: [{logits_adhoc.min():.2f}, {logits_adhoc.max():.2f}]")

    # Compare training
    metrics_pure = model_pure.train_step(input_ids, targets, n_vfe_steps=2)
    metrics_adhoc = model_adhoc.train_step(input_ids, targets, n_vfe_steps=2)

    print(f"\n    PURE FEP:  PPL = {metrics_pure['perplexity']:.2f}")
    print(f"    AD HOC:    PPL = {metrics_adhoc['perplexity']:.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Pure FEP Transformer Components")
    print("=" * 70)
    print("""
    PRINCIPLED (No Ad Hoc Structures):
    ✓ PriorBank: Unified embedding/output via token priors
      - Encode: q ← π_token (prior belief)
      - Decode: p(y|q) ∝ exp(-KL(q||π_y)/τ)

    ✓ GaugePositionEncoder: Position in φ ∈ so(3)
      - Transport Ω_ij encodes relative position
      - Shift-invariant attention with position awareness

    ✓ VFE with β_ij term:
      F = α·KL(q||p) + λ·Σ_ij β_ij·KL(q_i||Ω_ij·q_j) + E[-log p(y|z)]

    ✓ Softmax coupling gradient replaces GELU/ReLU nonlinearity

    TOGGLEABLE (for comparison):
    - embedding_mode: 'prior_bank', 'learned', 'hybrid'
    - position_mode: 'gauge_frame', 'sinusoidal_mu', 'both'
    - output_mode: 'kl_to_prior', 'linear', 'both'
    """)
    print("=" * 70)
    print("✓ ALL TESTS COMPLETE")
    print("=" * 70)