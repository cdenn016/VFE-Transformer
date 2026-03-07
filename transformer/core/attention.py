"""
KL-Divergence Based Attention Mechanism (0D Gauge Transformer)
===============================================================

Implements attention via information geometry instead of learned Q, K projections:

    β_ij = softmax(-KL(q_i || Ω_ij[q_j]) / κ)

where:
    - q_i = N(μ_i, Σ_i): Agent i's belief distribution
    - Ω_ij: Parallel transport operator (gauge connection)
    - KL: Kullback-Leibler divergence (information distance)
    - κ: Temperature parameter

Key Insight: NO W_Q, W_K matrices! Attention emerges from geometry.

0D Architecture:
    - All agents at single point c*
    - β_ij are scalars (not spatial fields)
    - No spatial integrals, just sums over agents

Author: Implementation from plan.py
Date: November 2025
"""

# Suppress noisy warnings BEFORE other imports
import warnings
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from transformer.core.gauge_utils import stable_matrix_exp_pair
from transformer.core.sanitization import san

# Import our fast math kernels
try:
    from math_utils.numba_kernels import (
        kl_gaussian_numba,
        compute_kl_transported_numba,
        push_gaussian_numba,
    )
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba kernels not available - falling back to PyTorch (slower)")

# Import transport operators
try:
    from math_utils.transport import compute_transport
    from math_utils.generators import generate_so3_generators
    TRANSPORT_AVAILABLE = True
except ImportError:
    TRANSPORT_AVAILABLE = False
    print("⚠️  Transport module not available")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core attention functions
    'compute_attention_weights',
    'aggregate_messages',

    # KL divergence computation
    'compute_kl_matrix',

    # Multi-head attention class
    'IrrepMultiHeadAttention',

    # Utilities
    'create_attention_mask',
    'compute_transport_operators',
    'estimate_chunk_size',

    # Constants for checking availability
    'NUMBA_AVAILABLE',
    'TRANSPORT_AVAILABLE',
]


# =============================================================================
# Rotary Position Embeddings (RoPE) for KL-Divergence Attention
# =============================================================================
# RoPE applies position-dependent SO(2)^{K/2} rotations to belief means μ
# before computing KL divergences. This makes attention position-sensitive
# without affecting gauge transport Ω_ij.
#
# In the gauge-theoretic framework (see GL(K)_attention.tex §3):
#   Ω_ij^{RoPE} = R(θ_{j-i}) · Ω_ij^{content}
# where R(θ) ∈ SO(2)^{K/2} ⊂ GL(K) is the position-dependent rotation.
# =============================================================================

def _build_rope_freqs(K: int, base: float = 10000.0,
                      device: torch.device = None,
                      dtype: torch.dtype = None) -> torch.Tensor:
    """Compute RoPE frequency bands for K-dimensional beliefs.

    Returns:
        freqs: (K//2,) inverse frequency bands
    """
    half_K = K // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_K, device=device, dtype=dtype) / half_K))
    return freqs


def _apply_rope(mu: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Apply Rotary Position Embeddings to belief means.

    Rotates consecutive pairs of dimensions by position-dependent angles,
    making KL divergences sensitive to relative position.

    Args:
        mu: (B, N, K) belief means
        base: RoPE frequency base (default 10000.0)

    Returns:
        mu_rotated: (B, N, K) position-rotated belief means
    """
    B, N, K = mu.shape
    half_K = K // 2

    # Compute position-dependent angles: θ_n(pos) = pos * freq_n
    freqs = _build_rope_freqs(K, base, device=mu.device, dtype=mu.dtype)  # (K//2,)
    positions = torch.arange(N, device=mu.device, dtype=mu.dtype)  # (N,)
    angles = torch.outer(positions, freqs)  # (N, K//2)

    cos_angles = torch.cos(angles)  # (N, K//2)
    sin_angles = torch.sin(angles)  # (N, K//2)

    # Split μ into even/odd pairs and apply 2D rotation
    mu_even = mu[:, :, :2*half_K:2]   # (B, N, K//2) - dims 0,2,4,...
    mu_odd = mu[:, :, 1:2*half_K:2]   # (B, N, K//2) - dims 1,3,5,...

    # R(θ) @ [x, y]^T = [x·cos(θ) - y·sin(θ), x·sin(θ) + y·cos(θ)]
    mu_rotated = mu.clone()
    mu_rotated[:, :, :2*half_K:2] = mu_even * cos_angles - mu_odd * sin_angles
    mu_rotated[:, :, 1:2*half_K:2] = mu_even * sin_angles + mu_odd * cos_angles

    return mu_rotated


# =============================================================================
# Sparse Attention Patterns
# =============================================================================

def create_attention_mask(
    num_agents: int,
    pattern: str = 'full',
    window: int = 64,
    device: torch.device = torch.device('cpu'),
    causal: bool = True,
) -> torch.Tensor:
    """
    Create attention mask.

    Args:
        num_agents: Number of agents (sequence length)
        pattern: 'full' (only supported pattern)
        window: Unused, kept for API compatibility
        device: Device to create tensor on
        causal: If True, apply causal masking (i can't attend to j>i)

    Returns:
        mask: (N, N) binary mask where 1 = can attend, 0 = cannot attend
    """
    N = num_agents
    mask = torch.ones(N, N, device=device)

    if causal:
        causal_mask = torch.tril(torch.ones(N, N, device=device))
        mask = mask * causal_mask

    return mask


# =============================================================================
# Transport Operator Caching (for evolve_phi=False optimization)
# =============================================================================

def compute_transport_operators(
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    enforce_orthogonal: bool = False,  # If True, project to SO(K) via Newton-Schulz
    gauge_mode: str = 'learned',  # 'learned' or 'trivial'
) -> dict:
    """
    Precompute transport operators for caching when phi is fixed.

    Works for SO(3), SO(N), and GL⁺(K) gauge groups:
    - SO(3): n_gen = 3, phi ∈ ℝ³, enforce_orthogonal=True
    - SO(N): n_gen = N(N-1)/2, phi ∈ ℝ^{N(N-1)/2}, enforce_orthogonal=True
    - GL⁺(K): n_gen = K², phi ∈ ℝ^{K²}, enforce_orthogonal=False
      (exp parameterization reaches identity component GL⁺(K) only;
       det(exp(X)) = exp(tr(X)) > 0 always. The product Ω_ij =
       exp(X_i)·exp(-X_j) covers all of GL⁺(K).)

    Gauge Modes:
    - 'learned': Standard mode where φ is learned per-token. Transport Ω_ij
                 encodes relative frame transformations between tokens.
    - 'trivial': Global frame mode where Ω = I for all pairs. This is the
                 mathematically principled "trivial gauge" or "gauge fixing"
                 that recovers standard attention as a special case.
                 Equivalent to setting φ = 0 everywhere.

    When evolve_phi=False, these operators are constant across layers.
    Computing once saves 2 matrix exponentials per head per layer.

    Args:
        phi: Gauge frames (B, N, n_gen) in Lie algebra
             - For SO(3): shape (B, N, 3)
             - For SO(N): shape (B, N, N*(N-1)/2)
             - For GL(K): shape (B, N, K²)
        generators: Lie algebra generators (n_gen, K, K)
        enforce_orthogonal: If True, apply Newton-Schulz to ensure Ω ∈ SO(K).
                           If False, allow Ω ∈ GL⁺(K) (faster, still gauge-invariant).
        gauge_mode: 'learned' for per-token frames, 'trivial' for global frame (Ω=I)

    Returns:
        dict with:
            'exp_phi': (B, N, K, K) - exp(φ·G) for each token
            'exp_neg_phi': (B, N, K, K) - exp(-φ·G) for each token
            'Omega': (B, N, N, K, K) - full pairwise transport Ω_ij = exp(φ_i)exp(-φ_j)
    """
    B, N, _ = phi.shape
    K = generators.shape[1]
    dtype = phi.dtype
    device = phi.device

    # =================================================================
    # TRIVIAL GAUGE: φ = 0, Ω = I (global frame, standard attention limit)
    # =================================================================
    # This is the mathematically principled gauge fixing where all tokens
    # share a single coordinate frame. No transport between frames.
    # KL(q_i || Ω_ij[q_j]) = KL(q_i || q_j) when Ω = I.
    if gauge_mode == 'trivial':
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = eye_K.expand(B, N, K, K).contiguous()      # (B, N, K, K)
        exp_neg_phi = eye_K.expand(B, N, K, K).contiguous()  # (B, N, K, K)
        Omega = eye_K.expand(B, N, N, K, K).contiguous()     # (B, N, N, K, K)
        return {
            'exp_phi': exp_phi,
            'exp_neg_phi': exp_neg_phi,
            'Omega': Omega,
        }

    # =================================================================
    # LEARNED GAUGE: φ per-token, Ω_ij = exp(φ_i)·exp(-φ_j)
    # =================================================================
    # φ·G: combine gauge frames with generators
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)

    # Check if generators are skew-symmetric (SO(K) gauge group).
    # For skew-symmetric A: exp(-A) = exp(A)^T, saving one matrix_exp call.
    _is_skew = torch.allclose(
        generators + generators.transpose(-1, -2),
        torch.zeros_like(generators), atol=1e-5
    )

    # Float64 matrix_exp for GL(K) numerical stability (prevents NaN
    # from Padé scaling-squaring overflow when phi values grow large).
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    if _is_skew:
        # SO(K): exp(-A) = exp(A)^T for skew-symmetric A
        exp_neg_phi = exp_phi.transpose(-1, -2)

    # Re-orthogonalization for SO(K) gauge groups
    # NOTE: For GL⁺(K), this is NOT required - VFE is invariant under GL(K)!
    # Only enable if you explicitly want SO(K) (e.g., for Haar measure averaging)
    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    # Full pairwise transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
    Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

    return {
        'exp_phi': exp_phi,
        'exp_neg_phi': exp_neg_phi,
        'Omega': Omega,
    }


# =============================================================================
# Core Attention: KL-Based Weights
# =============================================================================

def compute_attention_weights(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) or (B, N, K) if diagonal_covariance=True
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
    kappa: float,              # Temperature
    epsilon: float = 1e-8,     # Numerical stability
    mask: Optional[torch.Tensor] = None,  # (B, N, N) causal mask
    use_numba: bool = True,
    return_kl: bool = False,   # Return KL matrix for loss computation
    diagonal_covariance: bool = False,  # Use diagonal sigma (B,N,K) instead of full (B,N,K,K)
    cached_transport: Optional[dict] = None,  # Precomputed transport operators (from compute_transport_operators)
    # Memory-efficient options
    irrep_dims: Optional[List[int]] = None,  # Block-diagonal structure [d₁, d₂, ...] for principled KL decomposition
    chunk_size: Optional[int] = None,  # Chunk size for memory-efficient computation (None = auto)
    # Identity transport mode (diagnostic/simplification)
    use_identity_transport: bool = False,  # If True, Ω_ij = I (no gauge transport)
    # Self-attention masking (prevents attention collapse)
    mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
    # Gauge group control
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
    # Rotary Position Embeddings (RoPE)
    use_rope: bool = False,            # If True, apply RoPE rotations to μ before KL computation
    rope_base: float = 10000.0,        # RoPE frequency base
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute attention weights from KL divergences (0D version).

    Formula:
        β_ij = softmax_j(-KL(q_i || Ω_ij[q_j]) / κ)

    where Ω_ij = exp(φ_i) · exp(-φ_j) transports q_j to i's frame.

    0D Structure:
        - All agents at single point c*, so β_ij are SCALARS
        - No spatial fields β_ij(c), just one number per pair
        - No spatial integration, just O(N²) agent-pair loop

    Args:
        mu_q: Query belief means, shape (B, N, K)
              N = num_agents at single point c*
        sigma_q: Query covariances, shape (B, N, K, K) if full, (B, N, K) if diagonal
        phi: Gauge frames, shape (B, N, 3) in so(3)
        generators: SO(3) generators for irrep, shape (3, K, K)
        kappa: Temperature parameter (higher = softer attention)
        epsilon: Softmax stability constant
        mask: Optional causal mask (B, N, N) - 0 masks out position
        use_numba: Use fast Numba kernels if available
        diagonal_covariance: If True, sigma_q is (B,N,K) diagonal variances.
                            Uses O(N²×K) memory instead of O(N²×K²)!
        cached_transport: Optional precomputed transport operators from compute_transport_operators().
                         When evolve_phi=False, caching avoids redundant matrix exponentials.
        irrep_dims: Optional list of irrep block dimensions [d₁, d₂, ...].
                   When provided, uses block-diagonal KL computation which is:
                   1. Theoretically principled (respects gauge structure)
                   2. Memory efficient: O(N² × Σᵢdᵢ²) vs O(N² × K²)
                   For K=255 with 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂: ~82× memory savings!
        chunk_size: Optional chunk size for memory-efficient processing.
                   When provided, processes N×N attention in C×C chunks.
                   None = no chunking (fast but memory-hungry)
        use_identity_transport: If True, bypass gauge transport entirely.
                               Sets Ω_ij = I for all pairs, computing raw
                               KL(q_i || q_j) without any rotation.
                               Useful for diagnostics or when transport is
                               not desired.
        mask_self_attention: If True, mask out diagonal of attention matrix.
                            This prevents the model from attending to itself,
                            forcing it to attend to other tokens. Critical for
                            preventing attention collapse since KL(q_i||q_i)=0
                            always makes self-attention the most attractive.

    Returns:
        beta: Attention weights, shape (B, N, N)
              beta[b, i, j] = attention from agent i to agent j
        kl_matrix: (Optional) KL divergence matrix (B, N, N) if return_kl=True
                   kl_matrix[b, i, j] = KL(q_i || Ω_ij[q_j])

    Example:
        >>> B, N, K = 2, 10, 32
        >>> mu = torch.randn(B, N, K)
        >>> sigma = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        >>> phi = torch.randn(B, N, 3) * 0.1
        >>> G = torch.from_numpy(generate_so3_generators(K)).float()
        >>> beta = compute_attention_weights(mu, sigma, phi, G, kappa=1.0)
        >>> beta.shape
        torch.Size([2, 10, 10])
        >>> beta.sum(dim=-1)  # Should sum to 1 (plus epsilon)
        tensor([[1.0000, 1.0000, ...], ...])
    """
    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # =========================================================================
    # RoPE: Apply position-dependent SO(2)^{K/2} rotations to belief means
    # This makes KL(q_i || Ω_ij[q_j]) sensitive to relative position (j-i).
    # Applied ONLY to attention scores, NOT to message aggregation values.
    # =========================================================================
    if use_rope:
        mu_q = _apply_rope(mu_q, base=rope_base)

    # =========================================================================
    # Compute all pairwise KL divergences: KL(q_i || Ω_ij[q_j])
    # Helper functions return KL tensors (not in-place) to preserve autograd graph
    # =========================================================================

    # CRITICAL: Never use Numba path on CUDA devices!
    # Numba kernels are CPU-only and would cause GPU→CPU→GPU transfer bottleneck.
    # The PyTorch path is fully vectorized and runs efficiently on GPU.
    is_cuda = device.type == 'cuda'

    # =========================================================================
    # MEMORY-EFFICIENT PATHS (NEW!)
    # Priority: block-diagonal > chunked > diagonal > full
    # =========================================================================
    if irrep_dims is not None and diagonal_covariance:
        # BLOCK-DIAGONAL + DIAGONAL MODE: Best of both worlds!
        # Block processing for small Omega + diagonal KL formulas (no inv/Cholesky)
        kl_matrix = _compute_kl_matrix_block_diagonal_diag(
            mu_q, sigma_q, phi, generators, irrep_dims,
            enforce_orthogonal=enforce_orthogonal
        )
    elif irrep_dims is not None and not diagonal_covariance:
        # BLOCK-DIAGONAL MODE: Principled + memory-efficient!
        # Uses O(N² × Σᵢdᵢ²) instead of O(N² × K²) - massive savings!
        if chunk_size is not None:
            # Block-diagonal + chunked: maximum memory efficiency
            kl_matrix = _compute_kl_matrix_block_diagonal_chunked(
                mu_q, sigma_q, phi, generators, irrep_dims, chunk_size
            )
        else:
            # Block-diagonal only (still big savings vs full K×K)
            kl_matrix = _compute_kl_matrix_block_diagonal(
                mu_q, sigma_q, phi, generators, irrep_dims
            )
    elif chunk_size is not None and not diagonal_covariance:
        # CHUNKED MODE: Full covariance but memory-efficient
        kl_matrix = _compute_kl_matrix_chunked(
            mu_q, sigma_q, phi, generators, chunk_size
        )
    elif diagonal_covariance:
        # DIAGONAL MODE: O(N²×K) memory instead of O(N²×K²)!
        # sigma_q is (B, N, K) not (B, N, K, K)
        if chunk_size is not None:
            kl_matrix = _compute_kl_matrix_diagonal_chunked(
                mu_q, sigma_q, phi, generators, chunk_size,
                enforce_orthogonal=enforce_orthogonal
            )
        else:
            kl_matrix = _compute_kl_matrix_diagonal(
                mu_q, sigma_q, phi, generators, cached_transport,
                enforce_orthogonal=enforce_orthogonal
            )
    elif use_numba and NUMBA_AVAILABLE and TRANSPORT_AVAILABLE and not is_cuda:
        # Fast path: Use Numba kernels (CPU only)
        # Note: Numba path doesn't support cached_transport (CPU-only fallback)
        # Numba operates on numpy so pre-allocate buffer (no autograd needed)
        kl_matrix = torch.zeros(batch_size, num_agents, num_agents, device=device, dtype=dtype)
        _compute_kl_matrix_numba(
            mu_q, sigma_q, phi, generators, kl_matrix
        )
    else:
        # GPU path OR CPU fallback: Pure PyTorch (fully vectorized, CUDA-compatible)
        kl_matrix = _compute_kl_matrix_torch(
            mu_q, sigma_q, phi, generators, cached_transport,
            use_identity_transport=use_identity_transport,
            enforce_orthogonal=enforce_orthogonal
        )

    # =========================================================================
    # Convert KL distances to attention weights
    # =========================================================================

    # DIMENSION-AWARE KL NORMALIZATION (τ = √K):
    # KL between K-dimensional Gaussians scales as O(K).
    # For large K, raw KL magnitudes cause softmax saturation
    # (one-hot attention → zero gradients). Normalizing by √K
    # keeps logits in a well-conditioned range.
    dim_scale = math.sqrt(max(K, 1))

    # Attention logits: -KL / (κ · √K)
    logits = -kl_matrix / (kappa * dim_scale)  # (B, N, N)

    # Apply causal mask if provided (BEFORE self-attention masking)
    if mask is not None:
        # mask[b, i, j] = 0 means agent i CANNOT attend to agent j
        # masked_fill is not in-place (returns new tensor), safe for autograd
        logits = logits.masked_fill(mask == 0, float('-inf'))

    # ==========================================================================
    # SELF-ATTENTION MASKING: Force model to attend to other tokens
    # KL(q_i||q_i)=0 always, making diagonal dominant. Masking diagonal forces
    # the model to learn meaningful attention patterns over other tokens.
    #
    # IMPORTANT: Only mask diagonal where there are OTHER valid targets!
    # With causal masking, position 0 can only attend to itself - masking it
    # would leave no valid targets, causing NaN in softmax.
    # ==========================================================================
    if mask_self_attention:
        B, N, _ = logits.shape
        diag_idx = torch.arange(N, device=logits.device)
        # Check which positions have at least one other valid target
        # A position has other targets if any off-diagonal element is not -inf
        has_other_targets = (logits != float('-inf')).sum(dim=-1) > 1  # (B, N)
        # Clone to avoid inplace modification (needed for gradient computation)
        logits = logits.clone()
        # Apply masking only where safe (where there are other targets)
        diag_vals = logits[:, diag_idx, diag_idx]  # (B, N)
        masked_diag_vals = torch.where(
            has_other_targets,
            torch.full_like(diag_vals, float('-inf')),
            diag_vals
        )
        logits[:, diag_idx, diag_idx] = masked_diag_vals

    # Softmax over keys (dimension 2)
    beta = F.softmax(logits, dim=-1)  # (B, N, N)

    # Clamp only non-masked positions to epsilon for numerical stability,
    # preserving exact zeros from -inf masked positions (e.g. causal mask)
    masked_positions = (logits == float('-inf'))
    # Apply clamp only where positions are NOT masked, keeping masked positions at 0
    beta = torch.where(masked_positions, beta, beta.clamp(min=epsilon))
    # Re-normalize (guard against all-masked rows producing zero sums)
    beta_sum = beta.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    beta = beta / beta_sum

    if return_kl:
        return beta, kl_matrix
    else:
        return beta


def compute_kl_matrix(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) or (B, N, K) if diagonal_covariance=True
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators
    diagonal_covariance: bool = False,
    irrep_dims: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    use_identity_transport: bool = False,
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> torch.Tensor:
    """
    Compute pairwise KL divergence matrix: KL(q_i || Ω_ij[q_j]).

    This is a convenience function that directly returns the KL matrix
    without computing attention weights. Useful for debugging, loss
    computation, and analysis.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) full covariances or (B, N, K) diagonal variances
        phi: (B, N, n_gen) gauge frames
        generators: (n_gen, K, K) Lie algebra generators
        diagonal_covariance: If True, sigma_q is (B,N,K) diagonal variances
        irrep_dims: Optional list of irrep block dimensions for block-diagonal mode
        chunk_size: Optional chunk size for memory-efficient computation
        use_identity_transport: If True, Ω_ij = I (skip gauge transport)
        enforce_orthogonal: If True, enforce Ω ∈ SO(K) via Newton-Schulz iteration

    Returns:
        kl_matrix: (B, N, N) pairwise KL divergences
                   kl_matrix[b, i, j] = KL(q_i || Ω_ij[q_j])

    Example:
        >>> B, N, K = 2, 10, 32
        >>> mu = torch.randn(B, N, K)
        >>> sigma = torch.ones(B, N, K)  # Diagonal variances
        >>> phi = torch.randn(B, N, 3) * 0.1
        >>> G = generate_so3_generators(K)  # (3, K, K)
        >>> kl = compute_kl_matrix(mu, sigma, phi, G, diagonal_covariance=True)
        >>> kl.shape
        torch.Size([2, 10, 10])
    """
    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Select appropriate backend based on parameters
    # Helper functions return KL tensors (not in-place) to preserve autograd graph
    is_cuda = device.type == 'cuda'

    if irrep_dims is not None and diagonal_covariance:
        kl_matrix = _compute_kl_matrix_block_diagonal_diag(
            mu_q, sigma_q, phi, generators, irrep_dims,
            enforce_orthogonal=enforce_orthogonal
        )
    elif irrep_dims is not None and not diagonal_covariance:
        if chunk_size is not None:
            kl_matrix = _compute_kl_matrix_block_diagonal_chunked(
                mu_q, sigma_q, phi, generators, irrep_dims, chunk_size
            )
        else:
            kl_matrix = _compute_kl_matrix_block_diagonal(
                mu_q, sigma_q, phi, generators, irrep_dims
            )
    elif chunk_size is not None and not diagonal_covariance:
        kl_matrix = _compute_kl_matrix_chunked(
            mu_q, sigma_q, phi, generators, chunk_size
        )
    elif diagonal_covariance:
        if chunk_size is not None:
            kl_matrix = _compute_kl_matrix_diagonal_chunked(
                mu_q, sigma_q, phi, generators, chunk_size,
                enforce_orthogonal=enforce_orthogonal
            )
        else:
            kl_matrix = _compute_kl_matrix_diagonal(
                mu_q, sigma_q, phi, generators, None,
                enforce_orthogonal=enforce_orthogonal
            )
    elif NUMBA_AVAILABLE and TRANSPORT_AVAILABLE and not is_cuda:
        # Numba operates on numpy so pre-allocate buffer (no autograd needed)
        kl_matrix = torch.zeros(batch_size, num_agents, num_agents, device=device, dtype=dtype)
        _compute_kl_matrix_numba(
            mu_q, sigma_q, phi, generators, kl_matrix
        )
    else:
        kl_matrix = _compute_kl_matrix_torch(
            mu_q, sigma_q, phi, generators, None,
            use_identity_transport=use_identity_transport,
            enforce_orthogonal=enforce_orthogonal
        )

    return kl_matrix


def _compute_kl_matrix_numba(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    kl_matrix: torch.Tensor,
) -> None:
    """
    Fast KL matrix computation using Numba kernels.

    Computes KL(q_i || Ω_ij[q_j]) for all pairs (i,j) using:
    1. Transport q_j → i's frame via Ω_ij
    2. Compute KL divergence

    Modifies kl_matrix in-place.
    """
    batch_size, num_agents, K = mu_q.shape

    # Convert to numpy for Numba
    mu_np = mu_q.detach().cpu().numpy().astype(np.float64)
    sigma_np = sigma_q.detach().cpu().numpy().astype(np.float64)
    phi_np = phi.detach().cpu().numpy().astype(np.float64)
    G_np = generators.detach().cpu().numpy().astype(np.float64)

    # Compute KL matrix
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(num_agents):
                # Compute transport operator Ω_ij
                Omega_ij = compute_transport(
                    phi_np[b, i],      # φ_i
                    phi_np[b, j],      # φ_j
                    G_np,
                    validate=False,
                    eps=1e-8
                )  # (K, K)

                # Compute KL(q_i || Ω_ij[q_j]) in one shot
                kl_ij = compute_kl_transported_numba(
                    mu_np[b, i],       # μ_i
                    sigma_np[b, i],    # Σ_i
                    mu_np[b, j],       # μ_j
                    sigma_np[b, j],    # Σ_j
                    Omega_ij           # Ω_ij
                )

                kl_matrix[b, i, j] = kl_ij


def _compute_kl_matrix_torch(
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    phi: torch.Tensor,
    generators: torch.Tensor,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    use_identity_transport: bool = False,  # If True, bypass transport (Ω = I)
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> torch.Tensor:
    """
    VECTORIZED KL matrix computation using pure PyTorch.

    Computes all pairwise KL divergences without Python loops.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) belief covariances
        phi: (B, N, 3) gauge fields
        generators: (3, K, K) SO(3) generators
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()
        use_identity_transport: If True, skip transport and compute raw KL(q_i || q_j)
        enforce_orthogonal: If True, apply Newton-Schulz to ensure Ω ∈ SO(K)

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # =========================================================================
    # Step 1: Get transport operators (use cached if available)
    # =========================================================================
    if use_identity_transport:
        # IDENTITY TRANSPORT: Ω_ij = I for all pairs
        # Skip expensive matrix exponentials - just use raw beliefs
        # μ_transported = μ_j (no rotation)
        # Σ_transported = Σ_j (no rotation)
        # Use .clone() after expand to avoid view-related gradient issues
        mu_transported = mu_q[:, None, :, :].expand(-1, N, -1, -1).clone()  # (B, N, N, K)
        Sigma_transported = sigma_q[:, None, :, :, :].expand(-1, N, -1, -1, -1).clone()  # (B, N, N, K, K)
    else:
        if cached_transport is not None and 'Omega' in cached_transport:
            # Use precomputed transport operators (saves 2 matrix exponentials!)
            Omega = cached_transport['Omega']
        else:
            # Compute transport operators
            # phi: (B, N, n_gen) -> phi_matrix: (B, N, K, K)
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)

            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 16:
                eye_K = torch.eye(K, device=device, dtype=dtype)
                exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
                exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

            # Omega_ij = exp(φ_i) @ exp(-φ_j)
            # Result: (B, N, N, K, K)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # =========================================================================
        # Step 2: Transport all means and covariances
        # =========================================================================
        # μ_j^{→i} = Ω_ij @ μ_j
        mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
        Sigma_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Symmetrize to correct numerical asymmetry from the triple einsum.
        # Without this, Sigma_transported can have asymmetric floating-point
        # errors that cause Cholesky to fail (not positive-definite).
        Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.transpose(-1, -2))

    # =========================================================================
    # Step 3: Expand mu_i and Sigma_i for pairwise comparison
    # Use .clone() after expand to avoid view-related gradient issues
    # =========================================================================
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1).clone()  # (B, N, N, K)
    Sigma_i = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)

    # =========================================================================
    # Step 4: Compute all KL divergences
    # KL(q_i || Ω_ij[q_j]) = KL(N(μ_i, Σ_i) || N(μ_j^{→i}, Σ_j^{→i}))
    # =========================================================================
    I = torch.eye(K, device=device, dtype=dtype)
    Sigma_i_reg = Sigma_i + max(eps, 1e-4) * I

    # Try Cholesky with escalating jitter to handle near-singular matrices
    # before falling back to eigenvalue-based computation.
    # Start jitter at 1e-4 (not eps=1e-8): transported covariances accumulate
    # numerical error from Ω @ Σ @ Ωᵀ and need stronger regularization,
    # especially when sigma embeddings are untrained (alpha=beta=0 in M-step).
    # NaN guard: replace any NaN entries with identity covariance.
    # NaNs propagate from matrix_exp overflow when phi grows very large.
    nan_mask = torch.isnan(Sigma_transported).any(dim=-1).any(dim=-1)  # (B, N, N)
    if nan_mask.any():
        Sigma_transported = torch.where(
            nan_mask.unsqueeze(-1).unsqueeze(-1),
            I.expand_as(Sigma_transported),
            Sigma_transported,
        )

    cholesky_success = False
    jitter = max(eps, 1e-4)
    for _attempt in range(4):
        Sigma_transported_reg = Sigma_transported + jitter * I
        try:
            # Cholesky of transported covariances (prior in KL)
            L_p = torch.linalg.cholesky(Sigma_transported_reg)
            cholesky_success = True
            break
        except RuntimeError:
            jitter *= 10.0  # escalate: 1e-4 -> 1e-3 -> 1e-2 -> 1e-1

    if cholesky_success:
        san.record('cholesky_jitter', value=jitter)
        try:
            # Trace term: tr(Σ_p⁻¹ Σ_q) where Σ_p = Σ_j^{→i}, Σ_q = Σ_i
            Y = torch.linalg.solve_triangular(L_p, Sigma_i_reg, upper=False)
            Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
            trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, N, N)

            # Mahalanobis term: (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q)
            delta_mu = mu_transported - mu_i  # (B, N, N, K)
            v = torch.linalg.solve_triangular(
                L_p, delta_mu.unsqueeze(-1), upper=False
            ).squeeze(-1)
            mahal_term = torch.sum(v ** 2, dim=-1)  # (B, N, N)

            # Log determinant terms
            logdet_p = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )
            # Cholesky of Sigma_i (query covariance) with progressive fallback
            try:
                L_q = torch.linalg.cholesky(Sigma_i_reg)
            except RuntimeError:
                reg = eps
                for attempt in range(5):
                    reg *= 10.0
                    Sigma_i_fallback = Sigma_i + reg * I
                    Sigma_i_fallback = 0.5 * (Sigma_i_fallback + Sigma_i_fallback.transpose(-1, -2))
                    try:
                        L_q = torch.linalg.cholesky(Sigma_i_fallback)
                        warnings.warn(
                            f"[attention/_compute_kl_matrix] Cholesky(Sigma_i) recovered "
                            f"at attempt {attempt+1} with reg={reg:.1e}"
                        )
                        break
                    except RuntimeError:
                        continue
                else:
                    warnings.warn(
                        "[attention/_compute_kl_matrix] Cholesky(Sigma_i) FAILED "
                        "after 5 attempts, falling back to identity"
                    )
                    L_q = torch.linalg.cholesky(I.expand_as(Sigma_i_reg) + eps * I)
            logdet_q = 2.0 * torch.sum(
                torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=1e-12)), dim=-1
            )
            logdet_term = logdet_p - logdet_q  # (B, N, N)

            # KL divergence for all pairs
            kl_all = 0.5 * (trace_term + mahal_term - K + logdet_term)  # (B, N, N)
            kl_ceil = 100.0
            n_clamped = int((kl_all > kl_ceil).sum().item()) + int((kl_all < 0).sum().item())
            if n_clamped > 0:
                san.record('kl_clamp', count=n_clamped, value=float(kl_all.max().item()))
            kl_all = torch.clamp(kl_all, min=0.0, max=kl_ceil)

            # NaN/Inf safety: replace any residual numerical failures with zero KL
            bad_mask = torch.isnan(kl_all) | torch.isinf(kl_all)
            if bad_mask.any():
                bad_count = bad_mask.sum().item()
                warnings.warn(
                    f"[attention/_compute_kl_matrix] {bad_count} NaN/Inf in KL output, "
                    f"replacing with safe values"
                )
            kl_all = kl_all.nan_to_num(nan=0.0, posinf=kl_ceil, neginf=0.0)

            return kl_all
        except RuntimeError:
            # Cholesky of Sigma_i_reg (or solve_triangular) failed;
            # fall through to eigenvalue-based fallback below.
            pass

    # Fallback: eigenvalue-based KL (no Cholesky, robust to near-singular matrices)
    san.record('cholesky_fallback')
    Sigma_transported_reg = Sigma_transported + jitter * I
    bad_mask_p = ~torch.isfinite(Sigma_transported_reg).all(dim=-1).all(dim=-1)  # (B, N, N)
    bad_mask_q = ~torch.isfinite(Sigma_i_reg).all(dim=-1).all(dim=-1)            # (B, N, N)
    has_bad = bad_mask_p | bad_mask_q
    if has_bad.any():
        n_bad = int(has_bad.sum().item())
        san.record('nan_inf_replacement', count=n_bad)
        safe_cov = jitter * I.expand_as(Sigma_transported_reg)
        Sigma_transported_reg = torch.where(
            bad_mask_p.unsqueeze(-1).unsqueeze(-1), safe_cov, Sigma_transported_reg
        )
        Sigma_i_reg = torch.where(
            bad_mask_q.unsqueeze(-1).unsqueeze(-1), safe_cov, Sigma_i_reg
        )
        mu_transported = torch.nan_to_num(mu_transported, nan=0.0, posinf=0.0, neginf=0.0)
        mu_i = torch.nan_to_num(mu_i, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals_p = torch.linalg.eigvalsh(Sigma_transported_reg)  # (B, N, N, K)
        eigvals_q = torch.linalg.eigvalsh(Sigma_i_reg)            # (B, N, N, K)
    except RuntimeError:
        # Last resort: if eigvalsh still crashes (extreme condition numbers,
        # CUDA internal errors), return zero KL rather than killing training.
        B, N = mu_q.shape[0], mu_q.shape[1]
        return torch.zeros(B, N, N, device=device, dtype=dtype)

    n_eig_clamped = int((eigvals_p < 1e-6).sum().item()) + int((eigvals_q < 1e-6).sum().item())
    if n_eig_clamped > 0:
        san.record('eigval_clamp', count=n_eig_clamped)
    eigvals_p = eigvals_p.clamp(min=1e-6)
    eigvals_q = eigvals_q.clamp(min=1e-6)

    # log|Σ| via eigenvalues
    logdet_p = torch.sum(torch.log(eigvals_p), dim=-1)  # (B, N, N)
    logdet_q = torch.sum(torch.log(eigvals_q), dim=-1)  # (B, N, N)

    # tr(Σ_p⁻¹ Σ_q) and Mahalanobis via pinv (robust to singular matrices)
    Sigma_p_inv = torch.linalg.pinv(Sigma_transported_reg)  # (B, N, N, K, K)
    trace_term = torch.einsum('bijkk->bij',
        torch.einsum('bijkl,bijlm->bijkm', Sigma_p_inv, Sigma_i_reg))
    delta_mu = mu_transported - mu_i
    mahal_term = torch.einsum('bijk,bijk->bij',
        delta_mu, torch.einsum('bijkl,bijl->bijk', Sigma_p_inv, delta_mu))

    kl_all = 0.5 * (trace_term + mahal_term - K + logdet_p - logdet_q)
    kl_ceil = 100.0
    n_clamped = int((kl_all > kl_ceil).sum().item()) + int((kl_all < 0).sum().item())
    if n_clamped > 0:
        san.record('kl_clamp', count=n_clamped, value=float(kl_all.max().item()))
    return torch.clamp(kl_all, min=0.0, max=kl_ceil)


def _transport_gaussian_torch(
    mu: torch.Tensor,         # (K,)
    sigma: torch.Tensor,      # (K, K)
    phi_dst: torch.Tensor,    # (3,)
    phi_src: torch.Tensor,    # (3,)
    generators: torch.Tensor, # (3, K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transport Gaussian via Ω = exp(φ_dst) · exp(-φ_src).

    Returns:
        mu_transported: Ω μ, shape (K,)
        sigma_transported: Ω Σ Ω^T, shape (K, K)
    """
    # Build transport operator Ω
    # X_dst = Σ_a φ_dst[a] * G_a
    X_dst = torch.einsum('a,aij->ij', phi_dst, generators)  # (K, K)
    X_src = torch.einsum('a,aij->ij', phi_src, generators)

    # Matrix exponential (float64 for GL(K) stability)
    K = generators.shape[-1]
    exp_dst, _ = stable_matrix_exp_pair(X_dst)
    _, exp_neg_src = stable_matrix_exp_pair(X_src)
    Omega = exp_dst @ exp_neg_src

    # Transport
    # Use torch.mv for proper matrix-vector product: (K,K) @ (K,) → (K,)
    mu_transported = torch.mv(Omega, mu)
    sigma_transported = Omega @ sigma @ Omega.T

    # Symmetrize
    sigma_transported = 0.5 * (sigma_transported + sigma_transported.T)

    return mu_transported, sigma_transported


def _kl_gaussian_torch(
    mu1: torch.Tensor,     # (K,)
    sigma1: torch.Tensor,  # (K, K)
    mu2: torch.Tensor,     # (K,)
    sigma2: torch.Tensor,  # (K, K)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    KL divergence between two Gaussians: KL(N(μ1,Σ1) || N(μ2,Σ2)).

    Formula:
        KL = 0.5 * [tr(Σ2^{-1} Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1) - K + log|Σ2|/|Σ1|]
    """
    K = mu1.shape[0]
    I_K = torch.eye(K, device=sigma1.device, dtype=sigma1.dtype)

    # Symmetrize inputs to correct any numerical asymmetry from transport
    sigma1 = 0.5 * (sigma1 + sigma1.T)
    sigma2 = 0.5 * (sigma2 + sigma2.T)

    # Cholesky with jitter escalation for robustness.
    # Start at 1e-4 (not eps=1e-8): transported covariances need stronger reg.
    cholesky_success = False
    jitter = max(eps, 1e-4)
    for _attempt in range(4):
        try:
            sigma1_reg = sigma1 + jitter * I_K
            sigma2_reg = sigma2 + jitter * I_K
            L1 = torch.linalg.cholesky(sigma1_reg)
            L2 = torch.linalg.cholesky(sigma2_reg)
            cholesky_success = True
            break
        except RuntimeError:
            jitter *= 10.0

    if cholesky_success:
        san.record('cholesky_jitter', value=jitter)
        # Log determinants: log|Σ| = 2*sum(log(diag(L)))
        logdet1 = 2.0 * torch.sum(torch.log(torch.diag(L1).clamp(min=1e-12)))
        logdet2 = 2.0 * torch.sum(torch.log(torch.diag(L2).clamp(min=1e-12)))

        # Trace term: tr(Σ2^{-1} Σ1)
        Y = torch.linalg.solve_triangular(L2, sigma1_reg, upper=False)
        Z = torch.linalg.solve_triangular(L2.T, Y, upper=True)
        trace_term = torch.trace(Z)

        # Quadratic term: (μ2-μ1)^T Σ2^{-1} (μ2-μ1) = ||L2^{-1} (μ2-μ1)||^2
        delta_mu = mu2 - mu1
        # solve_triangular needs 2D input - reshape (K,) → (K, 1)
        y = torch.linalg.solve_triangular(L2, delta_mu.unsqueeze(-1), upper=False).squeeze(-1)
        quad_term = torch.dot(y, y)
    else:
        # Eigenvalue fallback for non-SPD matrices
        san.record('cholesky_fallback')
        sigma1_reg = sigma1 + jitter * I_K
        sigma2_reg = sigma2 + jitter * I_K
        eigvals1 = torch.linalg.eigvalsh(sigma1_reg).clamp(min=1e-6)
        eigvals2 = torch.linalg.eigvalsh(sigma2_reg).clamp(min=1e-6)
        logdet1 = torch.sum(torch.log(eigvals1))
        logdet2 = torch.sum(torch.log(eigvals2))

        sigma2_inv = torch.linalg.pinv(sigma2_reg)
        trace_term = torch.trace(sigma2_inv @ sigma1_reg)
        delta_mu = mu2 - mu1
        quad_term = delta_mu @ sigma2_inv @ delta_mu

    # Combine
    kl = 0.5 * (trace_term + quad_term - K + logdet2 - logdet1)

    # Numerical safety: clamp to [0, ∞)
    return torch.clamp(kl, min=0.0)


def _compute_kl_matrix_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances (NOT K×K!)
    phi: torch.Tensor,         # (B, N, 3) gauge frames
    generators: torch.Tensor,  # (3, K, K) SO(3) generators
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> torch.Tensor:
    """
    DIAGONAL covariance KL computation - O(N²×K) instead of O(N²×K²).

    For diagonal Gaussians, KL simplifies to:
        KL(N(μ_q, diag(σ_q)) || N(μ_p, diag(σ_p))) =
        0.5 * (sum(σ_q/σ_p) + sum((μ_p - μ_q)²/σ_p) - K + sum(log(σ_p) - log(σ_q)))

    Key simplifications:
    - No Cholesky decomposition (O(K³) → O(K))
    - No matrix inversion
    - No N×N×K×K intermediate tensors!
    - Transport still rotates μ, but σ stays diagonal (approximation)

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K) diagonal variances (positive)
        phi: (B, N, 3) gauge fields
        generators: (3, K, K) SO(3) generators
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    # Squeeze trailing singleton dimensions for robustness
    # (handles case where sigma_q comes in as (B, N, K, 1) instead of (B, N, K))
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # Ensure sigma is positive
    sigma_q = sigma_q.clamp(min=eps)

    # =========================================================================
    # Step 1: Get transport operators (use cached if available)
    # =========================================================================
    if cached_transport is not None and 'Omega' in cached_transport:
        # Use precomputed transport operators (saves 2 matrix exponentials!)
        Omega = cached_transport['Omega']
    else:
        # Compute transport operators
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)

        # Clamp phi_matrix norm to prevent matrix_exp overflow -> NaN
        phi_norm = phi_matrix.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        max_norm = 10.0
        scale = (max_norm / phi_norm).clamp(max=1.0)
        phi_matrix = phi_matrix * scale

        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

        # Re-orthogonalization for SO(K) if requested
        if enforce_orthogonal and K >= 16:
            eye_K = torch.eye(K, device=mu_q.device, dtype=dtype)
            exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
            exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

        # Omega_ij = exp(φ_i) @ exp(-φ_j)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)  # (B, N, N, K, K)

    # =========================================================================
    # Step 2: Transport means (still needed for accurate KL)
    # =========================================================================
    # μ_j^{→i} = Ω_ij @ μ_j
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

    # =========================================================================
    # Step 3: Compute diagonal of transported covariance
    # Σ_j_transported = Ω @ diag(σ_j) @ Ω^T
    # diag(Σ_j_transported)_k = Σ_l Ω_kl² * σ_j[l]
    # This is more accurate than just using σ_j, especially for non-identity Ω
    # =========================================================================
    # σ_j expanded for all pairs - use .clone() after expand to avoid view issues
    sigma_j_orig = sigma_q[:, None, :, :].expand(-1, N, -1, -1).clone()  # (B, N, N, K)
    sigma_i = sigma_q[:, :, None, :].expand(-1, -1, N, -1).clone()  # (B, N, N, K)

    # Compute diagonal of transported covariance: diag(Ω @ diag(σ_j) @ Ω^T)_k = Σ_l Ω_kl² * σ_j[l]
    # Omega: (B, N, N, K, K), sigma_j_orig: (B, N, N, K)
    # Result: (B, N, N, K)
    Omega_sq = Omega ** 2  # (B, N, N, K, K)
    sigma_j_transported_diag = torch.einsum('bijkl,bijl->bijk', Omega_sq, sigma_j_orig)  # (B, N, N, K)

    # Clamp for numerical stability
    sigma_j_transported_diag = sigma_j_transported_diag.clamp(min=eps)

    # =========================================================================
    # Step 4: Diagonal KL divergence (vectorized)
    # KL(q_i || transported q_j) where q_i ~ N(μ_i, diag(σ_i))
    # transported q_j ~ N(μ_j^{→i}, diag(σ_j_transported))
    # =========================================================================
    mu_i = mu_q[:, :, None, :].expand(-1, -1, N, -1).clone()  # (B, N, N, K)

    # Trace term: sum(σ_i / σ_j_transported)
    trace_term = (sigma_i / sigma_j_transported_diag).sum(dim=-1)  # (B, N, N)

    # Mahalanobis term: sum((μ_j^{→i} - μ_i)² / σ_j_transported)
    delta_mu = mu_transported - mu_i  # (B, N, N, K)
    mahal_term = ((delta_mu ** 2) / sigma_j_transported_diag).sum(dim=-1)  # (B, N, N)

    # Log determinant term: sum(log(σ_j_transported) - log(σ_i))
    logdet_term = (torch.log(sigma_j_transported_diag) - torch.log(sigma_i)).sum(dim=-1)  # (B, N, N)

    # Full KL
    kl_all = 0.5 * (trace_term + mahal_term - K + logdet_term)
    kl_ceil = 100.0
    n_clamped = int((kl_all > kl_ceil).sum().item()) + int((kl_all < 0).sum().item())
    if n_clamped > 0:
        san.record('kl_clamp', count=n_clamped, value=float(kl_all.max().item()))
    kl_all = torch.clamp(kl_all, min=0.0, max=kl_ceil)

    return kl_all


# =============================================================================
# CHUNKED KL Computation (Memory-Efficient Full Attention)
# =============================================================================
# Trades compute time for memory by processing N×N attention in C×C chunks.
# Peak memory: O(C²K²) instead of O(N²K²), where C << N
# =============================================================================

def _compute_kl_matrix_chunked(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) belief covariances
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    chunk_size: int = 32,      # Process chunk_size × chunk_size blocks at a time
) -> torch.Tensor:
    """
    CHUNKED KL matrix computation - O(C²K²) memory instead of O(N²K²).

    Strategy:
    1. Precompute exp(φ_i) and exp(-φ_i) for all i: O(N×K²) memory
    2. For each (i_chunk, j_chunk) pair:
       - Compute Omega[i_chunk, j_chunk]: O(C²×K²) memory
       - Compute KL for chunk and write to output
       - Delete intermediate tensors

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) belief covariances
        phi: (B, N, n_gen) gauge fields
        generators: (n_gen, K, K) Lie algebra generators
        chunk_size: Size of chunks to process (smaller = less memory, slower)

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # =========================================================================
    # Step 1: Precompute matrix exponentials for ALL positions
    # This is O(N×K²) memory - much smaller than O(N²×K²)
    # =========================================================================
    # phi: (B, N, n_gen) -> phi_matrix: (B, N, K, K)
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix  # Free memory

    # Precompute Cholesky of query covariances (for logdet_q term)
    I = torch.eye(K, device=device, dtype=dtype)
    sigma_q_reg = sigma_q + eps * I
    L_q_all = torch.linalg.cholesky(sigma_q_reg)  # (B, N, K, K)
    logdet_q_all = 2.0 * torch.sum(
        torch.log(torch.diagonal(L_q_all, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
    )  # (B, N)

    # =========================================================================
    # Step 2: Process in chunks, collecting results for non-in-place assembly
    # =========================================================================
    row_chunks_list = []  # List of (B, n_i, N) tensors
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        # Get exp_phi for i-chunk - use .contiguous() to avoid inplace modification errors
        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()  # (B, n_i, K, K)

        # Get query beliefs for i-chunk - use .contiguous() to create copies
        mu_i = mu_q[:, i_start:i_end].contiguous()          # (B, n_i, K)
        sigma_i = sigma_q[:, i_start:i_end].contiguous()    # (B, n_i, K, K)
        sigma_i_reg = sigma_i + eps * I
        logdet_q_i = logdet_q_all[:, i_start:i_end].contiguous()  # (B, n_i)

        col_chunks_list = []  # List of (B, n_i, n_j) tensors
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Get exp_neg_phi for j-chunk - use .contiguous() to avoid inplace modification errors
            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()  # (B, n_j, K, K)

            # =================================================================
            # Compute Omega for this chunk only: (B, n_i, n_j, K, K)
            # =================================================================
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )  # (B, n_i, n_j, K, K)

            # Get key beliefs for j-chunk - use .contiguous() to create copies
            mu_j = mu_q[:, j_start:j_end].contiguous()        # (B, n_j, K)
            sigma_j = sigma_q[:, j_start:j_end].contiguous()  # (B, n_j, K, K)

            # =================================================================
            # Transport j's beliefs to i's frame
            # =================================================================
            # μ_j^{→i} = Ω_ij @ μ_j
            mu_transported = torch.einsum(
                'bijkl,bjl->bijk', Omega_chunk, mu_j
            )  # (B, n_i, n_j, K)

            # Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_chunk, sigma_j, Omega_chunk.transpose(-1, -2)
            )  # (B, n_i, n_j, K, K)

            # Symmetrize to correct numerical asymmetry from the triple einsum
            Sigma_transported = 0.5 * (Sigma_transported + Sigma_transported.transpose(-1, -2))

            del Omega_chunk  # Free memory immediately

            # =================================================================
            # Compute KL divergence for this chunk
            # =================================================================
            # Expand mu_i and sigma_i for pairwise comparison - use .clone() after expand
            mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()  # (B, n_i, n_j, K)
            sigma_i_exp = sigma_i_reg[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()

            # Try Cholesky with escalating jitter before falling back to loop
            cholesky_success = False
            jitter = eps
            for _attempt in range(4):
                Sigma_transported_reg = Sigma_transported + jitter * I
                try:
                    L_p = torch.linalg.cholesky(Sigma_transported_reg)
                    cholesky_success = True
                    break
                except RuntimeError:
                    jitter *= 10.0

            if cholesky_success:
                san.record('cholesky_jitter', value=jitter)
                # Trace term: tr(Σ_p⁻¹ Σ_q)
                Y = torch.linalg.solve_triangular(L_p, sigma_i_exp, upper=False)
                Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
                trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)  # (B, n_i, n_j)

                # Mahalanobis term
                delta_mu = mu_transported - mu_i_exp  # (B, n_i, n_j, K)
                v = torch.linalg.solve_triangular(
                    L_p, delta_mu.unsqueeze(-1), upper=False
                ).squeeze(-1)
                mahal_term = torch.sum(v ** 2, dim=-1)  # (B, n_i, n_j)

                # Log determinant terms
                logdet_p = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
                )  # (B, n_i, n_j)
                logdet_q_i_exp = logdet_q_i[:, :, None].expand(-1, -1, n_j)  # (B, n_i, n_j)
                logdet_term = logdet_p - logdet_q_i_exp

                # KL divergence for chunk
                kl_chunk = 0.5 * (trace_term + mahal_term - K + logdet_term)
                kl_chunk = torch.clamp(kl_chunk, min=0.0, max=100.0)

                col_chunks_list.append(kl_chunk)

            else:
                san.record('cholesky_fallback')
                # Fallback to element-wise computation if Cholesky fails
                # Collect values in list to preserve autograd graph
                fallback_vals = []
                for b in range(B):
                    batch_vals = []
                    for bi in range(n_i):
                        row_vals = []
                        for bj in range(n_j):
                            kl_ij = _kl_gaussian_torch(
                                mu_q[b, i_start + bi], sigma_q[b, i_start + bi],
                                mu_transported[b, bi, bj], Sigma_transported[b, bi, bj]
                            )
                            row_vals.append(kl_ij.unsqueeze(0))
                        batch_vals.append(torch.cat(row_vals, dim=0))  # (n_j,)
                    fallback_vals.append(torch.stack(batch_vals, dim=0))  # (n_i, n_j)
                col_chunks_list.append(torch.stack(fallback_vals, dim=0))  # (B, n_i, n_j)

            # Explicit cleanup of large intermediate tensors
            del Sigma_transported, mu_transported
            if 'L_p' in dir():
                del L_p
            if 'Y' in dir():
                del Y
            if 'Z' in dir():
                del Z

        # Assemble row from column chunks: (B, n_i, N)
        row_chunks_list.append(torch.cat(col_chunks_list, dim=2))

    # Assemble full matrix from row chunks: (B, N, N)
    return torch.cat(row_chunks_list, dim=1)


def _compute_kl_matrix_diagonal_chunked(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    chunk_size: int = 32,      # Process chunk_size × chunk_size blocks at a time
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> torch.Tensor:
    """
    CHUNKED diagonal covariance KL computation - O(C²K) memory instead of O(N²K).

    For diagonal Gaussians with chunked processing.

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K) diagonal variances (positive)
        phi: (B, N, n_gen) gauge fields
        generators: (n_gen, K, K) generators
        chunk_size: Size of chunks to process
        enforce_orthogonal: If True, apply Newton-Schulz to ensure Ω ∈ SO(K)

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # Ensure sigma is positive
    sigma_q = sigma_q.clamp(min=eps)

    # =========================================================================
    # Step 1: Precompute matrix exponentials for ALL positions
    # =========================================================================
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix

    # Re-orthogonalization for SO(K) if requested
    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    # =========================================================================
    # Step 2: Process in chunks, collecting results for non-in-place assembly
    # =========================================================================
    row_chunks_list = []  # List of (B, n_i, N) tensors
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        # Use .contiguous() to create copies and avoid inplace modification errors
        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()  # (B, n_i, K, K)
        mu_i = mu_q[:, i_start:i_end].contiguous()          # (B, n_i, K)
        sigma_i = sigma_q[:, i_start:i_end].contiguous()    # (B, n_i, K)

        col_chunks_list = []  # List of (B, n_i, n_j) tensors
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Use .contiguous() to create copies and avoid inplace modification errors
            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()  # (B, n_j, K, K)
            mu_j = mu_q[:, j_start:j_end].contiguous()                   # (B, n_j, K)
            sigma_j = sigma_q[:, j_start:j_end].contiguous()             # (B, n_j, K)

            # =================================================================
            # Compute Omega for this chunk: (B, n_i, n_j, K, K)
            # =================================================================
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )

            # =================================================================
            # Transport means
            # =================================================================
            mu_transported = torch.einsum(
                'bijkl,bjl->bijk', Omega_chunk, mu_j
            )  # (B, n_i, n_j, K)

            # =================================================================
            # Compute diagonal of transported covariance
            # diag(Ω @ diag(σ_j) @ Ω^T)_k = Σ_l Ω_kl² * σ_j[l]
            # =================================================================
            # Use .clone() after expand to create copies and avoid view issues
            sigma_j_exp = sigma_j[:, None, :, :].expand(-1, n_i, -1, -1).clone()  # (B, n_i, n_j, K)
            Omega_sq = Omega_chunk ** 2
            sigma_j_transported = torch.einsum(
                'bijkl,bijl->bijk', Omega_sq, sigma_j_exp
            ).clamp(min=eps)  # (B, n_i, n_j, K)

            del Omega_chunk, Omega_sq

            # =================================================================
            # Diagonal KL computation
            # =================================================================
            # Use .clone() after expand to create copies and avoid view issues
            mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()  # (B, n_i, n_j, K)
            sigma_i_exp = sigma_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()  # (B, n_i, n_j, K)

            # Trace term: sum(σ_i / σ_j_transported)
            trace_term = (sigma_i_exp / sigma_j_transported).sum(dim=-1)

            # Mahalanobis term
            delta_mu = mu_transported - mu_i_exp
            mahal_term = ((delta_mu ** 2) / sigma_j_transported).sum(dim=-1)

            # Log determinant term
            logdet_term = (torch.log(sigma_j_transported) - torch.log(sigma_i_exp)).sum(dim=-1)

            # Full KL
            kl_chunk = 0.5 * (trace_term + mahal_term - K + logdet_term)
            # Clamp KL to [0, max] for numerical stability (scale ceiling with K)
            kl_ceil = 100.0
            kl_chunk = torch.clamp(kl_chunk, min=0.0, max=kl_ceil)

            col_chunks_list.append(kl_chunk)

            del sigma_j_transported, mu_transported

        # Assemble row from column chunks: (B, n_i, N)
        row_chunks_list.append(torch.cat(col_chunks_list, dim=2))

    # Assemble full matrix from row chunks: (B, N, N)
    return torch.cat(row_chunks_list, dim=1)


def estimate_chunk_size(
    N: int,
    K: int,
    available_memory_gb: float = 8.0,
    dtype_bytes: int = 4,
    safety_factor: float = 0.5,
    diagonal_covariance: bool = False,
) -> int:
    """
    Estimate optimal chunk size based on available GPU memory.

    Peak memory per chunk (full covariance):
    - Omega: C² × K² × dtype_bytes
    - Sigma_transported: C² × K² × dtype_bytes
    - Intermediate: ~2-3 × C² × K² × dtype_bytes
    Total: ~5 × C² × K² × dtype_bytes

    Peak memory per chunk (diagonal covariance):
    - Omega: C² × K² × dtype_bytes
    - sigma_transported: C² × K × dtype_bytes
    Total: ~2 × C² × K² × dtype_bytes (Omega dominates)

    Args:
        N: Sequence length
        K: Embedding dimension
        available_memory_gb: Available GPU memory in GB
        dtype_bytes: Bytes per element (4 for float32)
        safety_factor: Fraction of memory to use (0.5 = use 50%)
        diagonal_covariance: Whether using diagonal mode

    Returns:
        Recommended chunk size C
    """
    available_bytes = available_memory_gb * 1e9 * safety_factor

    # Memory per chunk: ~5 × C² × K² × dtype_bytes (full) or ~2 × C² × K² (diagonal)
    multiplier = 2.0 if diagonal_covariance else 5.0
    bytes_per_c_squared = multiplier * K * K * dtype_bytes

    # Solve for C: C² ≤ available_bytes / bytes_per_c_squared
    max_c_squared = available_bytes / bytes_per_c_squared
    max_c = int(max_c_squared ** 0.5)

    # Round down to power of 2 for efficiency (optional)
    # chunk_size = 2 ** int(np.log2(max_c)) if max_c >= 2 else 1

    # Clamp to reasonable range
    chunk_size = max(4, min(max_c, N))

    return chunk_size


# =============================================================================
# BLOCK-DIAGONAL KL Computation (Principled & Memory-Efficient)
# =============================================================================
# This is the PRINCIPLED approach: exploit block-diagonal structure of irreps.
# For ρ = n₁ℓ₁ ⊕ n₂ℓ₂ ⊕ ... with block sizes d₁, d₂, ...:
# - Generators G are block-diagonal → Omega = exp(φ·G) is block-diagonal
# - Covariances Σ are block-diagonal
# - KL(q || p) = Σᵢ KL(qᵢ || pᵢ) decomposes additively
#
# Memory: O(N² × Σᵢ dᵢ²) instead of O(N² × K²)
# For K=255 with 75×ℓ₀ + 30×ℓ₁ + 18×ℓ₂: 795 vs 65025 = 82× savings!
# =============================================================================

def _compute_kl_matrix_block_diagonal_diag(
    mu_q: torch.Tensor,             # (B, N, K) belief means
    sigma_q: torch.Tensor,          # (B, N, K) diagonal variances
    phi: torch.Tensor,              # (B, N, n_gen) gauge frames
    generators: torch.Tensor,       # (n_gen, K, K) block-diagonal generators
    irrep_dims: List[int],          # [d₁, d₂, ...] dimensions of each irrep block
    enforce_orthogonal: bool = False,
) -> torch.Tensor:
    """
    Block-diagonal KL for diagonal covariance mode.

    Processes each irrep block separately with small d×d Omega tensors
    instead of one giant K×K Omega. Uses diagonal KL formulas within
    each block (no Cholesky/inv needed).

    Memory: O(N² × max(dᵢ²)) instead of O(N² × K²)
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    assert sum(irrep_dims) == K, f"irrep_dims sum {sum(irrep_dims)} != K={K}"

    sigma_q = sigma_q.clamp(min=eps)
    kl_total = torch.zeros(B, N, N, device=device, dtype=dtype)

    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d

        mu_block = mu_q[:, :, block_start:block_end].contiguous()        # (B, N, d)
        sigma_block = sigma_q[:, :, block_start:block_end].contiguous()  # (B, N, d)
        gen_block = generators[:, block_start:block_end, block_start:block_end].contiguous()

        # Block transport operators: (B, N, N, d, d) - much smaller than (B, N, N, K, K)
        phi_matrix_block = torch.einsum('bna,aij->bnij', phi, gen_block)
        exp_phi_block, exp_neg_phi_block = stable_matrix_exp_pair(phi_matrix_block)

        if enforce_orthogonal and d >= 16:
            eye_d = torch.eye(d, device=device, dtype=dtype)
            exp_phi_block = exp_phi_block @ ((3.0 * eye_d - exp_phi_block.transpose(-1, -2) @ exp_phi_block) / 2.0)
            exp_neg_phi_block = exp_neg_phi_block @ ((3.0 * eye_d - exp_neg_phi_block.transpose(-1, -2) @ exp_neg_phi_block) / 2.0)

        Omega_block = torch.einsum('bikl,bjlm->bijkm', exp_phi_block, exp_neg_phi_block)
        del phi_matrix_block, exp_phi_block, exp_neg_phi_block

        # Transport means
        mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_block)

        # Diagonal covariance transport: σ_t[k] = Σ_l Ω_kl² * σ[l]
        sigma_j_transported = torch.einsum(
            'bijkl,bijkl,bjl->bijk', Omega_block, Omega_block, sigma_block
        ).clamp(min=eps)

        del Omega_block

        # Diagonal KL for this block
        sigma_i = sigma_block[:, :, None, :].expand(-1, -1, N, -1).clone()
        mu_i = mu_block[:, :, None, :].expand(-1, -1, N, -1).clone()
        delta_mu = mu_transported - mu_i

        trace_term = (sigma_i / sigma_j_transported).sum(dim=-1)
        mahal_term = (delta_mu ** 2 / sigma_j_transported).sum(dim=-1)
        logdet_term = (torch.log(sigma_j_transported) - torch.log(sigma_i)).sum(dim=-1)

        kl_block = 0.5 * (trace_term + mahal_term - d + logdet_term)
        kl_block = kl_block.clamp(min=0.0, max=100.0)
        kl_total = kl_total + kl_block

        del sigma_j_transported, mu_transported
        block_start = block_end

    return kl_total


def _compute_kl_matrix_block_diagonal(
    mu_q: torch.Tensor,             # (B, N, K) belief means
    sigma_q: torch.Tensor,          # (B, N, K, K) block-diagonal covariances
    phi: torch.Tensor,              # (B, N, n_gen) gauge frames
    generators: torch.Tensor,       # (n_gen, K, K) block-diagonal generators
    irrep_dims: List[int],          # [d₁, d₂, ...] dimensions of each irrep block
) -> torch.Tensor:
    """
    BLOCK-DIAGONAL KL computation - exploits irrep structure for massive memory savings.

    Since generators and covariances are block-diagonal:
    - Omega_ij = exp(φ_i·G)·exp(-φ_j·G) is block-diagonal
    - Transport Ω @ Σ @ Ω^T is block-diagonal
    - KL = Σ_blocks KL_block (additive decomposition)

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) block-diagonal covariances
        phi: (B, N, n_gen) gauge fields
        generators: (n_gen, K, K) block-diagonal generators
        irrep_dims: List of block dimensions [d₁, d₂, d₃, ...]
                   Must sum to K.

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    # Validate irrep_dims
    assert sum(irrep_dims) == K, f"irrep_dims sum {sum(irrep_dims)} != K={K}"

    # Initialize accumulator (non-in-place addition preserves autograd graph)
    kl_total = torch.zeros(B, N, N, device=device, dtype=dtype)

    # =========================================================================
    # Process each irrep block separately
    # =========================================================================
    block_start = 0
    for block_idx, d in enumerate(irrep_dims):
        block_end = block_start + d

        # Extract block from beliefs - use .contiguous() to create copies and avoid
        # inplace modification errors during backward pass
        mu_block = mu_q[:, :, block_start:block_end].contiguous()  # (B, N, d)
        sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()  # (B, N, d, d)

        # Extract block from generators - contiguous for consistency
        gen_block = generators[:, block_start:block_end, block_start:block_end].contiguous()  # (n_gen, d, d)

        # =====================================================================
        # Compute block-wise transport operators
        # =====================================================================
        # phi_matrix_block: (B, N, d, d)
        phi_matrix_block = torch.einsum('bna,aij->bnij', phi, gen_block)
        exp_phi_block, exp_neg_phi_block = stable_matrix_exp_pair(phi_matrix_block)

        # Omega_block: (B, N, N, d, d) - MUCH smaller than (B, N, N, K, K)!
        Omega_block = torch.einsum(
            'bikl,bjlm->bijkm',
            exp_phi_block, exp_neg_phi_block
        )

        del phi_matrix_block, exp_phi_block, exp_neg_phi_block

        # =====================================================================
        # Transport means and covariances for this block
        # =====================================================================
        # μ_block_transported: (B, N, N, d)
        mu_block_transported = torch.einsum(
            'bijkl,bjl->bijk', Omega_block, mu_block
        )

        # Σ_block_transported: (B, N, N, d, d)
        sigma_block_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega_block, sigma_block, Omega_block.transpose(-1, -2)
        )
        # Symmetrize to fix numerical asymmetry from triple einsum
        sigma_block_transported = 0.5 * (sigma_block_transported + sigma_block_transported.transpose(-1, -2))

        del Omega_block

        # =====================================================================
        # Compute KL for this block
        # =====================================================================
        I_block = torch.eye(d, device=device, dtype=dtype)

        # Expand for pairwise comparison - use .clone() after expand to create copies
        # and avoid view-related gradient issues
        mu_block_i = mu_block[:, :, None, :].expand(-1, -1, N, -1).clone()  # (B, N, N, d)
        sigma_block_i = sigma_block[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, d, d)

        sigma_block_i_reg = sigma_block_i + eps * I_block
        sigma_block_transported_reg = sigma_block_transported + eps * I_block

        # Try Cholesky with escalating jitter before falling back to diagonal approx
        cholesky_succeeded = False
        jitter = 1e-4
        for _attempt in range(4):
            try:
                jittered_transported = sigma_block_transported_reg + jitter * I_block
                jittered_i = sigma_block_i_reg + jitter * I_block

                L_p = torch.linalg.cholesky(jittered_transported)
                L_q = torch.linalg.cholesky(jittered_i)

                # Trace term: tr(Σ_p⁻¹ Σ_q)
                Y = torch.linalg.solve_triangular(L_p, jittered_i, upper=False)
                Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
                trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

                # Mahalanobis term
                delta_mu = mu_block_transported - mu_block_i
                v = torch.linalg.solve_triangular(
                    L_p, delta_mu.unsqueeze(-1), upper=False
                ).squeeze(-1)
                mahal_term = torch.sum(v ** 2, dim=-1)

                # Log determinant terms
                logdet_p = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
                )
                logdet_q = 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
                )

                # KL for this block
                kl_block = 0.5 * (trace_term + mahal_term - d + logdet_p - logdet_q)
                kl_block = torch.clamp(kl_block, min=0.0, max=100.0)

                # ACCUMULATE to total KL (additive decomposition)
                # Use non-in-place addition to preserve autograd graph
                kl_total = kl_total + kl_block
                cholesky_succeeded = True
                san.record('cholesky_jitter', value=jitter)
                break
            except RuntimeError:
                jitter *= 10.0  # escalate: 1e-4 -> 1e-3 -> 1e-2 -> 1e-1

        if not cholesky_succeeded:
            san.record('cholesky_fallback')
            # Cholesky failed (ill-conditioned covariance) - use diagonal KL approximation.
            # CRITICAL: The fallback must depend on phi through the transported
            # quantities to preserve the autograd graph. A constant fallback would
            # break torch.autograd.grad() in the caller.
            sigma_diag_transported = torch.diagonal(
                sigma_block_transported_reg, dim1=-2, dim2=-1
            ).clamp(min=eps)  # (B, N, N, d)
            sigma_diag_i = torch.diagonal(
                sigma_block_i_reg, dim1=-2, dim2=-1
            ).clamp(min=eps)  # (B, N, N, d)
            delta_mu = mu_block_transported - mu_block_i  # (B, N, N, d)

            trace_term = (sigma_diag_i / sigma_diag_transported).sum(dim=-1)
            mahal_term = ((delta_mu ** 2) / sigma_diag_transported).sum(dim=-1)
            logdet_term = (
                torch.log(sigma_diag_transported) - torch.log(sigma_diag_i)
            ).sum(dim=-1)

            kl_block = 0.5 * (trace_term + mahal_term - d + logdet_term)
            kl_block = torch.clamp(kl_block, min=0.0, max=100.0)
            kl_total = kl_total + kl_block

        # Cleanup
        del sigma_block_transported, mu_block_transported

        block_start = block_end

    return kl_total


def _compute_kl_matrix_block_diagonal_chunked(
    mu_q: torch.Tensor,             # (B, N, K) belief means
    sigma_q: torch.Tensor,          # (B, N, K, K) block-diagonal covariances
    phi: torch.Tensor,              # (B, N, n_gen) gauge frames
    generators: torch.Tensor,       # (n_gen, K, K) block-diagonal generators
    irrep_dims: List[int],          # [d₁, d₂, ...] dimensions of each irrep block
    chunk_size: int = 64,           # Process N×N in chunks
) -> torch.Tensor:
    """
    BLOCK-DIAGONAL + CHUNKED KL computation - maximum memory efficiency.

    Combines:
    1. Block-diagonal structure: O(Σᵢ dᵢ²) instead of O(K²) per pair
    2. Chunking: O(C²) pairs at a time instead of O(N²)

    Total memory: O(C² × max(dᵢ²)) instead of O(N² × K²)

    Args:
        mu_q: (B, N, K) belief means
        sigma_q: (B, N, K, K) block-diagonal covariances
        phi: (B, N, n_gen) gauge fields
        generators: (n_gen, K, K) block-diagonal generators
        irrep_dims: List of block dimensions [d₁, d₂, d₃, ...]
        chunk_size: Process chunk_size × chunk_size position pairs at a time

    Returns:
        kl_matrix: (B, N, N) KL divergence matrix with autograd graph intact
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype
    eps = 1e-6

    assert sum(irrep_dims) == K, f"irrep_dims sum {sum(irrep_dims)} != K={K}"

    # =========================================================================
    # Precompute block-wise matrix exponentials for all positions
    # Memory: O(N × Σᵢ dᵢ²) - manageable
    # =========================================================================
    block_exp_phi = []      # List of (B, N, dᵢ, dᵢ) tensors
    block_exp_neg_phi = []  # List of (B, N, dᵢ, dᵢ) tensors

    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d
        gen_block = generators[:, block_start:block_end, block_start:block_end]

        phi_matrix_block = torch.einsum('bna,aij->bnij', phi, gen_block)  # (B, N, d, d)
        exp_phi_blk, exp_neg_phi_blk = stable_matrix_exp_pair(phi_matrix_block)
        block_exp_phi.append(exp_phi_blk)
        block_exp_neg_phi.append(exp_neg_phi_blk)

        block_start = block_end

    # =========================================================================
    # Process position pairs in chunks, accumulate KL across blocks
    # Collect chunks in lists for non-in-place assembly (preserves autograd)
    # =========================================================================
    row_chunks_list = []  # List of (B, n_i, N) tensors
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        col_chunks_list = []  # List of (B, n_i, n_j) tensors
        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            # Accumulate KL across blocks using non-in-place addition
            kl_chunk = torch.zeros(B, n_i, n_j, device=device, dtype=dtype)

            # Process each irrep block
            block_start = 0
            for block_idx, d in enumerate(irrep_dims):
                block_end = block_start + d

                # Get precomputed exponentials for this chunk - use .contiguous() to avoid
                # inplace modification errors during backward pass
                exp_phi_i = block_exp_phi[block_idx][:, i_start:i_end].contiguous()      # (B, n_i, d, d)
                exp_neg_phi_j = block_exp_neg_phi[block_idx][:, j_start:j_end].contiguous()  # (B, n_j, d, d)

                # Compute Omega for this block-chunk: (B, n_i, n_j, d, d)
                Omega_block = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)

                # Extract beliefs for this block-chunk - use .contiguous() to create copies
                mu_i = mu_q[:, i_start:i_end, block_start:block_end].contiguous()  # (B, n_i, d)
                mu_j = mu_q[:, j_start:j_end, block_start:block_end].contiguous()  # (B, n_j, d)
                sigma_i = sigma_q[:, i_start:i_end, block_start:block_end, block_start:block_end].contiguous()
                sigma_j = sigma_q[:, j_start:j_end, block_start:block_end, block_start:block_end].contiguous()

                # Transport
                mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_block, mu_j)
                sigma_transported = torch.einsum(
                    'bijkl,bjlm,bijmn->bijkn',
                    Omega_block, sigma_j, Omega_block.transpose(-1, -2)
                )
                # Symmetrize to fix numerical asymmetry from triple einsum
                sigma_transported = 0.5 * (sigma_transported + sigma_transported.transpose(-1, -2))

                del Omega_block

                # Compute KL for this block - use .clone() after expand to create copies
                I_d = torch.eye(d, device=device, dtype=dtype)
                mu_i_exp = mu_i[:, :, None, :].expand(-1, -1, n_j, -1).clone()
                sigma_i_exp = sigma_i[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()

                sigma_i_reg = sigma_i_exp + eps * I_d
                sigma_transported_reg = sigma_transported + eps * I_d

                # Try Cholesky with escalating jitter before falling back
                cholesky_succeeded = False
                jitter = 1e-4
                for _attempt in range(4):
                    try:
                        jittered_transported = sigma_transported_reg + jitter * I_d
                        jittered_i = sigma_i_reg + jitter * I_d

                        L_p = torch.linalg.cholesky(jittered_transported)
                        L_q = torch.linalg.cholesky(jittered_i)

                        # Trace term
                        Y = torch.linalg.solve_triangular(L_p, jittered_i, upper=False)
                        Z = torch.linalg.solve_triangular(L_p.transpose(-1, -2), Y, upper=True)
                        trace_term = torch.diagonal(Z, dim1=-2, dim2=-1).sum(dim=-1)

                        # Mahalanobis term
                        delta_mu = mu_transported - mu_i_exp
                        v = torch.linalg.solve_triangular(
                            L_p, delta_mu.unsqueeze(-1), upper=False
                        ).squeeze(-1)
                        mahal_term = torch.sum(v ** 2, dim=-1)

                        # Log det terms
                        logdet_p = 2.0 * torch.sum(
                            torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
                        )
                        logdet_q = 2.0 * torch.sum(
                            torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1
                        )

                        # Accumulate block KL (non-in-place to preserve autograd graph)
                        kl_block = 0.5 * (trace_term + mahal_term - d + logdet_p - logdet_q)
                        kl_chunk = kl_chunk + torch.clamp(kl_block, min=0.0, max=100.0)
                        cholesky_succeeded = True
                        san.record('cholesky_jitter', value=jitter)
                        break
                    except RuntimeError:
                        jitter *= 10.0  # escalate: 1e-4 -> 1e-3 -> 1e-2 -> 1e-1

                if not cholesky_succeeded:
                    san.record('cholesky_fallback')
                    # Cholesky failed - use diagonal KL approximation.
                    # Must depend on phi through transported quantities to
                    # preserve the autograd graph (a constant would break
                    # torch.autograd.grad in the caller).
                    sigma_diag_transported = torch.diagonal(
                        sigma_transported_reg, dim1=-2, dim2=-1
                    ).clamp(min=eps)  # (B, n_i, n_j, d)
                    sigma_diag_i = torch.diagonal(
                        sigma_i_reg, dim1=-2, dim2=-1
                    ).clamp(min=eps)  # (B, n_i, n_j, d)
                    delta_mu = mu_transported - mu_i_exp  # (B, n_i, n_j, d)

                    trace_term = (sigma_diag_i / sigma_diag_transported).sum(dim=-1)
                    mahal_term = ((delta_mu ** 2) / sigma_diag_transported).sum(dim=-1)
                    logdet_term = (
                        torch.log(sigma_diag_transported) - torch.log(sigma_diag_i)
                    ).sum(dim=-1)

                    kl_block = 0.5 * (trace_term + mahal_term - d + logdet_term)
                    kl_chunk = kl_chunk + torch.clamp(
                        kl_block, min=0.0, max=100.0
                    )

                del sigma_transported, mu_transported
                block_start = block_end

            col_chunks_list.append(kl_chunk)

        # Assemble row from column chunks: (B, n_i, N)
        row_chunks_list.append(torch.cat(col_chunks_list, dim=2))

    # Assemble full matrix from row chunks: (B, N, N)
    return torch.cat(row_chunks_list, dim=1)


# =============================================================================
# Message Aggregation with Parallel Transport
# =============================================================================

def aggregate_messages(
    mu_q: torch.Tensor,         # (B, N, K)
    sigma_q: torch.Tensor,      # (B, N, K, K) or (B, N, K) if diagonal
    phi: torch.Tensor,          # (B, N, 3)
    beta: torch.Tensor,         # (B, N, N) attention weights
    generators: torch.Tensor,   # (3, K, K)
    aggregate_mode: str = 'mean_only',  # 'mean_only' or 'full_distribution'
    diagonal_covariance: bool = False,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Aggregate messages with GL(K) metric correction.

    For SO(K): m_i = Σ_j β_ij Ω_ij μ_j            (Ω orthogonal)
    For GL(K): m_i = Σ_j β_ij Ω_ij^{-T} μ_j       (metric factor (ΩΩ^T)^{-1})

    The variational gradient ∂F/∂μ_i includes a metric factor (ΩΩ^T)^{-1},
    so the principled message from j uses Ω^{-T} μ_j rather than Ω μ_j.
    For SO(K), Ω^{-T} = Ω identically (orthogonal matrices). For GL(K),
    the correction is non-trivial and required for faithful variational inference.

    0D Version: Simple weighted sum over agents, no spatial integration!

    Two modes:
        1. 'mean_only': Only aggregate means (faster)
           Returns: (messages, None)

        2. 'full_distribution': Aggregate full distributions
           Returns: (mu_aggregated, sigma_aggregated)
           Uses mixture of Gaussians approximation

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief covariances (B, N, K, K)
        phi: Gauge frames (B, N, 3)
        beta: Attention weights (B, N, N) - SCALARS, not fields!
        generators: SO(3) generators (3, K, K)
        aggregate_mode: 'mean_only' or 'full_distribution'
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators()

    Returns:
        mu_agg: Aggregated means (B, N, K)
        sigma_agg: Aggregated covariances (B, N, K, K) or None

    Example:
        >>> mu_agg, _ = aggregate_messages(mu, sigma, phi, beta, G, mode='mean_only')
        >>> # mu_agg[b, i] = Σ_j β[b,i,j] * Ω_ij[μ[b,j]]
    """
    batch_size, num_agents, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # =========================================================================
    # VECTORIZED aggregation - no Python loops!
    # =========================================================================

    # Step 1: Get transport operators (use cached if available)
    if cached_transport is not None and 'Omega' in cached_transport:
        # Use precomputed transport operators (saves 2 matrix exponentials!)
        Omega = cached_transport['Omega']
    else:
        # Compute all pairwise transport operators Ω_ij = exp(φ_i) exp(-φ_j)
        phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
        exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

        # Omega_ij = exp(φ_i) @ exp(-φ_j)  ->  (B, N, N, K, K)
        Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

    # Step 1.5: GL(K) metric correction for message transport
    # The variational gradient ∂F/∂μ_i includes a metric factor (ΩΩ^T)^{-1},
    # so the principled message from j uses Ω^{-T} μ_j rather than Ω μ_j.
    # For SO(K): Ω is orthogonal ⟹ Ω^{-T} = Ω (no correction needed).
    # For GL(K): Ω_ij^{-1} = Ω_ji, so Ω_ij^{-T} = Ω_ji^T — no inversion required.
    _is_skew = torch.allclose(
        generators + generators.transpose(-1, -2),
        torch.zeros_like(generators), atol=1e-5
    )
    if _is_skew:
        # SO(K): orthogonal transport, no metric correction
        Omega_msg = Omega
    else:
        # GL(K): metric-corrected transport Ω^{-T} = Ω_ji^T
        Omega_msg = Omega.permute(0, 2, 1, 3, 4).transpose(-1, -2)

    # Step 2: Transport all means (with GL(K) metric correction if applicable)
    # SO(K): μ_j^{→i} = Ω_ij @ μ_j
    # GL(K): μ_j^{→i} = Ω_ij^{-T} @ μ_j  (variational metric factor)
    mu_transported = torch.einsum('bijkl,bjl->bijk', Omega_msg, mu_q)  # (B, N, N, K)

    # Step 3: Weighted aggregation: m_i = Σ_j β_ij * μ_j^{→i}
    # beta: (B, N, N), mu_transported: (B, N, N, K)
    mu_aggregated = torch.einsum('bij,bijk->bik', beta, mu_transported)  # (B, N, K)

    # Step 4: Covariance aggregation (if requested)
    if aggregate_mode == 'full_distribution':
        B, N, K = mu_q.shape
        if diagonal_covariance:
            # DIAGONAL MODE: sigma_q is (B, N, K)
            # IMPORTANT: Transport DOES change diagonal covariance!
            # Σ_j^{→i} = Ω_ij @ diag(σ_j) @ Ω_ij^T is generally FULL, but we
            # take its diagonal for the output to stay in diagonal mode.
            # This is the CORRECT approximation: diag(Ω @ diag(σ) @ Ω^T)

            # Squeeze trailing singleton dimensions for robustness
            sigma_q_diag = sigma_q
            while sigma_q_diag.dim() > 3 and sigma_q_diag.shape[-1] == 1:
                sigma_q_diag = sigma_q_diag.squeeze(-1)

            # Expand diagonal to full covariance for transport: (B, N, K, K)
            sigma_full = torch.diag_embed(sigma_q_diag)  # (B, N, K, K)

            # Transport covariances (uses Omega_msg for GL(K) metric correction)
            # SO(K): Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
            # GL(K): Σ_j^{→i} = Ω_ij^{-T} @ Σ_j @ Ω_ij^{-1}
            Sigma_transported_full = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_msg, sigma_full, Omega_msg.transpose(-1, -2)
            )  # (B, N, N, K, K)

            # Extract diagonal of transported covariance: (B, N, N, K)
            # Use .clone() to avoid view-related gradient issues
            sigma_transported_diag = torch.diagonal(Sigma_transported_full, dim1=-2, dim2=-1).clone()

            # Second moment: E[x²] = diag(Σ_transported) + μ²
            second_moment = sigma_transported_diag + mu_transported ** 2  # (B, N, N, K)

            # Weighted sum
            sigma_aggregated = torch.einsum('bij,bijk->bik', beta, second_moment)

            # Complete mixture variance: Var = E[x²] - E[x]²
            sigma_aggregated = sigma_aggregated - mu_aggregated ** 2  # (B, N, K)
            # Clamp to ensure positivity (numerical cancellation can produce negatives)
            n_neg = int((sigma_aggregated < 0).sum().item())
            if n_neg > 0:
                san.record('sigma_clamp_neg', count=n_neg)
            sigma_aggregated = sigma_aggregated.clamp(min=1e-6)
        else:
            # FULL COVARIANCE MODE: sigma_q is (B, N, K, K)
            # Transport covariances (uses Omega_msg for GL(K) metric correction)
            # SO(K): Σ_j^{→i} = Ω_ij @ Σ_j @ Ω_ij^T
            # GL(K): Σ_j^{→i} = Ω_ij^{-T} @ Σ_j @ Ω_ij^{-1}
            Sigma_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_msg, sigma_q, Omega_msg.transpose(-1, -2)
            )  # (B, N, N, K, K)

            # Second moment: E[x x^T] = Σ + μ μ^T
            # (B, N, N, K, K) + (B, N, N, K, 1) @ (B, N, N, 1, K)
            second_moment = Sigma_transported + torch.einsum(
                'bijk,bijl->bijkl', mu_transported, mu_transported
            )

            # Weighted sum of second moments
            # beta: (B, N, N), second_moment: (B, N, N, K, K)
            sigma_aggregated = torch.einsum('bij,bijkl->bikl', beta, second_moment)

            # Complete mixture variance: Σ_mix = E[x x^T] - E[x] E[x]^T
            sigma_aggregated = sigma_aggregated - torch.einsum(
                'bik,bil->bikl', mu_aggregated, mu_aggregated
            )
            # Ensure SPD: add small positive diagonal for numerical stability
            eps_eye = 1e-6 * torch.eye(sigma_aggregated.shape[-1],
                                        device=sigma_aggregated.device,
                                        dtype=sigma_aggregated.dtype)
            sigma_aggregated = sigma_aggregated + eps_eye
    else:
        sigma_aggregated = None

    return mu_aggregated, sigma_aggregated


# =============================================================================
# Multi-Head Attention with Irrep Structure
# =============================================================================

class IrrepMultiHeadAttention(nn.Module):
    """
    Multi-head attention where heads correspond to SO(3) irreducible representations.

    Standard Transformer:
        - n_heads separate (W_Q, W_K, W_V) projections
        - Head dim = embed_dim / n_heads
        - Free parameter choices

    Gauge Transformer:
        - NO W_Q, W_K! (attention from KL divergence)
        - Heads = irrep blocks (ℓ0, ℓ1, ℓ2, ℓ3, ...)
        - Constrained by SO(3) symmetry
        - Each irrep transforms with specific rule under gauge

    Irrep Decomposition:
        K = Σ_ℓ multiplicity_ℓ × dim_ℓ

    Example (96-dim embedding):
        K = 12×1 + 7×3 + 5×5 + 2×7 = 96
        ℓ0: 12 scalar channels (gauge-invariant)
        ℓ1: 7 vector channels (transform as vectors)
        ℓ2: 5 rank-2 tensor channels
        ℓ3: 2 rank-3 tensor channels
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        kappa_beta: float,
        epsilon: float = 1e-8,
        aggregate_mode: str = 'mean_only',
        diagonal_covariance: bool = False,
        attention_pattern: str = 'full',  # Kept for API compat; only 'full' supported
        attention_window: int = 64,  # Unused, kept for API compat
        gauge_group: str = 'SO3',  # 'SO3' or 'SON'
        gauge_dim: int = 3,        # N for SO(N) - only used when gauge_group='SON'
        global_generators: Optional[torch.Tensor] = None,  # (n_gen, K, K) for SO(N) mode
        use_identity_transport: bool = False,  # If True, Ω_ij = I (no gauge transport)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
        per_head_kappa: bool = False,  # If True, learn separate κ_h per head
        use_output_projection: bool = False,  # If True, add W_O linear projection after heads
        irrep_dims_override: Optional[List[int]] = None,  # Override block dims (for cross-head coupling)
        use_rope: bool = False,  # If True, apply RoPE rotations to μ before KL computation
        rope_base: float = 10000.0,  # RoPE frequency base
    ):
        """
        Initialize irrep-structured multi-head attention.

        Args:
            embed_dim: Total embedding dimension K
            irrep_spec: List of (label, multiplicity, dim) tuples
                Example: [('ℓ0', 12, 1), ('ℓ1', 7, 3), ...]
            kappa_beta: Temperature for attention softmax
            epsilon: Numerical stability constant
            aggregate_mode: 'mean_only' or 'full_distribution'
            diagonal_covariance: If True, sigma is (B,N,K) diagonal variances
            attention_pattern: Only 'full' is supported (kept for API compatibility)
            attention_window: Unused, kept for API compatibility
            gauge_group: 'SO3' for SO(3) Wigner D-matrices, 'SON' for SO(N) fundamentals
            gauge_dim: N for SO(N) mode - determines generator structure
            global_generators: Pre-computed generators for SO(N) mode (n_gen, K, K)
                              Required when gauge_group='SON'
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                This prevents attention collapse since KL(q_i||q_i)=0
                                always makes self-attention the most attractive.
            per_head_kappa: If True, learn separate κ_h per head (log-parameterized).
                           Each head gets its own attention temperature, enabling
                           different heads to cluster at different scales.
            use_output_projection: If True, add a learned W_O ∈ R^{K×K} linear
                                  projection after concatenating head outputs.
                                  Enables cross-head information mixing.
        """
        super().__init__()
        self.diagonal_covariance = diagonal_covariance
        self.embed_dim = embed_dim
        self.irrep_spec = irrep_spec
        self.kappa_beta = kappa_beta
        self.epsilon = epsilon
        self.aggregate_mode = aggregate_mode
        self.attention_pattern = attention_pattern
        self.attention_window = attention_window
        self.use_identity_transport = use_identity_transport
        self.mask_self_attention = mask_self_attention
        self.enforce_orthogonal = enforce_orthogonal
        self.use_rope = use_rope
        self.rope_base = rope_base

        # Build irrep block structure
        self.irrep_dims = []
        self.irrep_labels = []
        total_dim = 0

        # =================================================================
        # GL(K) MODE: Single-head or multi-head GL(K) attention
        # =================================================================
        # GL(K) has no natural irrep decomposition like SO(K). However, we can
        # still use multi-head attention via block-diagonal structure:
        #   GL(d_head)^H ⊂ GL(K)
        # where d_head = K/H and each head has its own GL(d_head) gauge.
        #
        # Determine number of GL(K) heads from irrep_spec:
        #   - [('fund', H, d_head)] → H heads of dimension d_head
        #   - [('full', 1, K)] → single head (original behavior)
        if gauge_group == 'GLK':
            # Check if multi-head is requested via irrep_spec
            if len(irrep_spec) == 1 and irrep_spec[0][0] == 'full':
                # Single-head GL(K): original behavior
                self.irrep_dims = [embed_dim]
                self.irrep_labels = ['full']
                total_dim = embed_dim
                self.glk_multihead = False
                print(f"[GL(K) mode] Single-head attention: dim={embed_dim}, generators={embed_dim}²={embed_dim**2}")
            else:
                # Multi-head GL(K): block-diagonal structure
                # Parse irrep_spec as [(label, n_heads, d_head)]
                label, n_heads, d_head = irrep_spec[0]
                if n_heads * d_head != embed_dim:
                    raise ValueError(
                        f"GL(K) multi-head: n_heads({n_heads}) × d_head({d_head}) = {n_heads * d_head} "
                        f"must equal embed_dim={embed_dim}"
                    )

                if irrep_dims_override is not None:
                    # Cross-head coupling: use super-block dims from merge_coupled_heads.
                    # Generators have been reordered so super-blocks are contiguous.
                    self.irrep_dims = list(irrep_dims_override)
                    self.irrep_labels = [f'glk_superblock_{i}' for i in range(len(irrep_dims_override))]
                    self.glk_multihead = True
                    self.glk_d_head = d_head
                    self.glk_cross_head = True
                    print(f"[GL(K) cross-head] super-blocks={irrep_dims_override}, "
                          f"d_head={d_head}")
                else:
                    self.irrep_dims = [d_head] * n_heads
                    self.irrep_labels = [f'glk_head_{h}' for h in range(n_heads)]
                    self.glk_multihead = True
                    self.glk_d_head = d_head
                    self.glk_cross_head = False
                    print(f"[GL(K) multi-head] {n_heads} heads × GL({d_head}), generators per head={d_head}²={d_head**2}")
        else:
            # SO(3) / SO(N) mode: Use irrep decomposition
            for label, multiplicity, dim in irrep_spec:
                for _ in range(multiplicity):
                    self.irrep_dims.append(dim)
                    self.irrep_labels.append(label)
                    total_dim += dim

            # Pad to embed_dim if needed - add SCALAR heads (dim=1), not one big head
            if total_dim < embed_dim:
                padding = embed_dim - total_dim
                for _ in range(padding):
                    self.irrep_dims.append(1)  # Each padding is a scalar head
                    self.irrep_labels.append('ℓ0_pad')
                total_dim = embed_dim
            elif total_dim > embed_dim:
                raise ValueError(
                    f"Irrep spec sums to {total_dim}, exceeds embed_dim={embed_dim}"
                )

        self.n_heads = len(self.irrep_dims)
        self.total_dim = total_dim

        # Store gauge group info
        self.gauge_group = gauge_group
        self.gauge_dim = gauge_dim

        # =================================================================
        # Create generators for each head dimension
        # =================================================================
        # SO(3) mode:
        #   - For ℓ=0 (dim=1): Zero generator (scalars don't transform)
        #   - For ℓ≥1 (dim=3,5,7,...): Proper Wigner D-matrix generators
        #
        # SO(N) mode:
        #   - Use global generators (block-diagonal structure)
        #   - Extract appropriate blocks for each head
        #
        # Store as a list of buffers (can't use ParameterList since non-trainable)
        self.head_generators = nn.ModuleList()  # Will hold generator-holding modules

        # Track cumulative dimension for extracting blocks in SO(N) mode
        cum_dim = 0

        for head_idx, dim in enumerate(self.irrep_dims):
            if gauge_group == 'SO3':
                # SO(3) mode: Create Wigner D-matrix generators per head
                if dim == 1:
                    # Scalar irrep: zero generator (no transformation)
                    gen = torch.zeros(3, 1, 1)
                elif dim % 2 == 1 and dim >= 3:
                    # Proper SO(3) irrep: use Wigner D-matrix generators
                    gen_np = generate_so3_generators(dim)
                    gen = torch.from_numpy(gen_np).float()
                else:
                    # Even dimension - not a valid SO(3) irrep!
                    raise ValueError(
                        f"Head {head_idx} has dim={dim}, which is not a valid SO(3) irrep dimension. "
                        f"SO(3) irreps must have odd dimensions (1, 3, 5, 7, ...). "
                        f"For even dimensions, use gauge_group='SON' with appropriate gauge_dim."
                    )
            elif gauge_group == 'GLK':
                # GL(K) mode: Single-head or multi-head
                if global_generators is None:
                    raise ValueError(
                        f"GL(K) mode requires global_generators to be provided."
                    )

                if hasattr(self, 'glk_multihead') and self.glk_multihead:
                    # Multi-head GL(K) (standard or cross-coupled):
                    # Extract the dim×dim spatial block from ALL generators.
                    # phi has coefficients for all n_gen generators, so we keep
                    # the full first axis to match phi's last dimension.
                    # Generators that are zero in this spatial block contribute
                    # nothing to the Lie algebra element Σ_a φ^a G_a[block].
                    gen = global_generators[:, cum_dim:cum_dim+dim, cum_dim:cum_dim+dim].clone()
                else:
                    # Single-head GL(K): Use full K² generators on entire space
                    gen = global_generators.clone()  # (K², K, K)
            else:
                # SO(N) mode: Extract block from global generators
                if global_generators is None:
                    raise ValueError(
                        f"SO(N) mode requires global_generators to be provided. "
                        f"Pass generators from model.py to IrrepMultiHeadAttention."
                    )
                # Extract the block for this head from the global block-diagonal generators
                # global_generators: (n_gen, K, K) where K = embed_dim
                n_gen = global_generators.shape[0]
                gen = global_generators[:, cum_dim:cum_dim+dim, cum_dim:cum_dim+dim].clone()
                # gen shape: (n_gen, dim, dim)

            # Wrap in a module to register as buffer
            gen_holder = nn.Module()
            gen_holder.register_buffer('gen', gen)
            self.head_generators.append(gen_holder)

            cum_dim += dim

        # =================================================================
        # Per-head κ (learned temperature per head)
        # =================================================================
        self.per_head_kappa = per_head_kappa
        if per_head_kappa:
            # Log-parameterized for positivity: κ_h = exp(log_κ_h)
            self.log_kappa = nn.Parameter(
                torch.full((self.n_heads,), math.log(max(kappa_beta, 1e-6)))
            )
            print(f"  Per-head κ: {self.n_heads} learnable temperatures (init={kappa_beta:.3f})")
        else:
            self.log_kappa = None

        # =================================================================
        # W_O output projection (optional cross-head mixing)
        # =================================================================
        self.use_output_projection = use_output_projection
        if use_output_projection:
            self.output_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            print(f"  W_O output projection: {embed_dim}×{embed_dim} = {embed_dim**2} params")
        else:
            self.output_proj = None

        # Print attention configuration
        if gauge_group == 'GLK':
            # GL(K) mode: single head with full generators
            n_gen = global_generators.shape[0] if global_generators is not None else embed_dim**2
            print(f"[GL(K) Attention] Single head, dim={embed_dim}, n_generators={n_gen}")
            print(f"  → Full GL({embed_dim}) transport on entire embedding space")
        else:
            # SO(3) / SO(N) mode: count scalar vs non-scalar heads
            n_scalar_heads = sum(1 for d in self.irrep_dims if d == 1)
            n_gauge_active_heads = self.n_heads - n_scalar_heads
            scalar_channels = sum(d for d in self.irrep_dims if d == 1)

            print(f"IrrepMultiHeadAttention: {self.n_heads} heads, dims={self.irrep_dims}")

            # Warn if a large fraction of channels are gauge-invariant
            if n_scalar_heads > 0:
                import warnings
                scalar_fraction = scalar_channels / embed_dim
                if scalar_fraction > 0.5:
                    warnings.warn(
                        f"IrrepMultiHeadAttention: {n_scalar_heads}/{self.n_heads} heads are ℓ=0 (scalar), "
                        f"comprising {scalar_channels}/{embed_dim} = {100*scalar_fraction:.1f}% of channels. "
                        f"Scalar channels are GAUGE-INVARIANT: transport Ω_ij acts as identity, "
                        f"so gauge frame evolution (update_phi=True) won't affect them. "
                        f"Consider increasing non-scalar irreps (ℓ≥1) for gauge-sensitive representations.",
                        UserWarning
                    )
                print(f"  → {n_scalar_heads} scalar (ℓ=0) heads: GAUGE-INVARIANT (Ω=I)")
                print(f"  → {n_gauge_active_heads} non-scalar heads: gauge-active (transport via Wigner D)")

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        cached_head_transports: Optional[List[dict]] = None,  # Cross-layer cache
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.

        Args:
            mu_q: (B, N, K) belief means
            sigma_q: (B, N, K, K) belief covariances
            phi: (B, N, 3) gauge frames
            generators: (3, K, K) SO(3) generators
            mask: (B, N, N) optional causal mask
            return_attention: If True, return attention weights and KL matrices
            cached_head_transports: Optional list of precomputed transport dicts, one per head.
                                   When evolve_phi=False, this can be computed once at model
                                   entry and reused across all layers (6× speedup).

        Returns:
            mu_out: (B, N, K) updated means
            sigma_out: (B, N, K, K) updated covariances (or None)
            attention_weights: (B, n_heads, N, N) for visualization (or None)
            kl_matrices: (B, n_heads, N, N) KL divergences (or None)
        """
        batch_size, num_agents, K = mu_q.shape

        # =====================================================================
        # Split into irrep blocks
        # =====================================================================
        mu_blocks = self._split_irreps(mu_q)       # List of (B, N, dim_ℓ)
        sigma_blocks = self._split_irreps_sigma(sigma_q)  # List of (B, N, dim_ℓ, dim_ℓ)

        # =====================================================================
        # Process each head (irrep block)
        # =====================================================================
        head_outputs_mu = []
        head_outputs_sigma = []
        all_attention_weights = []
        all_kl_matrices = []

        for head_idx, (mu_head, sigma_head, dim_head, label) in enumerate(
            zip(mu_blocks, sigma_blocks, self.irrep_dims, self.irrep_labels)
        ):
            # Use proper SO(3) generators for this irrep dimension
            # These were pre-computed in __init__ using Wigner D-matrices
            gen_head = self.head_generators[head_idx].gen.to(
                device=generators.device, dtype=generators.dtype
            )

            # Get transport operators: use cross-layer cache if provided, else compute
            if cached_head_transports is not None:
                # Cross-layer cache: reuse transport computed at model entry
                head_cached_transport = cached_head_transports[head_idx]
            else:
                # Within-layer cache: compute once, reuse for KL and aggregation
                head_cached_transport = compute_transport_operators(
                    phi, gen_head, enforce_orthogonal=self.enforce_orthogonal
                )

            # Per-head temperature: κ_h = exp(log_κ_h) if learned, else shared κ
            if self.per_head_kappa:
                kappa_h = torch.exp(self.log_kappa[head_idx]).item()
            else:
                kappa_h = self.kappa_beta

            # Compute attention for this head
            if return_attention:
                # Full O(N²) attention with KL return
                beta_head, kl_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    kappa_h,
                    self.epsilon,
                    mask,
                    return_kl=True,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport,

                    use_identity_transport=self.use_identity_transport,
                    mask_self_attention=self.mask_self_attention,
                    enforce_orthogonal=self.enforce_orthogonal,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                )  # (B, N, N), (B, N, N)
                all_attention_weights.append(beta_head)
                all_kl_matrices.append(kl_head)
            else:
                # Full O(N²) attention without KL return
                beta_head = compute_attention_weights(
                    mu_head,
                    sigma_head,
                    phi,
                    gen_head,
                    kappa_h,
                    self.epsilon,
                    mask,
                    return_kl=False,
                    diagonal_covariance=self.diagonal_covariance,
                    cached_transport=head_cached_transport,

                    use_identity_transport=self.use_identity_transport,
                    mask_self_attention=self.mask_self_attention,
                    enforce_orthogonal=self.enforce_orthogonal,
                    use_rope=self.use_rope,
                    rope_base=self.rope_base,
                )  # (B, N, N)
                kl_head = None  # Not computed

            # Aggregate messages for this head (reuse cached transport!)
            mu_agg, sigma_agg = aggregate_messages(
                mu_head,
                sigma_head,
                phi,
                beta_head,
                gen_head,
                aggregate_mode=self.aggregate_mode,
                diagonal_covariance=self.diagonal_covariance,
                cached_transport=head_cached_transport,
            )

            head_outputs_mu.append(mu_agg)
            if sigma_agg is not None:
                head_outputs_sigma.append(sigma_agg)

        # =====================================================================
        # Concatenate head outputs
        # =====================================================================
        mu_concat = torch.cat(head_outputs_mu, dim=-1)  # (B, N, K)

        if head_outputs_sigma:
            # Block-diagonal covariance
            sigma_concat = self._block_diag_sigma(head_outputs_sigma)  # (B, N, K, K)
        else:
            sigma_concat = None

        # =====================================================================
        # Optional W_O output projection (cross-head mixing)
        # =====================================================================
        if self.output_proj is not None:
            mu_out = self.output_proj(mu_concat)  # (B, N, K) - learned cross-head mixing
        else:
            mu_out = mu_concat  # (B, N, K) - pure VFE, no mixing

        # Stack attention weights and KL matrices for loss computation
        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, n_heads, N, N)
            kl_matrices = torch.stack(all_kl_matrices, dim=1)  # (B, n_heads, N, N)
        else:
            attention_weights = None
            kl_matrices = None

        return mu_out, sigma_concat, attention_weights, kl_matrices

    def _split_irreps(self, mu: torch.Tensor) -> List[torch.Tensor]:
        """Split embedding into irrep blocks.

        Returns contiguous copies to avoid inplace modification issues during backward.
        """
        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            # Use .contiguous() to create a copy, avoiding inplace modification issues
            blocks.append(mu[..., start_idx:start_idx+dim].contiguous())
            start_idx += dim
        return blocks

    def _split_irreps_sigma(self, sigma: torch.Tensor) -> List[torch.Tensor]:
        """Split covariance into irrep blocks.

        For full covariance (B, N, K, K): extracts diagonal blocks
        For diagonal (B, N, K): extracts slices

        Handles mismatches between expected and actual sigma format by converting
        between diagonal and full covariance representations.
        """
        # Squeeze trailing singleton dimensions first
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        # Detect actual sigma format based on shape
        sigma_is_diagonal = sigma.dim() == 3

        # Handle format mismatches
        if self.diagonal_covariance and not sigma_is_diagonal:
            # Attention expects diagonal (B, N, K) but got full covariance (B, N, K, K)
            # Extract diagonal from full covariance matrix
            # Use .clone() to avoid view-related gradient issues (diagonal returns a view)
            sigma = torch.diagonal(sigma, dim1=-2, dim2=-1).clone()  # (B, N, K, K) -> (B, N, K)
        elif not self.diagonal_covariance and sigma_is_diagonal:
            # Attention expects full (B, N, K, K) but got diagonal (B, N, K)
            # Convert diagonal to full covariance
            sigma = torch.diag_embed(sigma)  # (B, N, K) -> (B, N, K, K)

        blocks = []
        start_idx = 0
        for dim in self.irrep_dims:
            if self.diagonal_covariance:
                # Diagonal mode: sigma is (B, N, K), just slice
                # Use .contiguous() to create a copy, avoiding inplace modification issues
                blocks.append(sigma[..., start_idx:start_idx+dim].contiguous())
            else:
                # Full mode: sigma is (B, N, K, K), extract diagonal block
                # Use .contiguous() to create a copy, avoiding inplace modification issues
                blocks.append(
                    sigma[..., start_idx:start_idx+dim, start_idx:start_idx+dim].contiguous()
                )
            start_idx += dim
        return blocks

    def _block_diag_sigma(self, sigma_blocks: List[torch.Tensor]) -> torch.Tensor:
        """Construct covariance from irrep blocks.

        For diagonal mode: concatenates (B, N, dim) slices → (B, N, K)
        For full mode: builds block-diagonal (B, N, K, K)
        """
        batch_size, num_agents = sigma_blocks[0].shape[:2]
        K = sum(self.irrep_dims)

        if self.diagonal_covariance:
            # Diagonal mode: just concatenate along last dim
            return torch.cat(sigma_blocks, dim=-1)  # (B, N, K)
        else:
            # Full mode: build block-diagonal matrix
            sigma_full = torch.zeros(
                batch_size, num_agents, K, K,
                device=sigma_blocks[0].device,
                dtype=sigma_blocks[0].dtype
            )

            start_idx = 0
            for sigma_block, dim in zip(sigma_blocks, self.irrep_dims):
                sigma_full[..., start_idx:start_idx+dim, start_idx:start_idx+dim] = sigma_block
                start_idx += dim

            return sigma_full

    def precompute_head_transports(
        self,
        phi: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[dict]:
        """
        Precompute transport operators for all heads.

        Call this once at model entry when evolve_phi=False, then pass the result
        to forward() as cached_head_transports to skip redundant matrix exponentials.

        Args:
            phi: (B, N, 3) gauge frames
            device: Device for generators
            dtype: Dtype for generators

        Returns:
            List of transport dicts, one per head. Each dict contains:
                'exp_phi': (B, N, dim, dim)
                'exp_neg_phi': (B, N, dim, dim)
                'Omega': (B, N, N, dim, dim)
        """
        cached_transports = []
        for head_idx in range(self.n_heads):
            gen_head = self.head_generators[head_idx].gen.to(device=device, dtype=dtype)
            cached_transports.append(compute_transport_operators(
                phi, gen_head, enforce_orthogonal=self.enforce_orthogonal
            ))
        return cached_transports

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"n_heads={self.n_heads}, "
            f"irrep_dims={self.irrep_dims[:3]}..., "
            f"kappa={self.kappa_beta}"
        )


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("KL-BASED ATTENTION MECHANISM TEST")
    print("="*70)

    # Test config
    B, N, K = 2, 8, 16  # Small for testing
    kappa = 1.0

    print(f"\n[1] Creating test data...")
    print(f"    Batch size: {B}")
    print(f"    Num agents: {N} (all at single point c*)")
    print(f"    Embed dim:  {K}")

    # Create random beliefs
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
    phi = torch.randn(B, N, 3) * 0.1

    # Generate SO(3) generators (import from existing module)
    if TRANSPORT_AVAILABLE:
        G = torch.from_numpy(generate_so3_generators(K)).float()
        print(f"    ✓ SO(3) generators created: {G.shape}")
    else:
        # Fallback: random skew-symmetric matrices
        G = torch.randn(3, K, K)
        G = 0.5 * (G - G.transpose(-1, -2))  # Make skew-symmetric
        print(f"    ⚠️  Using random generators (transport module unavailable)")

    # Test attention weights
    print(f"\n[2] Computing KL-based attention weights...")
    beta = compute_attention_weights(
        mu_q, sigma_q, phi, G, kappa, use_numba=False  # Use PyTorch for testing
    )
    print(f"    β shape: {beta.shape}")
    print(f"    β sum over keys: {beta.sum(dim=-1)[0, 0].item():.4f} (should ≈ 1)")
    print(f"    β min: {beta.min().item():.6f}")
    print(f"    β max: {beta.max().item():.6f}")

    # Test causal mask
    print(f"\n[3] Testing causal mask...")
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
    beta_causal = compute_attention_weights(
        mu_q, sigma_q, phi, G, kappa, mask=mask, use_numba=False
    )
    print(f"    Causal β[0, 0, :5]: {beta_causal[0, 0, :5]}")
    print(f"    Future positions should be ~0: {beta_causal[0, 0, 5:].sum().item():.6f}")

    # Test message aggregation
    print(f"\n[4] Testing message aggregation...")
    mu_agg, _ = aggregate_messages(
        mu_q, sigma_q, phi, beta, G, aggregate_mode='mean_only'
    )
    print(f"    Aggregated means shape: {mu_agg.shape}")
    print(f"    ✓ Messages aggregated via parallel transport")

    # Test multi-head attention
    print(f"\n[5] Testing multi-head attention...")
    irrep_spec = [
        ('ℓ0', 4, 1),   # 4 scalars
        ('ℓ1', 2, 3),   # 2 vectors
        ('ℓ2', 1, 5),   # 1 rank-2 tensor
    ]  # Total: 4 + 6 + 5 = 15 → pad to 16

    mha = IrrepMultiHeadAttention(
        embed_dim=K,
        irrep_spec=irrep_spec,
        kappa_beta=kappa,
    )
    print(f"    {mha}")

    mu_out, sigma_out, attn_weights, kl_matrices = mha(
        mu_q, sigma_q, phi, G, return_attention=True
    )
    print(f"    Output μ shape: {mu_out.shape}")
    print(f"    Attention weights shape: {attn_weights.shape}")
    print(f"    ✓ Multi-head attention complete")

    # Parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n[6] Parameter count:")
    print(f"    Multi-head attention: {total_params:,} parameters")
    print(f"    (Compare to standard: 4×K² = {4*K*K:,} for Q,K,V,O projections)")
    print(f"    Reduction: {4*K*K / max(total_params, 1):.1f}x fewer parameters!")

    print("\n" + "="*70)
    print("✓ All attention mechanism tests passed!")
    print("="*70)