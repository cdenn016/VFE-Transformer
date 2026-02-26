"""
Variational Feed-Forward Networks for Gauge Transformer
========================================================

Pure-PyTorch VFE gradient computation for GPU-accelerated active inference.

All gradient computation (VFE, natural gradient projection, SPD retraction)
runs entirely on GPU via PyTorch — no NumPy bridge or CPU round-trips.

Mathematical Foundation:
-----------------------
Free Energy (E-STEP):
    F = α·Σ_i KL(q_i||p_i)                      # Prior consistency
      + λ_β·Σ_{i,j} β_ij·KL(q_i||Ω_{ij}q_j)    # Belief alignment
      + λ_γ·Σ_{i,j} γ_ij·KL(p_i||Ω_{ij}p_j)    # Prior alignment
      + CE(W_out·μ, targets)                    # DISCRETE OBSERVATIONS!

CRITICAL: The cross-entropy term is the SINGLE observation model!
- E-step: Minimize F w.r.t. μ, Σ → compute ∂CE/∂μ with W_out frozen
- M-step: Minimize F w.r.t. W_out, embeddings → compute ∂CE/∂W_out with μ frozen

This is classic EM:
- E-step: "Given model (W_out), what beliefs (μ) explain observations?"
- M-step: "Given beliefs (μ), what model parameters explain observations?"

The SAME cross-entropy appears in both steps, just optimizing different parameters!

Gradient Engine computes:
    ∂F/∂θ for θ = {μ_q, Σ_q, μ_p, Σ_p, φ}

With natural gradient projection:
    Δθ = -η · F⁻¹(θ) · ∇F(θ)

Where F(θ) is the Fisher-Rao metric.

Date: November 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from transformer.core.gauge_utils import stable_matrix_exp_pair
from transformer.core.sanitization import san

# Import attention computation for dynamic β
from transformer.core.attention import compute_attention_weights, compute_transport_operators

# Import SO(N) and GL(K) retraction for proper phi updates
try:
    from math_utils.generators import (
        retract_soN_torch,
        retract_glK_torch,
        is_soN_generators,
        is_glK_generators,
    )
    RETRACTION_AVAILABLE = True
except ImportError:
    RETRACTION_AVAILABLE = False


def _retract_phi(
    phi: torch.Tensor,
    delta_phi: torch.Tensor,
    generators: torch.Tensor,
    step_size: float,
    trust_region: float = None,  # None = auto-select based on gauge group
    max_norm: float = None,  # None = auto-select based on gauge group
    bch_order: int = None,  # None = auto-select based on gauge group
    eps: float = 1e-6,
    gauge_group: str = None,  # Explicit: 'GLK', 'SON', or None for auto-detect
) -> torch.Tensor:
    """
    Retract phi update using appropriate method for gauge group.

    When gauge_group is provided, uses it directly. Otherwise auto-selects
    based on n_gen:
    - n_gen = N(N-1)/2 → SO(N): compact, uses trust_region=0.3, max_norm=π, bch_order=1
    - n_gen = K²       → GL(K): non-compact, uses trust_region=0.1, max_norm=1.0, bch_order=0

    Args:
        phi: Current gauge frames (..., n_gen)
        delta_phi: Update direction (..., n_gen)
        generators: Lie algebra generators (n_gen, dim, dim)
        step_size: Learning rate
        trust_region: Maximum relative change per update (auto if None)
        max_norm: Maximum norm for phi (auto if None)
        bch_order: BCH expansion order (auto if None)
        eps: Numerical stability
        gauge_group: Explicit gauge group ('GLK' or 'SON'). Overrides
            n_gen-based auto-detection. Required when n_gen doesn't match
            standard formulas (e.g. cross-head coupled GL(K)).

    Returns:
        phi_new: Updated gauge frames
    """
    n_gen = generators.shape[0]
    if gauge_group == 'GLK':
        is_glk = RETRACTION_AVAILABLE
        is_son = False
    elif gauge_group == 'SON':
        is_glk = False
        is_son = RETRACTION_AVAILABLE
    else:
        # Auto-detect from n_gen (original heuristic).
        # When n_gen doesn't match SO(N) or GL(K) exactly, default to GL(K)
        # retraction (conservative settings). This covers cross-head coupled
        # GL(K) where n_gen = H*d² + n_cross*d² > K².
        is_son = RETRACTION_AVAILABLE and is_soN_generators(n_gen)
        is_glk = RETRACTION_AVAILABLE and (is_glK_generators(n_gen) or not is_son)

    # Auto-select defaults based on gauge group
    if trust_region is None:
        trust_region = 0.1 if is_glk else 0.3
    if max_norm is None:
        max_norm = 1.0 if is_glk else math.pi
    if bch_order is None:
        bch_order = 0 if is_glk else 1

    if not RETRACTION_AVAILABLE:
        # Fallback: simple gradient descent with trust region
        update = step_size * delta_phi
        phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
        update_norm = torch.norm(update, dim=-1, keepdim=True)
        scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
        phi_new = phi + scale * update
        # Clamp to max norm
        phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
        phi_new = torch.where(
            phi_new_norm > max_norm,
            phi_new * max_norm / (phi_new_norm + eps),
            phi_new
        )
        return phi_new

    # Check if this is GL(K) (n_gen = K²) or SO(N) (n_gen = N(N-1)/2)
    if is_glk:
        # GL(K) is non-compact - needs conservative settings
        return retract_glK_torch(
            phi=phi,
            delta_phi=delta_phi,
            generators=generators,
            step_size=step_size,
            trust_region=trust_region,
            max_norm=max_norm,
            bch_order=bch_order,
            eps=eps,
            grad_clip=10.0,
        )
    elif is_son:
        # SO(N) is compact - can use standard settings
        return retract_soN_torch(
            phi=phi,
            delta_phi=delta_phi,
            generators=generators,
            step_size=step_size,
            trust_region=trust_region,
            max_norm=max_norm,
            bch_order=bch_order,
            eps=eps,
        )
    else:
        # Unknown gauge group - use conservative fallback
        update = step_size * delta_phi
        phi_norm = torch.norm(phi, dim=-1, keepdim=True).clamp(min=0.1)
        update_norm = torch.norm(update, dim=-1, keepdim=True)
        scale = torch.clamp(trust_region * phi_norm / (update_norm + eps), max=1.0)
        phi_new = phi + scale * update
        phi_new_norm = torch.norm(phi_new, dim=-1, keepdim=True)
        phi_new = torch.where(
            phi_new_norm > max_norm,
            phi_new * max_norm / (phi_new_norm + eps),
            phi_new
        )
        return phi_new


# =============================================================================
# Memory-Efficient VFE Gradient Helpers
# =============================================================================

def _compute_vfe_gradients_block_diagonal(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K, K) full block-diagonal covariances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K, K) prior covariances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: 'float | torch.Tensor',
    lambda_belief: float,
    kappa: float,
    eps: float,
    irrep_dims: List[int],
    chunk_size: Optional[int],
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-diagonal VFE gradient computation with optional chunking.

    Processes each irrep block separately to reduce memory from O(N²K²) to O(N² × max(dᵢ²)).
    When chunk_size is provided, also chunks over query positions to reduce memory further.
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Default chunk size to N (no chunking) if not provided
    C = chunk_size if chunk_size is not None else N

    # Initialize output gradients
    grad_mu = torch.zeros(B, N, K, device=device, dtype=dtype)
    grad_sigma = torch.zeros(B, N, K, K, device=device, dtype=dtype)

    # =================================================================
    # 1. Self-Coupling Gradient (block-wise but simpler)
    # =================================================================
    # Use 1e-4 floor (not eps=1e-6) to prevent singularity after many
    # training steps when σ entries shrink near zero
    _reg = max(eps, 1e-4)
    sigma_p_reg = sigma_p + _reg * torch.eye(K, device=device, dtype=dtype)
    try:
        sigma_p_inv = torch.linalg.inv(sigma_p_reg)
    except (torch.linalg.LinAlgError, RuntimeError):
        sigma_p_inv = torch.linalg.pinv(sigma_p_reg)
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

    sigma_q_reg = sigma_q + _reg * torch.eye(K, device=device, dtype=dtype)
    try:
        sigma_q_inv = torch.linalg.inv(sigma_q_reg)
    except (torch.linalg.LinAlgError, RuntimeError):
        sigma_q_inv = torch.linalg.pinv(sigma_q_reg)
    # For full covariance (4D), alpha (B,N,1) needs extra dim to broadcast with (B,N,K,K)
    alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
    grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

    grad_mu = grad_mu + grad_mu_self
    grad_sigma = grad_sigma + grad_sigma_self

    # =================================================================
    # 2. Belief Alignment Gradient (block-diagonal + chunked processing)
    # =================================================================
    # Precompute matrix exponentials per block (these are (B, N, d, d))
    block_exp_phi = []
    block_exp_neg_phi = []
    block_start = 0
    for d in irrep_dims:
        block_end = block_start + d
        gen_block = generators[:, block_start:block_end, block_start:block_end]
        phi_matrix_block = torch.einsum('bna,aij->bnij', phi, gen_block)
        exp_phi_block, exp_neg_phi_block = stable_matrix_exp_pair(phi_matrix_block)
        # Re-orthogonalization for SO(K) if requested
        if enforce_orthogonal and d >= 16:
            eye_d = torch.eye(d, device=device, dtype=dtype)
            exp_phi_block = exp_phi_block @ ((3.0 * eye_d - exp_phi_block.transpose(-1, -2) @ exp_phi_block) / 2.0)
            exp_neg_phi_block = exp_neg_phi_block @ ((3.0 * eye_d - exp_neg_phi_block.transpose(-1, -2) @ exp_neg_phi_block) / 2.0)
        block_exp_phi.append(exp_phi_block)
        block_exp_neg_phi.append(exp_neg_phi_block)
        block_start = block_end

    # Accumulators for alignment gradients
    grad_mu_align = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # For KL values and gradients - we'll accumulate these in chunks
    # We need full (B, N, N) for final softmax coupling, so accumulate
    kl_values = torch.zeros(B, N, N, device=device, dtype=dtype)
    grad_kl_per_pair_full = torch.zeros(B, N, N, K, device=device, dtype=dtype)

    # Process in chunks over query positions (i dimension)
    for i_start in range(0, N, C):
        i_end = min(i_start + C, N)
        C_actual = i_end - i_start

        # Process each irrep block for this chunk of query positions
        block_start = 0
        for block_idx, d in enumerate(irrep_dims):
            block_end = block_start + d

            # Extract block beliefs - use .contiguous() to create copies and avoid
            # inplace modification errors during backward pass
            mu_block = mu_q[:, :, block_start:block_end].contiguous()  # (B, N, d)
            sigma_block = sigma_q[:, :, block_start:block_end, block_start:block_end].contiguous()  # (B, N, d, d)

            # Get chunked exponentials for query positions - use .contiguous() for same reason
            exp_phi_i = block_exp_phi[block_idx][:, i_start:i_end].contiguous()  # (B, C, d, d)
            exp_neg_phi_j = block_exp_neg_phi[block_idx].contiguous()  # (B, N, d, d)

            # Compute Omega for this chunk: (B, C, N, d, d)
            Omega_chunk = torch.einsum(
                'bikl,bjlm->bijkm',
                exp_phi_i, exp_neg_phi_j
            )  # (B, C, N, d, d)

            # Transport means and covariances for this chunk
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega_chunk, mu_block)  # (B, C, N, d)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega_chunk, sigma_block, Omega_chunk.transpose(-1, -2)
            )  # (B, C, N, d, d)

            del Omega_chunk

            # Regularize and invert
            I_d = torch.eye(d, device=device, dtype=dtype)
            sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * I_d
            try:
                sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, C, N, d, d)
            except (torch.linalg.LinAlgError, RuntimeError):
                sigma_j_inv = torch.linalg.pinv(sigma_j_reg)

            # Delta mu for this block (query chunk) - contiguous to avoid view issues
            mu_block_i = mu_block[:, i_start:i_end].contiguous()  # (B, C, d)
            delta_mu_block = mu_block_i[:, :, None, :] - mu_j_transported  # (B, C, N, d)

            # ∂KL_ij/∂μ_i for this block
            grad_kl_block = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_block)  # (B, C, N, d)
            grad_kl_per_pair_full[:, i_start:i_end, :, block_start:block_end] = grad_kl_block

            # KL terms for this block
            mahal_block = torch.einsum('bijk,bijk->bij', delta_mu_block, grad_kl_block)  # (B, C, N)

            # Use contiguous slice and clone for expand to avoid view issues
            sigma_i_block_slice = sigma_block[:, i_start:i_end].contiguous()  # (B, C, d, d)
            sigma_i_block = sigma_i_block_slice[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, C, N, d, d)
            trace_block = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_block))

            try:
                L_j = torch.linalg.cholesky(sigma_j_reg)
                logdet_j = 2.0 * torch.sum(torch.log(torch.diagonal(L_j, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)
            except (torch.linalg.LinAlgError, RuntimeError):
                san.record('cholesky_fallback')
                try:
                    sign_j, logdet_j = torch.linalg.slogdet(sigma_j_reg)
                    logdet_j = torch.where(sign_j > 0, logdet_j, torch.zeros_like(logdet_j))
                except (torch.linalg.LinAlgError, RuntimeError):
                    logdet_j = torch.zeros(sigma_j_reg.shape[:-2], device=device, dtype=dtype)

            sigma_i_block_diag = sigma_i_block_slice + eps * I_d  # (B, C, d, d)
            try:
                L_i = torch.linalg.cholesky(sigma_i_block_diag)
                logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)
            except (torch.linalg.LinAlgError, RuntimeError):
                san.record('cholesky_fallback')
                try:
                    sign_i, logdet_i = torch.linalg.slogdet(sigma_i_block_diag)
                    logdet_i = torch.where(sign_i > 0, logdet_i, torch.zeros_like(logdet_i))
                except (torch.linalg.LinAlgError, RuntimeError):
                    logdet_i = torch.zeros(sigma_i_block_diag.shape[:-2], device=device, dtype=dtype)

            kl_block = 0.5 * (trace_block + mahal_block - d + logdet_j - logdet_i[:, :, None])
            kl_values[:, i_start:i_end, :] = kl_values[:, i_start:i_end, :] + kl_block.clamp(min=0.0, max=100.0)

            # Sigma alignment gradient for this block
            if compute_sigma_align_grad:
                sigma_i_inv_block = torch.linalg.inv(sigma_i_block_diag + 1e-4 * I_d)  # (B, C, d, d)
                # Use .clone() after expand to avoid view-related gradient issues
                sigma_i_inv_exp = sigma_i_inv_block[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()
                grad_sigma_block = 0.5 * (sigma_j_inv - sigma_i_inv_exp)
                beta_chunk = beta[:, i_start:i_end, :].contiguous()  # (B, C, N)
                grad_sigma_block_weighted = lambda_belief * torch.einsum('bij,bijkl->bikl', beta_chunk, grad_sigma_block)
                grad_sigma_align[:, i_start:i_end, block_start:block_end, block_start:block_end] += grad_sigma_block_weighted

            del sigma_j_transported, sigma_j_inv, mu_j_transported
            block_start = block_end

    # Direct term
    grad_mu_direct = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)

    # Softmax coupling term
    # Scale kappa by √K to match attention temperature τ = √K
    kappa_scaled = kappa * math.sqrt(max(K, 1))
    avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair_full)
    grad_deviation = grad_kl_per_pair_full - avg_grad.unsqueeze(2)
    d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
    grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu + grad_mu_align
    grad_sigma = grad_sigma + grad_sigma_align

    return grad_mu, grad_sigma


def _compute_vfe_gradients_chunked(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) prior variances
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames
    generators: torch.Tensor,  # (n_gen, K, K) generators
    alpha: float,
    lambda_belief: float,
    kappa: float,
    eps: float,
    chunk_size: int,
    compute_sigma_align_grad: bool,
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked VFE gradient computation for diagonal covariance mode.

    Processes N×N pairs in C×C chunks to reduce peak memory.

    Uses single-pass with incremental avg_grad and normalization by beta sum.
    This matches the original stable algorithm - only the inplace modification
    issue is fixed by using non-inplace operations (= ... + instead of +=).
    """
    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    sigma_q_safe = sigma_q.clamp(min=eps)
    sigma_p_safe = sigma_p.clamp(min=eps)

    # =================================================================
    # 1. Self-Coupling Gradient (simple, no chunking needed)
    # =================================================================
    delta_mu = mu_q - mu_p
    grad_mu_self = alpha * delta_mu / sigma_p_safe
    grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)

    # =================================================================
    # 2. Alignment Gradient (chunked processing)
    # =================================================================
    # Precompute matrix exponentials for all positions
    # Float64 for GL(K) numerical stability (prevents NaN in matrix_exp)
    phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
    exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)
    del phi_matrix

    # Re-orthogonalization for SO(K) if requested
    if enforce_orthogonal and K >= 16:
        eye_K = torch.eye(K, device=device, dtype=dtype)
        exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
        exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

    # Expand diagonal to full for transport
    sigma_j_diag = torch.diag_embed(sigma_q_safe)  # (B, N, K, K)

    # Accumulators - use non-inplace operations throughout
    grad_mu_direct = torch.zeros_like(mu_q)
    grad_mu_softmax = torch.zeros_like(mu_q)
    grad_sigma_align = torch.zeros_like(sigma_q)

    # Scale kappa by √K to match attention temperature τ = √K
    kappa_scaled = kappa * math.sqrt(max(K, 1))

    # Two-pass approach for correct softmax coupling gradient:
    # Pass 1: Accumulate avg_grad (= Σ_j β_ij ∂KL_ij/∂μ_i) and per-chunk KL/grad_kl
    # Pass 2: Compute softmax coupling with the correct (full) avg_grad

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        n_i = i_end - i_start

        exp_phi_i = exp_phi[:, i_start:i_end].contiguous()
        mu_i = mu_q[:, i_start:i_end].contiguous()
        sigma_i = sigma_q_safe[:, i_start:i_end].contiguous()
        beta_i = beta[:, i_start:i_end, :].contiguous()

        # ============================================================
        # PASS 1: Accumulate direct gradient and avg_grad over all j
        # ============================================================
        avg_grad_i = torch.zeros(B, n_i, K, device=device, dtype=dtype)
        # Store per-j-chunk data for pass 2
        chunk_data = []

        for j_start in range(0, N, chunk_size):
            j_end = min(j_start + chunk_size, N)
            n_j = j_end - j_start

            exp_neg_phi_j = exp_neg_phi[:, j_start:j_end].contiguous()
            mu_j = mu_q[:, j_start:j_end].contiguous()
            sigma_j_diag_chunk = sigma_j_diag[:, j_start:j_end].contiguous()
            beta_chunk = beta_i[:, :, j_start:j_end].contiguous()

            # Compute Omega for this chunk
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi_i, exp_neg_phi_j)

            # Transport means
            mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_j)

            # Transport covariances (diagonal -> full -> for grad computation)
            sigma_j_transported = torch.einsum(
                'bijkl,bjlm,bijmn->bijkn',
                Omega, sigma_j_diag_chunk, Omega.transpose(-1, -2)
            )

            del Omega

            # Regularize and invert transported covariance
            # Use 1e-4 floor (not eps=1e-6) to prevent singularity after many
            # VFE iterations when σ_j entries shrink near zero
            sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * torch.eye(K, device=device, dtype=dtype)
            try:
                sigma_j_inv = torch.linalg.inv(sigma_j_reg)
            except (torch.linalg.LinAlgError, RuntimeError):
                sigma_j_inv = torch.linalg.pinv(sigma_j_reg)

            # Delta mu
            delta_mu_ij = mu_i[:, :, None, :] - mu_j_transported

            # ∂KL/∂μ_i
            grad_kl = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

            # Accumulate direct gradient
            direct_contrib = lambda_belief * torch.einsum('bij,bijk->bik', beta_chunk, grad_kl)
            grad_mu_direct[:, i_start:i_end] = grad_mu_direct[:, i_start:i_end] + direct_contrib

            # Accumulate avg_grad = Σ_j β_ij ∂KL_ij/∂μ_i (complete over ALL j)
            avg_grad_i = avg_grad_i + torch.einsum('bij,bijk->bik', beta_chunk, grad_kl)

            # Compute KL values for this chunk
            mahal = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl)

            sigma_i_diag_exp = torch.diag_embed(sigma_i)[:, :, None, :, :].expand(-1, -1, n_j, -1, -1).clone()
            trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_diag_exp))

            try:
                L_p = torch.linalg.cholesky(sigma_j_reg)
                logdet_p = 2.0 * torch.sum(torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)
            except (torch.linalg.LinAlgError, RuntimeError):
                san.record('cholesky_fallback')
                logdet_p = torch.zeros(B, n_i, n_j, device=device, dtype=dtype)

            logdet_q = torch.sum(torch.log(sigma_i), dim=-1)[:, :, None].expand(-1, -1, n_j).clone()

            kl_chunk = 0.5 * (trace_term + mahal - K + logdet_p - logdet_q).clamp(min=0.0, max=100.0)

            # Store for pass 2
            chunk_data.append((beta_chunk, grad_kl, kl_chunk))

            # Sigma alignment gradient
            if compute_sigma_align_grad:
                sigma_j_inv_diag = torch.diagonal(sigma_j_inv, dim1=-2, dim2=-1)
                sigma_i_inv = 1.0 / sigma_i
                sigma_i_inv_exp = sigma_i_inv[:, :, None, :].expand(-1, -1, n_j, -1).clone()
                grad_sigma_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv_exp)
                sigma_contrib = lambda_belief * torch.einsum('bij,bijk->bik', beta_chunk, grad_sigma_pair)
                grad_sigma_align[:, i_start:i_end] = grad_sigma_align[:, i_start:i_end] + sigma_contrib

            del sigma_j_transported, sigma_j_inv, mu_j_transported

        # ============================================================
        # PASS 2: Softmax coupling with correct (complete) avg_grad
        # ============================================================
        for beta_chunk, grad_kl, kl_chunk in chunk_data:
            grad_deviation = grad_kl - avg_grad_i.unsqueeze(2)
            d_beta_d_mu = beta_chunk.unsqueeze(-1) * grad_deviation / kappa_scaled
            softmax_contrib = lambda_belief * torch.einsum('bij,bijk->bik', kl_chunk, d_beta_d_mu)
            grad_mu_softmax[:, i_start:i_end] = grad_mu_softmax[:, i_start:i_end] + softmax_contrib

        del chunk_data  # Free stored tensors

    grad_mu_align = grad_mu_direct + grad_mu_softmax
    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align

    return grad_mu, grad_sigma


# =============================================================================
# GPU-Based Gradient Computation (PyTorch - FAST!)
# =============================================================================

def compute_vfe_gradients_gpu(
    mu_q: torch.Tensor,        # (B, N, K) belief means
    sigma_q: torch.Tensor,     # (B, N, K) diagonal variances or (B, N, K, K) full
    mu_p: torch.Tensor,        # (B, N, K) prior means
    sigma_p: torch.Tensor,     # (B, N, K) diagonal or (B, N, K, K) full
    beta: torch.Tensor,        # (B, N, N) attention weights
    phi: torch.Tensor,         # (B, N, n_gen) gauge frames where n_gen is # of generators
    generators: torch.Tensor,  # (n_gen, K, K) Lie algebra generators (SO(3) or SO(N))
    alpha: 'float | torch.Tensor' = 0.01,  # Self-coupling weight: scalar or (B, N, 1) Bayesian precision
    lambda_belief: float = 1.0,  # Belief alignment weight
    kappa: float = 1.0,        # Temperature (for normalization)
    eps: float = 1e-6,
    cached_transport: Optional[dict] = None,  # Precomputed transport operators
    compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
    # Memory-efficient options (NEW!)
    irrep_dims: Optional[List[int]] = None,  # Block dimensions for block-diagonal processing
    chunk_size: Optional[int] = None,  # Chunk size for memory-efficient processing
    enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute VFE gradients entirely on GPU using PyTorch.

    This is the FAST version that replaces the NumPy-based gradient_engine.
    Fully vectorized - no loops over batch or agents!

    Supports both SO(3) and SO(N) gauge groups. The number of generators
    determines the gauge group: 3 for SO(3), N(N-1)/2 for SO(N).

    Gradients computed:
    1. Self-coupling: ∂/∂μ_q [α · KL(q||p)]
    2. Belief alignment: ∂/∂μ_q [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]

    Args:
        mu_q: Belief means (B, N, K)
        sigma_q: Belief variances - diagonal (B, N, K) or full (B, N, K, K)
        mu_p: Prior means (B, N, K)
        sigma_p: Prior variances - diagonal (B, N, K) or full (B, N, K, K)
        beta: Attention weights (B, N, N), already normalized
        phi: Gauge frames (B, N, n_gen) where n_gen is # of generators
        generators: Lie algebra generators (n_gen, K, K) - SO(3) has 3, SO(N) has N(N-1)/2
        alpha: Weight for KL(q||p) self-coupling term. Can be a float (uniform)
               or a (B, N, 1) tensor from Bayesian precision estimation.
        lambda_belief: Weight for belief alignment term
        kappa: Temperature parameter
        eps: Numerical stability
        cached_transport: Optional dict with precomputed 'Omega' from compute_transport_operators().
                         When provided, avoids redundant matrix exponential computations.
        compute_sigma_align_grad: If True (default), compute sigma gradient from belief alignment term.
                                  This is the theoretically correct gradient:
                                    ∂KL/∂Σ_q = 0.5 * (Σ_transported^{-1} - Σ_q^{-1})
                                  Set to False for legacy behavior (zero sigma alignment gradient).
        irrep_dims: Optional list of block dimensions for memory-efficient block-diagonal processing.
                   When provided, processes each irrep block separately to reduce memory from
                   O(N²K²) to O(N² × max(dᵢ²)).
        chunk_size: Optional chunk size for processing N×N pairs in C×C chunks.

    Returns:
        grad_mu: Gradient w.r.t. μ_q, shape (B, N, K)
        grad_sigma: Gradient w.r.t. σ_q, shape (B, N, K) for diagonal
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)
    while sigma_p.dim() > 3 and sigma_p.shape[-1] == 1:
        sigma_p = sigma_p.squeeze(-1)

    B, N, K = mu_q.shape
    device = mu_q.device
    dtype = mu_q.dtype

    # Detect diagonal vs full covariance
    is_diagonal = sigma_q.dim() == 3

    # =================================================================
    # MEMORY-EFFICIENT PATH: Block-diagonal processing
    # =================================================================
    if irrep_dims is not None and not is_diagonal:
        return _compute_vfe_gradients_block_diagonal(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, irrep_dims, chunk_size,
            compute_sigma_align_grad, enforce_orthogonal
        )

    # =================================================================
    # MEMORY-EFFICIENT PATH: Chunked processing for diagonal mode
    # =================================================================
    if chunk_size is not None and is_diagonal:
        return _compute_vfe_gradients_chunked(
            mu_q, sigma_q, mu_p, sigma_p, beta, phi, generators,
            alpha, lambda_belief, kappa, eps, chunk_size,
            compute_sigma_align_grad, enforce_orthogonal
        )

    # =================================================================
    # 1. Self-Coupling Gradient: ∂/∂μ_q [α · KL(q||p)]
    # =================================================================
    # For diagonal Gaussians:
    #   KL(q||p) = 0.5 * Σ_k [ σ_q[k]/σ_p[k] + (μ_p[k]-μ_q[k])²/σ_p[k] - 1 + log(σ_p[k]/σ_q[k]) ]
    #   ∂KL/∂μ_q = (μ_q - μ_p) / σ_p
    #   ∂KL/∂σ_q = 0.5 * (1/σ_p - 1/σ_q)

    if is_diagonal:
        # Clamp for stability
        sigma_q_safe = sigma_q.clamp(min=eps)
        sigma_p_safe = sigma_p.clamp(min=eps)

        # Self-coupling gradient w.r.t. μ
        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * delta_mu / sigma_p_safe  # (B, N, K)

        # Self-coupling gradient w.r.t. σ (diagonal)
        grad_sigma_self = alpha * 0.5 * (1.0 / sigma_p_safe - 1.0 / sigma_q_safe)  # (B, N, K)
    else:
        # Full covariance - use matrix operations
        # ∂KL/∂μ_q = Σ_p^{-1} (μ_q - μ_p)
        # Use 1e-4 floor (not eps=1e-6) to prevent singularity after many
        # training steps when σ entries shrink near zero
        _reg = max(eps, 1e-4)
        sigma_p_reg = sigma_p + _reg * torch.eye(K, device=device, dtype=dtype)
        try:
            sigma_p_inv = torch.linalg.inv(sigma_p_reg)  # (B, N, K, K)
        except (torch.linalg.LinAlgError, RuntimeError):
            sigma_p_inv = torch.linalg.pinv(sigma_p_reg)

        delta_mu = mu_q - mu_p  # (B, N, K)
        grad_mu_self = alpha * torch.einsum('bnij,bnj->bni', sigma_p_inv, delta_mu)

        # ∂KL/∂Σ_q = 0.5 * (Σ_p^{-1} - Σ_q^{-1})
        sigma_q_reg = sigma_q + _reg * torch.eye(K, device=device, dtype=dtype)
        try:
            sigma_q_inv = torch.linalg.inv(sigma_q_reg)
        except (torch.linalg.LinAlgError, RuntimeError):
            sigma_q_inv = torch.linalg.pinv(sigma_q_reg)
        # For full covariance (4D), alpha (B,N,1) needs extra dim to broadcast with (B,N,K,K)
        alpha_4d = alpha.unsqueeze(-1) if isinstance(alpha, torch.Tensor) else alpha
        grad_sigma_self = alpha_4d * 0.5 * (sigma_p_inv - sigma_q_inv)

    # =================================================================
    # 2. Belief Alignment Gradient: ∂/∂μ_i [λ · Σ_j β_ij · KL(q_i || Ω_ij q_j)]
    # =================================================================
    # Full gradient via product rule:
    #   ∂/∂μ_i [Σ_j β_ij · KL_ij] = Σ_j β_ij · ∂KL_ij/∂μ_i + Σ_j KL_ij · ∂β_ij/∂μ_i
    #                                  ↑ direct term           ↑ SOFTMAX COUPLING (nonlinearity!)
    #
    # The softmax coupling gradient is the KEY nonlinearity (replaces GELU/ReLU):
    #   ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    if is_diagonal:
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            # Compute transport operators (vectorized)
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)  # (B, N, K, K)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 16:
                eye_K = torch.eye(K, device=device, dtype=dtype)
                exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
                exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

            # Transport: Ω_ij = exp(φ_i) @ exp(-φ_j)
            # For all pairs: (B, N, N, K, K)
            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport all μ_j to frame i: μ_j_transported[i,j] = Ω_ij @ μ_j
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)  # (B, N, N, K)

        # Difference: μ_i - μ_j_transported (for each pair i,j)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported  # (B, N, N, K)

        # =================================================================
        # PROPER COVARIANCE TRANSPORT: Σ_j_transported = Ω @ diag(σ_j) @ Ω^T
        # =================================================================
        # Even though input is diagonal, transported covariance is FULL!
        # This is crucial for gauge equivariance.

        # Expand diagonal to full: diag(σ_j) -> (B, N, K, K)
        sigma_j_diag = torch.diag_embed(sigma_q.clamp(min=eps))  # (B, N, K, K)

        # Transport covariance: Σ_j_transported = Ω_ij @ Σ_j @ Ω_ij^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_j_diag, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert transported covariance
        sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * torch.eye(K, device=device, dtype=dtype)
        # NaN/Inf safety: if transport produced NaN or Inf (e.g., matrix_exp overflow),
        # replace with identity covariance to prevent cascading failures
        bad_mask = torch.isnan(sigma_j_reg) | torch.isinf(sigma_j_reg)
        if bad_mask.any():
            bad_elements = bad_mask.any(dim=-1).any(dim=-1)  # (B, N, N)
            eye_K = torch.eye(K, device=device, dtype=dtype)
            sigma_j_reg = torch.where(
                bad_elements.unsqueeze(-1).unsqueeze(-1),
                eye_K.expand_as(sigma_j_reg),
                sigma_j_reg,
            )
        try:
            sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)
        except (torch.linalg.LinAlgError, RuntimeError):
            sigma_j_inv = torch.linalg.pinv(sigma_j_reg)

        # ∂KL_ij/∂μ_i = Σ_j_transported^{-1} @ (μ_i - μ_j_transported)
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)  # (B, N, N, K)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        # Σ_i is diagonal: diag(σ_q[i]) expanded to all pairs
        sigma_i_diag = torch.diag_embed(sigma_q.clamp(min=eps))  # (B, N, K, K)
        # Use .clone() after expand to avoid view-related gradient issues
        sigma_i_expanded = sigma_i_diag[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms
        # For transported covariance (full matrix): use Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)  # (B, N, N)
        except (torch.linalg.LinAlgError, RuntimeError):
            san.record('cholesky_fallback')
            try:
                eigvals = torch.linalg.eigvalsh(sigma_j_reg)  # (B, N, N, K)
                logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)  # (B, N, N)
            except (torch.linalg.LinAlgError, RuntimeError):
                logdet_j_t = torch.zeros(sigma_j_reg.shape[:-2], device=device, dtype=dtype)
        # For source covariance (diagonal): log|diag(σ)| = sum(log(σ))
        logdet_i = torch.sum(torch.log(sigma_q.clamp(min=eps)), dim=-1)  # (B, N)
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N).clone()  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        kl_values = kl_values.clamp(min=0.0, max=100.0)  # (B, N, N)

        # =================================================================
        # 2a. Direct term: Σ_j β_ij · ∂KL_ij/∂μ_i
        # =================================================================
        # avg_grad = Σ_k β_ik · ∂KL_ik/∂μ_i (used for both direct and softmax terms)
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)  # (B, N, K)
        grad_mu_direct = lambda_belief * avg_grad

        # =================================================================
        # 2b. Softmax coupling term (THE NONLINEARITY!):
        #     ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ
        #     grad_softmax = Σ_j KL_ij · ∂β_ij/∂μ_i
        # =================================================================
        # Deviation from average: ∂KL_ij/∂μ_i - avg_grad_i
        # grad_kl_per_pair: (B, N, N, K), avg_grad: (B, N, K) -> expand to (B, N, 1, K)
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)  # (B, N, N, K)

        # Softmax coupling gradient: ∂β_ij/∂μ_i = β_ij · grad_deviation / κ
        # Scale kappa by √K to match attention temperature scaling (τ = √K)
        # The factor of 2 accounts for the ½ prefactor in KL divergence:
        #   KL = ½(trace + mahal - K + logdet), so magnitudes are O(K/2), not O(K)
        kappa_scaled = kappa * math.sqrt(max(K, 1))
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled  # (B, N, N, K)

        # Weight by KL values and sum: Σ_j KL_ij · ∂β_ij/∂μ_i
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        # Total alignment gradient (direct + softmax coupling)
        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # For diagonal mode, we take the diagonal of the transported inverse
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Diagonal of inverse of transported covariance: diag(Σ_j_transported^{-1})
            # Use .clone() to avoid view-related gradient issues
            sigma_j_inv_diag = torch.diagonal(sigma_j_inv, dim1=-2, dim2=-1).clone()  # (B, N, N, K)

            # Inverse of diagonal Σ_i: 1/σ_i expanded for all pairs
            sigma_i_inv = 1.0 / sigma_q.clamp(min=eps)  # (B, N, K)
            # Use .clone() after expand to avoid view-related gradient issues
            sigma_i_inv_expanded = sigma_i_inv[:, :, None, :].expand(-1, -1, N, -1).clone()  # (B, N, N, K)

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1}_kk - 1/σ_i[k])
            grad_sigma_per_pair = 0.5 * (sigma_j_inv_diag - sigma_i_inv_expanded)  # (B, N, N, K)

            # Weight by attention and sum: Σ_j β_ij * ∂KL_ij/∂σ_i
            grad_sigma_align = lambda_belief * torch.einsum('bij,bijk->bik', beta, grad_sigma_per_pair)  # (B, N, K)
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)
    else:
        # Full covariance belief alignment
        # Get transport operators (use cached if available)
        if cached_transport is not None and 'Omega' in cached_transport:
            Omega = cached_transport['Omega']
        else:
            phi_matrix = torch.einsum('bna,aij->bnij', phi, generators)
            exp_phi, exp_neg_phi = stable_matrix_exp_pair(phi_matrix)

            # Re-orthogonalization for SO(K) if requested
            if enforce_orthogonal and K >= 16:
                eye_K = torch.eye(K, device=device, dtype=dtype)
                exp_phi = exp_phi @ ((3.0 * eye_K - exp_phi.transpose(-1, -2) @ exp_phi) / 2.0)
                exp_neg_phi = exp_neg_phi @ ((3.0 * eye_K - exp_neg_phi.transpose(-1, -2) @ exp_neg_phi) / 2.0)

            Omega = torch.einsum('bikl,bjlm->bijkm', exp_phi, exp_neg_phi)

        # Transport means
        mu_j_transported = torch.einsum('bijkl,bjl->bijk', Omega, mu_q)
        delta_mu_ij = mu_q.unsqueeze(2) - mu_j_transported

        # Transport covariances: Σ_j_transported = Ω @ Σ_j @ Ω^T
        sigma_j_transported = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega, sigma_q, Omega.transpose(-1, -2)
        )  # (B, N, N, K, K)

        # Regularize and invert
        sigma_j_reg = sigma_j_transported + max(eps, 1e-4) * torch.eye(K, device=device, dtype=dtype)
        # NaN/Inf safety: if transport produced NaN or Inf, replace with identity covariance
        bad_mask = torch.isnan(sigma_j_reg) | torch.isinf(sigma_j_reg)
        if bad_mask.any():
            bad_elements = bad_mask.any(dim=-1).any(dim=-1)  # (B, N, N)
            eye_K = torch.eye(K, device=device, dtype=dtype)
            sigma_j_reg = torch.where(
                bad_elements.unsqueeze(-1).unsqueeze(-1),
                eye_K.expand_as(sigma_j_reg),
                sigma_j_reg,
            )
        try:
            sigma_j_inv = torch.linalg.inv(sigma_j_reg)  # (B, N, N, K, K)
        except (torch.linalg.LinAlgError, RuntimeError):
            sigma_j_inv = torch.linalg.pinv(sigma_j_reg)

        # ∂KL_ij/∂μ_i
        grad_kl_per_pair = torch.einsum('bijkl,bijl->bijk', sigma_j_inv, delta_mu_ij)

        # Compute FULL KL values (not just Mahalanobis - include trace and logdet!)
        # KL(q_i || Ω_ij[q_j]) = 0.5 * (tr(Σ_j_t^{-1} Σ_i) + mahal - K + log|Σ_j_t| - log|Σ_i|)

        # Mahalanobis term: δμ^T @ Σ_j_transported^{-1} @ δμ
        mahal_term = torch.einsum('bijk,bijk->bij', delta_mu_ij, grad_kl_per_pair)  # (B, N, N)

        # Trace term: tr(Σ_j_transported^{-1} @ Σ_i)
        # Use .clone() after expand to avoid view-related gradient issues
        sigma_i_expanded = sigma_q[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)
        trace_term = torch.einsum('bijkk->bij', torch.einsum('bijkl,bijlm->bijkm', sigma_j_inv, sigma_i_expanded))

        # Log-determinant terms using Cholesky with fallback
        try:
            L_j_t = torch.linalg.cholesky(sigma_j_reg)  # (B, N, N, K, K)
            logdet_j_t = 2.0 * torch.sum(torch.log(torch.diagonal(L_j_t, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)  # (B, N, N)
        except (torch.linalg.LinAlgError, RuntimeError):
            san.record('cholesky_fallback')
            try:
                eigvals = torch.linalg.eigvalsh(sigma_j_reg)
                logdet_j_t = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
            except (torch.linalg.LinAlgError, RuntimeError):
                logdet_j_t = torch.zeros(sigma_j_reg.shape[:-2], device=device, dtype=dtype)

        sigma_i_reg = sigma_q + eps * torch.eye(K, device=device, dtype=dtype)
        try:
            L_i = torch.linalg.cholesky(sigma_i_reg)  # (B, N, K, K)
            logdet_i = 2.0 * torch.sum(torch.log(torch.diagonal(L_i, dim1=-2, dim2=-1).clamp(min=eps)), dim=-1)  # (B, N)
        except (torch.linalg.LinAlgError, RuntimeError):
            san.record('cholesky_fallback')
            try:
                eigvals = torch.linalg.eigvalsh(sigma_i_reg)
                logdet_i = torch.sum(torch.log(eigvals.clamp(min=eps)), dim=-1)
            except (torch.linalg.LinAlgError, RuntimeError):
                logdet_i = torch.zeros(sigma_i_reg.shape[:-2], device=device, dtype=dtype)
        logdet_i_expanded = logdet_i[:, :, None].expand(-1, -1, N).clone()  # (B, N, N)

        # Full KL divergence
        kl_values = 0.5 * (trace_term + mahal_term - K + logdet_j_t - logdet_i_expanded)
        kl_values = kl_values.clamp(min=0.0, max=100.0)  # (B, N, N)

        # avg_grad = Σ_k β_ik · ∂KL_ik/∂μ_i (used for both direct and softmax terms)
        avg_grad = torch.einsum('bij,bijk->bik', beta, grad_kl_per_pair)
        grad_mu_direct = lambda_belief * avg_grad

        # Softmax coupling term
        # Scale kappa by √K to match attention temperature scaling (τ = √K)
        # The factor of 2 accounts for the ½ prefactor in KL divergence
        kappa_scaled = kappa * math.sqrt(max(K, 1))
        grad_deviation = grad_kl_per_pair - avg_grad.unsqueeze(2)
        d_beta_d_mu = beta.unsqueeze(-1) * grad_deviation / kappa_scaled
        grad_mu_softmax = lambda_belief * torch.einsum('bij,bijk->bik', kl_values, d_beta_d_mu)

        grad_mu_align = grad_mu_direct + grad_mu_softmax

        # =================================================================
        # Sigma gradient from alignment term (full covariance case)
        # ∂KL/∂Σ_i = 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
        # Weighted by attention: Σ_j β_ij * ∂KL_ij/∂Σ_i
        # =================================================================
        if compute_sigma_align_grad:
            # Use Σ_i^{-1} computed earlier in self-coupling section (sigma_q_inv)
            # Use .clone() after expand to avoid view-related gradient issues
            sigma_i_inv_expanded = sigma_q_inv[:, :, None, :, :].expand(-1, -1, N, -1, -1).clone()  # (B, N, N, K, K)

            # Gradient per pair: 0.5 * (Σ_j_transported^{-1} - Σ_i^{-1})
            grad_sigma_per_pair = 0.5 * (sigma_j_inv - sigma_i_inv_expanded)  # (B, N, N, K, K)

            # Weight by attention and sum: Σ_j β_ij * ∂KL_ij/∂Σ_i
            grad_sigma_align = lambda_belief * torch.einsum('bij,bijkl->bikl', beta, grad_sigma_per_pair)  # (B, N, K, K)
        else:
            # Simplified: no sigma gradient from alignment (legacy behavior)
            grad_sigma_align = torch.zeros_like(sigma_q)

    # =================================================================
    # 3. Combine Gradients
    # =================================================================
    grad_mu = grad_mu_self + grad_mu_align
    grad_sigma = grad_sigma_self + grad_sigma_align

    return grad_mu, grad_sigma


def compute_natural_gradient_gpu(
    grad_mu: torch.Tensor,     # (B, N, K) Euclidean gradient
    grad_sigma: torch.Tensor,  # (B, N, K) or (B, N, K, K)
    sigma_q: torch.Tensor,     # (B, N, K) or (B, N, K, K)
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project Euclidean gradients to natural gradients using Fisher metric.

    For Gaussian distributions, the Fisher information metric is:
        F_μ = Σ^{-1}  →  natural_grad_μ = Σ @ euclidean_grad_μ
        F_σ = 2Σ^{-2} →  natural_grad_σ = 0.5 * Σ² @ euclidean_grad_σ (diagonal approx)

    Args:
        grad_mu: Euclidean gradient w.r.t. μ
        grad_sigma: Euclidean gradient w.r.t. σ
        sigma_q: Current covariance
        eps: Numerical stability

    Returns:
        nat_grad_mu: Natural gradient for μ
        nat_grad_sigma: Natural gradient for σ
    """
    # Squeeze trailing singleton dimensions for robustness
    while sigma_q.dim() > 3 and sigma_q.shape[-1] == 1:
        sigma_q = sigma_q.squeeze(-1)

    is_diagonal = sigma_q.dim() == 3

    if is_diagonal:
        # Diagonal case: simple element-wise multiplication
        sigma_safe = sigma_q.clamp(min=eps)
        nat_grad_mu = sigma_safe * grad_mu  # (B, N, K)
        nat_grad_sigma = 0.5 * sigma_safe * sigma_safe * grad_sigma  # (B, N, K)
    else:
        # Full covariance: matrix multiplication
        nat_grad_mu = torch.einsum('bnij,bnj->bni', sigma_q, grad_mu)
        # For sigma, use diagonal approximation for simplicity
        # Use .clone() after diagonal to avoid view-related gradient issues
        sigma_diag = torch.diagonal(sigma_q, dim1=-2, dim2=-1).clone()
        nat_grad_sigma = 0.5 * sigma_diag.unsqueeze(-1) * sigma_diag.unsqueeze(-2) * grad_sigma

    return nat_grad_mu, nat_grad_sigma


# =============================================================================
# SPD Retraction (PyTorch GPU version)
# =============================================================================

def retract_spd_torch(
    Sigma: torch.Tensor,
    delta_Sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SPD-preserving retraction for covariance matrices (PyTorch GPU).

    Uses exponential map with WHITENED trust region (like retraction.py).
    The whitening normalizes out the σ² scaling in natural gradients,
    preventing runaway growth.

    Args:
        Sigma: SPD matrices, shape (B, N, K, K) or (B*N, K, K)
        delta_Sigma: Symmetric tangent vectors, same shape as Sigma
        step_size: Learning rate τ (already applied to delta_Sigma typically)
        trust_region: Max norm of WHITENED tangent ||Σ^{-1/2} ΔΣ Σ^{-1/2}||_F
        eps: Regularization floor for numerical stability

    Returns:
        Sigma_new: SPD matrices, same shape as Sigma
    """
    # Handle different input shapes
    original_shape = Sigma.shape
    if Sigma.dim() == 4:
        B, N, K, _ = Sigma.shape
        Sigma = Sigma.reshape(B * N, K, K)
        delta_Sigma = delta_Sigma.reshape(B * N, K, K)

    batch_size, K, _ = Sigma.shape
    device = Sigma.device
    dtype = Sigma.dtype

    # Symmetrize inputs (numerical safety)
    Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    delta_Sigma = 0.5 * (delta_Sigma + delta_Sigma.transpose(-1, -2))

    # Use DIAGONAL whitening (robust to ill-conditioning, avoids eigh)
    # This approximates full whitening B = Σ^{-1/2} ΔΣ Σ^{-1/2}
    # For diagonal-dominant covariances (common case), this is accurate
    # Key insight: whitening normalizes out the σ² scaling in natural gradients
    # Use .clone() after diagonal to avoid view-related gradient issues
    diag_sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1).clone()  # (batch, K)
    diag_sigma = diag_sigma.clamp(min=eps)
    inv_sqrt_diag = 1.0 / torch.sqrt(diag_sigma)  # (batch, K)

    # Diagonal-whitened tangent: B ≈ D^{-1/2} ΔΣ D^{-1/2} where D = diag(Σ)
    B = (inv_sqrt_diag.unsqueeze(-1) * delta_Sigma) * inv_sqrt_diag.unsqueeze(-2)  # (batch, K, K)

    # Trust region on WHITENED Frobenius norm (this is the key fix!)
    if trust_region is not None and trust_region > 0:
        B_norm = torch.linalg.norm(B, ord='fro', dim=(-2, -1), keepdim=True)  # (batch, 1, 1)
        scale = torch.clamp(trust_region / (B_norm + eps), max=1.0)
        B = B * scale

    # Un-whiten to get scaled delta: ΔΣ_scaled = D^{1/2} B D^{1/2}
    sqrt_diag = torch.sqrt(diag_sigma)  # (batch, K)
    delta_scaled = (sqrt_diag.unsqueeze(-1) * B) * sqrt_diag.unsqueeze(-2)  # (batch, K, K)

    # Linear update with trust-region-scaled delta
    Sigma_new = Sigma + step_size * delta_scaled

    # Symmetrize
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.transpose(-1, -2))

    # Ensure SPD via iterative regularization
    # Try Cholesky; if it fails, add regularization and retry
    # This is more robust than eigendecomposition for ill-conditioned matrices
    eye_K = torch.eye(K, device=device, dtype=dtype)

    # Compute scale for adaptive regularization (based on diagonal magnitude)
    # Use .clone() after diagonal to avoid view-related gradient issues
    diag_mean = torch.diagonal(Sigma_new, dim1=-2, dim2=-1).clone().abs().mean(dim=-1, keepdim=True)  # (batch, 1)
    base_reg = torch.clamp(diag_mean, min=eps).unsqueeze(-1)  # (batch, 1, 1)

    # Ensure SPD using cholesky_ex for per-matrix diagnostics (avoids
    # entire-batch fallback when only a few matrices are non-SPD)
    reg_scales = [eps, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

    for reg_scale in reg_scales:
        Sigma_reg = Sigma_new + (reg_scale * base_reg) * eye_K
        # cholesky_ex returns info codes per matrix (0 = success)
        _, info = torch.linalg.cholesky_ex(Sigma_reg)
        if (info == 0).all():
            # All matrices are SPD
            Sigma_new = Sigma_reg
            break
        elif reg_scale == reg_scales[-1]:
            # Last attempt: selectively fix only the failing matrices
            bad_mask = (info != 0)
            Sigma_new[~bad_mask] = Sigma_reg[~bad_mask]
            Sigma_new[bad_mask] = Sigma[bad_mask] + 0.1 * base_reg[bad_mask] * eye_K
            break  # Prevent for/else from overwriting the selective repair
        # else: try heavier regularization

    # Restore original shape
    if len(original_shape) == 4:
        Sigma_new = Sigma_new.reshape(original_shape)

    return Sigma_new


def retract_spd_diagonal_torch(
    sigma_diag: torch.Tensor,
    delta_sigma: torch.Tensor,
    step_size: float = 1.0,
    trust_region: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SPD retraction for diagonal covariances (much simpler).

    For diagonal matrices, the exponential map reduces to:
        σ_new = σ * exp(τ * δσ / σ)

    This ensures positivity: exp(x) > 0 for all x.

    Args:
        sigma_diag: Diagonal variances, shape (B, N, K)
        delta_sigma: Tangent in diagonal form, shape (B, N, K)
        step_size: Learning rate τ
        trust_region: Max absolute value of exponent argument
        eps: Floor for sigma values

    Returns:
        sigma_new: Positive diagonal variances, shape (B, N, K)
    """
    sigma_safe = sigma_diag.clamp(min=eps)

    # Whitened tangent: δσ / σ (element-wise for diagonal)
    whitened = delta_sigma / sigma_safe

    # Trust region on whitened tangent
    if trust_region is not None and trust_region > 0:
        whitened = whitened.clamp(-trust_region, trust_region)

    # Exponential update: σ_new = σ * exp(τ * whitened)
    # Clip exponent to prevent overflow
    exp_arg = (step_size * whitened).clamp(-50.0, 50.0)
    sigma_new = sigma_safe * torch.exp(exp_arg)

    # Clamp to [eps, 100.0] for numerical stability
    # Floor prevents division by zero; ceiling prevents KL divergence explosion
    return sigma_new.clamp(min=eps, max=100.0)


# =============================================================================
# Dynamic-β VFE: Full Active Inference with Evolving Attention (RECOMMENDED!)
# =============================================================================

class VariationalFFNDynamic(nn.Module):
    """
    Dynamic-β Variational FFN: Recomputes attention at each VFE step.

    This is the theoretically correct implementation where beliefs and attention
    co-evolve. At each integration step:

        1. Compute β from current beliefs: β_ij = softmax(-KL(q_i||Ω_ij[q_j])/κ)
        2. Compute full VFE gradient: ∂F/∂θ (includes ∂β/∂θ nonlinearity)
        3. Update beliefs via natural gradient descent
        4. (Optional) M-step: update priors toward beliefs
        5. Repeat

    Key difference from VariationalFFNGradientEngine:
        - GradientEngine: β computed once, held fixed during descent
        - Dynamic: β recomputed at each step → attention-belief co-evolution

    This enables emergent block structure in β as beliefs cluster.

    The ∂β/∂μ term is the principled nonlinearity (replaces GELU):
        ∂β_ij/∂μ_i = β_ij · [∂KL_ij/∂μ_i - Σ_k β_ik · ∂KL_ik/∂μ_i] / κ

    With dynamic β, this creates positive feedback:
        - Tokens with similar beliefs → higher β between them
        - Higher β → beliefs pulled closer together
        - → Cluster formation (meta-agents!)

    Complexity: O(n_steps × N² × K) - more expensive but theoretically sound
    """

    def __init__(
        self,
        embed_dim: int,
        generators: torch.Tensor,  # (3, K, K) SO(3) generators
        alpha: float = 0.01,       # Self-coupling weight (KL(q||p))
        lambda_belief: float = 1.0,  # Belief alignment weight
        kappa: float = 1.0,        # Attention temperature
        n_iterations: int = 10,    # VFE descent steps (more steps = deeper equilibration)
        learnable_lr: bool = True, # Learn step size?
        update_sigma: bool = True, # Update covariances?
        diagonal_covariance: bool = False,  # Use diagonal Σ for efficiency
        compute_sigma_align_grad: bool = True,  # Compute sigma gradient from alignment term
        # Phi (gauge frame) evolution via VFE gradients
        update_phi: bool = False,  # If True, update phi via ∂F/∂φ (after E-step loop)
        update_phi_per_iteration: bool = False,  # If True, update phi during EACH E-step iteration
        phi_lr: float = 0.05,      # Learning rate for phi updates
        phi_max_norm: float = 3.14159,  # Max norm for phi (π = 180° rotation)
        # Pure FEP mode: learning via prior evolution (no backprop)
        max_seq_len: int = 512,    # Max sequence length for persistent priors
        pure_fep_mode: bool = False,  # Enable backprop-free learning
        prior_lr: float = 0.01,    # Learning rate for prior updates
        prior_bank: Optional[nn.Module] = None,  # Token-dependent PriorBank (if provided)
        use_prior_bank: bool = False,  # If True, use PriorBank (token-dependent) instead of position-dependent priors
        # Memory-efficient options (NEW!)
        irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        # Bayesian precision (learned prior self-coupling)
        learnable_alpha: bool = False,  # If True, use Gamma-Normal conjugate precision
        # Multi-head VFE: maintain per-head β through iterations
        multihead_vfe: bool = False,  # If True, compute separate β_h per irrep block
        per_head_kappa: bool = False,  # If True, learn separate κ_h per head
    ):
        """
        Initialize dynamic-β VFE FFN.

        Args:
            embed_dim: K - dimension of belief vectors
            generators: SO(3) generators for gauge transport (3, K, K)
            alpha: Weight for KL(q||p) self-coupling (prior anchoring)
            lambda_belief: Weight for belief alignment term Σ β_ij KL(q_i||q_j)
            kappa: Temperature for attention softmax (higher = softer)
            n_iterations: Number of VFE descent iterations per forward pass
            learnable_lr: If True, step size η is a learnable parameter
            update_sigma: If True, also update covariance matrices Σ
            diagonal_covariance: Use diagonal Σ for O(K) instead of O(K²)
            compute_sigma_align_grad: If True, compute sigma gradient from alignment term
            max_seq_len: Maximum sequence length for persistent priors (pure FEP mode)
            pure_fep_mode: If True, use persistent priors that evolve via prediction error
            prior_lr: Learning rate for prior updates in pure FEP mode
            prior_bank: Optional PriorBank module with token-dependent priors (recommended!)
            use_prior_bank: If True, use PriorBank (token-dependent priors) instead of
                           position-dependent priors. CRITICAL for language modeling!
            irrep_dims: Block dimensions [d₁, d₂, ...] for memory-efficient block-diagonal KL.
                       When provided, exploits O(N² × Σᵢdᵢ²) vs O(N² × K²) - massive savings!
            chunk_size: Chunk size for memory-efficient processing. Processes N×N in C×C chunks.
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0 always.
            learnable_alpha: If True, use Bayesian precision via Gamma-Normal conjugacy.
                            α_i = (a₀ + K/2) / (b₀ + ½ ∥μ_q - μ_p∥²_{Σ_p⁻¹})
                            Gauge-invariant, state-dependent, only 2 learnable parameters.
            multihead_vfe: If True, maintain per-head attention β_h through VFE iterations.
                          Each irrep block gets its own attention pattern instead of a single
                          collapsed β. Requires irrep_dims to be set.
            per_head_kappa: If True, learn separate κ_h per head (requires multihead_vfe).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.register_buffer('generators', generators)
        self.n_iterations = n_iterations
        self.mask_self_attention = mask_self_attention
        self.update_sigma = update_sigma
        self.diagonal_covariance = diagonal_covariance
        self.compute_sigma_align_grad = compute_sigma_align_grad

        # Phi evolution via VFE gradients (principled approach)
        self.update_phi = update_phi
        self.update_phi_per_iteration = update_phi_per_iteration  # Dynamical gauge frames
        if update_phi_per_iteration:
            print(f"[VariationalFFNDynamic] φ will evolve DURING E-step iterations (dynamical gauge frames)")
        self.phi_lr = phi_lr
        self.phi_max_norm = phi_max_norm

        # Memory-efficient options
        self.irrep_dims = irrep_dims
        self.chunk_size = chunk_size

        # VFE hyperparameters
        self.alpha = alpha
        self.lambda_belief = lambda_belief
        self.kappa = kappa

        # =================================================================
        # Multi-head VFE: per-block β through iterations
        # =================================================================
        self.multihead_vfe = multihead_vfe and irrep_dims is not None
        self.per_head_kappa_enabled = per_head_kappa
        if self.multihead_vfe:
            n_heads = len(irrep_dims)
            if per_head_kappa:
                # Per-head κ (log-parameterized for positivity)
                self.log_kappa_heads = nn.Parameter(
                    torch.full((n_heads,), math.log(max(kappa, 1e-6)))
                )
                print(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with learned κ_h")
            else:
                self.log_kappa_heads = None
                print(f"[VariationalFFNDynamic] Multi-head VFE: {n_heads} heads with shared κ={kappa}")
        else:
            self.log_kappa_heads = None

        # =================================================================
        # Bayesian Precision: Gamma-Normal conjugate prior for α
        # =================================================================
        # α_i = (a₀ + K/2) / (b₀ + ½ ∥μ_q,i - μ_p,i∥²_{Σ_p⁻¹})
        #
        # Gauge-invariant (Mahalanobis distance is a gauge scalar).
        # Self-regulating: α shrinks when beliefs diverge from priors.
        # Initialized so α ≈ alpha (the scalar value) when ∥δμ∥ ≈ 0.
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            # Initialize: a₀ = 1, b₀ = (a₀ + K/2) / alpha_init
            # so that α ≈ alpha when Mahalanobis distance ≈ 0
            a0_init = 1.0
            alpha_init = max(alpha, 0.01)  # avoid division by zero
            b0_init = (a0_init + embed_dim / 2.0) / alpha_init
            # Parameterize via softplus to ensure positivity
            self.raw_a0 = nn.Parameter(torch.tensor(self._softplus_inverse(a0_init)))
            self.raw_b0 = nn.Parameter(torch.tensor(self._softplus_inverse(b0_init)))
            print(f"[VariationalFFNDynamic] Bayesian precision enabled: "
                  f"a₀={a0_init:.1f}, b₀={b0_init:.1f}, "
                  f"initial α≈{alpha_init} (K={embed_dim})")

        # Pure FEP mode: learning via prior evolution
        self.pure_fep_mode = pure_fep_mode
        self.max_seq_len = max_seq_len
        self.prior_lr = prior_lr
        self.use_prior_bank = use_prior_bank
        self.prior_bank = prior_bank

        if pure_fep_mode:
            if use_prior_bank and prior_bank is not None:
                # Token-dependent priors via PriorBank (CORRECT for language!)
                self.prior_bank = prior_bank
                print(f"[VariationalFFNDynamic] Using PriorBank with token-dependent priors (vocab_size={prior_bank.vocab_size})")
            elif use_prior_bank and prior_bank is None:
                raise ValueError(
                    "use_prior_bank=True requires prior_bank to be provided! "
                    "Create a PriorBank and pass it to VariationalFFNDynamic."
                )
            else:
                # Legacy position-dependent priors (DEPRECATED for language!)
                import warnings
                warnings.warn(
                    "pure_fep_mode without PriorBank uses POSITION-DEPENDENT priors. "
                    "This doesn't work for language modeling! "
                    "Set use_prior_bank=True and provide a PriorBank for token-dependent priors.",
                    UserWarning
                )
                # Position-dependent persistent priors (the LEARNING happens here!)
                # These evolve based on prediction-error-weighted beliefs
                self.register_buffer('prior_mu', torch.zeros(max_seq_len, embed_dim))
                self.register_buffer('prior_sigma', torch.ones(max_seq_len, embed_dim))
                self.register_buffer('prior_update_count', torch.zeros(max_seq_len))
                self.register_buffer('prior_initialized', torch.tensor(False))

        # Learnable step size
        if learnable_lr:
            self.lr = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('lr', torch.tensor(0.1))

    @staticmethod
    def _softplus_inverse(x: float) -> float:
        """Compute inverse of softplus: log(exp(x) - 1)."""
        if x > 20.0:
            return x  # softplus ≈ identity for large x
        return math.log(math.expm1(x))

    def get_bayesian_alpha(
        self,
        mu_q: torch.Tensor,      # (B, N, K)
        mu_p: torch.Tensor,      # (B, N, K)
        sigma_p: torch.Tensor,   # (B, N, K) diagonal or (B, N, K, K) full
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute Bayesian precision via Gamma-Normal conjugacy.

        α_i = (a₀ + K/2) / (b₀ + ½ ∥μ_q,i - μ_p,i∥²_{Σ_p⁻¹})

        This is the posterior mean of τ ~ Gamma(a₀, b₀) given
        observation (μ_q - μ_p) with precision Σ_p⁻¹.

        Gauge-invariant: the Mahalanobis distance is a gauge scalar.

        Returns:
            alpha: (B, N, 1) per-token precision, broadcastable to (B, N, K)
        """
        a0 = F.softplus(self.raw_a0)
        b0 = F.softplus(self.raw_b0)

        delta_mu = mu_q - mu_p  # (B, N, K)
        is_diagonal = (sigma_p.dim() == 3)

        if is_diagonal:
            sigma_p_safe = sigma_p.clamp(min=eps)
            # Mahalanobis squared: Σ_k (δμ_k)² / σ_p_k
            mahal_sq = (delta_mu ** 2 / sigma_p_safe).sum(dim=-1, keepdim=True)  # (B, N, 1)
        else:
            # Full covariance: δμᵀ Σ_p⁻¹ δμ
            K = mu_q.shape[-1]
            sigma_p_reg = sigma_p + max(eps, 1e-4) * torch.eye(K, device=mu_q.device, dtype=mu_q.dtype)
            try:
                sigma_p_inv = torch.linalg.inv(sigma_p_reg)  # (B, N, K, K)
            except (torch.linalg.LinAlgError, RuntimeError):
                sigma_p_inv = torch.linalg.pinv(sigma_p_reg)
            mahal_sq = torch.einsum('bni,bnij,bnj->bn', delta_mu, sigma_p_inv, delta_mu)
            mahal_sq = mahal_sq.unsqueeze(-1)  # (B, N, 1)

        K = mu_q.shape[-1]
        alpha = (a0 + K / 2.0) / (b0 + 0.5 * mahal_sq)  # (B, N, 1)

        return alpha

    def forward(
        self,
        mu: torch.Tensor,          # (B, N, K) - current beliefs
        beta: torch.Tensor,        # (B, n_heads, N, N) - INITIAL attention (will be recomputed)
        mu_prior: torch.Tensor,    # (B, N, K) - embedding priors
        phi: torch.Tensor,         # (B, N, 3) - gauge frames
        sigma: Optional[torch.Tensor] = None,  # (B, N, K, K) or (B, N, K) if diagonal
        mask: Optional[torch.Tensor] = None,   # (B, N, N) - causal mask
        targets: Optional[torch.Tensor] = None,  # (B, N) - target token IDs
        W_out: Optional[torch.Tensor] = None,    # (V, K) - output projection
        token_ids: Optional[torch.Tensor] = None,  # (B, N) - token IDs for PriorBank lookup
        return_beta_history: bool = False,  # Return β evolution for analysis
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[list]]:
        """
        Dynamic VFE descent with β recomputation at each step.

        Flow at each iteration:
            1. β = softmax(-KL(q||Ω[q])/κ)  [RECOMPUTE from current beliefs]
            2. ∂F/∂μ = α(μ-μ_p)/σ_p + λΣβ(∂KL/∂μ) + Σ KL(∂β/∂μ) + ∂CE/∂μ
            3. μ ← μ - η·F⁻¹·∂F/∂μ  [Natural gradient descent]
            4. (Optional) σ ← retract_spd(σ, -η·∂F/∂σ)
            5. (Optional) φ ← φ - η_φ·∂F/∂φ  [VFE gradient descent on gauge frames]
            6. (Optional M-step) μ_p ← μ_p + rate·(μ - μ_p)

        Args:
            mu: Current belief means (B, N, K)
            beta: Initial attention weights (B, n_heads, N, N) - used only for first step
            mu_prior: Prior means from embeddings (B, N, K)
            phi: Gauge frames (B, N, phi_dim)
            sigma: Belief covariances - (B, N, K, K) full or (B, N, K) diagonal
            mask: Causal mask (B, N, N) where 0 = cannot attend
            targets: Target tokens for observation term (B, N)
            W_out: Output projection for ∂CE/∂μ computation
            token_ids: Token IDs for PriorBank lookup (required if use_prior_bank=True)
            return_beta_history: If True, return list of β at each step

        Returns:
            mu_new: Updated beliefs (B, N, K)
            sigma_new: Updated covariances (same shape as input) or None
            phi_new: Updated gauge frames (B, N, phi_dim)
            beta_history: List of β tensors if return_beta_history, else None
        """
        B, N, K = mu.shape
        device = mu.device
        dtype = mu.dtype
        eps = 1e-6

        # Initialize sigma if not provided
        if sigma is None:
            if self.diagonal_covariance:
                sigma = torch.ones(B, N, K, device=device, dtype=dtype) * 0.1
            else:
                sigma = 0.1 * torch.eye(K, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

        # Squeeze trailing singleton dimensions for robustness
        while sigma.dim() > 3 and sigma.shape[-1] == 1:
            sigma = sigma.squeeze(-1)

        is_diagonal = sigma.dim() == 3

        # =====================================================================
        # PURE FEP MODE: Use persistent priors instead of embedding priors
        # =====================================================================
        if self.pure_fep_mode and self.use_prior_bank:
            # Token-dependent priors via PriorBank (CORRECT for language!)
            if token_ids is None:
                raise ValueError(
                    "token_ids required when use_prior_bank=True! "
                    "Pass token_ids to forward() for PriorBank lookup."
                )

            # Get token-dependent priors from PriorBank
            mu_p_from_bank, sigma_p_from_bank, _ = self.prior_bank.encode(token_ids)  # (B, N, K)

            # Use PriorBank priors for VFE dynamics
            mu_p_current = mu_p_from_bank

            # Convert diagonal sigma_p to full covariance if needed
            if is_diagonal:
                sigma_p = sigma_p_from_bank
            else:
                sigma_p = torch.diag_embed(sigma_p_from_bank)  # (B, N, K) -> (B, N, K, K)

        elif self.pure_fep_mode:
            # Legacy position-dependent priors (DEPRECATED!)
            # Initialize persistent priors from embeddings on first call
            self.initialize_priors_from_embeddings(mu_prior)

            # Get persistent priors (these evolve via prediction-error updates)
            # Note: persistent priors are always stored as diagonal (B, N, K)
            persistent_mu, persistent_sigma = self.get_persistent_priors(N, B, device)

            # Use persistent priors for VFE dynamics
            mu_p_current = persistent_mu.clone()

            # Convert diagonal persistent_sigma to full covariance if needed
            if persistent_sigma is not None:
                if is_diagonal:
                    # Both are diagonal - use directly
                    sigma_p = persistent_sigma.clone()
                else:
                    # Need full covariance (B, N, K, K) from diagonal (B, N, K)
                    sigma_p = torch.diag_embed(persistent_sigma)  # (B, N, K) -> (B, N, K, K)
            else:
                sigma_p = sigma.clone()
        else:
            # Standard mode: use embedding priors
            mu_p_current = mu_prior.clone()
            sigma_p = sigma.clone()

        # Current state (will evolve)
        mu_current = mu.clone()
        sigma_current = sigma.clone()
        phi_current = phi.clone()  # Track phi for dynamical gauge frames

        # Track β evolution if requested
        beta_history = [] if return_beta_history else None

        # Store observation info for fresh gradient computation
        has_observations = targets is not None and W_out is not None

        # =====================================================================
        # VFE Descent Loop with Dynamic β
        # =====================================================================
        # CRITICAL FIX: Scale base learning rate by 1/n_iterations
        # Without this, n_iterations=10 causes ~10x total displacement vs n_iterations=1
        # This ensures total integrated step size is approximately constant regardless of n_iterations
        base_lr = self.lr / self.n_iterations

        for iteration in range(self.n_iterations):
            # Additional decay within loop: lr decreases as we approach convergence
            # iteration 0: factor=1.0, iteration n-1: factor=0.5 (for n>1)
            if self.n_iterations > 1:
                decay_factor = 1.0 - 0.5 * (iteration / (self.n_iterations - 1))
            else:
                decay_factor = 1.0
            effective_lr = base_lr * decay_factor

            # =================================================================
            # STEP 0: Precompute transport operators ONCE per iteration
            # =================================================================
            # Both compute_attention_weights and compute_vfe_gradients_gpu need
            # the same Ω_ij = exp(φ_i)·exp(-φ_j). Computing once and passing
            # cached_transport avoids redundant matrix exponentials.
            # Skip caching when using block-diagonal or chunked paths (they
            # compute transport internally in chunks to save memory).
            if self.irrep_dims is None and self.chunk_size is None and not self.multihead_vfe:
                cached_transport = compute_transport_operators(
                    phi=phi_current,
                    generators=self.generators,
                    enforce_orthogonal=getattr(self, 'enforce_orthogonal', False),
                )
            else:
                cached_transport = None

            # =================================================================
            # Determine alpha: Bayesian precision or fixed scalar
            # =================================================================
            if self.learnable_alpha:
                alpha_effective = self.get_bayesian_alpha(
                    mu_current, mu_p_current, sigma_p, eps=eps
                )  # (B, N, 1) - gauge-invariant, state-dependent
            else:
                alpha_effective = self.alpha  # scalar (backward compatible)

            if self.multihead_vfe:
                # =============================================================
                # MULTI-HEAD VFE: Per-block β_h through iterations
                # =============================================================
                # Each irrep block gets its own attention pattern.
                # This maintains head diversity through all VFE iterations,
                # enabling different heads to cluster at different scales.
                grad_mu = torch.zeros_like(mu_current)
                grad_sigma = torch.zeros_like(sigma_current)
                beta_heads = []  # For history tracking

                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h

                    # Extract per-head slices.
                    # .detach() prevents autograd graph from growing 3× deeper
                    # per iteration — VFE gradients are computed analytically,
                    # so we don't need autograd through the per-head attention/KL.
                    # Gradient flow to embeddings is preserved through the
                    # mu_current = mu_current + delta_mu update chain.
                    mu_h = mu_current[:, :, block_start:block_end].detach().contiguous()
                    mu_p_h = mu_p_current[:, :, block_start:block_end].contiguous()
                    if is_diagonal:
                        # Diagonal covariance: sigma is (B, N, K)
                        sigma_h = sigma_current[:, :, block_start:block_end].detach().contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end].contiguous()
                    else:
                        # Full covariance: sigma is (B, N, K, K) — slice both dims
                        sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach().contiguous()
                        sigma_p_h = sigma_p[:, :, block_start:block_end, block_start:block_end].contiguous()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]

                    # Per-head temperature
                    if self.log_kappa_heads is not None:
                        kappa_h = torch.exp(self.log_kappa_heads[h]).item()
                    else:
                        kappa_h = self.kappa

                    # Per-head attention: β_h = softmax(-KL_h / κ_h)
                    beta_h = compute_attention_weights(
                        mu_q=mu_h,
                        sigma_q=sigma_h,
                        phi=phi_current,
                        generators=gen_h,
                        kappa=kappa_h,
                        epsilon=eps,
                        mask=mask,
                        use_numba=False,
                        return_kl=False,
                        diagonal_covariance=is_diagonal,
                        mask_self_attention=self.mask_self_attention,
                    )  # (B, N, N)
                    beta_heads.append(beta_h)

                    # Per-head VFE gradients
                    grad_mu_h, grad_sigma_h = compute_vfe_gradients_gpu(
                        mu_q=mu_h,
                        sigma_q=sigma_h,
                        mu_p=mu_p_h,
                        sigma_p=sigma_p_h,
                        beta=beta_h,
                        phi=phi_current,
                        generators=gen_h,
                        alpha=alpha_effective,
                        lambda_belief=self.lambda_belief,
                        kappa=kappa_h,
                        eps=eps,
                        compute_sigma_align_grad=self.compute_sigma_align_grad,
                    )

                    grad_mu[:, :, block_start:block_end] = grad_mu_h
                    if is_diagonal:
                        grad_sigma[:, :, block_start:block_end] = grad_sigma_h
                    else:
                        grad_sigma[:, :, block_start:block_end, block_start:block_end] = grad_sigma_h
                    block_start = block_end

                # Use mean of per-head betas for history tracking
                if return_beta_history:
                    beta_avg = torch.stack(beta_heads, dim=0).mean(dim=0)
                    beta_history.append(beta_avg.detach().clone())
                # Store last head's beta for phi update (uses alignment loss)
                beta_current = beta_heads[-1]

            else:
                # =============================================================
                # SINGLE-β VFE: Original behavior (all blocks share one β)
                # =============================================================
                # STEP 1: Recompute attention β from current beliefs
                beta_current = compute_attention_weights(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    phi=phi_current,
                    generators=self.generators,
                    kappa=self.kappa,
                    epsilon=eps,
                    mask=mask,
                    use_numba=False,
                    return_kl=False,
                    diagonal_covariance=is_diagonal,
                    cached_transport=cached_transport,
                    irrep_dims=self.irrep_dims,
                    chunk_size=self.chunk_size,
                    mask_self_attention=self.mask_self_attention,
                )  # (B, N, N)

                if return_beta_history:
                    beta_history.append(beta_current.detach().clone())

                # STEP 2: Compute VFE gradients with current β
                grad_mu, grad_sigma = compute_vfe_gradients_gpu(
                    mu_q=mu_current,
                    sigma_q=sigma_current,
                    mu_p=mu_p_current,
                    sigma_p=sigma_p,
                    beta=beta_current,
                    phi=phi_current,
                    generators=self.generators,
                    alpha=alpha_effective,
                    lambda_belief=self.lambda_belief,
                    kappa=self.kappa,
                    eps=eps,
                    cached_transport=cached_transport,
                    compute_sigma_align_grad=self.compute_sigma_align_grad,
                    irrep_dims=self.irrep_dims,
                    chunk_size=self.chunk_size,
                )

            # Add FRESH observation gradient (recomputed from current beliefs)
            # Use .detach() on mu_current to avoid second-order gradients through the
            # observation gradient computation. Gradients still flow through VFE dynamics
            # (the natural gradient update), just not through how the obs grad was computed.
            # This is more stable than full gradient flow while still allowing embeddings
            # to learn from VFE dynamics via the mu_current → mu_new update chain.
            if has_observations:
                logits = torch.matmul(mu_current.detach(), W_out.T)
                probs = F.softmax(logits, dim=-1)
                targets_valid = targets.clone()
                targets_valid[targets == -1] = 0
                one_hot = F.one_hot(targets_valid, num_classes=W_out.shape[0]).float()
                mask_obs = (targets != -1).unsqueeze(-1).float()
                one_hot = one_hot * mask_obs
                grad_error = (probs - one_hot) * mask_obs
                discrete_obs_grad = torch.matmul(grad_error, W_out)
                grad_mu = grad_mu + discrete_obs_grad

            # Clip for stability
            grad_mu = torch.clamp(grad_mu, min=-1e3, max=1e3)
            grad_sigma = torch.clamp(grad_sigma, min=-1e3, max=1e3)

            # =================================================================
            # STEP 3: Natural gradient projection
            # =================================================================
            nat_grad_mu, nat_grad_sigma = compute_natural_gradient_gpu(
                grad_mu, grad_sigma, sigma_current, eps=eps
            )

            # =================================================================
            # STEP 4: Update beliefs (E-step) with WHITENED trust region
            # =================================================================
            # The natural gradient nat_grad_mu = Σ @ grad scales with σ
            # Use whitened trust region: ||δμ / √σ|| instead of raw norm
            delta_mu = -effective_lr * nat_grad_mu

            # Whitened trust region for mu
            if is_diagonal:
                sigma_sqrt = torch.sqrt(sigma_current.clamp(min=eps))
                whitened_delta = delta_mu / sigma_sqrt
            else:
                # Use .clone() after diagonal to avoid view-related gradient issues
                sigma_diag = torch.diagonal(sigma_current, dim1=-2, dim2=-1).clone().clamp(min=eps)
                whitened_delta = delta_mu / torch.sqrt(sigma_diag)

            whitened_norm = torch.linalg.norm(whitened_delta, dim=-1, keepdim=True)
            mu_trust_region = 2.0  # Trust region on whitened norm
            scale = torch.clamp(mu_trust_region / (whitened_norm + eps), max=1.0)
            mu_current = mu_current + scale * delta_mu

            if self.update_sigma:
                # Use SPD-preserving retraction for stability with multiple iterations
                # Much smaller lr for sigma (matches simulation_runner: 0.005 vs 0.1)
                sigma_lr = effective_lr * 0.05
                if is_diagonal:
                    sigma_current = retract_spd_diagonal_torch(
                        sigma_diag=sigma_current,
                        delta_sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.2,  # Max 20% change per iteration
                        eps=eps,
                    )
                else:
                    sigma_current = retract_spd_torch(
                        Sigma=sigma_current,
                        delta_Sigma=-nat_grad_sigma,
                        step_size=sigma_lr,
                        trust_region=0.1,  # Max 10% change per iteration
                        eps=eps,
                    )

            # =============================================================
            # STEP 4b: Optional Phi Evolution DURING E-step (dynamical gauge frames)
            # =============================================================
            # When update_phi_per_iteration=True, φ evolves at each iteration
            # This makes gauge frames dynamical, co-evolving with beliefs
            if self.update_phi_per_iteration and torch.is_grad_enabled():
                phi_for_grad = phi_current.clone().requires_grad_(True)

                if self.multihead_vfe:
                    # Multi-head phi update: sum alignment loss over heads
                    alignment_loss = torch.tensor(0.0, device=mu_current.device,
                                                  dtype=mu_current.dtype)
                    block_start = 0
                    for h, d_h in enumerate(self.irrep_dims):
                        block_end = block_start + d_h
                        mu_h = mu_current[:, :, block_start:block_end].detach()
                        if sigma_current is None:
                            sigma_h = None
                        elif is_diagonal:
                            sigma_h = sigma_current[:, :, block_start:block_end].detach()
                        else:
                            sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach()
                        gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                        kappa_h = torch.exp(self.log_kappa_heads[h]).item() if self.log_kappa_heads is not None else self.kappa

                        beta_phi_h_result = compute_attention_weights(
                            mu_q=mu_h, sigma_q=sigma_h,
                            phi=phi_for_grad, generators=gen_h,
                            kappa=kappa_h, epsilon=eps, mask=mask,
                            use_numba=False, return_kl=True,
                            diagonal_covariance=is_diagonal,
                            mask_self_attention=self.mask_self_attention,
                        )
                        if isinstance(beta_phi_h_result, tuple):
                            beta_phi_h, kl_h = beta_phi_h_result
                        else:
                            beta_phi_h = beta_phi_h_result
                            kl_h = beta_phi_h
                        alignment_loss = alignment_loss + self.lambda_belief * (beta_phi_h * kl_h).sum()
                        block_start = block_end
                else:
                    # Single-β phi update (original)
                    beta_for_phi_result = compute_attention_weights(
                        mu_q=mu_current.detach(),
                        sigma_q=sigma_current.detach() if sigma_current is not None else None,
                        phi=phi_for_grad,
                        generators=self.generators,
                        kappa=self.kappa,
                        epsilon=eps,
                        mask=mask,
                        use_numba=False,
                        return_kl=True,
                        diagonal_covariance=is_diagonal,
                        irrep_dims=self.irrep_dims,
                        chunk_size=self.chunk_size,
                        mask_self_attention=self.mask_self_attention,
                    )

                    if isinstance(beta_for_phi_result, tuple):
                        beta_phi, kl_matrix = beta_for_phi_result
                    else:
                        beta_phi = beta_for_phi_result
                        kl_matrix = beta_phi  # Fallback

                    alignment_loss = self.lambda_belief * (beta_phi * kl_matrix).sum()

                # Compute ∂F/∂φ (only if alignment_loss has a gradient path to phi)
                if alignment_loss.grad_fn is not None:
                    grad_phi = torch.autograd.grad(
                        alignment_loss,
                        phi_for_grad,
                        create_graph=False,
                        retain_graph=False,
                    )[0]

                    # Clip phi gradient to prevent explosions (especially for GL(K))
                    grad_phi_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
                    grad_phi = torch.where(
                        grad_phi_norm > 10.0,
                        grad_phi * 10.0 / (grad_phi_norm + 1e-6),
                        grad_phi
                    )

                    # Update phi with proper retraction (auto-selects SO(N) or GL(K))
                    # SO(N): trust_region=0.3, bch_order=1 (compact group)
                    # GL(K): trust_region=0.1, bch_order=0 (non-compact, needs care)
                    phi_lr_iter = self.phi_lr / self.n_iterations  # Scale by iterations
                    phi_current = _retract_phi(
                        phi=phi_current,
                        delta_phi=-grad_phi,
                        generators=self.generators,
                        step_size=phi_lr_iter,
                        max_norm=self.phi_max_norm,
                        # trust_region and bch_order auto-selected based on gauge group
                    )

        # =================================================================
        # STEP 5: Optional Phi Evolution via VFE Gradient (after loop)
        # =================================================================
        # This runs when update_phi=True but update_phi_per_iteration=False
        # The belief alignment term F_align = λ·Σ β_ij KL(q_i || Ω_ij[q_j])
        # depends on φ through the transport operator Ω_ij = exp(φ_i)·exp(-φ_j).
        # Only update phi during training (when gradients are enabled)
        # Skip if already updated per-iteration
        if self.update_phi and not self.update_phi_per_iteration and torch.is_grad_enabled():
            # Enable gradients for phi
            phi_for_grad = phi_current.clone().requires_grad_(True)

            if self.multihead_vfe:
                # Multi-head: sum alignment loss over heads
                alignment_loss = torch.tensor(0.0, device=mu_current.device,
                                              dtype=mu_current.dtype)
                block_start = 0
                for h, d_h in enumerate(self.irrep_dims):
                    block_end = block_start + d_h
                    mu_h = mu_current[:, :, block_start:block_end].detach()
                    if sigma_current is None:
                        sigma_h = None
                    elif is_diagonal:
                        sigma_h = sigma_current[:, :, block_start:block_end].detach()
                    else:
                        sigma_h = sigma_current[:, :, block_start:block_end, block_start:block_end].detach()
                    gen_h = self.generators[:, block_start:block_end, block_start:block_end]
                    kappa_h = torch.exp(self.log_kappa_heads[h]).item() if self.log_kappa_heads is not None else self.kappa

                    beta_phi_h_result = compute_attention_weights(
                        mu_q=mu_h, sigma_q=sigma_h,
                        phi=phi_for_grad, generators=gen_h,
                        kappa=kappa_h, epsilon=eps, mask=mask,
                        use_numba=False, return_kl=True,
                        diagonal_covariance=is_diagonal,
                        mask_self_attention=self.mask_self_attention,
                    )
                    if isinstance(beta_phi_h_result, tuple):
                        beta_phi_h, kl_h = beta_phi_h_result
                    else:
                        beta_phi_h = beta_phi_h_result
                        kl_h = beta_phi_h
                    alignment_loss = alignment_loss + self.lambda_belief * (beta_phi_h * kl_h).sum()
                    block_start = block_end
            else:
                # Single-β: original alignment loss
                beta_for_phi = compute_attention_weights(
                    mu_q=mu_current.detach(),
                    sigma_q=sigma_current.detach() if sigma_current is not None else None,
                    phi=phi_for_grad,
                    generators=self.generators,
                    kappa=self.kappa,
                    epsilon=eps,
                    mask=mask,
                    use_numba=False,
                    return_kl=True,
                    diagonal_covariance=is_diagonal,
                    irrep_dims=self.irrep_dims,
                    chunk_size=self.chunk_size,
                    mask_self_attention=self.mask_self_attention,
                )

                if isinstance(beta_for_phi, tuple):
                    beta_phi, kl_matrix = beta_for_phi
                else:
                    beta_phi = beta_for_phi
                    kl_matrix = compute_attention_weights(
                        mu_q=mu_current.detach(),
                        sigma_q=sigma_current.detach() if sigma_current is not None else None,
                        phi=phi_for_grad,
                        generators=self.generators,
                        kappa=self.kappa,
                        epsilon=eps,
                        mask=mask,
                        use_numba=False,
                        return_kl=True,
                        diagonal_covariance=is_diagonal,
                        mask_self_attention=self.mask_self_attention,
                    )[1]

                alignment_loss = self.lambda_belief * (beta_phi * kl_matrix).sum()

            # Compute ∂F/∂φ (only if alignment_loss has a gradient path to phi)
            if alignment_loss.grad_fn is not None:
                grad_phi = torch.autograd.grad(
                    alignment_loss,
                    phi_for_grad,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Clip phi gradient to prevent explosions (especially for GL(K))
                grad_phi_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
                grad_phi = torch.where(
                    grad_phi_norm > 10.0,
                    grad_phi * 10.0 / (grad_phi_norm + 1e-6),
                    grad_phi
                )

                # Proper retraction with trust region (auto-selects SO(N) or GL(K))
                # SO(N): trust_region=0.3, bch_order=1 (compact group)
                # GL(K): trust_region=0.1, bch_order=0 (non-compact, needs care)
                phi_current = _retract_phi(
                    phi=phi_current,
                    delta_phi=-grad_phi,  # Negative gradient for descent
                    generators=self.generators,
                    step_size=self.phi_lr,
                    max_norm=self.phi_max_norm,
                    # trust_region and bch_order auto-selected based on gauge group
                )

        # Return results
        # NOTE: Previously returned .detach() which BREAKS gradient flow!
        # The VFE descent is an "inner loop" optimization, but we still need
        # gradients to flow through the final result to train the embeddings.
        # The detach was likely added to prevent backprop through all iterations,
        # but it completely breaks learning. If memory is an issue, consider
        # gradient checkpointing instead.
        if self.update_sigma:
            return mu_current, sigma_current, phi_current, beta_history
        else:
            return mu_current, None, phi_current, beta_history

    # =========================================================================
    # Pure FEP Mode: Backprop-free Learning via Prior Evolution
    # =========================================================================

    def initialize_priors_from_embeddings(self, mu_embed: torch.Tensor):
        """
        Initialize persistent priors from embedding priors (first batch only).

        In pure FEP mode, priors start at embedding values and then evolve.
        This provides a warm start for prior learning.

        Args:
            mu_embed: (B, N, K) embedding means - we use mean across batch
        """
        if not self.pure_fep_mode:
            return

        if self.prior_initialized:
            return

        B, N, K = mu_embed.shape
        N_update = min(N, self.max_seq_len)

        # Initialize from mean of embedding priors
        with torch.no_grad():
            self.prior_mu[:N_update] = mu_embed[:, :N_update].mean(dim=0)
            self.prior_initialized.fill_(True)

    def get_persistent_priors(self, seq_len: int, batch_size: int, device: torch.device):
        """
        Get persistent priors for the current sequence, expanded across batch.

        Args:
            seq_len: Current sequence length N
            batch_size: Batch size B
            device: Target device

        Returns:
            mu_prior: (B, N, K) persistent prior means
            sigma_prior: (B, N, K) persistent prior variances (diagonal)
        """
        if not self.pure_fep_mode:
            return None, None

        N = min(seq_len, self.max_seq_len)

        # Expand priors across batch - use .clone() to avoid view issues
        mu_prior = self.prior_mu[:N].unsqueeze(0).expand(batch_size, -1, -1).clone()
        sigma_prior = self.prior_sigma[:N].unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Handle sequences longer than max_seq_len (pad with defaults)
        if seq_len > self.max_seq_len:
            pad_len = seq_len - self.max_seq_len
            mu_pad = torch.zeros(batch_size, pad_len, self.embed_dim, device=device)
            sigma_pad = torch.ones(batch_size, pad_len, self.embed_dim, device=device)
            mu_prior = torch.cat([mu_prior, mu_pad], dim=1)
            sigma_prior = torch.cat([sigma_prior, sigma_pad], dim=1)

        return mu_prior, sigma_prior

    def update_priors_from_beliefs(
        self,
        mu_beliefs: torch.Tensor,       # (B, N, K) final beliefs after VFE
        sigma_beliefs: torch.Tensor,    # (B, N, K) belief variances
        prediction_errors: torch.Tensor,  # (B, N) per-position CE loss
        lr: Optional[float] = None,
    ):
        """
        Update persistent priors toward beliefs that successfully predicted.

        This is the LEARNING mechanism in pure FEP mode (replaces backprop):
        - Beliefs with low prediction error are "good" - priors should move toward them
        - Beliefs with high prediction error are "bad" - priors should ignore them
        - Weighting by softmax(-error) gives soft EM-like updates

        The key insight: CE is INSIDE the VFE during forward pass, so beliefs
        have already adjusted to minimize prediction error. We now consolidate
        successful beliefs into priors for future use.

        Args:
            mu_beliefs: (B, N, K) evolved belief means
            sigma_beliefs: (B, N, K) evolved belief variances
            prediction_errors: (B, N) per-position cross-entropy loss
            lr: Learning rate (uses self.prior_lr if None)
        """
        if not self.pure_fep_mode:
            return

        lr = lr if lr is not None else self.prior_lr
        B, N, K = mu_beliefs.shape
        N_update = min(N, self.max_seq_len)
        eps = 1e-6

        with torch.no_grad():
            # Compute position-wise weights from prediction errors
            # Low error = high weight (successful predictions should update priors)
            # Use softmax over batch dimension for each position
            errors_clamped = prediction_errors[:, :N_update].clamp(min=eps, max=20.0)
            weights = F.softmax(-errors_clamped, dim=0)  # (B, N_update)

            # Weighted mean of beliefs across batch for each position
            # Shape: (B, N_update, K) * (B, N_update, 1) -> sum over B -> (N_update, K)
            weighted_mu = (mu_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)
            weighted_sigma = (sigma_beliefs[:, :N_update] * weights.unsqueeze(-1)).sum(dim=0)

            # Compute confidence: inverse mean error per position
            # Higher confidence = larger update
            mean_error = errors_clamped.mean(dim=0)  # (N_update,)
            confidence = 1.0 / (1.0 + mean_error)  # (N_update,) in [0, 0.5]

            # Adaptive learning rate: scale by confidence
            effective_lr = lr * confidence.unsqueeze(-1)  # (N_update, 1)

            # EMA update toward weighted beliefs
            # prior <- (1 - lr) * prior + lr * belief
            self.prior_mu[:N_update] = (
                (1.0 - effective_lr) * self.prior_mu[:N_update] +
                effective_lr * weighted_mu
            )

            # Update sigma with smaller learning rate (more stable)
            sigma_lr = effective_lr * 0.1
            self.prior_sigma[:N_update] = (
                (1.0 - sigma_lr) * self.prior_sigma[:N_update] +
                sigma_lr * weighted_sigma
            ).clamp(min=eps)

            # Track update counts
            self.prior_update_count[:N_update] += 1

    def get_prior_stats(self) -> Dict[str, float]:
        """Get statistics about persistent priors for logging."""
        if not self.pure_fep_mode:
            return {}

        with torch.no_grad():
            active_mask = self.prior_update_count > 0
            n_active = active_mask.sum().item()

            if n_active == 0:
                return {'prior_active_positions': 0}

            active_mu = self.prior_mu[active_mask]
            active_sigma = self.prior_sigma[active_mask]

            return {
                'prior_active_positions': n_active,
                'prior_mu_mean': active_mu.mean().item(),
                'prior_mu_std': active_mu.std().item(),
                'prior_sigma_mean': active_sigma.mean().item(),
                'prior_update_count_mean': self.prior_update_count[active_mask].mean().item(),
            }

    def extra_repr(self) -> str:
        base = (
            f"embed_dim={self.embed_dim}, n_iterations={self.n_iterations}, "
            f"alpha={self.alpha}, lambda_belief={self.lambda_belief}, kappa={self.kappa}"
        )
        if self.pure_fep_mode:
            base += f", pure_fep_mode=True, prior_lr={self.prior_lr}"
        return base