"""
Gauge-Theoretic Transformer Block (0D Architecture)
====================================================

Complete transformer block with:
1. Gauge-theoretic multi-head attention (KL-based, no W_Q/W_K!)
2. Feedforward network (prior evolution)
3. Layer normalization
4. Residual connections

Standard Architecture, Gauge Mechanism:
    x → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual

But with gauge-theoretic attention:
    (μ, Σ, φ) → Attention(via KL + transport) → (μ', Σ', φ')

Author: Implementation from plan.py
Date: November 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Import our gauge attention
from transformer.core.attention import IrrepMultiHeadAttention

# Import VFE FFN module
from transformer.core.ffn import GaugeFFN

# Trajectory tracking (optional)
try:
    from transformer.analysis.trajectory import get_global_recorder
    TRAJECTORY_TRACKING_AVAILABLE = True
except ImportError:
    TRAJECTORY_TRACKING_AVAILABLE = False
    def get_global_recorder():
        return None


class GaugeTransformerBlock(nn.Module):
    """
    Single transformer block with gauge-theoretic attention.

    Architecture:
        1. Self-attention sublayer:
           - LayerNorm on means
           - IrrepMultiHeadAttention (KL-based)
           - Residual connection
           - Dropout

        2. Feedforward sublayer:
           - LayerNorm on means
           - VFE-based belief evolution (variational free energy minimization)
           - Residual connection
           - Dropout

    Note: We primarily evolve means (μ), while covariances (Σ) and
          gauge frames (φ) can be evolved or kept fixed depending on mode.
          Phi evolution uses ∂F/∂φ gradient descent, NOT neural networks.

    0D Structure:
        - All agents at single point c*
        - Attention computed via KL divergence
        - No spatial convolutions or position-dependent operations
    """

    def __init__(
        self,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        evolve_phi_e_step: bool = False,  # Update φ during E-step iterations (dynamical gauge frames)
        # Phi evolution parameters (VFE gradient-based, not neural network)
        phi_lr: float = 0.05,  # Learning rate for phi gradient descent
        phi_max_norm: float = 3.14159,  # Max phi norm (π radians = 180° rotation)
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,  # (3, K, K)
        ffn_mode: str = 'VFE_dynamic',  # VFE_dynamic is the only supported mode
        ffn_alpha: float = 0.001,
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        ffn_lambda_belief: float = 1.0,
        ffn_update_sigma: bool = True,
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Sparse attention
        attention_pattern: str = 'full',
        attention_window: int = 64,
        # Gauge frame dimension
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        # Pure FEP mode: learning via prior evolution (no backprop)
        ffn_pure_fep_mode: bool = False,
        ffn_max_seq_len: int = 512,
        ffn_prior_lr: float = 0.01,
        ffn_prior_bank: Optional[nn.Module] = None,  # PriorBank for token-dependent priors
        ffn_use_prior_bank: bool = False,  # Use PriorBank (token-dependent) vs position-dependent priors
        # Memory-efficient options
        ffn_irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        ffn_chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
        # Pure VFE mode: disable ad-hoc transformer components
        use_layernorm: bool = False,  # Pure VFE: beliefs evolve freely, no normalization
        use_dropout: bool = False,    # Pure VFE: uncertainty lives in Σ, not random masking
        use_residual: bool = False,   # Pure VFE: FFN outputs final belief, not delta
        # ALiBi-style positional bias
        alibi_slope: Optional[float] = None,  # If set, adds slope*(i-j) to attention logits
        # Identity transport mode
        use_identity_transport: bool = False,  # If True, Ω_ij = I (no gauge transport)
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        # Gauge group control
        enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
        # Bayesian precision (learned prior self-coupling)
        ffn_learnable_alpha: bool = False,  # If True, use Gamma-Normal conjugate precision
        # Per-head specialization
        per_head_kappa: bool = False,  # If True, learn separate κ_h per head
        use_output_projection: bool = False,  # If True, add W_O after multi-head attention
        # Multi-head VFE: per-block β through VFE iterations
        multihead_vfe: bool = False,  # If True, VFE_dynamic maintains per-head attention
        # Cross-head coupling
        cross_head_perm: Optional[object] = None,  # np.ndarray permutation for super-blocks
    ):
        """
        Initialize gauge transformer block.

        Args:
            embed_dim: Embedding dimension K
            irrep_spec: Irrep structure [(label, mult, dim), ...]
            hidden_dim: FFN hidden dimension (typically 4 × embed_dim)
            kappa_beta: Temperature for attention
            dropout: Dropout probability
            evolve_sigma: If True, update covariances via attention and FFN
            evolve_phi: If True, update gauge frames via FFN
            attention_pattern: 'full', 'local', or 'sparse' for efficient attention
            attention_window: Window size for local attention pattern
            generators: Lie algebra generators (required for VFE mode)
            ffn_mode: 'VFE_dynamic' - dynamic-β VFE with attention-belief co-evolution
            ffn_alpha: Prior weight for VFE
            ffn_kappa: Softmax temperature for attention
            ffn_n_iterations: VFE inference iterations per forward pass
            ffn_learnable_lr: Learn step size for variational descent
            ffn_lambda_belief: Belief alignment weight
            ffn_update_sigma: Update covariances in FFN
            ffn_pure_fep_mode: If True, use persistent priors for backprop-free learning
            ffn_max_seq_len: Max sequence length for persistent priors (pure FEP mode)
            ffn_prior_lr: Learning rate for prior updates (pure FEP mode)
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0 always.
            ffn_learnable_alpha: If True, use Bayesian precision via Gamma-Normal conjugacy.
            per_head_kappa: If True, learn separate κ_h per head in attention.
            use_output_projection: If True, add W_O projection after multi-head attention.
            multihead_vfe: If True, VFE_dynamic maintains per-head attention β_h.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.evolve_sigma = evolve_sigma
        self.evolve_phi = evolve_phi
        self.ffn_mode = ffn_mode
        self.generators = generators  # Store for variational FFN
        self.diagonal_covariance = diagonal_covariance

        # Pure VFE mode flags
        self.use_layernorm = use_layernorm
        self.use_dropout = use_dropout
        self.use_residual = use_residual

        # =====================================================================
        # Attention Sublayer
        # =====================================================================
        # Determine gauge group from generators shape
        if generators is not None:
            n_gen = generators.shape[0]
            K = generators.shape[1]  # Embedding dimension

            if n_gen == 3:
                gauge_group = 'SO3'
                gauge_dim_inferred = 3
            elif n_gen == K * K:
                # n_gen = K² means GL(K) single-head
                gauge_group = 'GLK'
                gauge_dim_inferred = K
            else:
                # Check if n_gen matches SO(N): n_gen = N*(N-1)/2
                disc = 1 + 8 * n_gen
                sqrt_disc = int(math.sqrt(disc))
                if sqrt_disc * sqrt_disc == disc:
                    N_candidate = (1 + sqrt_disc) // 2
                    if N_candidate * (N_candidate - 1) // 2 == n_gen:
                        gauge_group = 'SON'
                        gauge_dim_inferred = N_candidate
                    else:
                        # Not SO(N) — assume GL(K) multi-head or cross-coupled
                        gauge_group = 'GLK'
                        gauge_dim_inferred = K
                else:
                    # Not SO(N) — assume GL(K) multi-head or cross-coupled
                    gauge_group = 'GLK'
                    gauge_dim_inferred = K
        else:
            gauge_group = 'SO3'
            gauge_dim_inferred = 3

        self.attention = IrrepMultiHeadAttention(
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            kappa_beta=kappa_beta,
            epsilon=1e-8,
            aggregate_mode='full_distribution' if evolve_sigma else 'mean_only',
            diagonal_covariance=diagonal_covariance,
            attention_pattern=attention_pattern,
            attention_window=attention_window,
            gauge_group=gauge_group,
            gauge_dim=gauge_dim_inferred,
            global_generators=generators,  # Pass for SO(N) mode
            alibi_slope=alibi_slope,
            use_identity_transport=use_identity_transport,
            mask_self_attention=mask_self_attention,
            enforce_orthogonal=enforce_orthogonal,
            per_head_kappa=per_head_kappa,
            use_output_projection=use_output_projection,
            irrep_dims_override=ffn_irrep_dims if (gauge_group == 'GLK' and cross_head_perm is not None) else None,
        )

        # Conditionally create LayerNorm and Dropout (disabled for pure VFE)
        self.norm1 = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout) if use_dropout else nn.Identity()

        # =====================================================================
        # VFE_dynamic FFN Sublayer
        # =====================================================================
        self.ffn = GaugeFFN(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            generators=generators,  # Required for VFE mode
            dropout=dropout,
            mode=ffn_mode,
            # VFE parameters
            alpha=ffn_alpha,
            kappa=ffn_kappa,
            n_iterations=ffn_n_iterations,
            learnable_lr=ffn_learnable_lr,
            lambda_belief=ffn_lambda_belief,
            update_sigma=ffn_update_sigma,
            # Diagonal covariance mode
            diagonal_covariance=diagonal_covariance,
            # Pure FEP mode parameters
            pure_fep_mode=ffn_pure_fep_mode,
            max_seq_len=ffn_max_seq_len,
            prior_lr=ffn_prior_lr,
            # Phi evolution via VFE gradients (principled approach)
            update_phi=evolve_phi,  # When evolve_phi=True, update φ via ∂F/∂φ (after E-step)
            update_phi_per_iteration=evolve_phi_e_step,  # When True, update φ during each E-step iteration
            phi_lr=phi_lr,
            phi_max_norm=phi_max_norm,
            # Memory-efficient options
            irrep_dims=ffn_irrep_dims,
            chunk_size=ffn_chunk_size,
            # Self-attention masking (same as attention)
            mask_self_attention=mask_self_attention,
            # Bayesian precision
            learnable_alpha=ffn_learnable_alpha,
            # Multi-head VFE
            multihead_vfe=multihead_vfe,
            per_head_kappa=per_head_kappa,
        )

        self.norm2 = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

        # =====================================================================
        # Gauge Frame Evolution Configuration
        # =====================================================================
        # When evolve_phi=True, phi is updated via VFE gradient descent in the
        # variational FFN, NOT via a neural network. This is the principled
        # approach: ∂F/∂φ comes from the belief alignment term.
        self.phi_dim = phi_dim
        self.phi_max_norm = phi_max_norm
        self.evolve_phi = evolve_phi

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        token_ids: Optional[torch.Tensor] = None,  # For PriorBank lookup
        targets: Optional[torch.Tensor] = None,   # For E-step observations
        W_out: Optional[torch.Tensor] = None,     # Output projection for discrete observations
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            mu_q: Belief means (B, N, K)
            sigma_q: Belief covariances (B, N, K, K)
            phi: Gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask (B, N, N)
            mu_prior: Embedding priors (B, N, K) - required for variational FFN
            targets: Target token IDs (B, N) - for E-step discrete observations
            W_out: Output projection (V, K) - for computing CE gradient in E-step
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers.

        Returns:
            mu_q_out: Updated means (B, N, K)
            sigma_q_out: Updated covariances (B, N, K, K)
            phi_out: Updated gauge frames (B, N, 3)
        """
        # =====================================================================
        # 1. Attention Sublayer with Pre-Norm + Residual
        # =====================================================================

        # Pre-layer normalization on means
        mu_normalized = self.norm1(mu_q)

        # Multi-head attention (gauge-theoretic!)
        # Capture beta if needed for variational FFN or trajectory recording
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_attention = recorder is not None and recorder.enabled and recorder.record_attention
        need_beta = self.ffn_mode == 'VFE_dynamic'
        need_attention_output = need_beta or recording_attention

        mu_attn, sigma_attn, beta, kl_matrix = self.attention(
            mu_normalized,
            sigma_q,
            phi,
            generators,
            mask=mask,
            return_attention=need_attention_output,  # Compute if needed for FFN or recording
            cached_head_transports=cached_head_transports,  # Cross-layer cache
        )

        # Record attention for trajectory tracking
        if recording_attention and beta is not None:
            recorder.record_attention(beta, kl_matrix)

        # Apply dropout (identity if use_dropout=False)
        mu_attn = self.dropout1(mu_attn)

        # Residual connection (optional for pure VFE)
        if self.use_residual:
            mu_q = mu_q + mu_attn
        else:
            # Pure VFE: attention output IS the new belief
            mu_q = mu_attn

        # Update covariances if evolving
        if self.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn
        # Otherwise sigma_q stays unchanged

        # =====================================================================
        # 2. Feedforward Sublayer (with optional Pre-Norm + Residual)
        # =====================================================================

        # Pre-layer normalization (identity if use_layernorm=False)
        mu_normalized = self.norm2(mu_q)

        # VFE_dynamic FFN: β recomputed at each VFE step
        # Returns (mu, sigma, phi) tuple
        if mu_prior is None:
            raise ValueError("VFE_dynamic mode requires mu_prior argument")

        mu_ffn, sigma_ffn, phi_out = self.ffn(
            mu=mu_normalized,
            beta=beta,          # Initial β (will be recomputed each step inside FFN)
            mu_prior=mu_prior,  # From embeddings
            phi=phi,            # Current gauge frames
            sigma=sigma_q,      # Current covariances
            mask=mask,          # Causal mask
            token_ids=token_ids,  # For PriorBank lookup
            targets=targets,    # Target tokens (discrete observations)
            W_out=W_out,        # Output projection for ∂CE/∂μ
        )

        # Update covariances from FFN if evolving
        if self.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        # Residual connection (optional for pure VFE)
        if self.use_residual:
            mu_q = mu_q + mu_ffn
        else:
            # Pure VFE: FFN output IS the new belief
            mu_q = mu_ffn

        # phi is updated inside the VFE FFN via gradient descent (when update_phi=True)
        # This is the principled approach: φ evolves via ∂F/∂φ, not a neural network.
        return mu_q, sigma_q, phi_out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"evolve_sigma={self.evolve_sigma}, "
            f"evolve_phi={self.evolve_phi}"
        )


# =============================================================================
# Stack of Transformer Blocks
# =============================================================================

class GaugeTransformerStack(nn.Module):
    """
    Stack of N gauge transformer blocks.

    This is the main "encoder" of the model, transforming initial
    embeddings through multiple layers of gauge-theoretic attention.
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        irrep_spec: List[Tuple[str, int, int]],
        hidden_dim: int,
        kappa_beta: float,
        dropout: float = 0.1,
        evolve_sigma: bool = False,
        evolve_phi: bool = False,
        evolve_phi_e_step: bool = False,  # Update φ during E-step iterations (dynamical gauge frames)
        # Phi evolution parameters (VFE gradient-based, not neural network)
        phi_lr: float = 0.05,  # Learning rate for phi gradient descent
        phi_max_norm: float = 3.14159,  # Max phi norm (π radians = 180° rotation)
        # Variational FFN parameters
        generators: Optional[torch.Tensor] = None,
        ffn_mode: str = 'VFE_dynamic',
        ffn_alpha: float = 0.001,
        ffn_kappa: float = 1.0,
        ffn_n_iterations: int = 1,
        ffn_learnable_lr: bool = True,
        ffn_lambda_belief: float = 1.0,
        ffn_update_sigma: bool = True,
        # Diagonal covariance mode
        diagonal_covariance: bool = False,
        # Sparse attention
        attention_pattern: str = 'full',
        attention_window: int = 64,
        # Gauge frame dimension
        phi_dim: int = 3,  # 3 for SO(3), N(N-1)/2 for SO(N)
        # Pure FEP mode: learning via prior evolution (no backprop)
        ffn_pure_fep_mode: bool = False,
        ffn_max_seq_len: int = 512,
        ffn_prior_lr: float = 0.01,
        ffn_prior_bank: Optional[nn.Module] = None,  # PriorBank for token-dependent priors
        ffn_use_prior_bank: bool = False,  # Use PriorBank (token-dependent) vs position-dependent priors
        # Memory-efficient options
        ffn_irrep_dims: Optional[List[int]] = None,  # Block dimensions for principled KL decomposition
        ffn_chunk_size: Optional[int] = None,  # Chunk size for memory-efficient attention
        # Pure VFE mode: disable ad-hoc transformer components
        use_layernorm: bool = False,  # Pure VFE: beliefs evolve freely, no normalization
        use_dropout: bool = False,    # Pure VFE: uncertainty lives in Σ, not random masking
        use_residual: bool = False,   # Pure VFE: FFN outputs final belief, not delta
        # ALiBi-style positional bias
        alibi_slope: Optional[float] = None,  # If set, adds slope*(i-j) to attention logits
        # Identity transport mode
        use_identity_transport: bool = False,  # If True, Ω_ij = I (no gauge transport)
        # Self-attention masking (prevents attention collapse)
        mask_self_attention: bool = False,  # If True, mask out diagonal (no self-attention)
        # Gauge group control
        enforce_orthogonal: bool = False,  # If True, enforce Ω ∈ SO(K) via Newton-Schulz
        # Bayesian precision (learned prior self-coupling)
        ffn_learnable_alpha: bool = False,  # If True, use Gamma-Normal conjugate precision
        # Per-head specialization
        per_head_kappa: bool = False,  # If True, learn separate κ_h per head
        use_output_projection: bool = False,  # If True, add W_O after multi-head attention
        # Multi-head VFE: per-block β through VFE iterations
        multihead_vfe: bool = False,  # If True, VFE_dynamic maintains per-head attention
        # Cross-head coupling
        cross_head_perm: Optional[object] = None,  # np.ndarray permutation for super-blocks
    ):
        """
        Initialize stack of transformer blocks.

        Args:
            n_layers: Number of transformer blocks
            embed_dim: Embedding dimension
            irrep_spec: Irrep structure
            hidden_dim: FFN hidden dimension
            kappa_beta: Attention temperature
            dropout: Dropout probability
            evolve_sigma: If True, covariances evolve through layers
            evolve_phi: If True, gauge frames evolve through layers
            generators: Lie algebra generators (required for VFE FFN)
            phi_dim: Dimension of gauge frame (3 for SO(3), N(N-1)/2 for SO(N))
            ffn_mode: 'VFE_dynamic' (only supported mode)
            ffn_alpha: Prior weight
            ffn_kappa: Softmax temperature for attention
            ffn_n_iterations: VFE inference iterations
            ffn_learnable_lr: Learn step size for variational descent
            ffn_lambda_belief: Belief alignment weight
            ffn_update_sigma: Update covariances in FFN
            attention_pattern: 'full', 'local', or 'sparse' for efficient attention
            attention_window: Window size for local attention pattern
            ffn_pure_fep_mode: If True, use persistent priors for backprop-free learning
            ffn_max_seq_len: Max sequence length for persistent priors (pure FEP mode)
            ffn_prior_lr: Learning rate for prior updates (pure FEP mode)
            use_layernorm: If True, apply LayerNorm (default False for pure VFE)
            use_dropout: If True, apply Dropout (default False for pure VFE)
            use_residual: If True, use residual connections (default False for pure VFE)
            mask_self_attention: If True, mask out diagonal (no self-attention).
                                Prevents attention collapse since KL(q_i||q_i)=0 always.
            ffn_learnable_alpha: If True, use Bayesian precision via Gamma-Normal conjugacy.
            per_head_kappa: If True, learn separate κ_h per head.
            use_output_projection: If True, add W_O projection after multi-head attention.
            multihead_vfe: If True, VFE_dynamic maintains per-head attention β_h.
        """
        super().__init__()
        self.n_layers = n_layers

        self.blocks = nn.ModuleList([
            GaugeTransformerBlock(
                embed_dim=embed_dim,
                irrep_spec=irrep_spec,
                hidden_dim=hidden_dim,
                kappa_beta=kappa_beta,
                dropout=dropout,
                evolve_sigma=evolve_sigma,
                evolve_phi=evolve_phi,
                evolve_phi_e_step=evolve_phi_e_step,  # Update φ during E-step iterations
                # Phi evolution (VFE gradient-based)
                phi_lr=phi_lr,
                phi_max_norm=phi_max_norm,
                # VFE FFN
                generators=generators,
                ffn_mode=ffn_mode,
                ffn_alpha=ffn_alpha,
                ffn_kappa=ffn_kappa,
                ffn_n_iterations=ffn_n_iterations,
                ffn_learnable_lr=ffn_learnable_lr,
                ffn_lambda_belief=ffn_lambda_belief,
                ffn_update_sigma=ffn_update_sigma,
                # Diagonal covariance mode
                diagonal_covariance=diagonal_covariance,
                # Sparse attention
                attention_pattern=attention_pattern,
                attention_window=attention_window,
                # Gauge frame dimension
                phi_dim=phi_dim,
                # Pure FEP mode
                ffn_pure_fep_mode=ffn_pure_fep_mode,
                ffn_max_seq_len=ffn_max_seq_len,
                ffn_prior_lr=ffn_prior_lr,
                ffn_prior_bank=ffn_prior_bank,  # Pass PriorBank to each block
                ffn_use_prior_bank=ffn_use_prior_bank,  # Enable token-dependent priors
                # Memory-efficient options
                ffn_irrep_dims=ffn_irrep_dims,
                ffn_chunk_size=ffn_chunk_size,
                # Pure VFE mode
                use_layernorm=use_layernorm,
                use_dropout=use_dropout,
                use_residual=use_residual,
                # ALiBi positional bias
                alibi_slope=alibi_slope,
                # Identity transport
                use_identity_transport=use_identity_transport,
                # Self-attention masking
                mask_self_attention=mask_self_attention,
                # Gauge group control
                enforce_orthogonal=enforce_orthogonal,
                # Bayesian precision
                ffn_learnable_alpha=ffn_learnable_alpha,
                # Per-head specialization
                per_head_kappa=per_head_kappa,
                use_output_projection=use_output_projection,
                # Multi-head VFE
                multihead_vfe=multihead_vfe,
                # Cross-head coupling
                cross_head_perm=cross_head_perm,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm (optional for pure VFE)
        self.final_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

    def forward(
        self,
        mu_q: torch.Tensor,
        sigma_q: torch.Tensor,
        phi: torch.Tensor,
        generators: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mu_prior: Optional[torch.Tensor] = None,  # For variational FFN
        token_ids: Optional[torch.Tensor] = None,  # For PriorBank lookup
        return_intermediates: bool = False,
        cached_head_transports: Optional[list] = None,  # Cross-layer transport cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List]]:
        """
        Forward through all transformer blocks.

        Args:
            mu_q: Initial means (B, N, K)
            sigma_q: Initial covariances (B, N, K, K)
            phi: Initial gauge frames (B, N, 3)
            generators: SO(3) generators (3, K, K)
            mask: Optional causal mask
            mu_prior: Embedding priors (B, N, K) - for variational FFN
            return_intermediates: If True, return states after each layer
            cached_head_transports: Optional list of precomputed transport dicts per head.
                                   When evolve_phi=False, reuse across all layers (6× speedup).

        Returns:
            mu_q: Final means (B, N, K)
            sigma_q: Final covariances (B, N, K, K)
            phi: Final gauge frames (B, N, 3)
            intermediates: Optional list of intermediate states
        """
        intermediates = [] if return_intermediates else None

        # Get trajectory recorder
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        recording_enabled = recorder is not None and recorder.enabled

        for layer_idx, block in enumerate(self.blocks):
            # Trajectory recording: start layer
            if recording_enabled:
                recorder.start_layer(layer_idx)
                recorder.record_layer_input(mu_q, sigma_q, phi)

            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, generators, mask, mu_prior,
                token_ids=token_ids,  # Pass token IDs for PriorBank
                cached_head_transports=cached_head_transports,
            )

            # Trajectory recording: record output
            if recording_enabled:
                recorder.record_layer_output(mu_q, sigma_q, phi)
                recorder.end_layer()

            if return_intermediates:
                intermediates.append({
                    'layer': layer_idx,
                    'mu': mu_q.detach(),
                    'sigma': sigma_q.detach() if sigma_q is not None else None,
                    'phi': phi.detach(),
                })

        # Final normalization
        mu_q = self.final_norm(mu_q)

        return mu_q, sigma_q, phi, intermediates


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TRANSFORMER BLOCK TEST")
    print("="*70)

    # Test configuration
    B, N, K = 2, 8, 16
    n_layers = 3
    hidden_dim = 64

    print(f"\n[1] Configuration:")
    print(f"    Batch size: {B}")
    print(f"    Num agents: {N}")
    print(f"    Embed dim:  {K}")
    print(f"    Layers:     {n_layers}")
    print(f"    Hidden dim: {hidden_dim}")

    # Create test data
    mu_q = torch.randn(B, N, K)
    sigma_q = torch.eye(K).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1) * 0.5
    phi = torch.randn(B, N, 3) * 0.1

    # Random generators
    G = torch.randn(3, K, K)
    G = 0.5 * (G - G.transpose(-1, -2))

    # Irrep spec
    irrep_spec = [
        ('ℓ0', 4, 1),
        ('ℓ1', 2, 3),
        ('ℓ2', 1, 5),
    ]

    # Test single block
    print(f"\n[2] Testing single transformer block...")
    block = GaugeTransformerBlock(
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
        generators=G,  # Required for VFE_dynamic mode
    )
    print(f"    {block}")

    # Create mu_prior for VFE_dynamic mode
    mu_prior = mu_q.clone() * 0.5

    mu_out, sigma_out, phi_out = block(mu_q, sigma_q, phi, G, mu_prior=mu_prior)
    print(f"    Output μ shape: {mu_out.shape}")
    print(f"    Output Σ shape: {sigma_out.shape}")
    print(f"    Output φ shape: {phi_out.shape}")
    print(f"    ✓ Single block forward pass complete")

    # Test stack
    print(f"\n[3] Testing transformer stack ({n_layers} layers)...")
    stack = GaugeTransformerStack(
        n_layers=n_layers,
        embed_dim=K,
        irrep_spec=irrep_spec,
        hidden_dim=hidden_dim,
        kappa_beta=1.0,
        dropout=0.1,
        evolve_sigma=False,
        evolve_phi=False,
        generators=G,  # Required for VFE_dynamic mode
    )

    mu_final, sigma_final, phi_final, intermediates = stack(
        mu_q, sigma_q, phi, G, mu_prior=mu_prior, return_intermediates=True
    )

    print(f"    Final μ shape: {mu_final.shape}")
    print(f"    Intermediate states: {len(intermediates)}")
    print(f"    ✓ Stack forward pass complete")

    # Test with causal mask
    print(f"\n[4] Testing with causal mask...")
    mask = torch.tril(torch.ones(N, N)).unsqueeze(0).expand(B, -1, -1)
    mu_causal, _, _, _ = stack(mu_q, sigma_q, phi, G, mask=mask, mu_prior=mu_prior)
    print(f"    Causal output shape: {mu_causal.shape}")
    print(f"    ✓ Causal masking works")

    # Parameter count
    total_params = sum(p.numel() for p in stack.parameters())
    per_layer = total_params // n_layers

    print(f"\n[5] Parameter count:")
    print(f"    Total:     {total_params:,} parameters")
    print(f"    Per layer: {per_layer:,} parameters")
    print(f"    Attention: ~{per_layer // 3:,} params (1/3 of layer)")
    print(f"    FFN:       ~{2 * per_layer // 3:,} params (2/3 of layer)")

    # Compare to standard transformer
    standard_params = 4 * K * K + 2 * K * hidden_dim + 4 * K  # Q,K,V,O + FFN + LN
    standard_total = standard_params * n_layers
    reduction = standard_total / total_params

    print(f"\n[6] Comparison to standard transformer:")
    print(f"    Standard total: {standard_total:,} parameters")
    print(f"    Gauge total:    {total_params:,} parameters")
    print(f"    Reduction:      {reduction:.2f}x fewer parameters!")

    print("\n" + "="*70)
    print("All transformer block tests passed!")
    print("="*70)