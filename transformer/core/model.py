"""
Complete Gauge-Theoretic Language Model (0D Architecture)
==========================================================

Full transformer language model using gauge theory and active inference.

Architecture:
    Token Embedding → N × Transformer Blocks → Output Projection

Key Innovation: Attention via KL divergence on statistical manifold,
                no learned W_Q, W_K matrices!

0D Structure: All agents at single point c* (standard transformer topology)

Author: Implementation from plan.py
Date: November 2025
"""

# Suppress noisy warnings BEFORE torch import (torch may trigger imports)
import warnings
warnings.filterwarnings("ignore", message="Failed to find cuobjdump", module="triton")
warnings.filterwarnings("ignore", message="Failed to find nvdisasm", module="triton")
warnings.filterwarnings("ignore", message="CUDA path could not be detected", module="cupy")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import numpy as np

# Import our components
from transformer.core.embeddings import GaugeTokenEmbedding
from transformer.core.blocks import GaugeTransformerStack
from transformer.core.attention import create_attention_mask

# Trajectory tracking (optional)
try:
    from transformer.analysis.trajectory import get_global_recorder
    TRAJECTORY_TRACKING_AVAILABLE = True
except ImportError:
    TRAJECTORY_TRACKING_AVAILABLE = False
    def get_global_recorder():
        return None

# Try to import generators (fallback to random if unavailable)
try:
    from math_utils.generators import (
        generate_so3_generators,
        generate_soN_generators,
        generate_multi_irrep_generators,
        generate_multi_irrep_soN_generators,
        generate_glK_generators,
        generate_glK_multihead_generators,
        generate_glK_cross_head_generators,
        merge_coupled_heads,
        reorder_cross_head_generators,
    )
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False


class GaugeTransformerLM(nn.Module):
    """
    Complete gauge-theoretic language model.

    Architecture Flow:
        token_ids → (μ, Σ, φ) → Transformer Stack → μ_final → logits

    Components:
        1. GaugeTokenEmbedding: Maps tokens to beliefs
        2. GaugeTransformerStack: N layers of gauge attention
        3. Output projection: μ → logits over vocabulary

    0D Structure:
        - All N tokens → N agents at single point c*
        - No spatial structure, sequence via agent index
        - Attention β_ij are scalars (not fields)
    """

    def __init__(self, config: Dict):
        """
        Initialize gauge transformer language model.

        Args:
            config: Dictionary with model hyperparameters:
                - vocab_size: Vocabulary size
                - embed_dim: Embedding dimension K
                - n_layers: Number of transformer blocks
                - irrep_spec: Irrep structure [(label, mult, dim), ...]
                - hidden_dim: FFN hidden dimension
                - max_seq_len: Maximum sequence length
                - kappa_beta: Attention temperature
                - dropout: Dropout probability
                - evolve_sigma: If True, evolve covariances
                - evolve_phi: If True, evolve gauge frames
                - tie_embeddings: If True, tie input/output embeddings
        """
        super().__init__()
        self.config = config

        # Extract config
        vocab_size = config['vocab_size']
        embed_dim = config['embed_dim']
        n_layers = config['n_layers']
        irrep_spec = config['irrep_spec']
        hidden_dim = config['hidden_dim']
        max_seq_len = config['max_seq_len']
        kappa_beta = config['kappa_beta']
        dropout = config.get('dropout', 0.1)
        evolve_sigma = config.get('evolve_sigma', True)
        evolve_phi = config.get('evolve_phi', True)
        evolve_phi_e_step = config.get('evolve_phi_e_step', False)  # Update φ during E-step iterations
        tie_embeddings = config.get('tie_embeddings', True)

        # VFE FFN config
        ffn_mode = config.get('ffn_mode', 'VFE_dynamic')
        # Allow separate alpha for FFN E-step vs external loss
        # ffn_alpha controls the self-coupling strength INSIDE the VFE loop
        # config['alpha'] controls the external KL(q||p) loss term
        # By default they're the same (backward compatible), but decoupling
        # enables proper EM: VFE handles self-coupling internally, external loss is pure CE
        ffn_alpha = config.get('ffn_alpha', config.get('alpha', 0.001))  # E-step prior weight
        ffn_kappa = kappa_beta  # Unified: use same temperature for attention and FFN
        ffn_n_iterations = config.get('ffn_n_iterations', 1)
        ffn_learnable_lr = config.get('ffn_learnable_lr', True)
        ffn_lambda_belief = config.get('ffn_lambda_belief', 1.0)
        ffn_update_sigma = config.get('ffn_update_sigma', True)

        # Bayesian precision: Gamma-Normal conjugate prior for α
        ffn_learnable_alpha = config.get('learnable_alpha', False)

        # Pure FEP mode: learning via prior evolution (no backprop)
        ffn_pure_fep_mode = config.get('ffn_pure_fep_mode', False)
        ffn_max_seq_len = config.get('ffn_max_seq_len', max_seq_len)
        ffn_prior_lr = config.get('ffn_prior_lr', 0.01)

        # PriorBank positioning: token-dependent (True) vs position-dependent (False, legacy)
        # CRITICAL: Token-dependent is REQUIRED for language modeling!
        use_prior_bank = config.get('use_prior_bank', False)  # Toggle for PriorBank

        # Pure FEP requires untied embeddings!
        # - Input embeddings (W_in) = priors, updated via p-flow
        # - Output embeddings (W_out) = observation anchors, fixed
        if ffn_pure_fep_mode and tie_embeddings:
            print("Pure FEP mode: automatically untying embeddings (priors ≠ observations)")
            tie_embeddings = False

        # Gauge-fixed priors (for gauge covariance)
        gauge_fixed_priors = config.get('gauge_fixed_priors', False)

        # Diagonal covariance mode (memory optimization)
        diagonal_covariance = config.get('diagonal_covariance', False)
        self.diagonal_covariance = diagonal_covariance

        # Identity transport mode: Ω_ij = I for all pairs (bypasses gauge transport)
        # This is now primarily controlled by gauge_mode='trivial' for principled use.
        # Direct use_identity_transport config is kept for backward compatibility.
        use_identity_transport = config.get('use_identity_transport', False)

        # Store evolve_phi for cross-layer transport caching optimization
        self.evolve_phi = evolve_phi

        # Sparse attention/FFN config
        self.attention_pattern = config.get('attention_pattern', 'full')
        self.attention_window = config.get('attention_window', 64)
        self.ffn_pattern = config.get('ffn_pattern', 'full')
        self.ffn_window = config.get('ffn_window', 64)

        # =================================================================
        # Gauge Group and Mode (SO(3), SO(N), or GL(K))
        # =================================================================
        gauge_group = config.get('gauge_group', 'SO3')
        gauge_dim = config.get('gauge_dim', 3)  # N for SO(N), K for GL(K)
        use_multi_irrep = config.get('use_multi_irrep', False)

        # =================================================================
        # Gauge Mode: Controls transport operator behavior
        # =================================================================
        # 'learned': Per-token gauge frames φ_i, transport Ω_ij = exp(φ_i)·exp(-φ_j)
        #            This is the full gauge-theoretic attention.
        # 'trivial': Global frame (φ = 0), transport Ω = I (identity)
        #            This is the "trivial gauge fixing" that recovers standard
        #            attention as a special case. Mathematically principled:
        #            choosing a gauge where all tokens share one coordinate frame.
        #            KL(q_i || Ω[q_j]) = KL(q_i || q_j) when Ω = I.
        gauge_mode = config.get('gauge_mode', 'learned')
        if gauge_mode not in ('learned', 'trivial'):
            raise ValueError(f"gauge_mode must be 'learned' or 'trivial', got '{gauge_mode}'")

        # Store gauge group info for position encoding and other components
        self.gauge_group = gauge_group
        self.gauge_dim = gauge_dim
        self.gauge_mode = gauge_mode

        # Trivial gauge mode → Ω = I, no phi evolution
        # This is the mathematically principled "gauge fixing" to a global frame
        if gauge_mode == 'trivial':
            use_identity_transport = True  # Ω = I for all pairs
            evolve_phi = False  # No point updating φ when transport is identity
            evolve_phi_e_step = False
            print(f"[INFO] Trivial gauge mode: φ = 0, Ω = I (global frame / standard attention limit)")
            print(f"       This recovers standard KL-attention: KL(q_i || q_j) with no transport.")

        # =================================================================
        # Cross-Head Coupling (sparse off-diagonal gauge mixing)
        # =================================================================
        # cross_couplings: list of (head_a, head_b) pairs enabling gauge
        # transport between those heads. Empty list = block-diagonal (default).
        cross_couplings = config.get('cross_couplings', [])
        self.cross_couplings = cross_couplings

        # Compute phi dimension (number of generators)
        if gauge_group == 'SO3':
            self.phi_dim = 3  # SO(3) has 3 generators
        elif gauge_group == 'GLK':
            # GL(K): Check if multi-head requested
            is_glk_multihead = (
                use_multi_irrep and
                irrep_spec is not None and
                len(irrep_spec) == 1 and
                irrep_spec[0][0] != 'full' and
                irrep_spec[0][1] > 1  # n_heads > 1
            )
            if is_glk_multihead:
                # Multi-head GL(K): H × d_head² generators + cross-coupling generators
                _, n_heads, d_head = irrep_spec[0]
                n_cross_gen = len(cross_couplings) * d_head * d_head
                self.phi_dim = n_heads * d_head * d_head + n_cross_gen
            else:
                # Single-head GL(K): K² generators (cross_couplings ignored)
                self.phi_dim = embed_dim * embed_dim
        else:  # SO(N)
            self.phi_dim = gauge_dim * (gauge_dim - 1) // 2  # SO(N) has N(N-1)/2 generators

        if GENERATORS_AVAILABLE:
            if gauge_group == 'SO3':
                if use_multi_irrep and irrep_spec is not None:
                    generators = generate_multi_irrep_generators(irrep_spec)
                else:
                    generators = generate_so3_generators(embed_dim)
            elif gauge_group == 'GLK':
                # GL(K): Check if multi-head requested via irrep_spec
                # Multi-head: irrep_spec = [('fund', n_heads, d_head)] where n_heads * d_head = embed_dim
                # Single-head: irrep_spec = [('full', 1, embed_dim)] or no special format
                is_multihead = (
                    use_multi_irrep and
                    irrep_spec is not None and
                    len(irrep_spec) == 1 and
                    irrep_spec[0][0] != 'full' and
                    irrep_spec[0][1] > 1  # n_heads > 1
                )

                if is_multihead:
                    _, n_heads, d_head = irrep_spec[0]

                    if cross_couplings:
                        # Cross-head coupling: sparse off-diagonal generators
                        generators = generate_glK_cross_head_generators(
                            embed_dim, n_heads, cross_couplings
                        )
                        # Compute super-block structure
                        super_block_dims, super_block_head_groups = merge_coupled_heads(
                            n_heads, d_head, cross_couplings
                        )
                        # Reorder so merged heads are contiguous
                        generators, perm = reorder_cross_head_generators(
                            generators, n_heads, d_head,
                            cross_couplings, super_block_head_groups,
                        )
                        self._cross_head_perm = perm  # Stored for embedding reordering
                        self._super_block_dims = super_block_dims
                        self._super_block_head_groups = super_block_head_groups

                        n_cross = len(cross_couplings) * d_head**2
                        print(f"[INFO] GL(K) cross-head: {n_heads} heads × GL({d_head}), "
                              f"{n_heads * d_head**2} diag + {n_cross} cross generators = "
                              f"{generators.shape[0]} total")
                        print(f"       Super-blocks: {super_block_dims} "
                              f"(groups: {super_block_head_groups})")
                    else:
                        # Standard multi-head GL(K): block-diagonal generators
                        generators = generate_glK_multihead_generators(embed_dim, n_heads)
                        self._cross_head_perm = None
                        self._super_block_dims = None
                        self._super_block_head_groups = None
                        print(f"[INFO] GL(K) multi-head: {n_heads} heads × GL({d_head}), "
                              f"{n_heads * d_head**2} generators (vs {embed_dim**2} single-head)")
                else:
                    # Single-head GL(K): full K² generators
                    generators = generate_glK_generators(embed_dim)
                    print(f"[INFO] GL(K) single-head: {embed_dim}² = {embed_dim**2} generators")
            else:  # SO(N)
                if use_multi_irrep and irrep_spec is not None:
                    generators = generate_multi_irrep_soN_generators(irrep_spec, gauge_dim)
                else:
                    generators = generate_soN_generators(gauge_dim)
                    # For single-irrep SO(N), embed_dim should equal gauge_dim
                    if embed_dim != gauge_dim:
                        print(f"[WARNING] SO(N) with N={gauge_dim} but embed_dim={embed_dim}. "
                              f"Consider using multi-irrep mode for embed_dim != N.")
        else:
            # Fallback: random skew-symmetric matrices (should never happen!)
            # math_utils/generators.py should always be available
            import warnings
            warnings.warn(
                "GENERATORS_AVAILABLE=False: math_utils/generators.py import failed! "
                "Using random fallback generators. This may indicate a broken installation.",
                RuntimeWarning
            )
            n_generators = self.phi_dim
            # Use a fixed seed for reproducibility even if global seed wasn't set
            rng = np.random.RandomState(seed=42)
            generators = rng.randn(n_generators, embed_dim, embed_dim)
            generators = 0.5 * (generators - generators.transpose(0, 2, 1))

        self.register_buffer(
            'generators',
            torch.from_numpy(generators).float()
        )

        # =================================================================
        # Embedding Layers
        # =================================================================
        self.token_embed = GaugeTokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            init_std=config.get('mu_init_std', None),  # Embedding init std (None = default 2.0)
            init_sigma_scale=1.0,  # Scaled to match init_std for O(1) KL
            learnable_sigma=config.get('evolve_sigma', True),  # Learn per-token covariances
            learnable_phi=True,  # Always learn phi for gauge structure. Required for non-trivial transport Ω_ij.
            gauge_fixed_priors=gauge_fixed_priors,
            generators=self.generators,  # Always pass generators for gauge transport
            diagonal_covariance=diagonal_covariance,
            max_seq_len=max_seq_len,
            phi_dim=self.phi_dim,  # SO(3): 3, SO(N): N(N-1)/2
            phi_scale=config.get('phi_scale', 0.3),  # Gauge frame init scale (higher for clustering)
            # Mean embedding normalization options
            mu_normalize=config.get('mu_normalize', False),
            mu_max_norm=config.get('mu_max_norm', None),
        )

        # =================================================================
        # PriorBank for Pure FEP (Token-Dependent Priors)
        # =================================================================
        self.prior_bank = None
        if ffn_pure_fep_mode and use_prior_bank:
            from transformer.core.prior_bank import PriorBank

            self.prior_bank = PriorBank(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                init_std=config.get('mu_init_std', None),
                init_sigma_scale=1.0,
                learnable_sigma=config.get('evolve_sigma', True),
                gauge_fixed_priors=gauge_fixed_priors,  # Can use gauge-fixed priors in PriorBank too
                generators=self.generators if gauge_fixed_priors else None,
                phi_dim=self.phi_dim,
            )
            print(f"[GaugeTransformerLM] Created PriorBank with token-dependent priors (vocab_size={vocab_size})")
            print(f"                     gauge_fixed_priors={gauge_fixed_priors}")

        # =================================================================
        # Transformer Stack
        # =================================================================
        self.transformer = GaugeTransformerStack(
            n_layers=n_layers,
            embed_dim=embed_dim,
            irrep_spec=irrep_spec,
            hidden_dim=hidden_dim,
            kappa_beta=kappa_beta,
            dropout=dropout,
            evolve_sigma=evolve_sigma,
            evolve_phi=evolve_phi,
            evolve_phi_e_step=evolve_phi_e_step,  # Update φ during E-step iterations
            # Phi evolution parameters (VFE gradient-based)
            phi_lr=config.get('phi_lr', 0.05),  # Learning rate for ∂F/∂φ descent
            phi_max_norm=config.get('phi_max_norm', math.pi),  # Default: π radians
            # VFE FFN parameters
            generators=self.generators,
            ffn_mode=ffn_mode,
            ffn_alpha=ffn_alpha,
            ffn_kappa=ffn_kappa,
            ffn_n_iterations=ffn_n_iterations,
            ffn_learnable_lr=ffn_learnable_lr,
            ffn_lambda_belief=ffn_lambda_belief,
            ffn_update_sigma=ffn_update_sigma,
            diagonal_covariance=diagonal_covariance,
            # Sparse attention
            attention_pattern=self.attention_pattern,
            attention_window=self.attention_window,
            # Gauge frame dimension
            phi_dim=self.phi_dim,
            # Pure FEP mode parameters
            ffn_pure_fep_mode=ffn_pure_fep_mode,
            ffn_max_seq_len=ffn_max_seq_len,
            ffn_prior_lr=ffn_prior_lr,
            ffn_prior_bank=self.prior_bank,  # Pass PriorBank to FFN layers
            ffn_use_prior_bank=use_prior_bank,  # Enable token-dependent priors
            # Memory-efficient options
            ffn_irrep_dims=self._get_effective_irrep_dims(irrep_spec) if config.get('use_block_diagonal_kl', True) else None,
            ffn_chunk_size=config.get('ffn_chunk_size', None),
            # Pure VFE mode: disable ad-hoc transformer components
            use_layernorm=config.get('use_layernorm', True),
            use_dropout=config.get('use_dropout', True),
            use_residual=config.get('use_residual', True),
            # Identity transport mode
            use_identity_transport=use_identity_transport,
            # Self-attention masking (prevents attention collapse)
            mask_self_attention=config.get('mask_self_attention', False),
            # Gauge group control
            enforce_orthogonal=config.get('enforce_orthogonal', False),
            # Bayesian precision
            ffn_learnable_alpha=ffn_learnable_alpha,
            # Per-head specialization
            per_head_kappa=config.get('per_head_kappa', False),
            use_output_projection=config.get('use_output_projection', False),
            # Multi-head VFE
            multihead_vfe=config.get('multihead_vfe', False),
            # Cross-head coupling
            cross_head_perm=getattr(self, '_cross_head_perm', None),
        )

        # =================================================================
        # Output Projection
        # =================================================================
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie input/output embeddings (standard practice)
        # Note: Can't tie weights when gauge_fixed_priors=True since there's
        # no per-token embedding - just a single base_mu rotated per token
        if tie_embeddings and not gauge_fixed_priors:
            self.out_proj.weight = self.token_embed.mu_embed.weight
        elif tie_embeddings and gauge_fixed_priors:
            print("Warning: tie_embeddings disabled because gauge_fixed_priors=True")

        # =================================================================
        # Initialize Weights
        # =================================================================
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GaugeTransformerLM initialized: {n_params/1e6:.2f}M parameters")

    def _compute_irrep_dims(self, irrep_spec: List[Tuple[str, int, int]]) -> List[int]:
        """
        Compute flat list of block dimensions from irrep_spec.

        For irrep_spec = [('ℓ0', 75, 1), ('ℓ1', 30, 3), ('ℓ2', 18, 5)]:
        Returns: [1, 1, ...(75 times)..., 3, 3, ...(30 times)..., 5, 5, ...(18 times)...]

        This is used for block-diagonal KL computation which exploits
        the gauge structure for massive memory savings.
        """
        irrep_dims = []
        for label, mult, dim in irrep_spec:
            irrep_dims.extend([dim] * mult)
        return irrep_dims

    def _get_effective_irrep_dims(self, irrep_spec: List[Tuple[str, int, int]]) -> List[int]:
        """
        Get effective block dimensions, accounting for cross-head super-blocks.

        When cross_couplings are active, coupled heads are merged into larger
        super-blocks. The super-block dims replace the per-head dims for the
        coupled groups while uncoupled heads keep their original dimensions.

        Falls back to _compute_irrep_dims when no cross-coupling is active.
        """
        if getattr(self, '_super_block_dims', None) is not None:
            return self._super_block_dims
        return self._compute_irrep_dims(irrep_spec)

    def _init_weights(self, module):
        """Initialize weights following best practices.

        Note: Skip nn.Embedding modules — their initialization is handled
        by GaugeTokenEmbedding.__init__() with calibrated std values
        (e.g., init_std=2.0 for mu_embed, scaled phi_embed).
        Overwriting with std=0.02 would destroy the gauge-theoretic init.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_agents: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through 0D gauge transformer.

        Args:
            token_ids: (batch, seq_len) token indices
                       seq_len = number of agents at the single point c*
            return_agents: If True, return intermediate agent states

        Returns:
            logits: (batch, num_agents, vocab_size) next-token predictions
            agents: Optional dict with mu, sigma, phi for each agent

        0D STRUCTURE:
            - All agents exist at single base manifold point c*
            - No spatial variation: mu[i], sigma[i], phi[i] are per-agent, not per-location
            - Attention β_ij are scalars, not spatial fields
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # =================================================================
        # Trajectory Recording: Start forward pass
        # =================================================================
        recorder = get_global_recorder() if TRAJECTORY_TRACKING_AVAILABLE else None
        if recorder is not None and recorder.enabled:
            ffn_mode = self.config.get('ffn_mode', 'VFE_dynamic')
            recorder.start_forward(batch_size, num_agents, ffn_mode=ffn_mode)

        # =================================================================
        # 1. Token Embeddings (0D: one per agent at c*, not per spatial point)
        # =================================================================
        mu_q, sigma_q, phi = self.token_embed(token_ids)

        # =================================================================
        # 1b. Cross-Head Permutation (reorder dims for super-block contiguity)
        # =================================================================
        # When cross-head coupling is active, generators were reordered so that
        # coupled heads are contiguous. We must apply the same permutation to
        # the embedding dimensions so mu/sigma align with the generator blocks.
        if getattr(self, '_cross_head_perm', None) is not None:
            perm = torch.from_numpy(self._cross_head_perm).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, perm]
            if sigma_q is not None:
                if sigma_q.dim() == 3:
                    # Diagonal: (B, N, K)
                    sigma_q = sigma_q[:, :, perm]
                else:
                    # Full: (B, N, K, K)
                    sigma_q = sigma_q[:, :, perm][:, :, :, perm]

        # =================================================================
        # 2. Save Priors (position-independent semantics)
        # =================================================================
        # Priors represent "expected meaning of token" - independent of position.
        # This is the correct VFE setup: prior = semantic, belief = contextualized.
        mu_prior = mu_q.clone()

        # Record embeddings for trajectory tracking
        if recorder is not None and recorder.enabled:
            recorder.record_embeddings(mu_q, sigma_q, phi)

        # =================================================================
        # 4. Attention Mask (causal + optional sparsity)
        # =================================================================
        # Create attention mask based on pattern (full, local, strided)
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.attention_pattern,
            window=self.attention_window,
            device=device,
            causal=True,  # Always use causal for autoregressive LM
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)

        # =================================================================
        # 5. Precompute Transport Operators (when evolve_phi=False)
        # =================================================================
        # When phi doesn't evolve, we can compute transport operators once
        # and reuse across all layers, saving ~6× matrix exponential calls.
        if not self.evolve_phi:
            # Get the first block's attention layer to access head generators
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            cached_head_transports = None

        # =================================================================
        # 6. Forward Through Transformer Stack
        # =================================================================
        mu_q, sigma_q, phi, intermediates = self.transformer(
            mu_q,
            sigma_q,
            phi,
            self.generators,
            mask=mask,
            mu_prior=mu_prior,  # Pass priors for variational FFN
            token_ids=token_ids,  # Pass token IDs for PriorBank lookup
            return_intermediates=return_agents,
            cached_head_transports=cached_head_transports,
        )

        # =================================================================
        # 6b. Inverse Cross-Head Permutation (restore original dim order)
        # =================================================================
        if getattr(self, '_cross_head_perm', None) is not None:
            inv_perm = torch.from_numpy(
                np.argsort(self._cross_head_perm)
            ).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, inv_perm]

        # =================================================================
        # 7. Project to Vocabulary (one prediction per agent)
        # =================================================================
        logits = self.out_proj(mu_q)  # (B, N, V)

        # =================================================================
        # Trajectory Recording: End forward pass
        # =================================================================
        if recorder is not None and recorder.enabled:
            recorder.end_forward(mu_q, logits)

        if return_agents:
            agent_states = {
                'mu': mu_q.detach(),
                'sigma': sigma_q.detach() if sigma_q is not None else None,
                'phi': phi.detach(),
                'intermediates': intermediates,
            }
            return logits, agent_states

        return logits

    def forward_with_attention(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass that returns attention weights and KL matrices for loss computation.

        This is used during training to compute the attention-weighted free energy:
            F = Σ_ij β_ij · KL(q_i || Ω_ij[q_j]) - E[log p(o|x)]
                                                     ↑ Observations!

        Args:
            token_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens - used as observations in E-step

        Returns:
            logits: (batch, num_agents, vocab_size) predictions
            attention_info: Dict with:
                - 'beta': (B, n_heads, N, N) attention weights per head
                - 'kl': (B, n_heads, N, N) KL divergences per head
                - 'mu': (B, N, K) final belief means
                - 'sigma': (B, N, K, K) final covariances
                - 'phi': (B, N, 3) final gauge frames
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Embeddings
        mu_q, sigma_q, phi = self.token_embed(token_ids)

        # Cross-head permutation (same as in forward())
        if getattr(self, '_cross_head_perm', None) is not None:
            perm = torch.from_numpy(self._cross_head_perm).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, perm]
            if sigma_q is not None:
                if sigma_q.dim() == 3:
                    sigma_q = sigma_q[:, :, perm]
                else:
                    sigma_q = sigma_q[:, :, perm][:, :, :, perm]

        # Save priors (position-independent semantics) before position encoding
        mu_prior = mu_q.clone()
        sigma_prior = sigma_q.clone() if sigma_q is not None else None
        phi_prior = phi.clone()

        # Attention mask (causal + optional sparsity)
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.attention_pattern,
            window=self.attention_window,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Precompute transport operators when evolve_phi=False (saves ~6× matrix exps)
        if not self.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            cached_head_transports = None

        # Forward through transformer blocks (all but last without attention tracking)
        for block in self.transformer.blocks[:-1]:
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, self.generators, mask, mu_prior,
                targets=targets,  # Pass targets for E-step
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
                cached_head_transports=cached_head_transports,
            )

        # Final block WITH attention tracking
        final_block = self.transformer.blocks[-1]

        # Pre-norm + attention with tracking
        mu_normalized = final_block.norm1(mu_q)
        mu_attn, sigma_attn, beta, kl = final_block.attention(
            mu_normalized,
            sigma_q,
            phi,
            self.generators,
            mask=mask,
            return_attention=True,  # Get β_ij and KL_ij
            cached_head_transports=cached_head_transports,
        )

        # Complete final block forward (residual + FFN)
        # Must respect use_residual flag (same logic as block.forward in blocks.py)
        mu_attn = final_block.dropout1(mu_attn)
        if final_block.use_residual:
            mu_q = mu_q + mu_attn
        else:
            mu_q = mu_attn

        if final_block.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn

        # FFN sublayer
        mu_normalized = final_block.norm2(mu_q)

        # VFE_dynamic FFN returns (mu, sigma, phi) tuple
        mu_ffn, sigma_ffn, phi_ffn = final_block.ffn(
            mu=mu_normalized,
            beta=beta,
            mu_prior=mu_prior,
            phi=phi,
            sigma=sigma_q,
            mask=mask,
            targets=targets,
            W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
        )
        # Update covariances if evolving
        if final_block.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        if final_block.use_residual:
            mu_q = mu_q + mu_ffn
        else:
            mu_q = mu_ffn

        # Final norm
        mu_q = self.transformer.final_norm(mu_q)

        # Inverse cross-head permutation
        if getattr(self, '_cross_head_perm', None) is not None:
            inv_perm = torch.from_numpy(
                np.argsort(self._cross_head_perm)
            ).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, inv_perm]

        # Project to vocabulary
        logits = self.out_proj(mu_q)

        attention_info = {
            'beta': beta,      # (B, n_heads, N, N)
            'kl': kl,          # (B, n_heads, N, N)
            'mu': mu_q,        # (B, N, K) - evolved beliefs
            'sigma': sigma_q,  # (B, N, K, K) or None
            'phi': phi,        # (B, N, 3)
            # Priors for gamma term (saved before position encoding)
            'mu_prior': mu_prior,        # (B, N, K) - initial embedding means
            'sigma_prior': sigma_prior,  # (B, N, K, K) - initial embedding covariances
            'phi_prior': phi_prior,      # (B, N, 3) - initial gauge frames (pre-positional)
        }

        return logits, attention_info

    def forward_with_rg_tracking(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass that tracks RG flow across VFE iterations.

        This is for detailed RG analysis - captures beta_history showing
        how attention evolves during VFE descent within a single forward pass.

        Only meaningful when ffn has n_iterations > 1.

        Args:
            token_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target tokens

        Returns:
            logits: (batch, num_agents, vocab_size) predictions
            rg_info: Dict with:
                - 'beta_history': List of (B, N, N) attention at each VFE step
                - 'mu': Final belief means
                - 'sigma': Final covariances
                - 'phi': Final gauge frames
                - 'n_iterations': Number of VFE steps
        """
        batch_size, num_agents = token_ids.shape
        device = token_ids.device

        # Embeddings
        mu_q, sigma_q, phi = self.token_embed(token_ids)

        # Cross-head permutation (same as in forward())
        if getattr(self, '_cross_head_perm', None) is not None:
            perm = torch.from_numpy(self._cross_head_perm).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, perm]
            if sigma_q is not None:
                if sigma_q.dim() == 3:
                    sigma_q = sigma_q[:, :, perm]
                else:
                    sigma_q = sigma_q[:, :, perm][:, :, :, perm]

        mu_prior = mu_q.clone()

        # Attention mask
        mask = create_attention_mask(
            num_agents=num_agents,
            pattern=self.attention_pattern,
            window=self.attention_window,
            device=device,
            causal=True,
        )
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Precompute transport operators
        if not self.evolve_phi:
            first_attention = self.transformer.blocks[0].attention
            cached_head_transports = first_attention.precompute_head_transports(
                phi, device, mu_q.dtype
            )
        else:
            cached_head_transports = None

        # Forward through all but last block (no RG tracking needed)
        for block in self.transformer.blocks[:-1]:
            mu_q, sigma_q, phi = block(
                mu_q, sigma_q, phi, self.generators, mask, mu_prior,
                targets=targets,
                W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
                cached_head_transports=cached_head_transports,
            )

        # Final block WITH beta_history tracking
        final_block = self.transformer.blocks[-1]

        # Pre-norm + attention
        mu_normalized = final_block.norm1(mu_q)
        mu_attn, sigma_attn, beta, kl = final_block.attention(
            mu_normalized, sigma_q, phi, self.generators,
            mask=mask, return_attention=True,
            cached_head_transports=cached_head_transports,
        )

        # Complete attention sublayer (respect use_residual flag)
        mu_attn = final_block.dropout1(mu_attn)
        if final_block.use_residual:
            mu_q = mu_q + mu_attn
        else:
            mu_q = mu_attn

        if final_block.evolve_sigma and sigma_attn is not None:
            sigma_q = sigma_attn

        # FFN sublayer WITH beta_history
        mu_normalized = final_block.norm2(mu_q)

        # Call FFN with return_beta_history=True
        ffn_result = final_block.ffn(
            mu=mu_normalized,
            beta=beta,
            mu_prior=mu_prior,
            phi=phi,
            sigma=sigma_q,
            mask=mask,
            targets=targets,
            W_out=self.out_proj.weight if hasattr(self.out_proj, 'weight') else None,
            return_beta_history=True,  # <-- Key difference!
        )

        # Unpack result (4 values when return_beta_history=True)
        mu_ffn, sigma_ffn, phi_ffn, beta_history = ffn_result

        # Update covariances if evolving
        if final_block.evolve_sigma and sigma_ffn is not None:
            sigma_q = sigma_ffn

        if final_block.use_residual:
            mu_q = mu_q + mu_ffn
        else:
            mu_q = mu_ffn

        # Final norm
        mu_q = self.transformer.final_norm(mu_q)

        # Inverse cross-head permutation
        if getattr(self, '_cross_head_perm', None) is not None:
            inv_perm = torch.from_numpy(
                np.argsort(self._cross_head_perm)
            ).to(device=device, dtype=torch.long)
            mu_q = mu_q[:, :, inv_perm]

        # Project to vocabulary
        logits = self.out_proj(mu_q)

        # Get n_iterations from VFE FFN
        n_iterations = getattr(final_block.ffn.variational_ffn, 'n_iterations', 1)

        rg_info = {
            'beta_history': beta_history,  # List of (B, N, N) at each VFE step
            'mu': mu_q,
            'sigma': sigma_q,
            'phi': phi,
            'n_iterations': n_iterations,
            'beta_final': beta_history[-1] if beta_history else beta,
        }

        return logits, rg_info

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            prompt_ids: (1, prompt_len) initial tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)

        Returns:
            generated: (1, prompt_len + max_new_tokens) full sequence
        """
        was_training = self.training
        self.eval()
        try:
            generated = prompt_ids.clone()

            for _ in range(max_new_tokens):
                # Truncate if exceeds max_seq_len
                if generated.shape[1] > self.config['max_seq_len']:
                    generated = generated[:, -self.config['max_seq_len']:]

                # Forward pass - handle both tuple and single return value
                result = self.forward(generated)
                logits = result[0] if isinstance(result, tuple) else result  # (1, T, V)

                # Get logits for last token
                logits_next = logits[:, -1, :] / temperature  # (1, V)

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                    logits_next[logits_next < v[:, [-1]]] = -float('inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right to keep first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits_next[indices_to_remove] = -float('inf')

                # Sample
                probs = F.softmax(logits_next, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

            return generated
        finally:
            if was_training:
                self.train()

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return number of parameters.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            n_params: Total parameter count
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Exclude embedding parameters
            if hasattr(self.token_embed, 'mu_embed'):
                # Standard per-token embeddings
                n_params -= self.token_embed.mu_embed.weight.numel()
            elif hasattr(self.token_embed, 'base_mu'):
                # Gauge-fixed priors: base_mu + base_log_sigma_diag + phi_embed
                n_params -= self.token_embed.base_mu.numel()
                n_params -= self.token_embed.base_log_sigma_diag.numel()
                n_params -= self.token_embed.phi_embed.weight.numel()

        return n_params

    # =========================================================================
    # P-FLOW: EMA update of token embeddings toward successful beliefs
    # =========================================================================
    def p_flow_update(
        self,
        token_ids: torch.Tensor,           # (B, N) token IDs
        mu_beliefs: torch.Tensor,          # (B, N, K) final beliefs after VFE
        prediction_errors: torch.Tensor,   # (B, N) per-position CE loss
        ema_decay: float = 0.99,           # EMA decay (higher = slower)
    ):
        """
        P-flow: Update token embeddings toward successful beliefs.

        This is the key learning mechanism from fep_transformer.py:
        - After VFE dynamics produce final beliefs
        - Update token priors (embeddings) toward beliefs that predicted well
        - Uses EMA for stable, gradual updates

        Args:
            token_ids: (B, N) token indices
            mu_beliefs: (B, N, K) final belief means after VFE
            prediction_errors: (B, N) per-position CE loss
            ema_decay: EMA decay rate (0.99 = slow, 0.9 = faster)
        """
        if hasattr(self.token_embed, 'update_embeddings_from_beliefs'):
            self.token_embed.update_embeddings_from_beliefs(
                token_ids=token_ids,
                mu_beliefs=mu_beliefs,
                prediction_errors=prediction_errors,
                ema_decay=ema_decay,
            )

    def delta_rule_update_w_out(
        self,
        mu_beliefs: torch.Tensor,          # (B, N, K) final beliefs after VFE
        targets: torch.Tensor,             # (B, N) target token IDs
        lr: float = 0.001,                 # Learning rate for delta rule
    ):
        """
        Delta rule update for W_out - backprop-free learning.

        Instead of backpropagating through the full computation graph,
        update W_out using the local delta rule (Widrow-Hoff):

            ΔW = η · (target - prediction) ⊗ μ^T

        This is biologically plausible and doesn't require storing
        intermediate activations for backprop.

        Args:
            mu_beliefs: (B, N, K) final belief means after VFE
            targets: (B, N) target token indices
            lr: Learning rate for delta rule update
        """
        with torch.no_grad():
            B, N, K = mu_beliefs.shape
            V = self.config['vocab_size']

            # Get current predictions: softmax(W_out @ mu)
            logits = self.out_proj(mu_beliefs)  # (B, N, V)
            predictions = F.softmax(logits, dim=-1)  # (B, N, V)

            # One-hot encode targets
            targets_onehot = F.one_hot(targets, num_classes=V).float()  # (B, N, V)

            # Prediction error: (target - prediction)
            error = targets_onehot - predictions  # (B, N, V)

            # Delta rule: ΔW = error^T @ mu (outer product averaged over batch & positions)
            # W_out shape is (V, K), so we need: (V, K) += (B*N, V)^T @ (B*N, K)
            error_flat = error.reshape(-1, V)  # (B*N, V)
            mu_flat = mu_beliefs.reshape(-1, K)  # (B*N, K)

            # Compute delta: (V, K) = (V, B*N) @ (B*N, K)
            delta_W = error_flat.t() @ mu_flat  # (V, K)
            delta_W /= (B * N)  # Average over batch and positions

            # Apply update to W_out
            self.out_proj.weight.add_(lr * delta_W)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("GAUGE TRANSFORMER LANGUAGE MODEL TEST")
    print("="*70)

    # Test configuration (small for quick testing)
    config = {
        'vocab_size': 100,
        'embed_dim': 32,
        'n_layers': 2,
        'hidden_dim': 128,
        'max_seq_len': 16,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'evolve_sigma': False,
        'evolve_phi': False,
        'tie_embeddings': True,
        'irrep_spec': [
            ('ℓ0', 8, 1),
            ('ℓ1', 4, 3),
            ('ℓ2', 2, 5),
        ],  # Total: 8 + 12 + 10 = 30 → pad to 32
    }

    print(f"\n[1] Creating model...")
    print(f"    Config: vocab={config['vocab_size']}, K={config['embed_dim']}, "
          f"layers={config['n_layers']}")

    model = GaugeTransformerLM(config)
    print(f"    ✓ Model created")

    # Test forward pass
    print(f"\n[2] Testing forward pass...")
    batch_size = 2
    seq_len = 8
    token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    logits = model(token_ids)
    print(f"    Input shape:  {token_ids.shape}")
    print(f"    Output shape: {logits.shape}")
    print(f"    ✓ Forward pass complete")

    # Test with return_agents
    print(f"\n[3] Testing with agent state tracking...")
    logits, agents = model(token_ids, return_agents=True)
    print(f"    Agent states: μ={agents['mu'].shape}, "
          f"φ={agents['phi'].shape}")
    print(f"    Intermediates: {len(agents['intermediates'])} layers")
    print(f"    ✓ Agent tracking works")

    # Test generation
    print(f"\n[4] Testing autoregressive generation...")
    prompt = torch.randint(0, config['vocab_size'], (1, 4))
    generated = model.generate(
        prompt,
        max_new_tokens=8,
        temperature=1.0,
        top_k=10,
    )
    print(f"    Prompt length:    {prompt.shape[1]}")
    print(f"    Generated length: {generated.shape[1]}")
    print(f"    Tokens: {generated[0].tolist()}")
    print(f"    ✓ Generation works")

    # Parameter count
    total_params = model.get_num_params(non_embedding=False)
    non_embed_params = model.get_num_params(non_embedding=True)

    print(f"\n[5] Parameter count:")
    print(f"    Total:         {total_params:,} parameters")
    print(f"    Non-embedding: {non_embed_params:,} parameters")
    print(f"    Embedding:     {total_params - non_embed_params:,} parameters")

    # Compare to standard transformer
    standard_params = (
        config['vocab_size'] * config['embed_dim'] +  # Token embedding
        config['max_seq_len'] * config['embed_dim'] +  # Position embedding
        config['n_layers'] * (
            4 * config['embed_dim'] ** 2 +  # Q,K,V,O
            2 * config['embed_dim'] * config['hidden_dim'] +  # FFN
            4 * config['embed_dim']  # LayerNorm
        )
    )

    print(f"\n[6] Comparison to standard transformer:")
    print(f"    Standard (est): {standard_params:,} parameters")
    print(f"    Gauge:          {total_params:,} parameters")
    print(f"    Reduction:      {standard_params / total_params:.2f}x")

    print("\n" + "="*70)
    print("✓ All model tests passed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Implement training loop (free energy loss)")
    print("  2. Create data pipeline (WikiText-2)")
    print("  3. Train the model!")
    print("="*70)
