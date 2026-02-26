# -*- coding: utf-8 -*-
"""
Tests for cross-head gauge coupling (sparse off-diagonal transport).

Tests the full pipeline:
  1. Generator construction (generate_glK_cross_head_generators)
  2. Super-block merging (merge_coupled_heads)
  3. Generator reordering (reorder_cross_head_generators)
  4. Model construction with cross_couplings config
  5. Forward + backward pass through the full model
"""
import pytest
import numpy as np
import torch

from math_utils.generators import (
    generate_glK_cross_head_generators,
    generate_glK_multihead_generators,
    merge_coupled_heads,
    reorder_cross_head_generators,
)


# =========================================================================
# Generator construction tests
# =========================================================================

class TestGeneratorConstruction:
    """Test generate_glK_cross_head_generators."""

    def test_shape(self):
        K, H, d = 24, 4, 6
        couplings = [(0, 1), (1, 0)]
        G = generate_glK_cross_head_generators(K, H, couplings)
        n_diag = H * d * d
        n_cross = len(couplings) * d * d
        assert G.shape == (n_diag + n_cross, K, K)

    def test_diagonal_matches_standard(self):
        """Diagonal generators should match generate_glK_multihead_generators."""
        K, H = 24, 4
        d = K // H
        couplings = [(0, 2), (2, 0)]
        G_cross = generate_glK_cross_head_generators(K, H, couplings)
        G_std = generate_glK_multihead_generators(K, H)

        n_diag = H * d * d
        np.testing.assert_allclose(G_cross[:n_diag], G_std, atol=1e-10)

    def test_cross_generators_correct_block(self):
        """Cross generators should have non-zero entries only in the target block."""
        K, H, d = 24, 4, 6
        couplings = [(1, 3)]  # head 1 → head 3
        G = generate_glK_cross_head_generators(K, H, couplings)

        n_diag = H * d * d
        # The cross generators are at indices [n_diag:]
        for g_idx in range(n_diag, G.shape[0]):
            g = G[g_idx]
            # Should be non-zero only at rows [6:12] (head 1) and cols [18:24] (head 3)
            mask = np.zeros_like(g)
            mask[6:12, 18:24] = 1.0
            assert np.allclose(g * (1 - mask), 0.0), f"Generator {g_idx} leaks outside target block"

    def test_self_coupling_raises(self):
        with pytest.raises(ValueError, match="Self-coupling"):
            generate_glK_cross_head_generators(24, 4, [(2, 2)])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            generate_glK_cross_head_generators(24, 4, [(0, 5)])

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            generate_glK_cross_head_generators(25, 4, [(0, 1)])


# =========================================================================
# Super-block merging tests
# =========================================================================

class TestMergeCoupledHeads:
    """Test merge_coupled_heads."""

    def test_no_coupling(self):
        dims, groups = merge_coupled_heads(4, 6, [])
        assert dims == [6, 6, 6, 6]
        assert groups == [[0], [1], [2], [3]]

    def test_symmetric_pair(self):
        dims, groups = merge_coupled_heads(4, 6, [(0, 1), (1, 0)])
        assert dims == [12, 6, 6]
        assert groups == [[0, 1], [2], [3]]

    def test_two_pairs(self):
        dims, groups = merge_coupled_heads(4, 6, [(0, 1), (1, 0), (2, 3), (3, 2)])
        assert sum(dims) == 24
        assert len(groups) == 2

    def test_transitive(self):
        """Coupling 0-1 and 1-2 should merge all three."""
        dims, groups = merge_coupled_heads(4, 6, [(0, 1), (1, 2)])
        assert dims == [18, 6]
        assert groups == [[0, 1, 2], [3]]

    def test_asymmetric(self):
        """Single directed coupling still merges the pair."""
        dims, groups = merge_coupled_heads(4, 6, [(0, 2)])
        assert dims == [12, 6, 6]
        assert [0, 2] in groups


# =========================================================================
# Reorder tests
# =========================================================================

class TestReorderGenerators:
    """Test reorder_cross_head_generators."""

    def test_reordered_is_block_diagonal(self):
        """After reordering, generators should be block-diagonal in super-blocks."""
        K, H, d = 24, 4, 6
        couplings = [(0, 1), (1, 0), (2, 3), (3, 2)]
        G = generate_glK_cross_head_generators(K, H, couplings)
        _, groups = merge_coupled_heads(H, d, couplings)
        G_reord, perm = reorder_cross_head_generators(G, H, d, couplings, groups)

        # Super-blocks: [0:12] and [12:24]
        for g_idx in range(G_reord.shape[0]):
            cross_01 = np.abs(G_reord[g_idx, :12, 12:]).max()
            cross_10 = np.abs(G_reord[g_idx, 12:, :12]).max()
            assert cross_01 < 1e-10, f"Generator {g_idx} has cross-super-block entries"
            assert cross_10 < 1e-10, f"Generator {g_idx} has cross-super-block entries"

    def test_permutation_is_bijection(self):
        K, H, d = 24, 4, 6
        couplings = [(0, 2)]
        G = generate_glK_cross_head_generators(K, H, couplings)
        _, groups = merge_coupled_heads(H, d, couplings)
        _, perm = reorder_cross_head_generators(G, H, d, couplings, groups)
        assert len(set(perm)) == K


# =========================================================================
# Full model integration tests
# =========================================================================

class TestModelIntegration:
    """Test full model forward/backward with cross-head coupling."""

    @pytest.fixture
    def base_config(self):
        return {
            'vocab_size': 50,
            'embed_dim': 24,
            'n_layers': 1,
            'irrep_spec': [('fund', 4, 6)],
            'hidden_dim': 48,
            'max_seq_len': 32,
            'kappa_beta': 1.0,
            'evolve_sigma': False,
            'evolve_phi': True,
            'gauge_group': 'GLK',
            'use_multi_irrep': True,
            'use_block_diagonal_kl': True,
            'diagonal_covariance': True,
            'use_layernorm': True,
            'use_dropout': False,
            'use_residual': True,
        }

    def test_standard_multihead_still_works(self, base_config):
        """Regression: standard multihead (no coupling) works."""
        from transformer.core.model import GaugeTransformerLM
        model = GaugeTransformerLM(base_config)
        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()

    def test_symmetric_coupling(self, base_config):
        """Symmetric coupling: heads 0<->1 and 2<->3."""
        from transformer.core.model import GaugeTransformerLM
        base_config['cross_couplings'] = [(0, 1), (1, 0), (2, 3), (3, 2)]
        model = GaugeTransformerLM(base_config)
        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()
        # Check super-block structure
        attn = model.transformer.blocks[0].attention
        assert attn.irrep_dims == [12, 12]

    def test_asymmetric_coupling(self, base_config):
        """Asymmetric coupling: only head 0→2."""
        from transformer.core.model import GaugeTransformerLM
        base_config['cross_couplings'] = [(0, 2)]
        model = GaugeTransformerLM(base_config)
        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()
        attn = model.transformer.blocks[0].attention
        assert attn.irrep_dims == [12, 6, 6]

    def test_full_covariance_with_coupling(self, base_config):
        """Cross-coupling with full (non-diagonal) covariance."""
        from transformer.core.model import GaugeTransformerLM
        base_config['cross_couplings'] = [(0, 1), (1, 0)]
        base_config['diagonal_covariance'] = False
        base_config['evolve_sigma'] = True
        model = GaugeTransformerLM(base_config)
        x = torch.randint(0, 50, (2, 8))
        logits = model(x)
        assert logits.shape == (2, 8, 50)
        logits.sum().backward()

    def test_phi_dim_includes_cross_generators(self, base_config):
        """phi_dim should account for cross-coupling generators."""
        from transformer.core.model import GaugeTransformerLM
        d_head = 6
        n_heads = 4
        # Standard: 4 * 36 = 144
        model_std = GaugeTransformerLM(base_config)
        assert model_std.phi_dim == n_heads * d_head**2

        # With 2 coupling pairs: 144 + 2*36 = 216
        base_config['cross_couplings'] = [(0, 1), (2, 3)]
        model_cross = GaugeTransformerLM(base_config)
        assert model_cross.phi_dim == n_heads * d_head**2 + 2 * d_head**2

    def test_forward_with_attention(self, base_config):
        """forward_with_attention works with cross-coupling."""
        from transformer.core.model import GaugeTransformerLM
        base_config['cross_couplings'] = [(0, 1), (1, 0)]
        base_config['n_layers'] = 2
        model = GaugeTransformerLM(base_config)
        x = torch.randint(0, 50, (2, 8))
        logits, info = model.forward_with_attention(x)
        assert logits.shape == (2, 8, 50)
        assert 'beta' in info
        logits.sum().backward()
