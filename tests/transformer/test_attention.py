# -*- coding: utf-8 -*-
"""
Attention Tests
===============

Tests for transformer.core.attention module.
"""

import pytest
import torch
import math


class TestComputeAttentionWeights:
    """Test compute_attention_weights function."""

    @staticmethod
    def _make_so3_generators(K, device):
        """Create skew-symmetric SO(3) generators for K dimensions."""
        generators = torch.randn(3, K, K, device=device)
        generators = generators - generators.transpose(-1, -2)
        return generators

    def test_basic_computation(self, cpu_device):
        """Test basic attention weight computation."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)
        kappa = 1.0

        beta = compute_attention_weights(
            mu, sigma, phi, generators, kappa=kappa,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Check shape
        assert beta.shape == (B, N, N)

    def test_output_is_probability(self, cpu_device):
        """Test attention weights are valid probabilities."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)
        kappa = 1.0

        beta = compute_attention_weights(
            mu, sigma, phi, generators, kappa=kappa,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Check non-negative
        assert (beta >= 0).all(), "Attention weights should be non-negative"

        # Check sums to 1 along last dim
        sums = beta.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Attention weights should sum to 1"

    def test_output_finite(self, cpu_device):
        """Test output contains no NaN/Inf."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)
        kappa = 1.0

        beta = compute_attention_weights(
            mu, sigma, phi, generators, kappa=kappa,
            diagonal_covariance=True, use_identity_transport=True,
        )

        assert torch.isfinite(beta).all(), "Output contains NaN or Inf"

    def test_with_mask(self, cpu_device):
        """Test attention with causal mask."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)
        kappa = 1.0

        # Create causal mask
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        beta = compute_attention_weights(
            mu, sigma, phi, generators, kappa=kappa, mask=mask,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Check masked positions are near zero
        upper_tri = torch.triu(torch.ones(N, N, device=cpu_device), diagonal=1).bool()
        assert torch.allclose(beta[:, upper_tri], torch.zeros_like(beta[:, upper_tri]), atol=1e-6), \
            "Masked positions should be zero"

    def test_different_kappa_values(self, cpu_device):
        """Test with different temperature values."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)

        for kappa in [0.1, 1.0, 10.0]:
            beta = compute_attention_weights(
                mu, sigma, phi, generators, kappa=kappa,
                diagonal_covariance=True, use_identity_transport=True,
            )
            assert torch.isfinite(beta).all()
            assert (beta >= 0).all()

    def test_kappa_temperature_effect(self, cpu_device):
        """Test that lower kappa gives sharper distributions."""
        from transformer.core.attention import compute_attention_weights

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = self._make_so3_generators(K, cpu_device)

        beta_low = compute_attention_weights(
            mu, sigma, phi, generators, kappa=0.1,
            diagonal_covariance=True, use_identity_transport=True,
        )
        beta_high = compute_attention_weights(
            mu, sigma, phi, generators, kappa=10.0,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Lower kappa should give higher max attention (sharper)
        max_low = beta_low.max(dim=-1).values.mean()
        max_high = beta_high.max(dim=-1).values.mean()

        assert max_low > max_high, "Lower kappa should give sharper attention"


class TestComputeKLMatrix:
    """Test compute_kl_matrix function."""

    def test_basic_computation(self, cpu_device):
        """Test basic KL matrix computation."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = torch.zeros(3, K, K, device=cpu_device)

        kl_matrix = compute_kl_matrix(
            mu, sigma, phi, generators,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Check shape
        assert kl_matrix.shape == (B, N, N)

    def test_self_kl_is_zero(self, cpu_device):
        """Test KL(p||p) = 0 on diagonal."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = torch.zeros(3, K, K, device=cpu_device)

        kl_matrix = compute_kl_matrix(
            mu, sigma, phi, generators,
            diagonal_covariance=True, use_identity_transport=True,
        )

        # Diagonal should be zero (KL divergence to self)
        diag = torch.diagonal(kl_matrix, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-5), \
            "KL(p||p) should be 0"

    def test_kl_non_negative(self, cpu_device):
        """Test KL divergence is non-negative."""
        from transformer.core.attention import compute_kl_matrix

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = torch.zeros(3, K, K, device=cpu_device)

        kl_matrix = compute_kl_matrix(
            mu, sigma, phi, generators,
            diagonal_covariance=True, use_identity_transport=True,
        )

        assert (kl_matrix >= -1e-6).all(), "KL divergence should be non-negative"


class TestCreateAttentionMask:
    """Test create_attention_mask function."""

    def test_full_causal_mask(self, cpu_device):
        """Test full causal attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 8
        mask = create_attention_mask(N, pattern='full', causal=True, device=cpu_device)

        # Should be lower triangular
        expected = torch.tril(torch.ones(N, N, device=cpu_device))
        assert torch.allclose(mask, expected)

    def test_full_bidirectional_mask(self, cpu_device):
        """Test full bidirectional attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 8
        mask = create_attention_mask(N, pattern='full', causal=False, device=cpu_device)

        # Should be all ones
        expected = torch.ones(N, N, device=cpu_device)
        assert torch.allclose(mask, expected)

    def test_local_attention_mask(self, cpu_device):
        """Test local attention mask."""
        from transformer.core.attention import create_attention_mask

        N = 16
        window = 4
        mask = create_attention_mask(
            N, pattern='local', window=window, causal=True, device=cpu_device
        )

        # Check shape
        assert mask.shape == (N, N)

        # Check it's a valid mask (0 or 1)
        assert ((mask == 0) | (mask == 1)).all()


class TestIrrepMultiHeadAttention:
    """Test IrrepMultiHeadAttention module."""

    @pytest.fixture
    def attention_module(self, cpu_device):
        """Create attention module."""
        from transformer.core.attention import IrrepMultiHeadAttention

        K = 16
        # irrep_spec: 2 heads of 8 scalars each = 16 total
        irrep_spec = [('l0', 8, 1), ('l0b', 8, 1)]
        kappa_beta = 1.0

        # Create simple antisymmetric generators
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        attention = IrrepMultiHeadAttention(
            embed_dim=K,
            irrep_spec=irrep_spec,
            kappa_beta=kappa_beta,
            diagonal_covariance=True,
        )
        return attention.to(cpu_device), generators

    def test_creation(self, cpu_device):
        """Test creating attention module."""
        from transformer.core.attention import IrrepMultiHeadAttention

        K = 16
        irrep_spec = [('l0', 8, 1), ('l0b', 8, 1)]

        attention = IrrepMultiHeadAttention(
            embed_dim=K,
            irrep_spec=irrep_spec,
            kappa_beta=1.0,
            diagonal_covariance=True,
        )
        assert attention is not None

    def test_forward_pass(self, attention_module, cpu_device):
        """Test forward pass through attention."""
        attention, generators = attention_module
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1

        # Create causal mask
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, beta, kl = attention(
            mu, sigma, phi, generators, mask=mask, return_attention=True,
        )

        # Check output shapes
        assert mu_out.shape == (B, N, K)
        assert beta.shape[-2:] == (N, N)

    def test_forward_output_finite(self, attention_module, cpu_device):
        """Test forward outputs are finite."""
        attention, generators = attention_module
        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.abs(torch.randn(B, N, K, device=cpu_device)) + 0.1
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1
        mask = torch.tril(torch.ones(B, N, N, device=cpu_device))

        mu_out, sigma_out, beta, kl = attention(
            mu, sigma, phi, generators, mask=mask, return_attention=True,
        )

        assert torch.isfinite(mu_out).all(), "mu contains NaN/Inf"
        if beta is not None:
            assert torch.isfinite(beta).all(), "beta contains NaN/Inf"


class TestAggregateMessages:
    """Test aggregate_messages function."""

    def test_basic_aggregation(self, cpu_device):
        """Test basic message aggregation with identity transport."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)
        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only', cached_transport=transport,
        )

        # Check shape
        assert mu_agg.shape == (B, N, K)

    def test_aggregation_with_identity_beta(self, cpu_device):
        """Test aggregation with identity attention (copies input)."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        B, N, K = 2, 8, 16
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        phi = torch.zeros(B, N, 3, device=cpu_device)
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        # Identity attention: each position attends only to itself
        beta = torch.eye(N, device=cpu_device).unsqueeze(0).expand(B, -1, -1)

        transport = compute_transport_operators(phi, generators)
        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only', cached_transport=transport,
        )

        # Should be same as input (identity attention + identity transport)
        assert torch.allclose(mu_agg, mu, atol=1e-5)


class TestGLKMetricCorrection:
    """Test GL(K) metric correction in aggregate_messages.

    The variational gradient includes a metric factor (ΩΩ^T)^{-1}, so the
    aggregation should use Ω^{-T} μ_j (not Ω μ_j) for GL(K).
    For SO(K), Ω^{-T} = Ω identically, so no correction is needed.
    """

    @staticmethod
    def _make_glk_generators(K, device):
        """Create GL(K) generators: K² elementary matrices (not skew-symmetric)."""
        n_gen = K * K
        generators = torch.zeros(n_gen, K, K, device=device)
        idx = 0
        for a in range(K):
            for b in range(K):
                generators[idx, a, b] = 1.0
                idx += 1
        return generators

    @staticmethod
    def _make_sok_generators(K, device):
        """Create SO(K) generators: K(K-1)/2 skew-symmetric basis matrices."""
        n_gen = K * (K - 1) // 2
        generators = torch.zeros(n_gen, K, K, device=device)
        idx = 0
        for a in range(K):
            for b in range(a + 1, K):
                generators[idx, a, b] = 1.0
                generators[idx, b, a] = -1.0
                idx += 1
        return generators

    def test_glk_aggregation_uses_omega_inv_transpose(self, cpu_device):
        """For GL(K), aggregation should use Ω^{-T} μ_j, not Ω μ_j."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        torch.manual_seed(42)
        B, N, K = 1, 3, 4
        generators = self._make_glk_generators(K, cpu_device)
        n_gen = generators.shape[0]

        phi = torch.randn(B, N, n_gen, device=cpu_device) * 0.1
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)
        Omega = transport['Omega']  # (B, N, N, K, K)

        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only',
            cached_transport=transport,
        )

        # Manually compute CORRECT result: Σ_j β_ij Ω_ij^{-T} μ_j
        Omega_inv_T = Omega.permute(0, 2, 1, 3, 4).transpose(-1, -2)
        mu_correct = torch.einsum('bijkl,bjl->bijk', Omega_inv_T, mu)
        mu_correct_agg = torch.einsum('bij,bijk->bik', beta, mu_correct)

        # Manually compute WRONG (uncorrected) result: Σ_j β_ij Ω_ij μ_j
        mu_wrong = torch.einsum('bijkl,bjl->bijk', Omega, mu)
        mu_wrong_agg = torch.einsum('bij,bijk->bik', beta, mu_wrong)

        # Output should match the metric-corrected version
        assert torch.allclose(mu_agg, mu_correct_agg, atol=1e-5), \
            "GL(K) aggregation should use Ω^{-T} μ_j"

        # Output should NOT match the uncorrected version
        assert not torch.allclose(mu_agg, mu_wrong_agg, atol=1e-3), \
            "GL(K) aggregation should differ from naive Ω μ_j"

    def test_sok_aggregation_matches_direct_transport(self, cpu_device):
        """For SO(K), Ω^{-T} = Ω, so the correction is a no-op."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        torch.manual_seed(42)
        B, N, K = 1, 3, 4
        generators = self._make_sok_generators(K, cpu_device)
        n_gen = generators.shape[0]

        phi = torch.randn(B, N, n_gen, device=cpu_device) * 0.1
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)
        Omega = transport['Omega']

        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only',
            cached_transport=transport,
        )

        # For SO(K), Ω μ_j and Ω^{-T} μ_j should be identical
        mu_direct = torch.einsum('bijkl,bjl->bijk', Omega, mu)
        mu_direct_agg = torch.einsum('bij,bijk->bik', beta, mu_direct)

        assert torch.allclose(mu_agg, mu_direct_agg, atol=1e-5), \
            "SO(K) aggregation should match direct Ω μ_j (Ω^{-T} = Ω)"

    def test_glk_identity_omega_no_correction(self, cpu_device):
        """When phi=0 (Ω=I), result is simple weighted sum regardless of group."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        torch.manual_seed(42)
        B, N, K = 1, 4, 3
        generators = self._make_glk_generators(K, cpu_device)

        phi = torch.zeros(B, N, K * K, device=cpu_device)  # Ω = I
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)

        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only',
            cached_transport=transport,
        )

        # With Ω = I, result is just Σ_j β_ij μ_j
        expected = torch.einsum('bij,bjk->bik', beta, mu)
        assert torch.allclose(mu_agg, expected, atol=1e-4), \
            "With Ω=I, aggregation should be simple weighted sum"

    def test_glk_covariance_metric_correction(self, cpu_device):
        """For GL(K) full_distribution mode, covariance uses Ω^{-T} Σ Ω^{-1}."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        torch.manual_seed(42)
        B, N, K = 1, 3, 4
        generators = self._make_glk_generators(K, cpu_device)
        n_gen = generators.shape[0]

        phi = torch.randn(B, N, n_gen, device=cpu_device) * 0.1
        mu = torch.randn(B, N, K, device=cpu_device)
        # Positive definite covariance
        A = torch.randn(B, N, K, K, device=cpu_device)
        sigma = A @ A.transpose(-1, -2) + 0.1 * torch.eye(K, device=cpu_device)
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)
        Omega = transport['Omega']

        mu_agg, sigma_agg = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='full_distribution',
            cached_transport=transport,
        )

        # Manually compute metric-corrected transport
        Omega_inv_T = Omega.permute(0, 2, 1, 3, 4).transpose(-1, -2)  # Ω^{-T}
        Omega_inv = Omega.permute(0, 2, 1, 3, 4)  # Ω^{-1}

        # Corrected mean: Ω^{-T} μ_j
        mu_corrected = torch.einsum('bijkl,bjl->bijk', Omega_inv_T, mu)
        mu_expected = torch.einsum('bij,bijk->bik', beta, mu_corrected)

        # Corrected cov: Ω^{-T} Σ Ω^{-1}
        Sigma_corrected = torch.einsum(
            'bijkl,bjlm,bijmn->bijkn',
            Omega_inv_T, sigma, Omega_inv
        )

        # Second moment of mixture
        second_moment = Sigma_corrected + torch.einsum(
            'bijk,bijl->bijkl', mu_corrected, mu_corrected
        )
        sigma_expected = torch.einsum('bij,bijkl->bikl', beta, second_moment) \
            - torch.einsum('bik,bil->bikl', mu_expected, mu_expected)

        assert torch.allclose(mu_agg, mu_expected, atol=1e-5), \
            "GL(K) mean aggregation incorrect in full_distribution mode"
        assert torch.allclose(sigma_agg, sigma_expected, atol=1e-4), \
            "GL(K) covariance should use Ω^{-T} Σ Ω^{-1}"

    def test_glk_output_finite(self, cpu_device):
        """GL(K) aggregation should produce finite outputs."""
        from transformer.core.attention import aggregate_messages, compute_transport_operators

        torch.manual_seed(123)
        B, N, K = 2, 6, 4
        generators = self._make_glk_generators(K, cpu_device)
        n_gen = generators.shape[0]

        phi = torch.randn(B, N, n_gen, device=cpu_device) * 0.1
        mu = torch.randn(B, N, K, device=cpu_device)
        sigma = torch.eye(K, device=cpu_device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
        beta = torch.softmax(torch.randn(B, N, N, device=cpu_device), dim=-1)

        transport = compute_transport_operators(phi, generators)
        mu_agg, _ = aggregate_messages(
            mu, sigma, phi, beta, generators,
            aggregate_mode='mean_only',
            cached_transport=transport,
        )

        assert torch.isfinite(mu_agg).all(), "GL(K) aggregation produced NaN/Inf"
        assert mu_agg.shape == (B, N, K)


class TestComputeTransportOperators:
    """Test compute_transport_operators function."""

    def test_basic_computation(self, cpu_device):
        """Test basic transport operator computation."""
        from transformer.core.attention import compute_transport_operators

        B, N = 2, 8
        K = 16
        phi = torch.randn(B, N, 3, device=cpu_device) * 0.1

        # Create generators
        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        result = compute_transport_operators(phi, generators)
        omega = result['Omega']

        # Check shape: (B, N, N, K, K)
        assert omega.shape == (B, N, N, K, K)

    def test_identity_for_zero_phi(self, cpu_device):
        """Test transport is identity when phi=0."""
        from transformer.core.attention import compute_transport_operators

        B, N = 2, 8
        K = 16
        phi = torch.zeros(B, N, 3, device=cpu_device)

        generators = torch.randn(3, K, K, device=cpu_device)
        generators = generators - generators.transpose(-1, -2)

        result = compute_transport_operators(phi, generators)
        omega = result['Omega']

        # When phi_i = phi_j = 0, transport should be identity
        identity = torch.eye(K, device=cpu_device)
        diag_omega = omega[:, range(N), range(N)]  # Self-transport

        for b in range(B):
            for n in range(N):
                assert torch.allclose(diag_omega[b, n], identity, atol=1e-4), \
                    "Self-transport should be identity"
