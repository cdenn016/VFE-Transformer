# -*- coding: utf-8 -*-
"""
Embeddings Tests
================

Tests for transformer.core.embeddings module.
"""

import pytest
import torch


class TestGaugeTokenEmbedding:
    """Test GaugeTokenEmbedding class."""

    def test_creation(self, cpu_device):
        """Test creating token embedding."""
        from transformer.core.embeddings import GaugeTokenEmbedding

        V, K = 100, 16
        embed = GaugeTokenEmbedding(V, K)
        assert embed is not None

    def test_forward(self, cpu_device):
        """Test forward pass."""
        from transformer.core.embeddings import GaugeTokenEmbedding

        V, K = 100, 16
        B, N = 2, 8
        embed = GaugeTokenEmbedding(V, K).to(cpu_device)

        token_ids = torch.randint(0, V, (B, N), device=cpu_device)
        mu, sigma, phi = embed(token_ids)

        # Check shapes
        assert mu.shape == (B, N, K)
        assert phi.shape[0] == B
        assert phi.shape[1] == N

    def test_output_finite(self, cpu_device):
        """Test outputs are finite."""
        from transformer.core.embeddings import GaugeTokenEmbedding

        V, K = 100, 16
        B, N = 2, 8
        embed = GaugeTokenEmbedding(V, K).to(cpu_device)

        token_ids = torch.randint(0, V, (B, N), device=cpu_device)
        mu, sigma, phi = embed(token_ids)

        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()
        assert torch.isfinite(phi).all()

    def test_same_token_same_embedding(self, cpu_device):
        """Test same token produces same embedding."""
        from transformer.core.embeddings import GaugeTokenEmbedding

        V, K = 100, 16
        embed = GaugeTokenEmbedding(V, K).to(cpu_device)

        # Same token at different positions
        token_ids = torch.tensor([[5, 5, 5]], device=cpu_device)
        mu, sigma, phi = embed(token_ids)

        # All positions should have same mu (before positional encoding)
        assert torch.allclose(mu[0, 0], mu[0, 1])
        assert torch.allclose(mu[0, 1], mu[0, 2])


class TestGaugePositionalEncoding:
    """Test GaugePositionalEncoding class."""

    def test_creation_learned(self, cpu_device):
        """Test creating learned positional encoding."""
        from transformer.core.embeddings import GaugePositionalEncoding

        max_len, phi_dim = 128, 3
        pos_enc = GaugePositionalEncoding(
            max_seq_len=max_len,
            phi_dim=phi_dim,
            mode='learned',
        )
        assert pos_enc is not None

    def test_creation_sinusoidal(self, cpu_device):
        """Test creating sinusoidal positional encoding."""
        from transformer.core.embeddings import GaugePositionalEncoding

        max_len, phi_dim = 128, 3
        pos_enc = GaugePositionalEncoding(
            max_seq_len=max_len,
            phi_dim=phi_dim,
            mode='sinusoidal',
        )
        assert pos_enc is not None

    def test_compose(self, cpu_device):
        """Test compose method."""
        from transformer.core.embeddings import GaugePositionalEncoding

        max_len, phi_dim = 128, 3
        B, N = 2, 16

        pos_enc = GaugePositionalEncoding(
            max_seq_len=max_len,
            phi_dim=phi_dim,
            mode='learned',
        ).to(cpu_device)

        phi = torch.randn(B, N, phi_dim, device=cpu_device)
        phi_out = pos_enc.compose(phi, N, device=cpu_device)

        # Check shape preserved
        assert phi_out.shape == (B, N, phi_dim)

    def test_compose_output_finite(self, cpu_device):
        """Test compose produces finite output."""
        from transformer.core.embeddings import GaugePositionalEncoding

        max_len, phi_dim = 128, 3
        B, N = 2, 16

        pos_enc = GaugePositionalEncoding(
            max_seq_len=max_len,
            phi_dim=phi_dim,
            mode='learned',
        ).to(cpu_device)

        phi = torch.randn(B, N, phi_dim, device=cpu_device)
        phi_out = pos_enc.compose(phi, N, device=cpu_device)

        assert torch.isfinite(phi_out).all()

    def test_different_positions_different_encoding(self, cpu_device):
        """Test different positions get different encodings."""
        from transformer.core.embeddings import GaugePositionalEncoding

        max_len, phi_dim = 128, 3
        B, N = 1, 16

        pos_enc = GaugePositionalEncoding(
            max_seq_len=max_len,
            phi_dim=phi_dim,
            mode='learned',
        ).to(cpu_device)

        # Zero input phi
        phi = torch.zeros(B, N, phi_dim, device=cpu_device)
        phi_out = pos_enc.compose(phi, N, device=cpu_device)

        # Different positions should have different phi
        # (at least some should differ)
        same_count = 0
        for i in range(N - 1):
            if torch.allclose(phi_out[0, i], phi_out[0, i + 1], atol=1e-5):
                same_count += 1

        assert same_count < N - 1, "Positional encoding should differ across positions"
