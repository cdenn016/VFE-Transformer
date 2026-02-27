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


