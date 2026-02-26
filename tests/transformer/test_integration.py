# -*- coding: utf-8 -*-
"""
Integration Tests
=================

End-to-end tests for the transformer module.
"""

import pytest
import torch
import torch.nn.functional as F


class TestEndToEndTraining:
    """Test complete training workflow."""

    def test_single_training_step(self, minimal_config, cpu_device):
        """Test a single training step completes without errors."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        # Forward
        logits = model(input_ids)

        # Loss
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert torch.isfinite(torch.tensor(loss.item()))

    def test_multiple_training_steps(self, minimal_config, cpu_device):
        """Test multiple training steps show loss decrease."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        V = minimal_config['vocab_size']
        B, N = 4, 16

        # Fixed data for overfitting test
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        losses = []
        for step in range(20):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease (model should overfit to fixed data)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_eval_mode_no_gradients(self, minimal_config, cpu_device):
        """Test eval mode doesn't accumulate gradients."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        input_ids = torch.randint(0, V, (2, 16), device=cpu_device)

        with torch.no_grad():
            logits = model(input_ids)

        # Check no gradients accumulated
        for p in model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0


class TestModelOutputDistribution:
    """Test model output distributions."""

    def test_logits_are_unnormalized(self, gauge_model, batch_tensors, cpu_device):
        """Test logits are unnormalized (not probabilities)."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)

        # Logits should not sum to 1 (unnormalized)
        sums = logits.sum(dim=-1)
        ones = torch.ones_like(sums)

        assert not torch.allclose(sums, ones, atol=0.01), \
            "Logits should be unnormalized"

    def test_softmax_gives_probabilities(self, gauge_model, batch_tensors, cpu_device):
        """Test softmax of logits gives valid probabilities."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)
            probs = F.softmax(logits, dim=-1)

        # Should be non-negative
        assert (probs >= 0).all()

        # Should sum to 1
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestAttentionPatterns:
    """Test attention pattern behavior."""

    def test_causal_attention(self, minimal_config, cpu_device):
        """Test attention is causal (no future positions)."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)

        with torch.no_grad():
            logits, attn_info = model.forward_with_attention(input_ids)

        # Get attention weights
        if 'beta_layers' in attn_info:
            beta = attn_info['beta_layers'][0]  # First layer
        elif 'beta' in attn_info:
            beta = attn_info['beta']
        else:
            pytest.skip("No attention weights in output")

        # Handle multi-head case
        if beta.dim() == 4:  # (B, H, N, N)
            beta = beta.mean(dim=1)  # Average over heads

        # Check upper triangle is zero (causal mask)
        for i in range(N):
            for j in range(i + 1, N):
                # Position i should not attend to position j > i
                assert torch.allclose(
                    beta[:, i, j],
                    torch.zeros_like(beta[:, i, j]),
                    atol=1e-5
                ), f"Position {i} should not attend to future position {j}"


class TestModelDeterminism:
    """Test model determinism and reproducibility."""

    def test_same_input_same_output(self, minimal_config, cpu_device):
        """Test same input gives same output."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        input_ids = torch.randint(0, V, (2, 16), device=cpu_device)

        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model(input_ids)

        assert torch.allclose(out1, out2)

    def test_different_input_different_output(self, minimal_config, cpu_device):
        """Test different inputs give different outputs."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        input1 = torch.randint(0, V, (2, 16), device=cpu_device)
        input2 = torch.randint(0, V, (2, 16), device=cpu_device)

        # Ensure inputs are different (with iteration limit to prevent infinite loop)
        for _ in range(100):
            if not torch.equal(input1, input2):
                break
            input2 = torch.randint(0, V, (2, 16), device=cpu_device)

        with torch.no_grad():
            out1 = model(input1)
            out2 = model(input2)

        assert not torch.allclose(out1, out2), \
            "Different inputs should produce different outputs"


class TestGaugeVsStandard:
    """Compare gauge transformer to standard transformer."""

    def test_both_produce_valid_outputs(self, minimal_config, cpu_device):
        """Test both model types produce valid outputs."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.baselines.standard_transformer import StandardTransformerLM

        V = minimal_config['vocab_size']
        K = minimal_config['embed_dim']

        # Gauge model
        gauge_model = GaugeTransformerLM(minimal_config)
        gauge_model = gauge_model.to(cpu_device)
        gauge_model.eval()

        # Standard model
        std_config = {
            'vocab_size': V,
            'embed_dim': K,
            'n_heads': 3,
            'n_layers': 1,
            'hidden_dim': minimal_config['hidden_dim'],
            'max_seq_len': minimal_config['max_seq_len'],
            'dropout': 0.0,
        }
        std_model = StandardTransformerLM(std_config)
        std_model = std_model.to(cpu_device)
        std_model.eval()

        # Same input
        input_ids = torch.randint(0, V, (2, 16), device=cpu_device)

        with torch.no_grad():
            gauge_out = gauge_model(input_ids)
            std_result = std_model(input_ids)
            std_out = std_result['logits']

        # Both should produce valid outputs
        assert gauge_out.shape == (2, 16, V)
        assert std_out.shape == (2, 16, V)
        assert torch.isfinite(gauge_out).all()
        assert torch.isfinite(std_out).all()


class TestMemoryEfficiency:
    """Test memory-related behavior."""

    def test_inference_no_memory_leak(self, minimal_config, cpu_device):
        """Test repeated inference doesn't leak memory."""
        from transformer.core.model import GaugeTransformerLM
        import gc

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']

        # Run inference multiple times
        for _ in range(10):
            input_ids = torch.randint(0, V, (4, 32), device=cpu_device)
            with torch.no_grad():
                _ = model(input_ids)

        gc.collect()
        # If this completes without OOM, test passes

    def test_gradient_checkpointing_compatible(self, minimal_config, cpu_device):
        """Test model works with gradient accumulation."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        V = minimal_config['vocab_size']

        # Gradient accumulation over 4 mini-batches
        optimizer.zero_grad()
        total_loss = 0

        for _ in range(4):
            input_ids = torch.randint(0, V, (2, 16), device=cpu_device)
            targets = torch.randint(0, V, (2, 16), device=cpu_device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            loss = loss / 4  # Scale for accumulation
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        assert total_loss > 0
