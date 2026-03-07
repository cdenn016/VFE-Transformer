# -*- coding: utf-8 -*-
"""
Model Tests
===========

Tests for transformer.core.model.GaugeTransformerLM
"""

import pytest
import torch
import torch.nn as nn


class TestGaugeTransformerLMCreation:
    """Test model creation and configuration."""

    def test_create_minimal_model(self, minimal_config, cpu_device):
        """Test creating model with minimal config."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_has_required_components(self, minimal_config, cpu_device):
        """Test model has all required submodules."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        # Check core components exist
        assert hasattr(model, 'token_embed')
        assert hasattr(model, 'pos_encoding')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'out_proj')

    def test_model_config_stored(self, minimal_config, cpu_device):
        """Test model stores config correctly."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        assert model.config == minimal_config

    def test_model_with_different_ffn_modes(self, cpu_device):
        """Test model creation with different FFN modes."""
        from transformer.core.model import GaugeTransformerLM

        for ffn_mode in ['VFE_dynamic']:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'dropout': 0.0,
                'use_diagonal_covariance': True,
                'ffn_mode': ffn_mode,
            }
            model = GaugeTransformerLM(config)
            assert model is not None

    def test_model_with_tied_embeddings(self, cpu_device):
        """Test model with tied embeddings."""
        from transformer.core.model import GaugeTransformerLM

        config = {
            'vocab_size': 100,
            'embed_dim': 15,
            'n_layers': 1,
            'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
            'hidden_dim': 32,
            'max_seq_len': 32,
            'kappa_beta': 1.0,
            'use_diagonal_covariance': True,
            'tie_embeddings': True,
        }
        model = GaugeTransformerLM(config)

        # Check output projection exists
        assert model.out_proj is not None

    def test_model_parameter_count(self, minimal_config, cpu_device):
        """Test model has reasonable parameter count."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # Minimal model should have < 1M parameters
        assert n_params < 1_000_000


class TestGaugeTransformerLMForward:
    """Test model forward pass."""

    def test_forward_basic(self, gauge_model, batch_tensors, cpu_device):
        """Test basic forward pass."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)

        # Check output shape
        B, N = input_ids.shape
        V = gauge_model.config['vocab_size']
        assert logits.shape == (B, N, V)

    def test_forward_output_finite(self, gauge_model, batch_tensors, cpu_device):
        """Test forward pass produces finite outputs."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits = gauge_model(input_ids)

        assert torch.isfinite(logits).all(), "Output contains NaN or Inf"

    def test_forward_with_agents(self, gauge_model, batch_tensors, cpu_device):
        """Test forward pass returning agent states."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        with torch.no_grad():
            logits, agents = gauge_model(input_ids, return_agents=True)

        # Check logits
        B, N = input_ids.shape
        V = gauge_model.config['vocab_size']
        assert logits.shape == (B, N, V)

        # Check agent states
        assert 'mu' in agents
        assert 'sigma' in agents
        assert agents['mu'].shape[0] == B
        assert agents['mu'].shape[1] == N

    def test_forward_different_batch_sizes(self, minimal_config, cpu_device):
        """Test forward with different batch sizes."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        N = 16

        for B in [1, 2, 4, 8]:
            input_ids = torch.randint(0, V, (B, N), device=cpu_device)
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (B, N, V)

    def test_forward_different_sequence_lengths(self, minimal_config, cpu_device):
        """Test forward with different sequence lengths."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        B = 2
        max_len = minimal_config['max_seq_len']

        for N in [4, 8, 16, max_len]:
            input_ids = torch.randint(0, V, (B, N), device=cpu_device)
            with torch.no_grad():
                logits = model(input_ids)
            assert logits.shape == (B, N, V)

    def test_forward_with_attention(self, minimal_config, cpu_device):
        """Test forward_with_attention method."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.eval()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)

        with torch.no_grad():
            logits, attn_info = model.forward_with_attention(input_ids)

        assert logits.shape == (B, N, V)
        assert 'beta_layers' in attn_info or 'beta' in attn_info


class TestGaugeTransformerLMGradients:
    """Test model gradient computation."""

    def test_gradients_flow(self, minimal_config, cpu_device):
        """Test gradients flow through model."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )
        loss.backward()

        # Check some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No parameters received gradients"

    def test_gradients_finite(self, minimal_config, cpu_device):
        """Test gradients are finite."""
        from transformer.core.model import GaugeTransformerLM

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)
        model.train()

        V = minimal_config['vocab_size']
        B, N = 2, 16
        input_ids = torch.randint(0, V, (B, N), device=cpu_device)
        targets = torch.randint(0, V, (B, N), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )
        loss.backward()

        # Check all gradients are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Gradient for {name} contains NaN/Inf"


class TestGaugeTransformerLMEvalMode:
    """Test model behavior in eval mode."""

    def test_eval_mode_deterministic(self, gauge_model, batch_tensors, cpu_device):
        """Test model is deterministic in eval mode."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        gauge_model.eval()

        with torch.no_grad():
            out1 = gauge_model(input_ids)
            out2 = gauge_model(input_ids)

        assert torch.allclose(out1, out2), "Model not deterministic in eval mode"

    def test_train_eval_toggle(self, gauge_model, batch_tensors, cpu_device):
        """Test switching between train and eval mode."""
        input_ids = batch_tensors['input_ids'].to(cpu_device)

        # Start in eval mode
        gauge_model.eval()
        with torch.no_grad():
            eval_out = gauge_model(input_ids)

        # Switch to train and back
        gauge_model.train()
        gauge_model.eval()

        with torch.no_grad():
            eval_out2 = gauge_model(input_ids)

        assert torch.allclose(eval_out, eval_out2)


class TestGaugeTransformerLMConfigurations:
    """Test various model configurations."""

    def test_config_evolve_sigma(self, cpu_device):
        """Test evolve_sigma configuration."""
        from transformer.core.model import GaugeTransformerLM

        for evolve_sigma in [True, False]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'evolve_sigma': evolve_sigma,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_evolve_phi(self, cpu_device):
        """Test evolve_phi configuration."""
        from transformer.core.model import GaugeTransformerLM

        for evolve_phi in [True, False]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'evolve_phi': evolve_phi,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_different_kappa(self, cpu_device):
        """Test different kappa_beta values."""
        from transformer.core.model import GaugeTransformerLM

        for kappa in [0.1, 1.0, 10.0]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': 1,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': kappa,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_multiple_layers(self, cpu_device):
        """Test model with multiple layers."""
        from transformer.core.model import GaugeTransformerLM

        for n_layers in [1, 2, 3]:
            config = {
                'vocab_size': 100,
                'embed_dim': 15,
                'n_layers': n_layers,
                'hidden_dim': 32,
                'max_seq_len': 32,
                'kappa_beta': 1.0,
                'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
                'use_diagonal_covariance': True,
                'ffn_mode': 'VFE_dynamic',
            }
            model = GaugeTransformerLM(config)
            input_ids = torch.randint(0, 100, (2, 16))

            with torch.no_grad():
                logits = model(input_ids)

            assert torch.isfinite(logits).all()

    def test_config_diagonal_covariance(self, cpu_device):
        """Test diagonal covariance mode."""
        from transformer.core.model import GaugeTransformerLM

        config = {
            'vocab_size': 100,
            'embed_dim': 15,
            'n_layers': 1,
            'hidden_dim': 32,
            'max_seq_len': 32,
            'kappa_beta': 1.0,
            'diagonal_covariance': True,
            'irrep_spec': [('l0', 6, 1), ('l1', 3, 3)],
            'ffn_mode': 'VFE_dynamic',
        }
        model = GaugeTransformerLM(config)
        input_ids = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            logits = model(input_ids)

        assert torch.isfinite(logits).all()


class TestGaugeTransformerLMSaveLoad:
    """Test model save/load functionality."""

    def test_state_dict_roundtrip(self, minimal_config, cpu_device, tmp_path):
        """Test saving and loading state dict."""
        from transformer.core.model import GaugeTransformerLM

        # Create and run model
        model1 = GaugeTransformerLM(minimal_config)
        model1 = model1.to(cpu_device)

        input_ids = torch.randint(0, 100, (2, 16), device=cpu_device)
        with torch.no_grad():
            out1 = model1(input_ids)

        # Save state dict
        state_dict = model1.state_dict()
        save_path = tmp_path / "model.pt"
        torch.save(state_dict, save_path)

        # Create new model and load
        model2 = GaugeTransformerLM(minimal_config)
        model2 = model2.to(cpu_device)
        model2.load_state_dict(torch.load(save_path, weights_only=True))

        # Check outputs match
        with torch.no_grad():
            out2 = model2(input_ids)

        assert torch.allclose(out1, out2)
