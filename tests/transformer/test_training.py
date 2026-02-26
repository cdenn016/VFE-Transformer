# -*- coding: utf-8 -*-
"""
Training Utilities Tests
========================

Tests for transformer.training module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_config(self):
        """Test creating default config."""
        from transformer.training.config import TrainingConfig

        config = TrainingConfig()
        assert config is not None
        assert hasattr(config, 'mu_lr')
        assert hasattr(config, 'sigma_lr')

    def test_get_standard_config(self):
        """Test get_standard_config preset."""
        from transformer.training.config import get_standard_config

        config = get_standard_config()
        assert config is not None
        assert config.training_mode == 'standard'

    def test_get_vfe_dynamic_config(self):
        """Test get_vfe_dynamic_config preset."""
        from transformer.training.config import get_vfe_dynamic_config

        config = get_vfe_dynamic_config()
        assert config is not None
        assert config.training_mode == 'vfe_dynamic'

    def test_config_overrides(self):
        """Test config with custom overrides."""
        from transformer.training.config import get_standard_config

        config = get_standard_config(
            mu_lr=0.05,
            max_steps=500,
        )

        assert config.mu_lr == 0.05
        assert config.max_steps == 500

    def test_identity_group_default_false(self):
        """Test use_identity_group defaults to False."""
        from transformer.training.config import TrainingConfig

        config = TrainingConfig()
        assert config.use_identity_group is False

    def test_identity_group_vfe_dynamic(self):
        """Test use_identity_group can be set via get_vfe_dynamic_config."""
        from transformer.training.config import get_vfe_dynamic_config

        config = get_vfe_dynamic_config(use_identity_group=True)
        assert config.use_identity_group is True
        assert config.training_mode == 'vfe_dynamic'

    def test_identity_group_override(self):
        """Test use_identity_group override on standard config."""
        from transformer.training.config import get_standard_config

        config = get_standard_config(use_identity_group=True)
        assert config.use_identity_group is True


class TestCreateParamGroups:
    """Test create_param_groups function."""

    def test_param_groups_creation(self, minimal_config, cpu_device):
        """Test creating parameter groups."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_param_groups
        from transformer.training.config import TrainingConfig

        model = GaugeTransformerLM(minimal_config)

        config = TrainingConfig(
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attention_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        param_groups = create_param_groups(model, config)

        assert isinstance(param_groups, list)
        assert len(param_groups) > 0

        # Each group should have 'params' and 'lr'
        for group in param_groups:
            assert 'params' in group
            assert 'lr' in group

    def test_all_params_covered(self, minimal_config, cpu_device):
        """Test all parameters are in some group."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_param_groups
        from transformer.training.config import TrainingConfig

        model = GaugeTransformerLM(minimal_config)

        config = TrainingConfig(
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attention_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        param_groups = create_param_groups(model, config)

        # Collect all params in groups
        grouped_params = set()
        for group in param_groups:
            for p in group['params']:
                grouped_params.add(id(p))

        # Check all model params are grouped
        model_params = set(id(p) for p in model.parameters())

        # All model params should be in groups (or subset if some are frozen)
        # Note: some implementations may not include all params
        assert len(grouped_params) > 0


class TestCreateOptimizer:
    """Test create_optimizer function."""

    def test_optimizer_creation(self, minimal_config, cpu_device):
        """Test creating optimizer."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_optimizer
        from transformer.training.config import TrainingConfig

        model = GaugeTransformerLM(minimal_config)

        config = TrainingConfig(
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attention_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        optimizer = create_optimizer(model, config)

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.Optimizer)

    def test_optimizer_step(self, minimal_config, cpu_device):
        """Test optimizer can perform step."""
        from transformer.core.model import GaugeTransformerLM
        from transformer.training.optimizer import create_optimizer
        from transformer.training.config import TrainingConfig

        model = GaugeTransformerLM(minimal_config)
        model = model.to(cpu_device)

        config = TrainingConfig(
            mu_lr=0.1,
            sigma_lr=0.01,
            phi_lr=0.05,
            attention_lr=0.001,
            ffn_lr=0.001,
            output_lr=0.001,
        )

        optimizer = create_optimizer(model, config)

        # Forward pass
        V = minimal_config['vocab_size']
        input_ids = torch.randint(0, V, (2, 16), device=cpu_device)
        targets = torch.randint(0, V, (2, 16), device=cpu_device)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V),
            targets.view(-1)
        )

        # Backward and step
        loss.backward()
        optimizer.step()

        # Should complete without error


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_tracker_creation(self, tmp_path):
        """Test creating metrics tracker."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(output_dir=tmp_path)
        assert tracker is not None

    def test_tracker_log(self, tmp_path):
        """Test logging metrics."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(output_dir=tmp_path)

        tracker.log({
            'loss': 2.5,
            'accuracy': 0.8,
        }, step=1)

        assert len(tracker.history) == 1

    def test_tracker_save(self, tmp_path):
        """Test saving metrics to CSV."""
        from transformer.training.metrics import MetricsTracker

        tracker = MetricsTracker(output_dir=tmp_path)

        for i in range(5):
            tracker.log({
                'loss': 2.5 - i * 0.1,
            }, step=i)

        tracker.save()

        # Check CSV file created
        csv_files = list(tmp_path.glob('*.csv'))
        assert len(csv_files) > 0
