# -*- coding: utf-8 -*-
"""
Import Tests
============

Verify all import paths work correctly, including backward compatibility.
"""

import pytest


class TestPackageImports:
    """Test main package imports."""

    def test_main_package_import(self):
        """Test importing the main transformer package."""
        import transformer
        assert hasattr(transformer, 'GaugeTransformerLM')
        assert hasattr(transformer, 'Trainer')
        assert hasattr(transformer, 'TrainingConfig')

    def test_main_package_exports(self):
        """Test __all__ exports are accessible."""
        from transformer import (
            GaugeTransformerLM,
            Trainer,
            TrainingConfig,
            create_optimizer,
            create_param_groups,
            MetricsTracker,
            create_dataloaders,
        )
        assert GaugeTransformerLM is not None
        assert callable(create_dataloaders)


class TestCoreImports:
    """Test core module imports."""

    def test_core_model_import(self):
        """Test importing from transformer.core.model."""
        from transformer.core.model import GaugeTransformerLM
        assert GaugeTransformerLM is not None

    def test_core_attention_import(self):
        """Test importing from transformer.core.attention."""
        from transformer.core.attention import (
            compute_attention_weights,
            IrrepMultiHeadAttention,
            create_attention_mask,
            compute_transport_operators,
        )
        assert callable(compute_attention_weights)
        assert IrrepMultiHeadAttention is not None

    def test_core_blocks_import(self):
        """Test importing from transformer.core.blocks."""
        from transformer.core.blocks import GaugeTransformerBlock, GaugeTransformerStack
        assert GaugeTransformerBlock is not None
        assert GaugeTransformerStack is not None

    def test_core_embeddings_import(self):
        """Test importing from transformer.core.embeddings."""
        from transformer.core.embeddings import (
            GaugeTokenEmbedding,
            GaugePositionalEncoding,
        )
        assert GaugeTokenEmbedding is not None
        assert GaugePositionalEncoding is not None

    def test_core_ffn_import(self):
        """Test importing from transformer.core.ffn."""
        from transformer.core.ffn import GaugeFFN
        assert GaugeFFN is not None

    def test_core_prior_bank_import(self):
        """Test importing from transformer.core.prior_bank."""
        from transformer.core.prior_bank import PriorBank
        assert PriorBank is not None


class TestDataImports:
    """Test data module imports."""

    def test_data_package_import(self):
        """Test importing from transformer.data."""
        from transformer.data import (
            create_dataloaders,
            create_char_dataloaders,
            create_byte_dataloaders,
        )
        assert callable(create_dataloaders)
        assert callable(create_char_dataloaders)
        assert callable(create_byte_dataloaders)

    def test_data_datasets_import(self):
        """Test importing from transformer.data.datasets."""
        from transformer.data.datasets import WikiText2Dataset
        assert WikiText2Dataset is not None


class TestAnalysisImports:
    """Test analysis module imports."""

    def test_analysis_rg_metrics_import(self):
        """Test importing from transformer.analysis.rg_metrics."""
        from transformer.analysis.rg_metrics import (
            compute_rg_diagnostics,
            RGDiagnostics,
            RGFlowSummary,
        )
        assert callable(compute_rg_diagnostics)
        assert RGDiagnostics is not None

    def test_analysis_trajectory_import(self):
        """Test importing from transformer.analysis.trajectory."""
        from transformer.analysis.trajectory import (
            TrajectoryRecorder,
            get_global_recorder,
        )
        assert TrajectoryRecorder is not None
        assert callable(get_global_recorder)

    def test_analysis_publication_metrics_import(self):
        """Test importing from transformer.analysis.publication_metrics."""
        from transformer.analysis.publication_metrics import (
            PublicationMetrics,
            ExperimentResult,
        )
        assert PublicationMetrics is not None
        assert ExperimentResult is not None


class TestTrainingImports:
    """Test training module imports."""

    def test_training_package_import(self):
        """Test importing from transformer.training."""
        from transformer.training import (
            create_optimizer,
            create_param_groups,
            MetricsTracker,
        )
        assert callable(create_optimizer)
        assert callable(create_param_groups)
        assert MetricsTracker is not None

    def test_training_config_import(self):
        """Test importing from transformer.training.config."""
        from transformer.training.config import (
            TrainingConfig,
            get_standard_config,
            get_vfe_dynamic_config,
            get_pure_fep_config,
        )
        assert TrainingConfig is not None
        assert callable(get_standard_config)


class TestBaselinesImports:
    """Test baselines module imports."""

    def test_baselines_standard_transformer_import(self):
        """Test importing from transformer.baselines."""
        from transformer.baselines.standard_transformer import StandardTransformerLM
        assert StandardTransformerLM is not None


class TestBackwardCompatibility:
    """Test backward compatibility imports (old paths still work)."""

    def test_old_model_import(self):
        """Test old import path: from transformer.model import ..."""
        from transformer.model import GaugeTransformerLM
        from transformer.core.model import GaugeTransformerLM as NewGaugeTransformerLM
        assert GaugeTransformerLM is NewGaugeTransformerLM

    def test_old_attention_import(self):
        """Test old import path: from transformer.attention import ..."""
        from transformer.attention import (
            compute_attention_weights,
            IrrepMultiHeadAttention,
        )
        from transformer.core.attention import (
            compute_attention_weights as new_compute,
            IrrepMultiHeadAttention as NewAttention,
        )
        assert compute_attention_weights is new_compute
        assert IrrepMultiHeadAttention is NewAttention

    def test_old_embeddings_import(self):
        """Test old import path: from transformer.embeddings import ..."""
        from transformer.embeddings import GaugeTokenEmbedding
        from transformer.core.embeddings import GaugeTokenEmbedding as NewEmbed
        assert GaugeTokenEmbedding is NewEmbed

    def test_old_transformer_block_import(self):
        """Test old import path: from transformer.transformer_block import ..."""
        from transformer.transformer_block import GaugeTransformerBlock, GaugeTransformerStack
        from transformer.core.blocks import (
            GaugeTransformerBlock as NewBlock,
            GaugeTransformerStack as NewStack,
        )
        assert GaugeTransformerBlock is NewBlock
        assert GaugeTransformerStack is NewStack

    def test_old_standard_transformer_import(self):
        """Test old import path: from transformer.standard_transformer import ..."""
        from transformer.standard_transformer import StandardTransformerLM
        from transformer.baselines.standard_transformer import StandardTransformerLM as NewStd
        assert StandardTransformerLM is NewStd

    def test_old_rg_metrics_import(self):
        """Test old import path: from transformer.rg_metrics import ..."""
        from transformer.rg_metrics import compute_rg_diagnostics, RGDiagnostics
        from transformer.analysis.rg_metrics import (
            compute_rg_diagnostics as new_compute,
            RGDiagnostics as NewDiag,
        )
        assert compute_rg_diagnostics is new_compute
        assert RGDiagnostics is NewDiag

    def test_old_trajectory_tracking_import(self):
        """Test old import path: from transformer.trajectory_tracking import ..."""
        from transformer.trajectory_tracking import TrajectoryRecorder, get_global_recorder
        from transformer.analysis.trajectory import (
            TrajectoryRecorder as NewRecorder,
            get_global_recorder as new_get,
        )
        assert TrajectoryRecorder is NewRecorder
        assert get_global_recorder is new_get

    def test_old_prior_bank_import(self):
        """Test old import path: from transformer.prior_bank import ..."""
        from transformer.prior_bank import PriorBank
        from transformer.core.prior_bank import PriorBank as NewPriorBank
        assert PriorBank is NewPriorBank

    def test_old_ffn_import(self):
        """Test old import path: from transformer.ffn import ..."""
        from transformer.ffn import GaugeFFN
        from transformer.core.ffn import GaugeFFN as NewFFN
        assert GaugeFFN is NewFFN
