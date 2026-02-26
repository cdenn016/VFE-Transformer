# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.analysis.publication_metrics for backward compatibility.
New code should import directly from transformer.analysis.publication_metrics.
"""

from transformer.analysis.publication_metrics import (
    PublicationMetrics,
    ExperimentResult,
)

__all__ = [
    'PublicationMetrics',
    'ExperimentResult',
]
