# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.analysis.rg_metrics for backward compatibility.
New code should import directly from transformer.analysis.rg_metrics.
"""

from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)

__all__ = [
    'compute_rg_diagnostics',
    'RGDiagnostics',
    'RGFlowSummary',
]
