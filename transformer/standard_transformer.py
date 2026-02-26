# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.baselines.standard_transformer for backward compatibility.
New code should import directly from transformer.baselines.standard_transformer.
"""

from transformer.baselines.standard_transformer import StandardTransformerLM

__all__ = ['StandardTransformerLM']
