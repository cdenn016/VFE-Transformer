# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.model for backward compatibility.
New code should import directly from transformer.core.model.
"""

from transformer.core.model import GaugeTransformerLM

__all__ = ['GaugeTransformerLM']
