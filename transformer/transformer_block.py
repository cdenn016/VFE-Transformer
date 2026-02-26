# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.blocks for backward compatibility.
New code should import directly from transformer.core.blocks.
"""

from transformer.core.blocks import (
    GaugeTransformerBlock,
    GaugeTransformerStack,
)

__all__ = ['GaugeTransformerBlock', 'GaugeTransformerStack']
