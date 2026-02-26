# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.ffn for backward compatibility.
New code should import directly from transformer.core.ffn.
"""

from transformer.core.ffn import GaugeFFN

__all__ = ['GaugeFFN']
