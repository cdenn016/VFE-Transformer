# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.embeddings for backward compatibility.
New code should import directly from transformer.core.embeddings.
"""

from transformer.core.embeddings import (
    GaugeTokenEmbedding,
    GaugePositionalEncoding,
    so3_compose_bch,
)

__all__ = [
    'GaugeTokenEmbedding',
    'GaugePositionalEncoding',
    'so3_compose_bch',
]
