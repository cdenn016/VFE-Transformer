# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.attention for backward compatibility.
New code should import directly from transformer.core.attention.
"""

from transformer.core.attention import (
    compute_attention_weights,
    aggregate_messages,
    compute_kl_matrix,
    IrrepMultiHeadAttention,
    create_attention_mask,
    compute_transport_operators,
    estimate_chunk_size,
    NUMBA_AVAILABLE,
    TRANSPORT_AVAILABLE,
)

__all__ = [
    'compute_attention_weights',
    'aggregate_messages',
    'compute_kl_matrix',
    'IrrepMultiHeadAttention',
    'create_attention_mask',
    'compute_transport_operators',
    'estimate_chunk_size',
    'NUMBA_AVAILABLE',
    'TRANSPORT_AVAILABLE',
]
