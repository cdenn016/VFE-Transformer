# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.variational_ffn for backward compatibility.
New code should import directly from transformer.core.variational_ffn.
"""

from transformer.core.variational_ffn import (
    VariationalFFNDynamic,
    compute_vfe_gradients_gpu,
    compute_natural_gradient_gpu,
)

__all__ = [
    'VariationalFFNDynamic',
    'compute_vfe_gradients_gpu',
    'compute_natural_gradient_gpu',
]
