# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.core.prior_bank for backward compatibility.
New code should import directly from transformer.core.prior_bank.
"""

from transformer.core.prior_bank import PriorBank

__all__ = ['PriorBank']
