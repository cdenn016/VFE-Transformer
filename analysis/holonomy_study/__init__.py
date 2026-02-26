"""
Holonomy Study: Measuring Gauge Curvature in Pretrained Transformers
====================================================================

Tests the flat bundle conjecture: if language has evolved toward minimal
holonomy, then the effective multi-layer transport in standard transformers
should be approximately flat for literal language and show measurable
curvature for ironic/pragmatic language where meaning transport is
path-dependent.

Holonomy H_{ijk} = T_{ij} T_{jk} T_{ki} around closed token loops,
where T_{ij} = dh_i^{(L)} / dh_j^{(0)} is the effective Jacobian transport.

Modules:
    transport       Extract transport operators from pretrained models
    holonomy        Compute holonomy and curvature metrics
    datasets        Irony/literal/control sentence pairs
    experiment      Main experiment pipeline
    visualization   Distribution plots and layer profiles
"""

from .holonomy import loop_holonomy, sentence_holonomy, multilayer_holonomy
from .experiment import run_holonomy_study
