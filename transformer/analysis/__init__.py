"""
Analysis and Metrics Module
===========================

Tools for analyzing transformer behavior:
- RG metrics: Renormalization group flow analysis
- Publication metrics: Comprehensive metrics for papers
- Trajectory: Belief trajectory tracking
- Semantics: Gauge semantics analysis
"""

from transformer.analysis.rg_metrics import (
    compute_rg_diagnostics,
    RGDiagnostics,
    RGFlowSummary,
)
from transformer.analysis.trajectory import (
    TrajectoryRecorder,
    LeapfrogSnapshot,
    LayerTrajectory,
    ForwardTrajectory,
)

__all__ = [
    # RG metrics
    'compute_rg_diagnostics',
    'RGDiagnostics',
    'RGFlowSummary',

    # Trajectory tracking
    'TrajectoryRecorder',
    'LeapfrogSnapshot',
    'LayerTrajectory',
    'ForwardTrajectory',
]
