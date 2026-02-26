# -*- coding: utf-8 -*-
"""
Backward Compatibility Module
=============================

This module re-exports from transformer.analysis.trajectory for backward compatibility.
New code should import directly from transformer.analysis.trajectory.
"""

from transformer.analysis.trajectory import (
    ForwardTrajectory,
    LayerTrajectory,
    LeapfrogSnapshot,
    TrajectoryRecorder,
    get_global_recorder,
    set_global_recorder,
)

__all__ = [
    'ForwardTrajectory',
    'LayerTrajectory',
    'LeapfrogSnapshot',
    'TrajectoryRecorder',
    'get_global_recorder',
    'set_global_recorder',
]
