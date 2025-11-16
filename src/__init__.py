"""
High-Precision UAV-Satellite Image Matching System

This package implements a multi-stage pipeline for matching UAV and satellite images
with subpixel accuracy using IMU constraints, factor graph optimization, and
template matching.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .stage2_imu_constraint import IMUConstraintOptimizer
from .stage3_factor_graph import FactorGraphOptimizer
from .stage5_subpixel_matching import SubpixelMatcher
from .pipeline import MatchingPipeline

__all__ = [
    "IMUConstraintOptimizer",
    "FactorGraphOptimizer",
    "SubpixelMatcher",
    "MatchingPipeline",
]
