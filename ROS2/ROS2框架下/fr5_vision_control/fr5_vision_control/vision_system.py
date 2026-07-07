"""Backward-compatibility re-exports from ``nasal_endoscopy_algorithms``.

New code should import directly from the algorithm package.
"""

from nasal_endoscopy_algorithms.vision.outside_vision import (
    VisionSystem,
    NostrilDetectionResult,
)

__all__ = ["VisionSystem", "NostrilDetectionResult"]
