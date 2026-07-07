"""Pure geometry / math utilities with zero ROS dependencies."""

import math
from typing import Tuple


def clamp(val: float, vmin: float, vmax: float) -> float:
    """Clamp *val* to the inclusive range [*vmin*, *vmax*]."""
    return max(vmin, min(vmax, val))


def pixel_to_mm(
    pixel_dist: float,
    reference_mm: float,
    reference_px: float,
    min_mm_per_px: float = 0.01,
    max_mm_per_px: float = 0.50,
) -> float:
    """Convert a pixel distance to millimetres using a known reference.

    Parameters
    ----------
    pixel_dist : float
        Distance in pixels to convert.
    reference_mm : float
        Known real-world size (mm) of the reference feature.
    reference_px : float
        Measured size (px) of the same reference feature in the current frame.
    min_mm_per_px, max_mm_per_px : float
        Sanity bounds on the computed scale factor.

    Returns
    -------
    float
        Distance in mm.
    """
    if reference_px < 1e-6:
        return 0.0
    scale = reference_mm / reference_px
    scale = clamp(scale, min_mm_per_px, max_mm_per_px)
    return pixel_dist * scale


def euclidean_distance(
    a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])
