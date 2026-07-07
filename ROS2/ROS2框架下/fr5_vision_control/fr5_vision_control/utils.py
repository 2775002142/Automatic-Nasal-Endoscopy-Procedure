"""Backward-compatibility re-exports from ``nasal_endoscopy_algorithms``.

New code should import directly from the algorithm package.
These re-exports exist so existing scripts (e.g. calibrate_*.py)
continue to work without modification.
"""

from nasal_endoscopy_algorithms.utils.state_enums import SystemState
from nasal_endoscopy_algorithms.utils.geometry import clamp
from nasal_endoscopy_algorithms.filters.ema_filter import EMAFilter
from nasal_endoscopy_algorithms.filters.kalman_filter_2d import KalmanFilter2D

__all__ = ["SystemState", "clamp", "EMAFilter", "KalmanFilter2D"]
