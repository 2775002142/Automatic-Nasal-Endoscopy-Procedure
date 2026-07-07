"""Exponential Moving Average filter for 1-D and N-D signals."""

import numpy as np
from typing import Optional, Union


class EMAFilter:
    """Simple exponential moving average filter.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1].  Larger → less smoothing.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha: float = float(alpha)
        self.inited: bool = False
        self.value: Optional[np.ndarray] = None

    # ── public API ──────────────────────────────────────────

    def reset(self) -> None:
        """Discard accumulated state."""
        self.inited = False
        self.value = None

    def update(self, x: Union[float, np.ndarray, list]) -> np.ndarray:
        """Accept a new measurement, return the filtered value."""
        x_arr = np.asarray(x, dtype=np.float32)
        if not self.inited:
            self.value = x_arr
            self.inited = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * x_arr
        return self.value
