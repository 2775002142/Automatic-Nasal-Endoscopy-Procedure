"""2-D Kalman filter for (x, y) point tracking.

State vector: [x, y, vx, vy]ᵀ
Measurement vector: [x, y]ᵀ
"""

import numpy as np
from typing import Optional


class KalmanFilter2D:
    """Constant-velocity Kalman filter for noisy 2-D measurements.

    Parameters
    ----------
    dt : float
        Time step (seconds) between predictions.
    process_noise_cov : float
        Process-noise scale (scalar, applied to a 4×4 identity).
    measurement_noise_cov : float
        Measurement-noise scale (scalar, applied to a 2×2 identity).
    """

    def __init__(
        self,
        dt: float = 0.033,
        process_noise_cov: float = 1e-2,
        measurement_noise_cov: float = 1e0,
    ) -> None:
        self.dt: float = float(dt)

        # State-transition matrix (4×4)
        self.A: np.ndarray = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float64,
        )

        # Observation matrix (2×4)
        self.H: np.ndarray = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float64,
        )

        self.Q: np.ndarray = float(process_noise_cov) * np.eye(4, dtype=np.float64)
        self.R: np.ndarray = float(measurement_noise_cov) * np.eye(2, dtype=np.float64)

        self.P: np.ndarray = np.eye(4, dtype=np.float64)
        self.x_hat: np.ndarray = np.zeros((4, 1), dtype=np.float64)
        self.initialized: bool = False

    # ── public API ──────────────────────────────────────────

    def reset(self) -> None:
        """Clear all internal state."""
        self.initialized = False
        self.x_hat = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def is_initialized(self) -> bool:
        """Return True if at least one measurement has been ingested."""
        return self.initialized

    def predict_only(self) -> Optional[np.ndarray]:
        """Run a pure prediction step (no measurement).

        Returns
        -------
        np.ndarray or None
            Predicted (x, y) position, or *None* if not yet initialised.
        """
        if not self.initialized:
            return None
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat[0:2].flatten()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Predict + update with a new measurement.

        Returns
        -------
        np.ndarray
            Filtered (x, y) position.
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(2, 1)

        if not self.initialized:
            self.x_hat[0:2] = z
            self.initialized = True
            return self.x_hat[0:2].flatten()

        # ── predict ──
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q

        # ── update ──
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y_residual = z - (self.H @ self.x_hat)
        self.x_hat = self.x_hat + (K @ y_residual)
        I = np.eye(self.A.shape[0], dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.x_hat[0:2].flatten()
