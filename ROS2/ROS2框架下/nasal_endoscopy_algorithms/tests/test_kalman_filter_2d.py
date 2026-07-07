"""Tests for KalmanFilter2D."""

import numpy as np
import pytest
from nasal_endoscopy_algorithms.filters.kalman_filter_2d import KalmanFilter2D


class TestKalmanFilter2D:
    def test_init_uninitialised(self):
        kf = KalmanFilter2D()
        assert not kf.is_initialized()
        assert kf.predict_only() is None

    def test_first_update_sets_position(self):
        kf = KalmanFilter2D(dt=1.0)
        out = kf.update([10.0, 20.0])
        assert kf.is_initialized()
        np.testing.assert_array_almost_equal(out, np.array([10.0, 20.0]))

    def test_predict_only_after_init(self):
        kf = KalmanFilter2D(dt=1.0)
        kf.update([10.0, 20.0])
        pred = kf.predict_only()
        assert pred is not None
        # Position should advance by velocity (which is still ~0)
        assert pred.shape == (2,)

    def test_reset(self):
        kf = KalmanFilter2D()
        kf.update([5.0, 5.0])
        kf.reset()
        assert not kf.is_initialized()

    def test_smoothing(self):
        """Noisy measurements should be smoothed."""
        kf = KalmanFilter2D(
            dt=1.0, process_noise_cov=1e-4, measurement_noise_cov=100.0
        )
        results = []
        for _ in range(20):
            results.append(kf.update([100.0 + np.random.randn() * 10,
                                       50.0 + np.random.randn() * 10]))
        final = results[-1]
        # After 20 updates the estimate should be near the true mean
        assert abs(final[0] - 100.0) < 15.0
        assert abs(final[1] - 50.0) < 15.0
