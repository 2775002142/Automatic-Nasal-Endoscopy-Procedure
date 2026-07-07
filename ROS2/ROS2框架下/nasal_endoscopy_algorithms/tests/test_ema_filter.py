"""Tests for EMAFilter."""

import numpy as np
import pytest
from nasal_endoscopy_algorithms.filters.ema_filter import EMAFilter


class TestEMAFilter:
    def test_init_uninitialised(self):
        f = EMAFilter(alpha=0.3)
        assert not f.inited
        assert f.value is None

    def test_first_update_sets_value(self):
        f = EMAFilter(alpha=0.5)
        out = f.update(10.0)
        assert f.inited
        np.testing.assert_array_equal(out, np.array([10.0], dtype=np.float32))

    def test_convergence(self):
        f = EMAFilter(alpha=0.5)
        f.update(0.0)
        out = f.update(10.0)
        # After one non-init update: (1-0.5)*0 + 0.5*10 = 5
        assert float(out[0]) == pytest.approx(5.0, abs=1e-6)

    def test_reset(self):
        f = EMAFilter(alpha=0.3)
        f.update(42.0)
        f.reset()
        assert not f.inited
        assert f.value is None

    def test_nd_array_input(self):
        f = EMAFilter(alpha=0.4)
        out = f.update([100.0, 200.0])
        assert out.shape == (2,)
