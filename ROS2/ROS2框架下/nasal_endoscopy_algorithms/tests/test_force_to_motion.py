"""Tests for ForceToMotionConverter."""

import numpy as np
import pytest
from nasal_endoscopy_algorithms.control.force_to_motion import (
    ForceToMotionConverter,
    MotionCommand,
)


class TestForceToMotionConverter:
    @pytest.fixture
    def converter(self):
        return ForceToMotionConverter(
            force_deadzone=10.0,
            max_force_for_scale=1150.0,
            min_rotation_gain=0.001,
            max_rotation_gain=0.012,
            max_rotation_deg=0.5,
            min_translate_step_mm=0.02,
            max_translate_step_mm=0.15,
            max_translate_per_phase_mm=1.0,
        )

    def test_deadzone_returns_zero(self, converter):
        cmd = converter.convert(np.array([5.0, 0.0]), 0.0, 0.0)
        assert cmd.rx_deg == 0.0
        assert cmd.ry_deg == 0.0
        assert cmd.dx_mm == 0.0
        assert cmd.dy_mm == 0.0

    def test_positive_fx_produces_positive_ry(self, converter):
        """Target to the right → positive yaw."""
        cmd = converter.convert(np.array([500.0, 0.0]), 0.0, 0.0)
        assert cmd.ry_deg > 0
        assert cmd.dx_mm > 0

    def test_positive_fy_produces_negative_rx(self, converter):
        """Target below centre → negative pitch."""
        cmd = converter.convert(np.array([0.0, 500.0]), 0.0, 0.0)
        assert cmd.rx_deg < 0
        assert cmd.dy_mm > 0

    def test_rotation_clamped(self, converter):
        cmd = converter.convert(np.array([5000.0, 0.0]), 0.0, 0.0)
        assert abs(cmd.ry_deg) <= converter.max_rotation_deg

    def test_translate_phase_cap(self, converter):
        """Once the phase cumulative translation is at the cap,
        no further translation should be produced."""
        cmd = converter.convert(
            np.array([500.0, 0.0]),
            converter.max_translate_per_phase_mm,  # already saturated
            0.0,
        )
        assert cmd.dx_mm == 0.0

    def test_dataclass_defaults(self):
        cmd = MotionCommand()
        assert cmd.rx_deg == 0.0
        assert cmd.ry_deg == 0.0
        assert cmd.dx_mm == 0.0
        assert cmd.dy_mm == 0.0
