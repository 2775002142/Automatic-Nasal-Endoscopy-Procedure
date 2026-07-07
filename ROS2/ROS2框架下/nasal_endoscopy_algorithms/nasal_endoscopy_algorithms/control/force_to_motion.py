"""Force-field → robot-motion converter.

Converts pixel-space APF force vectors into tool-frame rotation angles
and translation steps with dynamic, non-linear gain scheduling.
"""

import math
from dataclasses import dataclass

from nasal_endoscopy_algorithms.utils.geometry import clamp


@dataclass
class MotionCommand:
    """Single-cycle motion command in tool-frame coordinates.

    All fields are **incremental** (delta), not absolute.
    """
    rx_deg: float = 0.0   # rotation about tool X (pitch)
    ry_deg: float = 0.0   # rotation about tool Y (yaw)
    dx_mm: float = 0.0    # translation along tool X
    dy_mm: float = 0.0    # translation along tool Y


class ForceToMotionConverter:
    """Maps a 2-D pixel force vector to tool-frame rotation + translation.

    Parameters
    ----------
    force_deadzone : float
        Force magnitude below which no motion is produced.
    max_force_for_scale : float
        Assumed maximum force magnitude for gain normalisation.
    min_rotation_gain, max_rotation_gain : float
        Rotation gain bounds (deg / pixel).
    rotation_gain_curve_factor : float
        Non-linearity exponent for rotation gain ( <1 → more aggressive
        at low forces).
    max_rotation_deg : float
        Hard ceiling on single-step rotation magnitude.
    min_translate_step_mm, max_translate_step_mm : float
        Translation step bounds (mm).
    translate_step_curve_factor : float
        Non-linearity exponent for translation step size.
    max_translate_per_phase_mm : float
        Per-phase cumulative-translation limit (mm).
    """

    def __init__(
        self,
        force_deadzone: float = 10.0,
        max_force_for_scale: float = 1150.0,
        min_rotation_gain: float = 0.001,
        max_rotation_gain: float = 0.012,
        rotation_gain_curve_factor: float = 0.6,
        max_rotation_deg: float = 0.5,
        min_translate_step_mm: float = 0.02,
        max_translate_step_mm: float = 0.15,
        translate_step_curve_factor: float = 0.7,
        max_translate_per_phase_mm: float = 1.0,
    ) -> None:
        self.force_deadzone = float(force_deadzone)
        self.max_force_for_scale = float(max_force_for_scale)
        self.min_rotation_gain = float(min_rotation_gain)
        self.max_rotation_gain = float(max_rotation_gain)
        self.rotation_gain_curve_factor = float(rotation_gain_curve_factor)
        self.max_rotation_deg = float(max_rotation_deg)
        self.min_translate_step_mm = float(min_translate_step_mm)
        self.max_translate_step_mm = float(max_translate_step_mm)
        self.translate_step_curve_factor = float(translate_step_curve_factor)
        self.max_translate_per_phase_mm = float(max_translate_per_phase_mm)

    # ── public API ──────────────────────────────────────────

    def convert(
        self,
        force_vector: "np.ndarray",
        current_phase_dx: float,
        current_phase_dy: float,
    ) -> "MotionCommand":
        """Convert a single force vector into a motion command.

        Parameters
        ----------
        force_vector : np.ndarray
            Force in pixel units, shape (2,).
        current_phase_dx, current_phase_dy : float
            Cumulative translation already executed in the *current* phase
            (used to enforce ``max_translate_per_phase_mm``).

        Returns
        -------
        MotionCommand
        """
        import numpy as np

        fx, fy = float(force_vector[0]), float(force_vector[1])
        force_mag = math.sqrt(fx * fx + fy * fy)

        dyn_rot_gain, dyn_trans_step = self._calculate_dynamic_params(force_mag)
        if dyn_rot_gain == 0.0 and dyn_trans_step == 0.0:
            return MotionCommand()

        # Rotation:  +fy (target below centre) → -Rx (pitch down)
        #            +fx (target right of centre) → +Ry (yaw right)
        rx_deg = -fy * dyn_rot_gain
        ry_deg = fx * dyn_rot_gain
        rx_deg = clamp(rx_deg, -self.max_rotation_deg, self.max_rotation_deg)
        ry_deg = clamp(ry_deg, -self.max_rotation_deg, self.max_rotation_deg)

        dx_mm, dy_mm = 0.0, 0.0
        if force_mag > 1e-6:
            dir_x = fx / force_mag
            dir_y = fy / force_mag

            dx_cand = dir_x * dyn_trans_step
            dy_cand = dir_y * dyn_trans_step

            if abs(current_phase_dx + dx_cand) <= self.max_translate_per_phase_mm:
                dx_mm = dx_cand
            if abs(current_phase_dy + dy_cand) <= self.max_translate_per_phase_mm:
                dy_mm = dy_cand

        return MotionCommand(
            rx_deg=rx_deg, ry_deg=ry_deg, dx_mm=dx_mm, dy_mm=dy_mm
        )

    # ── internal ────────────────────────────────────────────

    def _calculate_dynamic_params(
        self, force_mag: float
    ) -> "tuple[float, float]":
        if force_mag < self.force_deadzone:
            return 0.0, 0.0

        norm_force = clamp(
            (force_mag - self.force_deadzone)
            / (self.max_force_for_scale - self.force_deadzone),
            0.0,
            1.0,
        )

        rot_gain = self.min_rotation_gain + (
            self.max_rotation_gain - self.min_rotation_gain
        ) * (norm_force ** self.rotation_gain_curve_factor)

        trans_step = self.min_translate_step_mm + (
            self.max_translate_step_mm - self.min_translate_step_mm
        ) * (norm_force ** self.translate_step_curve_factor)

        return rot_gain, trans_step
