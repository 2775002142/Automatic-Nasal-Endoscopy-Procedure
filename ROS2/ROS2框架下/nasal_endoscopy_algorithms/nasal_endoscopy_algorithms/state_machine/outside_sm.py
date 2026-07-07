"""Outside-navigation state machine (体外导航状态机).

Encapsulates the full state-transition logic from ``move_outside_node.py``
as a pure, testable object with zero ROS dependencies.
"""

from dataclasses import dataclass
from typing import Optional

from nasal_endoscopy_algorithms.utils.state_enums import SystemState


@dataclass
class OutsideSMConfig:
    """Tunable parameters for the outside state machine."""

    align_tolerance_enter: float = 20.0   # px — error above which we re-enter ALIGN
    align_tolerance_exit: float = 10.0    # px — error below which we exit ALIGN
    min_align_frames: int = 3             # minimum frames in ALIGN before transition
    min_approach_frames: int = 3          # minimum frames in APPROACH before transition
    transition_frames: int = 5            # frames for cross-fade transitions


class OutsideStateMachine:
    """Hysteretic state machine for outside (nostril-approach) navigation.

    Usage
    -----
    .. code-block:: python

        sm = OutsideStateMachine(OutsideSMConfig())
        for frame in camera:
            detection = vision.detect_nose_target(...)
            new_state = sm.evaluate(
                has_target=(detection.nose_pos is not None),
                dist_err_px=...,
                filtered_w=detection.feature_width,
                is_auto_run=True,
                is_finished=False,
                target_width_threshold=300,
            )
    """

    def __init__(self, config: OutsideSMConfig = OutsideSMConfig()) -> None:
        self.cfg = config
        self.current_state: SystemState = SystemState.IDLE
        self.frame_counter: int = 0

    # ── public API ──────────────────────────────────────────

    def reset(self) -> None:
        """Reset to IDLE and clear the frame counter."""
        self.current_state = SystemState.IDLE
        self.frame_counter = 0

    def evaluate(
        self,
        has_target: bool,
        dist_err_px: float,
        filtered_w: float,
        is_auto_run: bool,
        is_finished: bool,
        target_width_threshold: float,
    ) -> SystemState:
        """Evaluate one frame and return the (possibly new) state.

        Parameters
        ----------
        has_target : bool
            Whether a valid nostril detection is available this frame.
        dist_err_px : float
            Euclidean pixel error from image centre to target.
        filtered_w : float
            EMA-filtered feature width (pixels).
        is_auto_run : bool
            Whether auto-navigation is engaged.
        is_finished : bool
            External "task complete" flag (e.g. depth limit hit).
        target_width_threshold : float
            Feature width (px) that signals arrival.

        Returns
        -------
        SystemState
        """
        # ── terminal / override conditions ──
        if is_finished:
            new_state = SystemState.TARGET_REACHED
        elif not is_auto_run:
            new_state = SystemState.IDLE
        elif not has_target:
            new_state = SystemState.TARGET_LOST
        else:
            new_state = self._transition_from_valid(dist_err_px, filtered_w,
                                                    target_width_threshold)

        # ── frame-counter management ──
        if new_state != self.current_state:
            self.current_state = new_state
            self.frame_counter = 0
        else:
            self.frame_counter += 1

        return self.current_state

    # ── internal ────────────────────────────────────────────

    def _transition_from_valid(
        self,
        dist_err: float,
        filtered_w: float,
        target_width_threshold: float,
    ) -> SystemState:
        cfg = self.cfg
        prev = self.current_state
        cnt = self.frame_counter

        if prev == SystemState.APPROACH_Z:
            if (dist_err > cfg.align_tolerance_enter
                    and cnt >= cfg.min_approach_frames):
                return SystemState.TRANSITION_TO_ALIGN
            return SystemState.APPROACH_Z

        if prev == SystemState.ALIGN_XY:
            if (dist_err <= cfg.align_tolerance_exit
                    and cnt >= cfg.min_align_frames):
                if filtered_w < target_width_threshold:
                    return SystemState.TRANSITION_TO_APPROACH
                return SystemState.TARGET_REACHED
            return SystemState.ALIGN_XY

        if prev in (SystemState.IDLE, SystemState.TARGET_LOST):
            if dist_err > cfg.align_tolerance_exit:
                return SystemState.ALIGN_XY
            return SystemState.APPROACH_Z

        # Transition states — count up, then switch
        if prev == SystemState.TRANSITION_TO_APPROACH:
            if cnt >= cfg.transition_frames:
                return SystemState.APPROACH_Z
            return SystemState.TRANSITION_TO_APPROACH

        if prev == SystemState.TRANSITION_TO_ALIGN:
            if cnt >= cfg.transition_frames:
                return SystemState.ALIGN_XY
            return SystemState.TRANSITION_TO_ALIGN

        # Fallback
        if dist_err > cfg.align_tolerance_exit:
            return SystemState.ALIGN_XY
        return SystemState.APPROACH_Z
