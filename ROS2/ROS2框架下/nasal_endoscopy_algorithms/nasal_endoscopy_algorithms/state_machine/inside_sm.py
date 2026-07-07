"""Inside-navigation state machine (体内导航状态机).

Encapsulates the full state-transition logic from ``move_inside_node.py``
as a pure, testable object with zero ROS dependencies.

The machine covers three phases:
1. **Blind entry** — advance blindly until a lumen is confirmed or the
   blind distance budget is exhausted.
2. **Normal navigation** — hysteretic switching between ROTATE_ALIGN and
   ADVANCE_Z with transition cross-fades.
3. **Safety / recovery** — BLOCKED timeout → RETREAT, MAX_DEPTH_REACHED.
"""

import time
from dataclasses import dataclass
from typing import Optional

from nasal_endoscopy_algorithms.utils.state_enums import SystemState


@dataclass
class InsideSMConfig:
    """Tunable parameters for the inside state machine."""

    # Hysteresis thresholds (pixels)
    align_dist_start_px: float = 180.0   # enter ROTATE when dist above this
    align_dist_stop_px: float = 130.0    # exit ROTATE when dist below this

    # Frame-count debounce
    min_rotate_frames: int = 2
    min_advance_frames: int = 5
    transition_frames: int = 10

    # Blocked / retreat
    blocked_timeout_sec: float = 1.5

    # Blind-entry strategy
    blind_entry_max_mm: float = 20.0
    blind_goal_confirm_frames: int = 20


class InsideStateMachine:
    """Hysteretic state machine for inside (lumen-following) navigation.

    Usage
    -----
    .. code-block:: python

        sm = InsideStateMachine(InsideSMConfig())
        for frame in camera:
            result: APFResult = vision.process_frame(frame)
            new_state = sm.evaluate(
                goal_exists=(result.goal is not None),
                pixel_dist=...,
                is_auto_run=True,
                current_depth_mm=12.3,
                max_depth_mm=50.0,
                rotation_safe=True,
            )
    """

    def __init__(self, config: InsideSMConfig = InsideSMConfig()) -> None:
        self.cfg = config
        self.current_state: SystemState = SystemState.IDLE
        self.frame_counter: int = 0

        # Blind-entry tracking
        self.blind_entry_completed: bool = False
        self.blind_goal_consecutive: int = 0

        # Blocked timer
        self.blocked_start_time: Optional[float] = None

    # ── public API ──────────────────────────────────────────

    def reset(self) -> None:
        """Full reset — state, counters, timers, blind-entry progress."""
        self.current_state = SystemState.IDLE
        self.frame_counter = 0
        self.blind_entry_completed = False
        self.blind_goal_consecutive = 0
        self.blocked_start_time = None

    def evaluate(
        self,
        goal_exists: bool,
        pixel_dist: float,
        is_auto_run: bool,
        current_depth_mm: float,
        max_depth_mm: float,
        rotation_safe: bool,
    ) -> SystemState:
        """Evaluate one frame and return the (possibly new) state.

        Parameters
        ----------
        goal_exists : bool
            Whether a valid lumen goal was detected this frame.
        pixel_dist : float
            Distance (px) from image centre to (filtered) goal.
        is_auto_run : bool
            Whether auto-navigation is engaged.
        current_depth_mm : float
            Current cumulative insertion depth.
        max_depth_mm : float
            Safety depth limit.
        rotation_safe : bool
            Whether the per-phase rotation budget has headroom.

        Returns
        -------
        SystemState
        """
        # ── 1. Pause ──
        if not is_auto_run:
            new_state = SystemState.IDLE
        else:
            new_state = self._evaluate_active(
                goal_exists, pixel_dist, current_depth_mm,
                max_depth_mm, rotation_safe,
            )

        # ── frame-counter management ──
        if new_state != self.current_state:
            self.current_state = new_state
            self.frame_counter = 0
        else:
            self.frame_counter += 1

        return self.current_state

    # ── internal ────────────────────────────────────────────

    def _evaluate_active(
        self,
        goal_exists: bool,
        pixel_dist: float,
        current_depth_mm: float,
        max_depth_mm: float,
        rotation_safe: bool,
    ) -> SystemState:
        cfg = self.cfg
        prev = self.current_state

        # ── 2. Blind-entry phase ──
        if not self.blind_entry_completed:
            return self._evaluate_blind(goal_exists)

        # ── 3. Safety ceilings ──
        if current_depth_mm >= max_depth_mm:
            return SystemState.MAX_DEPTH_REACHED

        # ── 4. Goal lost → BLOCKED / RETREAT ──
        if not goal_exists:
            if prev in (SystemState.RETREAT, SystemState.BLOCKED):
                return self._evaluate_blocked(prev)
            return self._enter_blocked()

        # ── 5. Normal hysteretic switching ──
        # Only ROTATE states care about the rotation budget
        in_rotate = prev in (SystemState.ROTATE_ALIGN,
                             SystemState.TRANSITION_TO_ROTATE)
        if in_rotate and not rotation_safe:
            # Force transition out of rotation when budget exhausted
            if prev == SystemState.ROTATE_ALIGN and \
               self.frame_counter >= cfg.min_rotate_frames:
                return SystemState.TRANSITION_TO_ADVANCE
            return prev

        return self._hysteretic_switch(prev, pixel_dist)

    def _evaluate_blind(self, goal_exists: bool) -> SystemState:
        """State evaluation within the blind-entry phase."""
        cfg = self.cfg

        if goal_exists:
            self.blind_goal_consecutive += 1
        else:
            self.blind_goal_consecutive = 0

        if self.blind_goal_consecutive >= cfg.blind_goal_confirm_frames:
            self.blind_entry_completed = True
            return SystemState.TRANSITION_TO_ROTATE

        # Blind-entry distance cap is checked *externally* (the node
        # tracks ``blind_entry_distance``).  When the cap is hit the
        # node sets ``blind_entry_completed = True`` on the SM directly.
        return SystemState.BLIND_ENTRY

    def _enter_blocked(self) -> SystemState:
        if self.blocked_start_time is None:
            self.blocked_start_time = time.time()
        return SystemState.BLOCKED

    def _evaluate_blocked(self, prev: SystemState) -> SystemState:
        cfg = self.cfg
        elapsed = time.time() - (self.blocked_start_time or time.time())
        if elapsed > cfg.blocked_timeout_sec:
            return SystemState.RETREAT
        return SystemState.BLOCKED

    def _hysteretic_switch(
        self, prev: SystemState, pixel_dist: float
    ) -> SystemState:
        cfg = self.cfg
        cnt = self.frame_counter

        # From advance / idle: go to rotate if far enough
        if prev in (SystemState.ADVANCE_Z, SystemState.IDLE):
            if (pixel_dist > cfg.align_dist_start_px
                    and cnt >= cfg.min_advance_frames):
                return SystemState.TRANSITION_TO_ROTATE
            return SystemState.ADVANCE_Z

        # From rotate: go to advance if close enough (or rotation unsafe,
        # which is handled upstream)
        if prev == SystemState.ROTATE_ALIGN:
            if (pixel_dist < cfg.align_dist_stop_px
                    and cnt >= cfg.min_rotate_frames):
                return SystemState.TRANSITION_TO_ADVANCE
            return SystemState.ROTATE_ALIGN

        # Transition states — count up, then switch
        if prev == SystemState.TRANSITION_TO_ROTATE:
            if cnt >= cfg.transition_frames:
                return SystemState.ROTATE_ALIGN
            return SystemState.TRANSITION_TO_ROTATE

        if prev == SystemState.TRANSITION_TO_ADVANCE:
            if cnt >= cfg.transition_frames:
                return SystemState.ADVANCE_Z
            return SystemState.TRANSITION_TO_ADVANCE

        # Recovery from BLOCKED / RETREAT / other
        return SystemState.TRANSITION_TO_ROTATE

    # ── manual overrides (for the node to call) ─────────────

    def force_blind_complete(self) -> None:
        """Called by the node when the blind-entry distance cap is reached."""
        self.blind_entry_completed = True

    def clear_blocked(self) -> None:
        """Called when a retreat succeeds and we have a target again."""
        self.blocked_start_time = None
