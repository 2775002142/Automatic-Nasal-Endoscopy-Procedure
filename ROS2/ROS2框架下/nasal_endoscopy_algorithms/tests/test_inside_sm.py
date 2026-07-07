"""Tests for InsideStateMachine."""

import pytest
from nasal_endoscopy_algorithms.state_machine.inside_sm import (
    InsideSMConfig,
    InsideStateMachine,
)
from nasal_endoscopy_algorithms.utils.state_enums import SystemState


class TestInsideStateMachine:
    @pytest.fixture
    def sm(self):
        return InsideStateMachine(
            InsideSMConfig(
                align_dist_start_px=180.0,
                align_dist_stop_px=130.0,
                min_rotate_frames=2,
                min_advance_frames=5,
                transition_frames=10,
                blocked_timeout_sec=1.5,
                blind_entry_max_mm=20.0,
                blind_goal_confirm_frames=20,
            )
        )

    # ── pause ──

    def test_idle_when_not_auto(self, sm):
        assert sm.evaluate(True, 200.0, is_auto_run=False,
                           current_depth_mm=10.0, max_depth_mm=50.0,
                           rotation_safe=True) == SystemState.IDLE

    # ── blind entry phase ──

    def test_blind_entry_initially(self, sm):
        """A fresh machine should start in BLIND_ENTRY if auto-running."""
        st = sm.evaluate(False, 0.0, is_auto_run=True,
                         current_depth_mm=0.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.BLIND_ENTRY

    def test_blind_goal_confirm_transitions(self, sm):
        """After enough consecutive detections, blind entry completes."""
        sm.blind_goal_consecutive = sm.cfg.blind_goal_confirm_frames
        st = sm.evaluate(True, 100.0, is_auto_run=True,
                         current_depth_mm=0.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert sm.blind_entry_completed
        assert st == SystemState.TRANSITION_TO_ROTATE

    # ── safety: max depth ──

    def test_max_depth_reached(self, sm):
        sm.blind_entry_completed = True
        st = sm.evaluate(True, 100.0, is_auto_run=True,
                         current_depth_mm=55.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.MAX_DEPTH_REACHED

    # ── goal lost → BLOCKED → RETREAT ──

    def test_goal_lost_enters_blocked(self, sm):
        sm.blind_entry_completed = True
        st = sm.evaluate(False, 0.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.BLOCKED

    # ── hysteretic switch: ADVANCE → ROTATE ──

    def test_advance_to_rotate_when_far(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.ADVANCE_Z
        sm.frame_counter = 10  # >= min_advance_frames
        st = sm.evaluate(True, 200.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.TRANSITION_TO_ROTATE

    def test_advance_stays_when_not_enough_frames(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.ADVANCE_Z
        sm.frame_counter = 1  # < min_advance_frames
        st = sm.evaluate(True, 200.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.ADVANCE_Z

    # ── hysteretic switch: ROTATE → ADVANCE ──

    def test_rotate_to_advance_when_close(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.ROTATE_ALIGN
        sm.frame_counter = 5  # >= min_rotate_frames
        st = sm.evaluate(True, 100.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.TRANSITION_TO_ADVANCE

    # ── transition counting ──

    def test_transition_to_rotate_completes(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.TRANSITION_TO_ROTATE
        sm.frame_counter = 20  # >= transition_frames
        st = sm.evaluate(True, 200.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=True)
        assert st == SystemState.ROTATE_ALIGN

    # ── rotation safety ──

    def test_rotate_transitions_when_unsafe(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.ROTATE_ALIGN
        sm.frame_counter = 5
        st = sm.evaluate(True, 200.0, is_auto_run=True,
                         current_depth_mm=10.0, max_depth_mm=50.0,
                         rotation_safe=False)
        assert st == SystemState.TRANSITION_TO_ADVANCE

    # ── reset ──

    def test_reset(self, sm):
        sm.blind_entry_completed = True
        sm.current_state = SystemState.ADVANCE_Z
        sm.frame_counter = 10
        sm.reset()
        assert sm.current_state == SystemState.IDLE
        assert sm.frame_counter == 0
        assert not sm.blind_entry_completed
        assert sm.blind_goal_consecutive == 0
        assert sm.blocked_start_time is None
