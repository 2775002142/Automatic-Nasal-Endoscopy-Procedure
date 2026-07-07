"""Tests for OutsideStateMachine."""

import pytest
from nasal_endoscopy_algorithms.state_machine.outside_sm import (
    OutsideSMConfig,
    OutsideStateMachine,
)
from nasal_endoscopy_algorithms.utils.state_enums import SystemState


class TestOutsideStateMachine:
    @pytest.fixture
    def sm(self):
        return OutsideStateMachine(
            OutsideSMConfig(
                align_tolerance_enter=20.0,
                align_tolerance_exit=10.0,
                min_align_frames=3,
                min_approach_frames=3,
                transition_frames=5,
            )
        )

    # ── basic terminal conditions ──

    def test_idle_when_not_auto(self, sm):
        assert sm.evaluate(True, 50.0, 100.0, is_auto_run=False,
                           is_finished=False, target_width_threshold=300) \
               == SystemState.IDLE

    def test_target_reached_when_finished(self, sm):
        assert sm.evaluate(True, 5.0, 100.0, is_auto_run=True,
                           is_finished=True, target_width_threshold=300) \
               == SystemState.TARGET_REACHED

    def test_target_lost_when_no_detection(self, sm):
        assert sm.evaluate(False, 0.0, 0.0, is_auto_run=True,
                           is_finished=False, target_width_threshold=300) \
               == SystemState.TARGET_LOST

    # ── hysteretic switching (IDLE / LOST → steady state) ──

    def test_idle_to_align_when_far(self, sm):
        """When starting from IDLE with large error, go to ALIGN_XY."""
        st = sm.evaluate(True, 50.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.ALIGN_XY

    def test_idle_to_approach_when_close(self, sm):
        """When starting from IDLE with small error, go directly to APPROACH_Z."""
        st = sm.evaluate(True, 5.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.APPROACH_Z

    # ── hysteresis: ALIGN → APPROACH ──

    def test_align_to_transition_after_min_frames(self, sm):
        """After min_align_frames of small error, transition to approach."""
        sm.current_state = SystemState.ALIGN_XY
        sm.frame_counter = 5  # >= min_align_frames
        st = sm.evaluate(True, 5.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.TRANSITION_TO_APPROACH

    def test_align_stays_when_not_enough_frames(self, sm):
        sm.current_state = SystemState.ALIGN_XY
        sm.frame_counter = 1  # < min_align_frames
        st = sm.evaluate(True, 5.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.ALIGN_XY

    # ── hysteresis: APPROACH → back to ALIGN ──

    def test_approach_to_transition_back_when_far(self, sm):
        sm.current_state = SystemState.APPROACH_Z
        sm.frame_counter = 5  # >= min_approach_frames
        st = sm.evaluate(True, 50.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.TRANSITION_TO_ALIGN

    # ── transition state counting ──

    def test_transition_completes_after_enough_frames(self, sm):
        sm.current_state = SystemState.TRANSITION_TO_APPROACH
        sm.frame_counter = 10  # >= transition_frames
        st = sm.evaluate(True, 5.0, 100.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.APPROACH_Z

    # ── Target reached on width ──

    def test_approach_to_reached_when_width_met(self, sm):
        sm.current_state = SystemState.APPROACH_Z
        sm.frame_counter = 5
        st = sm.evaluate(True, 5.0, 350.0, is_auto_run=True,
                         is_finished=False, target_width_threshold=300)
        assert st == SystemState.TARGET_REACHED

    # ── reset ──

    def test_reset(self, sm):
        sm.current_state = SystemState.ALIGN_XY
        sm.frame_counter = 10
        sm.reset()
        assert sm.current_state == SystemState.IDLE
        assert sm.frame_counter == 0
