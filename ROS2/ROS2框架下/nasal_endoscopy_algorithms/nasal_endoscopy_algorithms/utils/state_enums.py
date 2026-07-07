"""Shared state-machine enumerations.

These are the *only* enums shared between outside and inside navigation.
They live here (zero-ROS package) so every layer can import them without
pulling in rclpy or robot-specific logic.
"""

from enum import Enum, auto


class SystemState(Enum):
    """Master state enum — covers outside, inside, and shared states."""

    # ── shared ──────────────────────────────────────────────
    IDLE = auto()           # idle / paused
    SYSTEM_ERROR = auto()   # unrecoverable error
    TERMINATED = auto()     # program exit

    # ── outside (体外) ──────────────────────────────────────
    ALIGN_XY = auto()               # lateral alignment
    APPROACH_Z = auto()             # depth-wise approach
    TRANSITION_TO_APPROACH = auto() # fade from ALIGN → APPROACH
    TRANSITION_TO_ALIGN = auto()    # fade from APPROACH → ALIGN
    TARGET_LOST = auto()            # no detection
    RETREAT = auto()                # back-off on prolonged loss
    TARGET_REACHED = auto()         # feature-width threshold met

    # ── inside (体内) ───────────────────────────────────────
    BLIND_ENTRY = auto()            # blind insertion phase
    ROTATE_ALIGN = auto()           # rotational alignment via APF
    ADVANCE_Z = auto()              # pure forward advance
    TRANSITION_TO_ADVANCE = auto()  # fade ROTATE → ADVANCE
    TRANSITION_TO_ROTATE = auto()   # fade ADVANCE → ROTATE
    BLOCKED = auto()                # temporary obstruction
    MAX_DEPTH_REACHED = auto()      # safety depth limit hit
