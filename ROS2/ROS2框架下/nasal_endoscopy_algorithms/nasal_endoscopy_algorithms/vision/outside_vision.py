"""Outside (体外) vision: MediaPipe-based nostril detection.

Returns a ``NostrilDetectionResult`` dataclass instead of a raw tuple.
"""

import cv2
import math
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NostrilDetectionResult:
    """Structured output from a single outside-vision frame.

    Attributes
    ----------
    nose_pos : (int, int) or None
        Pixel coordinates of the target nostril centre (in the *cropped* image).
    feature_width : float
        Nostril wing-to-wing pixel width (landmark 358 → 129).
    nostril_distance_px : float
        Pixel distance between left (279) and right (49) nostril base points.
    """
    nose_pos: Optional[Tuple[int, int]]
    feature_width: float
    nostril_distance_px: float


class VisionSystem:
    """MediaPipe FaceMesh wrapper for nostril landmark detection."""

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[Vision] MediaPipe model loaded")

    # ── preprocessing ───────────────────────────────────────

    def crop_effective_area(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Remove black borders and return (cropped, (offset_x, offset_y))."""
        if image is None:
            return None, (0, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image, (0, 0)

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # Reject noise: skip crop if effective area is too small
        if w * h < (image.shape[0] * image.shape[1] * 0.05):
            return image, (0, 0)

        return image[y : y + h, x : x + w], (x, y)

    # ── detection ───────────────────────────────────────────

    def detect_nose_target(
        self,
        effective_image: np.ndarray,
        target_side: str = "center",
    ) -> NostrilDetectionResult:
        """Locate the nostril target in *effective_image*.

        Parameters
        ----------
        effective_image : np.ndarray
            BGR image (preferably already cropped).
        target_side : str
            ``'left'``, ``'right'``, or ``'center'``.

        Returns
        -------
        NostrilDetectionResult
        """
        if effective_image is None:
            return NostrilDetectionResult(None, 0.0, 0.0)

        h, w = effective_image.shape[:2]
        rgb = cv2.cvtColor(effective_image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return NostrilDetectionResult(None, 0.0, 0.0)

        landmarks = results.multi_face_landmarks[0].landmark

        # Key landmarks
        pt_left_base = landmarks[279]   # left nostril base
        pt_right_base = landmarks[49]   # right nostril base

        # Columella midpoint
        mid_x = (pt_left_base.x + pt_right_base.x) * 0.5
        mid_y = (pt_left_base.y + pt_right_base.y) * 0.5

        target_side_str = str(target_side).strip().lower()
        if target_side_str == "left":
            target_norm_x = (pt_left_base.x + mid_x) * 0.5
            target_norm_y = (pt_left_base.y + mid_y) * 0.5
        elif target_side_str == "right":
            target_norm_x = (pt_right_base.x + mid_x) * 0.5
            target_norm_y = (pt_right_base.y + mid_y) * 0.5
        else:
            target_norm_x = mid_x
            target_norm_y = mid_y

        tx = int(target_norm_x * w)
        ty = int(target_norm_y * h)

        # Nostril reference distance
        nostril_px = math.hypot(
            (pt_left_base.x - pt_right_base.x) * w,
            (pt_left_base.y - pt_right_base.y) * h,
        )

        # Full wing width (for arrival threshold)
        left_wing = landmarks[358]
        right_wing = landmarks[129]
        feature_width = math.hypot(
            (left_wing.x - right_wing.x) * w,
            (left_wing.y - right_wing.y) * h,
        )

        return NostrilDetectionResult(
            nose_pos=(tx, ty) if tx >= 0 and ty >= 0 else None,
            feature_width=float(feature_width),
            nostril_distance_px=float(nostril_px),
        )

    # ── cleanup ─────────────────────────────────────────────

    def release(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, "mp_face_mesh"):
            self.mp_face_mesh.close()
            print("[Vision] MediaPipe resources released")
