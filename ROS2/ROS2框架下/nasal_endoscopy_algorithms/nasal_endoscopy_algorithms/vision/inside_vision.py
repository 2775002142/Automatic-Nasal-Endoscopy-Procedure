"""Inside (体内) vision: Artificial Potential Field (APF) based lumen detection.

Returns an ``APFResult`` dataclass instead of a raw tuple.
"""

import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from nasal_endoscopy_algorithms.utils.geometry import clamp


@dataclass
class APFResult:
    """Structured output from a single inside-vision frame.

    Attributes
    ----------
    vis_image : np.ndarray or None
        Annotated image for display (may be *None* if input was None).
    force_vector : np.ndarray
        Net force vector (2,) in pixel units.
    goal : (int, int) or None
        Pixel coordinates of the target dark region centre.
    """
    vis_image: Optional[np.ndarray]
    force_vector: np.ndarray
    goal: Optional[Tuple[int, int]]


class APFVisionSystem:
    """Artificial Potential Field visual processor for lumen-following."""

    def __init__(self, debug: bool = False) -> None:
        self.K_ATT: float = 5.0
        self.K_REP: float = 3.0
        self.REP_RANGE: float = 80.0
        self.MIN_AREA_RATIO: float = 0.005

        self.BASE_DARK_PERCENTILE: int = 8
        self.BASE_BRIGHT_PERCENTILE: int = 80

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.debug: bool = debug

        # Per-frame spatial references (updated on each process_frame call)
        self.center: Tuple[int, int] = (0, 0)
        self.crop_offset: Tuple[int, int] = (0, 0)
        self.width: int = 0
        self.height: int = 0

    # ── public API ──────────────────────────────────────────

    def process_frame(self, image: np.ndarray) -> APFResult:
        """Run the full APF pipeline on one BGR image.

        Returns
        -------
        APFResult
        """
        if image is None:
            return APFResult(None, np.array([0.0, 0.0]), None)

        image_cropped, self.crop_offset = self._crop_black_borders(image)
        self.height, self.width = image_cropped.shape[:2]
        if self.width == 0 or self.height == 0:
            return APFResult(image, np.array([0.0, 0.0]), None)

        self.center = (self.width // 2, self.height // 2)

        gray = self._preprocess_image(image_cropped)
        dark_regions, bright_regions = self._find_regions(gray)
        goal = self._select_goal(dark_regions)

        if goal:
            att_force = self._calculate_attractive_force(goal)
            rep_force = self._calculate_repulsive_force(bright_regions)
            total_force = att_force + rep_force
        else:
            total_force = np.array([0.0, 0.0])

        vis = self._visualize_result(
            image, total_force, goal, dark_regions, bright_regions
        )
        return APFResult(vis_image=vis, force_vector=total_force, goal=goal)

    # ── internal helpers (public for node-side recombination) ─

    def _crop_black_borders(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image, (0, 0)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        if cv2.contourArea(max_contour) / (image.shape[0] * image.shape[1]) < 0.1:
            return image, (0, 0)
        return image[y : y + h, x : x + w], (x, y)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (7, 7), 0)

    def _find_regions(
        self, gray_img: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        dark_thresh = np.percentile(gray_img, self.BASE_DARK_PERCENTILE)
        bright_thresh = np.percentile(gray_img, self.BASE_BRIGHT_PERCENTILE)

        dark_mask = cv2.inRange(gray_img, 0, int(dark_thresh))
        bright_mask = cv2.inRange(gray_img, int(bright_thresh), 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)

        min_area = self.MIN_AREA_RATIO * self.height * self.width
        d_cnts, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        b_cnts, _ = cv2.findContours(
            bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return (
            [c for c in d_cnts if cv2.contourArea(c) > min_area],
            [c for c in b_cnts if cv2.contourArea(c) > min_area],
        )

    def _select_goal(
        self, regions: List[np.ndarray]
    ) -> Optional[Tuple[int, int]]:
        if not regions:
            return None
        best_goal, max_score = None, -1.0
        for region in regions:
            M = cv2.moments(region)
            if M["m00"] == 0:
                continue
            area = M["m00"]
            cx = int(M["m10"] / area)
            cy = int(M["m01"] / area)
            # Score: prefer lower (deeper) and larger regions
            score = (cy / self.height * 0.2) + (
                area / (self.width * self.height) * 0.8
            )
            if score > max_score:
                max_score, best_goal = score, (cx, cy)
        return best_goal

    def _calculate_attractive_force(
        self, goal: Tuple[int, int]
    ) -> np.ndarray:
        dx = goal[0] - self.center[0]
        dy = goal[1] - self.center[1]
        return self.K_ATT * np.array([dx, dy], dtype=np.float64)

    def _calculate_repulsive_force(
        self, obstacles: List[np.ndarray]
    ) -> np.ndarray:
        total_rep = np.array([0.0, 0.0], dtype=np.float64)
        center_pt = (float(self.center[0]), float(self.center[1]))

        for obs in obstacles:
            if cv2.pointPolygonTest(obs, center_pt, False) > 0:
                M = cv2.moments(obs)
                if M["m00"] == 0:
                    continue
                ox = int(M["m10"] / M["m00"])
                oy = int(M["m01"] / M["m00"])
                vec = np.array(
                    [self.center[0] - ox, self.center[1] - oy],
                    dtype=np.float64,
                )
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    total_rep += self.K_REP * 100.0 * (vec / norm)
                continue

            # Closest boundary point
            dist_sq_min = float("inf")
            closest_pt = None
            for pt_wrapper in obs:
                pt = pt_wrapper[0]
                d_sq = (self.center[0] - pt[0]) ** 2 + (
                    self.center[1] - pt[1]
                ) ** 2
                if d_sq < dist_sq_min:
                    dist_sq_min, closest_pt = d_sq, pt

            if closest_pt is None:
                continue
            dist = math.sqrt(dist_sq_min)
            if 0 < dist < self.REP_RANGE:
                vec = np.array(
                    [
                        self.center[0] - closest_pt[0],
                        self.center[1] - closest_pt[1],
                    ],
                    dtype=np.float64,
                )
                mag = (
                    self.K_REP
                    * ((1.0 / dist) - (1.0 / self.REP_RANGE))
                    * (1.0 / dist ** 2)
                )
                total_rep += mag * (vec / dist)
        return total_rep

    def _visualize_result(
        self,
        original_img: np.ndarray,
        force: np.ndarray,
        goal: Optional[Tuple[int, int]],
        darks: List[np.ndarray],
        brights: List[np.ndarray],
        filtered_goal: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        vis = original_img.copy()
        ox, oy = self.crop_offset
        if self.width == 0 or self.height == 0:
            return vis

        cx, cy = self.center[0] + ox, self.center[1] + oy

        # Centre crosshair (yellow)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        # Raw goal (orange hollow)
        if goal is not None:
            gx = goal[0] + ox
            gy = goal[1] + oy
            cv2.circle(vis, (gx, gy), 8, (0, 165, 255), 2)

        # Filtered goal (green filled)
        if filtered_goal is not None:
            fgx = int(filtered_goal[0] + ox)
            fgy = int(filtered_goal[1] + oy)
            cv2.circle(vis, (fgx, fgy), 7, (0, 255, 0), -1)

        # Dark / bright contours
        darks_abs = [c + (ox, oy) for c in darks]
        brights_abs = [c + (ox, oy) for c in brights]
        cv2.drawContours(vis, darks_abs, -1, (255, 0, 0), 1)
        cv2.drawContours(vis, brights_abs, -1, (0, 0, 255), 1)

        # Force vector arrow
        f_norm = float(np.linalg.norm(force))
        if f_norm > 1e-4:
            unit = force / f_norm
            end = (int(cx + unit[0] * 50), int(cy + unit[1] * 50))
            cv2.arrowedLine(vis, (cx, cy), end, (0, 255, 0), 2)

        return vis
