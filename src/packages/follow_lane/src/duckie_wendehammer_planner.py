#!/usr/bin/env python3

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


STATE_SEARCH = "SEARCH"
STATE_LANE_FOLLOW = "LANE_FOLLOW"
STATE_TRACK_TARGET = "TRACK_TARGET"
STATE_PASS_GAP = "PASS_GAP"
STATE_STOP = "STOP"
STATE_RECOVERY_STOP = "RECOVERY_STOP"
STATE_RECOVERY_REVERSE = "RECOVERY_REVERSE"
STATE_RECOVERY_TURN = "RECOVERY_TURN"
STATE_RECOVERY_SETTLE = "RECOVERY_SETTLE"

SPEED_STOP = "stop"
SPEED_SEARCH = "search"
SPEED_CRAWL = "crawl"
SPEED_SLOW = "slow"
SPEED_REVERSE = "reverse"


def clamp(value, low, high):
    return max(low, min(high, value))


def parameter_value(parameters, group, name, fallback=None):
    try:
        value = parameters[group][name]
    except (KeyError, TypeError):
        return fallback
    if isinstance(value, dict):
        return value.get("default", fallback)
    return value


def numeric_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def make_color_mask(hsv, parameters, group):
    low = (
        int(parameter_value(parameters, group, "hl", 0)),
        int(parameter_value(parameters, group, "sl", 0)),
        int(parameter_value(parameters, group, "vl", 0)),
    )
    high = (
        int(parameter_value(parameters, group, "hh", 255)),
        int(parameter_value(parameters, group, "sh", 255)),
        int(parameter_value(parameters, group, "vh", 255)),
    )
    mask = cv2.inRange(hsv, low, high)
    kernel_size = int(parameter_value(parameters, "planner", "mask_kernel_size", 5))
    kernel_size = max(1, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


@dataclass
class PlannerSettings:
    size: int
    slice_near_y: float
    slice_far_y: float
    slice_count: int
    line_window_half_height: int
    min_line_points: int
    yellow_right_percentile: float
    white_left_percentile: float
    yellow_fallback_fraction: float
    white_fallback_fraction: float
    line_guard_px: float
    corridor_margin_px: float
    min_gap_px: float
    min_box_height_px: float
    min_box_area_px: float
    min_box_conf: float
    max_box_age_sec: float
    fresh_box_age_sec: float
    duckie_margin_x_px: float
    duckie_margin_y_px: float
    stale_margin_gain_px: float
    min_duckie_y: float
    max_duckie_y: float
    require_duckie_for_target: bool
    allow_virtual_corridor_edges: bool
    target_smoothing: float
    target_reached_y: float
    target_reached_error_px: float
    target_centered_frames: int
    target_complete_search_frames: int
    gap_pass_speed: float
    gap_pass_frames: int
    crawl_speed: float
    max_speed: float
    search_omega: float
    omega_gain: float
    max_omega: float
    turn_sign_invert: bool
    gap_activation_min_box_height_px: float
    gap_activation_min_box_area_px: float
    gap_activation_frames: int
    gap_release_frames: int
    lane_follow_speed: float
    lane_fallback_speed: float
    lane_omega_gain: float
    lane_max_omega: float
    lane_turn_slowdown: float
    lane_lookahead_y: float
    lane_boundary_clearance_px: float
    lane_boundary_avoid_omega: float
    lane_emergency_min_y: float
    lane_min_turn_speed: float
    recovery_stop_seconds: float
    recovery_no_gap_confirm_frames: int
    recovery_reverse_speed: float
    recovery_reverse_seconds: float
    recovery_turn_omega: float
    recovery_turn_seconds: float
    recovery_settle_seconds: float

    @classmethod
    def from_parameters(cls, parameters):
        size = int(parameter_value(parameters, "birdseye", "size", 400))
        return cls(
            size=size,
            slice_near_y=float(parameter_value(parameters, "planner", "slice_near_y", 360.0)),
            slice_far_y=float(parameter_value(parameters, "planner", "slice_far_y", 190.0)),
            slice_count=int(parameter_value(parameters, "planner", "slice_count", 5)),
            line_window_half_height=int(parameter_value(parameters, "planner", "line_window_half_height", 24)),
            min_line_points=int(parameter_value(parameters, "planner", "min_line_points", 12)),
            yellow_right_percentile=float(parameter_value(parameters, "planner", "yellow_right_percentile", 75.0)),
            white_left_percentile=float(parameter_value(parameters, "planner", "white_left_percentile", 25.0)),
            yellow_fallback_fraction=float(parameter_value(parameters, "planner", "yellow_fallback_fraction", 0.08)),
            white_fallback_fraction=float(parameter_value(parameters, "planner", "white_fallback_fraction", 0.92)),
            line_guard_px=float(parameter_value(parameters, "planner", "line_guard_px", 10.0)),
            corridor_margin_px=float(parameter_value(parameters, "planner", "corridor_margin_px", 12.0)),
            min_gap_px=float(parameter_value(parameters, "safety", "min_gap_px", 105.0)),
            min_box_height_px=float(parameter_value(parameters, "safety", "min_box_height_px", 18.0)),
            min_box_area_px=float(parameter_value(parameters, "safety", "min_box_area_px", 420.0)),
            min_box_conf=float(parameter_value(parameters, "model", "conf", 0.25)),
            max_box_age_sec=float(parameter_value(parameters, "safety", "max_box_age_sec", 0.75)),
            fresh_box_age_sec=float(parameter_value(parameters, "safety", "fresh_box_age_sec", 0.25)),
            duckie_margin_x_px=float(parameter_value(parameters, "safety", "duckie_margin_x_px", 22.0)),
            duckie_margin_y_px=float(parameter_value(parameters, "safety", "duckie_margin_y_px", 44.0)),
            stale_margin_gain_px=float(parameter_value(parameters, "safety", "stale_margin_gain_px", 28.0)),
            min_duckie_y=float(parameter_value(parameters, "safety", "min_duckie_y", -40.0)),
            max_duckie_y=float(parameter_value(parameters, "safety", "max_duckie_y", size + 70.0)),
            require_duckie_for_target=numeric_bool(parameter_value(parameters, "planner", "require_duckie_for_target", 1)),
            allow_virtual_corridor_edges=numeric_bool(
                parameter_value(parameters, "planner", "allow_virtual_corridor_edges", 1)
            ),
            target_smoothing=float(parameter_value(parameters, "control", "target_smoothing", 0.35)),
            target_reached_y=float(parameter_value(parameters, "control", "target_reached_y", size * 0.88)),
            target_reached_error_px=float(parameter_value(parameters, "control", "target_reached_error_px", 22.0)),
            target_centered_frames=int(parameter_value(parameters, "control", "target_centered_frames", 4)),
            target_complete_search_frames=int(parameter_value(parameters, "control", "target_complete_search_frames", 3)),
            gap_pass_speed=float(parameter_value(parameters, "control", "gap_pass_speed", 0.11)),
            gap_pass_frames=int(parameter_value(parameters, "control", "gap_pass_frames", 12)),
            crawl_speed=float(parameter_value(parameters, "control", "crawl_speed", 0.025)),
            max_speed=float(parameter_value(parameters, "control", "max_speed", 0.045)),
            search_omega=float(parameter_value(parameters, "control", "search_omega", 0.32)),
            omega_gain=float(parameter_value(parameters, "control", "omega_gain", 1.15)),
            max_omega=float(parameter_value(parameters, "control", "max_omega", 1.15)),
            turn_sign_invert=numeric_bool(parameter_value(parameters, "control", "turn_sign_invert", 0)),
            gap_activation_min_box_height_px=float(parameter_value(parameters, "behavior", "gap_activation_min_box_height_px", 40.0)),
            gap_activation_min_box_area_px=float(parameter_value(parameters, "behavior", "gap_activation_min_box_area_px", 1500.0)),
            gap_activation_frames=int(parameter_value(parameters, "behavior", "gap_activation_frames", 3)),
            gap_release_frames=int(parameter_value(parameters, "behavior", "gap_release_frames", 18)),
            lane_follow_speed=float(parameter_value(parameters, "behavior", "lane_follow_speed", 0.12)),
            lane_fallback_speed=float(parameter_value(parameters, "behavior", "lane_fallback_speed", 0.09)),
            lane_omega_gain=float(parameter_value(parameters, "behavior", "lane_omega_gain", 1.15)),
            lane_max_omega=float(parameter_value(parameters, "behavior", "lane_max_omega", 1.0)),
            lane_turn_slowdown=float(parameter_value(parameters, "behavior", "lane_turn_slowdown", 0.45)),
            lane_lookahead_y=float(parameter_value(parameters, "lane", "lookahead_y", 300.0)),
            lane_boundary_clearance_px=float(
                parameter_value(parameters, "lane", "boundary_clearance_px", 55.0)
            ),
            lane_boundary_avoid_omega=float(
                parameter_value(parameters, "lane", "boundary_avoid_omega", 0.90)
            ),
            lane_emergency_min_y=float(
                parameter_value(parameters, "lane", "emergency_min_y", 260.0)
            ),
            lane_min_turn_speed=float(parameter_value(parameters, "lane", "min_turn_speed", 0.10)),
            recovery_stop_seconds=float(parameter_value(parameters, "recovery", "stop_seconds", 0.25)),
            recovery_no_gap_confirm_frames=int(
                parameter_value(parameters, "recovery", "no_gap_confirm_frames", 6)
            ),
            recovery_reverse_speed=float(parameter_value(parameters, "recovery", "reverse_speed", 0.10)),
            recovery_reverse_seconds=float(parameter_value(parameters, "recovery", "reverse_seconds", 0.45)),
            recovery_turn_omega=float(parameter_value(parameters, "recovery", "turn_omega", 0.75)),
            recovery_turn_seconds=float(parameter_value(parameters, "recovery", "turn_seconds", 0.75)),
            recovery_settle_seconds=float(parameter_value(parameters, "recovery", "settle_seconds", 0.25)),
        )

    def slice_y_values(self):
        count = max(1, self.slice_count)
        return np.linspace(self.slice_near_y, self.slice_far_y, count)


@dataclass
class RawDuckieDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    stamp_sec: float

    @property
    def width(self):
        return max(0.0, float(self.x2) - float(self.x1))

    @property
    def height(self):
        return max(0.0, float(self.y2) - float(self.y1))

    @property
    def area(self):
        return self.width * self.height

    def to_json(self):
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "confidence": float(self.conf),
            "height_px": float(self.height),
            "area_px": float(self.area),
        }


@dataclass
class DuckieObstacle:
    index: int
    x_left: float
    x_right: float
    y: float
    y_radius: float
    age: float
    state: str
    conf: float
    raw_box: Tuple[float, float, float, float]
    rejected: bool = False
    reject_reason: str = ""

    @property
    def center_x(self):
        return (self.x_left + self.x_right) * 0.5

    @property
    def width(self):
        return self.x_right - self.x_left

    def active_at_y(self, y):
        return abs(float(y) - self.y) <= self.y_radius

    def interval_at_y(self, y, left_limit, right_limit):
        if not self.active_at_y(y):
            return None
        left = clamp(self.x_left, left_limit, right_limit)
        right = clamp(self.x_right, left_limit, right_limit)
        if right <= left:
            return None
        return left, right

    def to_json(self):
        return {
            "index": int(self.index),
            "x_left": float(self.x_left),
            "x_right": float(self.x_right),
            "x": float(self.center_x),
            "y": float(self.y),
            "width_px": float(self.width),
            "y_radius_px": float(self.y_radius),
            "age_sec": float(self.age),
            "state": self.state,
            "confidence": float(self.conf),
            "raw_box": {
                "x1": float(self.raw_box[0]),
                "y1": float(self.raw_box[1]),
                "x2": float(self.raw_box[2]),
                "y2": float(self.raw_box[3]),
            },
            "rejected": bool(self.rejected),
            "reject_reason": self.reject_reason,
        }


@dataclass
class LaneBoundary:
    y: float
    yellow_x: float
    white_x: float
    yellow_found: bool
    white_found: bool
    corridor_left: float
    corridor_right: float
    valid: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def corridor_width(self):
        return max(0.0, self.corridor_right - self.corridor_left)

    def to_json(self):
        return {
            "y": float(self.y),
            "yellow_x": float(self.yellow_x),
            "white_x": float(self.white_x),
            "yellow_found": bool(self.yellow_found),
            "white_found": bool(self.white_found),
            "corridor_left": float(self.corridor_left),
            "corridor_right": float(self.corridor_right),
            "corridor_width": float(self.corridor_width),
            "valid": bool(self.valid),
            "warnings": list(self.warnings),
        }


@dataclass
class GapCandidate:
    y: float
    left_x: float
    right_x: float
    center_x: float
    width: float
    left_source: str
    right_source: str
    valid: bool
    reason: str
    score: float

    @property
    def has_duckie_edge(self):
        return self.left_source.startswith("duckie") or self.right_source.startswith("duckie")

    def to_json(self):
        return {
            "y": float(self.y),
            "left_x": float(self.left_x),
            "right_x": float(self.right_x),
            "center_x": float(self.center_x),
            "width_px": float(self.width),
            "left_source": self.left_source,
            "right_source": self.right_source,
            "valid": bool(self.valid),
            "reason": self.reason,
            "score": float(self.score),
        }


@dataclass
class PlannerResult:
    state: str
    reason: str
    speed_level: str
    has_target: bool
    target_x: Optional[float]
    target_y: Optional[float]
    target_error_px: float
    v: float
    omega: float
    chosen_gap: Optional[GapCandidate]
    candidate_gaps: List[GapCandidate]
    duckies: List[DuckieObstacle]
    rejected_duckies: List[DuckieObstacle]
    lane_boundaries: List[LaneBoundary]
    warnings: List[str] = field(default_factory=list)

    @property
    def chosen_gap_width_px(self):
        return None if self.chosen_gap is None else float(self.chosen_gap.width)

    def with_state(self, state, reason, speed_level=None, v=None, omega=None, has_target=None):
        return replace(
            self,
            state=state,
            reason=reason,
            speed_level=self.speed_level if speed_level is None else speed_level,
            v=self.v if v is None else float(v),
            omega=self.omega if omega is None else float(omega),
            has_target=self.has_target if has_target is None else bool(has_target),
        )

    def status_payload(self, include_candidates=True):
        target = None
        if self.target_x is not None and self.target_y is not None:
            target = {
                "x": float(self.target_x),
                "y": float(self.target_y),
                "error_px": float(self.target_error_px),
            }
        return {
            "state": self.state,
            "reason": self.reason,
            "speed_level": self.speed_level,
            "has_target": bool(self.has_target),
            "target": target,
            "chosen_gap_width_px": self.chosen_gap_width_px,
            "chosen_gap": None if self.chosen_gap is None else self.chosen_gap.to_json(),
            "candidate_gaps": [gap.to_json() for gap in self.candidate_gaps] if include_candidates else [],
            "duckies": [duckie.to_json() for duckie in self.duckies],
            "rejected_duckies": [duckie.to_json() for duckie in self.rejected_duckies],
            "lane_boundaries": [boundary.to_json() for boundary in self.lane_boundaries],
            "v": float(self.v),
            "omega": float(self.omega),
            "warnings": list(self.warnings),
        }


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: Optional[np.ndarray]
    image_size: Optional[Tuple[int, int]]
    rms_error: Optional[float]
    path: str

    def usable_for(self, image):
        if self.image_size is None:
            return True
        height, width = image.shape[:2]
        return (int(width), int(height)) == self.image_size


def load_camera_calibration(path):
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    with open(calibration_path, "r") as handle:
        payload = json.load(handle)
    camera_matrix = np.array(payload["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(payload["dist_coeffs"], dtype=np.float32)
    new_camera_matrix = payload.get("new_camera_matrix")
    if new_camera_matrix is not None:
        new_camera_matrix = np.array(new_camera_matrix, dtype=np.float32)
    image_size = None
    if payload.get("image_width") is not None and payload.get("image_height") is not None:
        image_size = (int(payload["image_width"]), int(payload["image_height"]))
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        new_camera_matrix=new_camera_matrix,
        image_size=image_size,
        rms_error=payload.get("rms_error"),
        path=str(calibration_path),
    )


def save_camera_calibration(path, image_size, camera_matrix, dist_coeffs, new_camera_matrix, roi, rms_error, board_cols, board_rows, square_size):
    calibration_path = Path(path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "board_cols": int(board_cols),
        "board_rows": int(board_rows),
        "square_size": float(square_size),
        "rms_error": float(rms_error),
        "camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "new_camera_matrix": np.asarray(new_camera_matrix, dtype=float).tolist(),
        "roi": [int(value) for value in roi],
    }
    with open(calibration_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


class WendehammerGapPlanner:
    def __init__(self, parameters):
        self.update(parameters)

    def update(self, parameters):
        self.parameters = parameters
        self.settings = PlannerSettings.from_parameters(parameters)

    def perspective(self):
        p = self.parameters
        size = self.settings.size
        src = np.float32(
            [
                [
                    parameter_value(p, "birdseye", "top_left_x", 159),
                    parameter_value(p, "birdseye", "top_left_y", 218),
                ],
                [
                    parameter_value(p, "birdseye", "top_right_x", 441),
                    parameter_value(p, "birdseye", "top_right_y", 218),
                ],
                [
                    parameter_value(p, "birdseye", "bottom_right_x", -29),
                    parameter_value(p, "birdseye", "bottom_right_y", 382),
                ],
                [
                    parameter_value(p, "birdseye", "bottom_left_x", 606),
                    parameter_value(p, "birdseye", "bottom_left_y", 382),
                ],
            ]
        )
        dst = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
        return cv2.getPerspectiveTransform(src, dst), size

    def crop_birdseye(self, image, matrix):
        return cv2.warpPerspective(image, matrix, (self.settings.size, self.settings.size))

    def masks(self, birdseye):
        hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)
        return make_color_mask(hsv, self.parameters, "white"), make_color_mask(hsv, self.parameters, "yellow")

    def suppress_detection_regions(self, white_mask, yellow_mask, raw_duckies, matrix):
        """Keep yellow duck bodies and their white tags out of the lane masks."""
        cleaned_white = white_mask.copy()
        cleaned_yellow = yellow_mask.copy()
        min_conf = self.settings.min_box_conf
        for raw in raw_duckies:
            if raw.conf < min_conf:
                continue
            corners = np.float32(
                [[(raw.x1, raw.y1), (raw.x2, raw.y1), (raw.x2, raw.y2), (raw.x1, raw.y2)]]
            )
            polygon = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
            polygon = np.rint(polygon).astype(np.int32)
            cv2.fillConvexPoly(cleaned_white, polygon, 0)
            cv2.fillConvexPoly(cleaned_yellow, polygon, 0)
        return cleaned_white, cleaned_yellow

    def plan_image(self, image, raw_duckies, now_sec, previous_target_x=None, gap_mode_active=True):
        matrix, _ = self.perspective()
        birdseye = self.crop_birdseye(image, matrix)
        white_mask, yellow_mask = self.masks(birdseye)
        white_mask, yellow_mask = self.suppress_detection_regions(
            white_mask, yellow_mask, raw_duckies, matrix
        )
        result = self.plan_birdseye(
            white_mask,
            yellow_mask,
            raw_duckies,
            matrix,
            now_sec,
            previous_target_x=previous_target_x,
            gap_mode_active=gap_mode_active,
        )
        return result, birdseye, white_mask, yellow_mask, matrix

    def empty_result(self, state=STATE_SEARCH, reason="no_image"):
        speed_level = SPEED_STOP if state == STATE_STOP else SPEED_SEARCH
        omega = 0.0 if state == STATE_STOP else self.search_omega()
        return PlannerResult(
            state=state,
            reason=reason,
            speed_level=speed_level,
            has_target=False,
            target_x=None,
            target_y=None,
            target_error_px=0.0,
            v=0.0,
            omega=omega,
            chosen_gap=None,
            candidate_gaps=[],
            duckies=[],
            rejected_duckies=[],
            lane_boundaries=[],
            warnings=[],
        )

    def search_omega(self):
        sign = -1.0 if self.settings.turn_sign_invert else 1.0
        return sign * self.settings.search_omega

    def find_mask_x(self, mask, y, percentile):
        s = self.settings
        height, _ = mask.shape[:2]
        y0 = int(clamp(y - s.line_window_half_height, 0, height - 1))
        y1 = int(clamp(y + s.line_window_half_height, y0 + 1, height))
        roi = mask[y0:y1, :]
        _, xs = np.where(roi > 0)
        if len(xs) < s.min_line_points:
            return None, False
        return float(np.percentile(xs, percentile)), True

    def build_lane_boundary(self, white_mask, yellow_mask, y):
        s = self.settings
        fallback_yellow = s.size * s.yellow_fallback_fraction
        fallback_white = s.size * s.white_fallback_fraction
        warnings = []
        yellow_x, yellow_found = self.find_mask_x(yellow_mask, y, s.yellow_right_percentile)
        white_x, white_found = self.find_mask_x(white_mask, y, s.white_left_percentile)
        if yellow_x is None:
            yellow_x = fallback_yellow
            warnings.append("yellow_fallback")
        if white_x is None:
            white_x = fallback_white
            warnings.append("white_fallback")

        corridor_left = clamp(yellow_x + s.line_guard_px, s.corridor_margin_px, s.size - s.corridor_margin_px)
        corridor_right = clamp(white_x - s.line_guard_px, s.corridor_margin_px, s.size - s.corridor_margin_px)
        valid = corridor_right - corridor_left >= max(s.min_gap_px * 0.6, 10.0)
        if not valid:
            warnings.append("corridor_too_narrow")
        return LaneBoundary(
            y=float(y),
            yellow_x=float(yellow_x),
            white_x=float(white_x),
            yellow_found=bool(yellow_found),
            white_found=bool(white_found),
            corridor_left=float(corridor_left),
            corridor_right=float(corridor_right),
            valid=bool(valid),
            warnings=warnings,
        )

    def build_lane_boundaries(self, white_mask, yellow_mask):
        return [
            self.build_lane_boundary(white_mask, yellow_mask, y)
            for y in self.settings.slice_y_values()
        ]

    def estimate_line_x(self, boundaries, x_attribute, found_attribute, y):
        """Interpolate/extrapolate a detected lane line to another BEV row."""
        points = sorted(
            (
                (float(boundary.y), float(getattr(boundary, x_attribute)))
                for boundary in boundaries
                if getattr(boundary, found_attribute)
            ),
            key=lambda item: item[0],
        )
        if not points:
            return None
        if len(points) == 1:
            return points[0][1]
        if y <= points[0][0]:
            first, second = points[0], points[1]
        elif y >= points[-1][0]:
            first, second = points[-2], points[-1]
        else:
            first, second = points[0], points[1]
            for lower, upper in zip(points, points[1:]):
                if lower[0] <= y <= upper[0]:
                    first, second = lower, upper
                    break
        dy = second[0] - first[0]
        if abs(dy) < 1e-6:
            return first[1]
        progress = (float(y) - first[0]) / dy
        return clamp(first[1] + progress * (second[1] - first[1]), 0.0, self.settings.size - 1.0)

    def infer_dynamic_boundary_lines(self, boundary, reference_boundaries):
        """Use visible line geometry when a dynamic duckie row misses the pixels."""
        s = self.settings
        warnings = list(boundary.warnings)
        yellow_x = boundary.yellow_x
        white_x = boundary.white_x
        yellow_found = boundary.yellow_found
        white_found = boundary.white_found

        if not yellow_found:
            estimate = self.estimate_line_x(reference_boundaries, "yellow_x", "yellow_found", boundary.y)
            if estimate is not None:
                yellow_x = estimate
                yellow_found = True
                warnings = [warning for warning in warnings if warning != "yellow_fallback"]
                warnings.append("yellow_inferred")
        if not white_found:
            estimate = self.estimate_line_x(reference_boundaries, "white_x", "white_found", boundary.y)
            if estimate is not None:
                white_x = estimate
                white_found = True
                warnings = [warning for warning in warnings if warning != "white_fallback"]
                warnings.append("white_inferred")

        corridor_left = clamp(yellow_x + s.line_guard_px, s.corridor_margin_px, s.size - s.corridor_margin_px)
        corridor_right = clamp(white_x - s.line_guard_px, s.corridor_margin_px, s.size - s.corridor_margin_px)
        valid = corridor_right - corridor_left >= max(s.min_gap_px * 0.6, 10.0)
        warnings = [warning for warning in warnings if warning != "corridor_too_narrow"]
        if not valid:
            warnings.append("corridor_too_narrow")
        return replace(
            boundary,
            yellow_x=float(yellow_x),
            white_x=float(white_x),
            yellow_found=bool(yellow_found),
            white_found=bool(white_found),
            corridor_left=float(corridor_left),
            corridor_right=float(corridor_right),
            valid=bool(valid),
            warnings=warnings,
        )

    def add_duckie_boundaries(self, white_mask, yellow_mask, lane_boundaries, obstacles):
        """Add gap cross-sections where obstacles actually are in bird's-eye space.

        Fixed planner slices are useful for lane following, but distant detections can
        project above all of them.  Without these rows the obstacle never intersects
        a slice and every otherwise open gap is incorrectly rejected as having no
        duckie edge.
        """
        s = self.settings
        boundaries = list(lane_boundaries)
        existing_y = [boundary.y for boundary in boundaries]
        min_separation = max(3.0, s.line_window_half_height * 0.25)
        for obstacle in obstacles:
            y = clamp(obstacle.y, 0.0, s.size - 1.0)
            if any(abs(y - value) < min_separation for value in existing_y):
                continue
            boundary = self.build_lane_boundary(white_mask, yellow_mask, y)
            boundary = self.infer_dynamic_boundary_lines(boundary, lane_boundaries)
            boundaries.append(boundary)
            existing_y.append(y)
        return boundaries

    def rejection_obstacle(self, index, raw, reason, now_sec):
        age = max(0.0, now_sec - float(raw.stamp_sec))
        return DuckieObstacle(
            index=index,
            x_left=float(raw.x1),
            x_right=float(raw.x2),
            y=float(raw.y2),
            y_radius=0.0,
            age=float(age),
            state="rejected",
            conf=float(raw.conf),
            raw_box=(float(raw.x1), float(raw.y1), float(raw.x2), float(raw.y2)),
            rejected=True,
            reject_reason=reason,
        )

    def project_detection_to_obstacle(self, index, raw, matrix, now_sec):
        s = self.settings
        age = max(0.0, now_sec - float(raw.stamp_sec))
        if raw.conf < s.min_box_conf:
            return None, self.rejection_obstacle(index, raw, "low_confidence", now_sec)
        if raw.height < s.min_box_height_px:
            return None, self.rejection_obstacle(index, raw, "box_too_short", now_sec)
        if raw.area < s.min_box_area_px:
            return None, self.rejection_obstacle(index, raw, "box_too_small", now_sec)
        if age > s.max_box_age_sec:
            return None, self.rejection_obstacle(index, raw, "box_too_old", now_sec)

        points = np.float32(
            [
                [(raw.x1, raw.y2)],
                [(raw.x2, raw.y2)],
                [((raw.x1 + raw.x2) * 0.5, raw.y2)],
                [(raw.x1, raw.y1)],
                [(raw.x2, raw.y1)],
            ]
        )
        warped = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
        bottom = warped[:3]
        top = warped[3:]
        x_min = float(np.min(bottom[:, 0]))
        x_max = float(np.max(bottom[:, 0]))
        y = float(np.mean(bottom[:, 1]))
        top_y = float(np.mean(top[:, 1]))
        height_bev = abs(y - top_y)

        if y < s.min_duckie_y or y > s.max_duckie_y:
            return None, self.rejection_obstacle(index, raw, "outside_birdseye_y", now_sec)

        if age <= s.fresh_box_age_sec:
            state = "fresh"
            age_extra = 0.0
        else:
            state = "aging"
            progress = clamp((age - s.fresh_box_age_sec) / max(s.max_box_age_sec - s.fresh_box_age_sec, 0.01), 0.0, 1.0)
            age_extra = progress * s.stale_margin_gain_px

        margin_x = s.duckie_margin_x_px + age_extra
        margin_y = s.duckie_margin_y_px + age_extra + height_bev * 0.20
        obstacle = DuckieObstacle(
            index=index,
            x_left=float(x_min - margin_x),
            x_right=float(x_max + margin_x),
            y=float(y),
            y_radius=float(margin_y),
            age=float(age),
            state=state,
            conf=float(raw.conf),
            raw_box=(float(raw.x1), float(raw.y1), float(raw.x2), float(raw.y2)),
        )
        return obstacle, None

    def build_obstacles(self, raw_duckies, matrix, now_sec):
        obstacles = []
        rejected = []
        for index, raw in enumerate(raw_duckies):
            obstacle, rejection = self.project_detection_to_obstacle(index, raw, matrix, now_sec)
            if obstacle is not None:
                obstacles.append(obstacle)
            if rejection is not None:
                rejected.append(rejection)
        obstacles.sort(key=lambda item: item.center_x)
        return obstacles, rejected

    def merged_intervals(self, intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda item: item[0])
        merged = [list(intervals[0])]
        for left, right, source in intervals[1:]:
            current = merged[-1]
            if left <= current[1]:
                current[1] = max(current[1], right)
                current[2] = current[2] + "+" + source
            else:
                merged.append([left, right, source])
        return [(float(left), float(right), str(source)) for left, right, source in merged]

    def build_gap_candidates_for_boundary(self, boundary, obstacles):
        s = self.settings
        if not boundary.valid:
            return []
        occupied = []
        for duckie in obstacles:
            interval = duckie.interval_at_y(boundary.y, boundary.corridor_left, boundary.corridor_right)
            if interval is not None:
                occupied.append((interval[0], interval[1], f"duckie_{duckie.index}"))
        occupied = self.merged_intervals(occupied)

        gaps = []
        previous_x = boundary.corridor_left
        previous_source = "yellow"

        def append_gap(left_x, right_x, left_source, right_source):
            width = right_x - left_x
            if width <= 0:
                return
            center_x = (left_x + right_x) * 0.5
            valid = width >= s.min_gap_px
            if s.require_duckie_for_target:
                left_is_duckie = left_source.startswith("duckie")
                right_is_duckie = right_source.startswith("duckie")
                has_duckie_edge = left_is_duckie or right_is_duckie
                missing_fallback_edge = (
                    (left_source == "yellow" and not boundary.yellow_found)
                    or (right_source == "white" and not boundary.white_found)
                )
                if width < s.min_gap_px:
                    valid = False
                    reason = "too_narrow"
                elif not has_duckie_edge:
                    valid = False
                    reason = "no_duckie_edge"
                elif (
                    missing_fallback_edge
                    and not s.allow_virtual_corridor_edges
                    and not (left_is_duckie and right_is_duckie)
                ):
                    valid = False
                    reason = "missing_lane_boundary"
                else:
                    valid = True
                    reason = "ok"
            else:
                reason = "ok" if valid else "too_narrow"
            missing_line_penalty = 0.0
            if not boundary.yellow_found:
                missing_line_penalty += s.min_gap_px * 0.10
            if not boundary.white_found:
                missing_line_penalty += s.min_gap_px * 0.10
            near_preference = boundary.y / max(s.size, 1.0) * 4.0
            score = width - missing_line_penalty + near_preference
            gaps.append(
                GapCandidate(
                    y=float(boundary.y),
                    left_x=float(left_x),
                    right_x=float(right_x),
                    center_x=float(center_x),
                    width=float(width),
                    left_source=left_source,
                    right_source=right_source,
                    valid=bool(valid),
                    reason=reason,
                    score=float(score),
                )
            )

        for left, right, source in occupied:
            if right <= boundary.corridor_left or left >= boundary.corridor_right:
                continue
            clipped_left = clamp(left, boundary.corridor_left, boundary.corridor_right)
            clipped_right = clamp(right, boundary.corridor_left, boundary.corridor_right)
            append_gap(previous_x, clipped_left, previous_source, f"{source}_left")
            previous_x = clipped_right
            previous_source = f"{source}_right"
        append_gap(previous_x, boundary.corridor_right, previous_source, "white")
        return gaps

    def build_gap_candidates(self, lane_boundaries, obstacles):
        candidates = []
        for boundary in lane_boundaries:
            candidates.extend(self.build_gap_candidates_for_boundary(boundary, obstacles))
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def nearest_obstacle_layer_candidates(self, candidates, obstacles):
        """Return only gaps at the closest obstacle cross-section.

        Comparing widths across unrelated depth rows is unsafe: a wide gap beside a
        distant duckie can be unreachable because a closer duckie blocks the path.
        Receding-horizon planning must clear the nearest obstacle layer first.
        """
        if not candidates or not obstacles:
            return candidates
        nearest_y = clamp(max(obstacle.y for obstacle in obstacles), 0.0, self.settings.size - 1.0)
        obstacle_candidates = [candidate for candidate in candidates if candidate.has_duckie_edge]
        if not obstacle_candidates:
            return []
        layer_y = min(
            {candidate.y for candidate in obstacle_candidates},
            key=lambda value: abs(value - nearest_y),
        )
        return [candidate for candidate in obstacle_candidates if abs(candidate.y - layer_y) < 0.5]

    def command_for_gap(self, gap):
        s = self.settings
        center_x = s.size * 0.5
        error_px = center_x - gap.center_x
        omega = error_px / max(center_x, 1.0) * s.omega_gain
        omega = clamp(omega, -s.max_omega, s.max_omega)
        if s.turn_sign_invert:
            omega = -omega

        extra_width = max(0.0, gap.width - s.min_gap_px)
        speed_progress = clamp(extra_width / max(s.min_gap_px, 1.0), 0.0, 1.0)
        speed = s.crawl_speed + (s.max_speed - s.crawl_speed) * speed_progress
        speed = clamp(speed, 0.0, s.max_speed)
        speed_level = SPEED_CRAWL if speed <= s.crawl_speed + 1e-6 else SPEED_SLOW
        return float(speed), float(omega), speed_level, float(error_px)

    def lane_follow_result(self, lane_boundaries, obstacles, rejected, warnings):
        s = self.settings
        usable = [boundary for boundary in lane_boundaries if boundary.valid]
        observed = [
            boundary
            for boundary in usable
            if boundary.yellow_found or boundary.white_found
        ]
        if observed:
            # Follow a forward lookahead instead of the closest visible row.
            # The closest row reacts too late to a strongly curving white line.
            boundary = min(
                observed,
                key=lambda item: (
                    abs(item.y - s.lane_lookahead_y),
                    -(int(item.yellow_found) + int(item.white_found)),
                ),
            )
        else:
            boundary = usable[0] if usable else (lane_boundaries[0] if lane_boundaries else None)
        if boundary is None:
            return self.empty_result(state=STATE_STOP, reason="missing_lane_geometry")

        lane_center = (boundary.corridor_left + boundary.corridor_right) * 0.5
        image_center = s.size * 0.5
        error_px = image_center - lane_center
        omega = error_px / max(image_center, 1.0) * s.lane_omega_gain

        # A line that enters the bot's forward safety corridor gets a nonlinear
        # steering override. Invalid/narrow farther slices still carry crucial
        # collision information and must not be ignored by the normal target
        # selection above.
        clearance_target = max(1.0, s.lane_boundary_clearance_px)
        white_urgency = 0.0
        yellow_urgency = 0.0
        for candidate in lane_boundaries:
            if candidate.y < s.lane_emergency_min_y:
                continue
            if candidate.white_found:
                white_clearance = candidate.white_x - s.line_guard_px - image_center
                white_urgency = max(
                    white_urgency,
                    clamp((clearance_target - white_clearance) / clearance_target, 0.0, 1.0),
                )
            if candidate.yellow_found:
                yellow_clearance = image_center - (candidate.yellow_x + s.line_guard_px)
                yellow_urgency = max(
                    yellow_urgency,
                    clamp((clearance_target - yellow_clearance) / clearance_target, 0.0, 1.0),
                )

        guard_omega = min(abs(s.lane_boundary_avoid_omega), abs(s.lane_max_omega))
        if white_urgency > yellow_urgency:
            omega = max(omega, guard_omega * white_urgency)
        elif yellow_urgency > white_urgency:
            omega = min(omega, -guard_omega * yellow_urgency)
        omega = clamp(omega, -s.lane_max_omega, s.lane_max_omega)
        if s.turn_sign_invert:
            omega = -omega

        detected_lines = int(boundary.yellow_found) + int(boundary.white_found)
        base_speed = s.lane_follow_speed if detected_lines > 0 else s.lane_fallback_speed
        turn_fraction = abs(omega) / max(s.lane_max_omega, 0.01)
        speed_scale = 1.0 - clamp(turn_fraction, 0.0, 1.0) * clamp(s.lane_turn_slowdown, 0.0, 0.95)
        minimum_turn_speed = clamp(s.lane_min_turn_speed, 0.0, base_speed)
        speed = max(minimum_turn_speed, base_speed * speed_scale)
        reason = "lane_lines"
        if detected_lines == 1:
            reason = "lane_single_line"
        elif detected_lines == 0:
            reason = "lane_fallback_no_lines"
        if white_urgency >= 0.5 and white_urgency > yellow_urgency:
            reason += "_white_guard"
        elif yellow_urgency >= 0.5 and yellow_urgency > white_urgency:
            reason += "_yellow_guard"
        if obstacles:
            reason += "_duckies_far"

        return PlannerResult(
            state=STATE_LANE_FOLLOW,
            reason=reason,
            speed_level=SPEED_SLOW,
            has_target=False,
            target_x=float(lane_center),
            target_y=float(boundary.y),
            target_error_px=float(error_px),
            v=float(speed),
            omega=float(omega),
            chosen_gap=None,
            candidate_gaps=[],
            duckies=obstacles,
            rejected_duckies=rejected,
            lane_boundaries=lane_boundaries,
            warnings=warnings,
        )

    def plan_birdseye(
        self,
        white_mask,
        yellow_mask,
        raw_duckies,
        matrix,
        now_sec,
        previous_target_x=None,
        gap_mode_active=True,
    ):
        s = self.settings
        lane_boundaries = self.build_lane_boundaries(white_mask, yellow_mask)
        obstacles, rejected = self.build_obstacles(raw_duckies, matrix, now_sec)
        gap_boundaries = self.add_duckie_boundaries(
            white_mask, yellow_mask, lane_boundaries, obstacles
        )
        warnings = sorted({warning for boundary in lane_boundaries for warning in boundary.warnings})

        if not gap_mode_active:
            return self.lane_follow_result(lane_boundaries, obstacles, rejected, warnings)

        all_candidates = self.build_gap_candidates(gap_boundaries, obstacles)
        candidates = self.nearest_obstacle_layer_candidates(all_candidates, obstacles)

        if not obstacles:
            return self.lane_follow_result(lane_boundaries, obstacles, rejected, warnings)

        valid_gaps = [gap for gap in candidates if gap.valid]
        if not valid_gaps:
            return PlannerResult(
                state=STATE_STOP,
                reason="no_gap_wide_enough",
                speed_level=SPEED_STOP,
                has_target=False,
                target_x=None,
                target_y=None,
                target_error_px=0.0,
                v=0.0,
                omega=0.0,
                chosen_gap=candidates[0] if candidates else None,
                candidate_gaps=candidates,
                duckies=obstacles,
                rejected_duckies=rejected,
                lane_boundaries=lane_boundaries,
                warnings=warnings,
            )

        chosen = valid_gaps[0]
        target_x = chosen.center_x
        if previous_target_x is not None:
            target_x = s.target_smoothing * float(previous_target_x) + (1.0 - s.target_smoothing) * target_x
            target_x = clamp(target_x, chosen.left_x, chosen.right_x)
            chosen = replace(chosen, center_x=float(target_x))
        speed, omega, speed_level, error_px = self.command_for_gap(chosen)

        return PlannerResult(
            state=STATE_TRACK_TARGET,
            reason="gap_selected",
            speed_level=speed_level,
            has_target=True,
            target_x=float(target_x),
            target_y=float(chosen.y),
            target_error_px=float(error_px),
            v=speed,
            omega=omega,
            chosen_gap=chosen,
            candidate_gaps=candidates,
            duckies=obstacles,
            rejected_duckies=rejected,
            lane_boundaries=lane_boundaries,
            warnings=warnings,
        )


class DuckieModeGate:
    """Debounces the transition between lane following and near-duckie gap mode."""

    def __init__(self, settings):
        self.settings = settings
        self.reset()

    def update_settings(self, settings):
        self.settings = settings

    def reset(self):
        self.active = False
        self.near_frames = 0
        self.missing_frames = 0
        self.near_now = False

    def qualifies(self, detection):
        s = self.settings
        return (
            detection.conf >= s.min_box_conf
            and detection.height >= s.gap_activation_min_box_height_px
            and detection.area >= s.gap_activation_min_box_area_px
        )

    def update(self, detections):
        s = self.settings
        self.near_now = any(self.qualifies(detection) for detection in detections)
        if self.near_now:
            self.near_frames += 1
            self.missing_frames = 0
        else:
            self.near_frames = 0
            if self.active:
                self.missing_frames += 1

        if not self.active and self.near_frames >= max(1, s.gap_activation_frames):
            self.active = True
            self.missing_frames = 0
        elif self.active and self.missing_frames >= max(1, s.gap_release_frames):
            self.active = False
            self.missing_frames = 0
        return self.active


class GapRecoveryController:
    """Produces a bounded stop/reverse/turn recovery when no gap is safe."""

    def __init__(self, settings):
        self.settings = settings
        self.next_turn_sign = 1.0
        self.reset()

    def update_settings(self, settings):
        self.settings = settings

    def reset(self):
        self.phase = "idle"
        self.deadline = 0.0
        self.turn_sign = 1.0
        self.no_gap_frames = 0

    def choose_turn_sign(self, result):
        s = self.settings
        sign = self.next_turn_sign
        if result.chosen_gap is not None:
            error = s.size * 0.5 - result.chosen_gap.center_x
            if abs(error) >= 3.0:
                sign = 1.0 if error > 0 else -1.0
        if s.turn_sign_invert:
            sign = -sign
        self.next_turn_sign = -sign
        return sign

    def phase_result(self, result):
        s = self.settings
        if self.phase == "stop":
            return result.with_state(
                STATE_RECOVERY_STOP,
                "no_gap_prepare_reverse",
                speed_level=SPEED_STOP,
                v=0.0,
                omega=0.0,
                has_target=False,
            )
        if self.phase == "reverse":
            return result.with_state(
                STATE_RECOVERY_REVERSE,
                "no_gap_reversing",
                speed_level=SPEED_REVERSE,
                v=-abs(s.recovery_reverse_speed),
                omega=0.0,
                has_target=False,
            )
        if self.phase == "turn":
            return result.with_state(
                STATE_RECOVERY_TURN,
                "no_gap_adjusting_angle",
                speed_level=SPEED_SEARCH,
                v=0.0,
                omega=self.turn_sign * abs(s.recovery_turn_omega),
                has_target=False,
            )
        return result.with_state(
            STATE_RECOVERY_SETTLE,
            "no_gap_settling",
            speed_level=SPEED_STOP,
            v=0.0,
            omega=0.0,
            has_target=False,
        )

    def update(self, result, now_sec, gap_mode_active):
        s = self.settings
        if not gap_mode_active:
            self.reset()
            return result

        no_gap = result.state == STATE_STOP and result.reason == "no_gap_wide_enough"

        # Recovery is only justified while the planner still sees no alternative.
        # A newly available gap cancels reverse/turn immediately.
        if not no_gap:
            self.reset()
            return result

        if self.phase == "idle":
            self.no_gap_frames += 1
            if self.no_gap_frames < max(1, s.recovery_no_gap_confirm_frames):
                return result.with_state(
                    STATE_STOP,
                    "no_gap_waiting_confirmation",
                    speed_level=SPEED_STOP,
                    v=0.0,
                    omega=0.0,
                    has_target=False,
                )
            self.phase = "stop"
            self.deadline = now_sec + max(0.0, s.recovery_stop_seconds)
            self.turn_sign = self.choose_turn_sign(result)
            return self.phase_result(result)

        if now_sec >= self.deadline:
            if self.phase == "stop":
                self.phase = "reverse"
                self.deadline = now_sec + max(0.0, s.recovery_reverse_seconds)
            elif self.phase == "reverse":
                self.phase = "turn"
                self.deadline = now_sec + max(0.0, s.recovery_turn_seconds)
            elif self.phase == "turn":
                self.phase = "settle"
                self.deadline = now_sec + max(0.0, s.recovery_settle_seconds)
            else:
                self.reset()
                self.no_gap_frames = 1
                return result.with_state(
                    STATE_STOP,
                    "no_gap_waiting_confirmation",
                    speed_level=SPEED_STOP,
                    v=0.0,
                    omega=0.0,
                    has_target=False,
                )
        return self.phase_result(result)


class GapTargetTracker:
    def __init__(self, settings):
        self.settings = settings
        self.centered_frames = 0
        self.search_hold_frames = 0
        self.was_tracking = False

    def update_settings(self, settings):
        self.settings = settings

    def reset(self):
        self.centered_frames = 0
        self.search_hold_frames = 0
        self.was_tracking = False

    def update(self, result):
        s = self.settings
        if self.search_hold_frames > 0:
            self.search_hold_frames -= 1
            self.was_tracking = False
            self.centered_frames = 0
            return result.with_state(
                STATE_PASS_GAP,
                "passing_gap",
                speed_level=SPEED_SLOW,
                v=s.gap_pass_speed,
                omega=0.0,
                has_target=False,
            )

        if result.state != STATE_TRACK_TARGET or not result.has_target:
            if self.was_tracking and result.state == STATE_SEARCH:
                result = result.with_state(
                    STATE_SEARCH,
                    "target_lost",
                    speed_level=SPEED_SEARCH,
                    v=0.0,
                    omega=(-s.search_omega if s.turn_sign_invert else s.search_omega),
                    has_target=False,
                )
            self.was_tracking = False
            self.centered_frames = 0
            return result

        self.was_tracking = True
        if abs(result.target_error_px) <= s.target_reached_error_px:
            self.centered_frames += 1
        else:
            self.centered_frames = 0

        target_near_bottom = result.target_y is not None and result.target_y >= s.target_reached_y
        target_stably_centered = self.centered_frames >= max(1, s.target_centered_frames)
        # Being centered only means the bot is aligned with the gap.  It must
        # keep moving until that centered target also reaches the near field.
        if target_near_bottom and target_stably_centered:
            self.search_hold_frames = max(0, s.gap_pass_frames)
            self.was_tracking = False
            self.centered_frames = 0
            return result.with_state(
                STATE_PASS_GAP,
                "target_reached",
                speed_level=SPEED_SLOW,
                v=s.gap_pass_speed,
                omega=0.0,
                has_target=False,
            )

        return result


def birdseye_to_image_points(points, matrix):
    if not points:
        return []
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return []
    source = np.float32(points).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(source, inverse).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in warped]


def render_birdseye_debug(birdseye, white_mask, yellow_mask, result, text_lines=None):
    debug = birdseye.copy()
    height, width = debug.shape[:2]
    debug[white_mask > 0] = (255, 255, 255)
    debug[yellow_mask > 0] = (0, 255, 255)

    for boundary in result.lane_boundaries:
        y = int(clamp(boundary.y, 0, height - 1))
        color = (0, 180, 255) if boundary.valid else (0, 0, 255)
        cv2.line(debug, (int(clamp(boundary.corridor_left, 0, width - 1)), y), (int(clamp(boundary.corridor_right, 0, width - 1)), y), color, 1)
        cv2.circle(debug, (int(clamp(boundary.yellow_x, 0, width - 1)), y), 3, (0, 255, 255), -1)
        cv2.circle(debug, (int(clamp(boundary.white_x, 0, width - 1)), y), 3, (255, 255, 255), -1)

    for gap in result.candidate_gaps:
        y = int(clamp(gap.y, 0, height - 1))
        color = (0, 160, 0) if gap.valid else (0, 0, 180)
        thickness = 2 if result.chosen_gap is not None and gap == result.chosen_gap else 1
        cv2.line(debug, (int(clamp(gap.left_x, 0, width - 1)), y), (int(clamp(gap.right_x, 0, width - 1)), y), color, thickness)
        cv2.circle(debug, (int(clamp(gap.center_x, 0, width - 1)), y), 4, color, -1)

    for duckie in result.duckies:
        x1 = int(clamp(duckie.x_left, 0, width - 1))
        x2 = int(clamp(duckie.x_right, 0, width - 1))
        y1 = int(clamp(duckie.y - duckie.y_radius, 0, height - 1))
        y2 = int(clamp(duckie.y + duckie.y_radius, 0, height - 1))
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(debug, (int(clamp(duckie.center_x, 0, width - 1)), int(clamp(duckie.y, 0, height - 1))), 4, (0, 0, 255), -1)
        cv2.putText(debug, f"d{duckie.index} {duckie.state}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)

    show_target = result.has_target or result.state == STATE_LANE_FOLLOW
    if show_target and result.target_x is not None and result.target_y is not None:
        target = (int(clamp(result.target_x, 0, width - 1)), int(clamp(result.target_y, 0, height - 1)))
        cv2.circle(debug, target, 8, (0, 255, 0), -1)
        cv2.line(debug, (width // 2, height - 1), target, (0, 255, 0), 2)

    lines = list(text_lines or [])
    if not lines:
        lines = [
            f"{result.state} {result.reason}",
            f"gap={result.chosen_gap_width_px} v={result.v:.3f} w={result.omega:.2f}",
            f"duckies={len(result.duckies)} rejected={len(result.rejected_duckies)}",
        ]
    put_text_lines(debug, lines)
    return debug


def render_masks_debug(white_mask, yellow_mask, result):
    debug = np.zeros((white_mask.shape[0], white_mask.shape[1], 3), dtype=np.uint8)
    debug[white_mask > 0] = (255, 255, 255)
    debug[yellow_mask > 0] = (0, 255, 255)
    return render_birdseye_debug(debug, white_mask, yellow_mask, result)


def render_camera_debug(image, raw_duckies, result, matrix, calibration_label=""):
    debug = image.copy()
    height, width = debug.shape[:2]
    for raw in raw_duckies:
        x1 = int(clamp(raw.x1, 0, width - 1))
        y1 = int(clamp(raw.y1, 0, height - 1))
        x2 = int(clamp(raw.x2, 0, width - 1))
        y2 = int(clamp(raw.y2, 0, height - 1))
        color = (0, 0, 255)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug,
            f"{raw.conf:.2f} h{raw.height:.0f} a{raw.area:.0f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
        )

    show_target = result.has_target or result.state == STATE_LANE_FOLLOW
    if show_target and result.target_x is not None and result.target_y is not None:
        target_points = birdseye_to_image_points([(result.target_x, result.target_y)], matrix)
        if target_points:
            target = target_points[0]
            cv2.circle(debug, (int(clamp(target[0], 0, width - 1)), int(clamp(target[1], 0, height - 1))), 8, (0, 255, 0), -1)

    put_text_lines(
        debug,
        [
            f"{result.state} {result.reason}",
            f"v={result.v:.3f} w={result.omega:.2f} {calibration_label}",
        ],
    )
    return debug


def put_text_lines(image, lines, origin=(8, 22), color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
        y += 18
