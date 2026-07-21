#!/usr/bin/env python3

"""Pure image-processing and control logic for the simple Kreiselfahrt mode."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


STATE_TRACK = "TRACK"
STATE_HOLD = "HOLD"
STATE_SEARCH = "SEARCH"
STATE_COUNTERSTEER = "COUNTERSTEER"
STATE_RECOVERY_FORWARD = "RECOVERY_FORWARD"
STATE_DUCKIE_CLEARANCE = "DUCKIE_CLEARANCE"
STATE_DISABLED = "DISABLED"
STATE_CAMERA_STALE = "CAMERA_STALE"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def parameter_value(parameters, group, name, fallback):
    value = parameters.get(group, {}).get(name, fallback)
    if isinstance(value, dict):
        return value.get("default", fallback)
    return value


@dataclass(frozen=True)
class KreiselfahrtSettings:
    size: int
    source_points: np.ndarray
    yellow_lower: Tuple[int, int, int]
    yellow_upper: Tuple[int, int, int]
    min_component_area: int
    segment_length: int
    line_width: int
    right_padding: int
    lookahead_min_y: int
    near_field_top: int
    near_field_min_aspect_ratio: float
    near_field_extra_padding: int
    roi_left: int
    roi_right: int
    roi_top: int
    roi_bottom: int
    target_x: float
    min_row_coverage: float
    boundary_percentile: float
    smoothing_alpha: float
    hold_seconds: float
    duckie_memory_seconds: float
    reacquire_frames: int
    kp: float
    ki: float
    kd: float
    nominal_speed: float
    reduced_speed: float
    search_speed: float
    search_omega: float
    reacquire_countersteer_angle: float
    reacquire_forward_distance: float
    reacquire_forward_speed: float
    max_omega: float
    integral_limit: float
    camera_timeout: float
    command_rate: float
    debug_rate: float

    @classmethod
    def from_parameters(cls, parameters):
        val = lambda group, name, fallback: parameter_value(parameters, group, name, fallback)
        size = max(100, int(val("birdseye", "size", 400)))

        points = np.float32(
            [
                [val("birdseye", "top_left_x", 159), val("birdseye", "top_left_y", 218)],
                [val("birdseye", "top_right_x", 441), val("birdseye", "top_right_y", 218)],
                [val("birdseye", "bottom_left_x", -29), val("birdseye", "bottom_left_y", 382)],
                [val("birdseye", "bottom_right_x", 606), val("birdseye", "bottom_right_y", 382)],
            ]
        )

        roi_left = int(clamp(int(val("tracking", "roi_left", 0)), 0, size - 1))
        roi_right = int(clamp(int(val("tracking", "roi_right", 300)), roi_left + 1, size))
        roi_top = int(clamp(int(val("tracking", "roi_top", 250)), 0, size - 1))
        roi_bottom = int(clamp(int(val("tracking", "roi_bottom", 350)), roi_top + 1, size))

        h_low = int(clamp(int(val("yellow", "hl", 15)), 0, 179))
        h_high = int(clamp(int(val("yellow", "hh", 60)), h_low, 179))
        s_low = int(clamp(int(val("yellow", "sl", 80)), 0, 255))
        s_high = int(clamp(int(val("yellow", "sh", 255)), s_low, 255))
        v_low = int(clamp(int(val("yellow", "vl", 130)), 0, 255))
        v_high = int(clamp(int(val("yellow", "vh", 255)), v_low, 255))

        return cls(
            size=size,
            source_points=points,
            yellow_lower=(h_low, s_low, v_low),
            yellow_upper=(h_high, s_high, v_high),
            min_component_area=max(1, int(val("virtual_line", "min_component_area", 20))),
            segment_length=max(1, int(val("virtual_line", "segment_length", 100))),
            line_width=max(1, int(val("virtual_line", "line_width", 7))),
            right_padding=max(0, int(val("virtual_line", "right_padding", 5))),
            lookahead_min_y=int(
                clamp(int(val("virtual_line", "lookahead_min_y", 180)), 0, size - 2)
            ),
            near_field_top=int(
                clamp(
                    int(val("virtual_line", "near_field_top", 330)),
                    int(clamp(int(val("virtual_line", "lookahead_min_y", 180)), 0, size - 2)) + 1,
                    size - 1,
                )
            ),
            near_field_min_aspect_ratio=max(
                0.1, float(val("virtual_line", "near_field_min_aspect_ratio", 0.8))
            ),
            near_field_extra_padding=max(
                0, int(val("virtual_line", "near_field_extra_padding", 25))
            ),
            roi_left=roi_left,
            roi_right=roi_right,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            target_x=float(clamp(float(val("tracking", "target_x", 80)), 0, size - 1)),
            min_row_coverage=float(clamp(float(val("tracking", "min_row_coverage", 0.15)), 0.0, 1.0)),
            boundary_percentile=float(clamp(float(val("tracking", "boundary_percentile", 80.0)), 0.0, 100.0)),
            smoothing_alpha=float(clamp(float(val("tracking", "smoothing_alpha", 0.4)), 0.0, 1.0)),
            hold_seconds=max(0.0, float(val("tracking", "hold_seconds", 0.35))),
            duckie_memory_seconds=float(
                clamp(float(val("tracking", "duckie_memory_seconds", 0.8)), 0.0, 5.0)
            ),
            reacquire_frames=max(1, int(val("tracking", "reacquire_frames", 2))),
            kp=float(val("control", "p", 5.0)),
            ki=float(val("control", "i", 0.04)),
            kd=float(val("control", "d", 1.0)),
            nominal_speed=max(0.0, float(val("control", "nominal_speed", 0.15))),
            reduced_speed=max(0.0, float(val("control", "reduced_speed", 0.08))),
            search_speed=max(0.0, float(val("control", "search_speed", 0.08))),
            search_omega=max(0.0, float(val("control", "search_omega", 1.0))),
            reacquire_countersteer_angle=max(
                0.0, float(val("control", "reacquire_countersteer_angle", 0.2))
            ),
            reacquire_forward_distance=float(
                clamp(float(val("control", "reacquire_forward_distance", 0.08)), 0.0, 1.0)
            ),
            reacquire_forward_speed=float(
                clamp(float(val("control", "reacquire_forward_speed", 0.08)), 0.01, 0.5)
            ),
            max_omega=max(0.0, float(val("control", "max_omega", 2.5))),
            integral_limit=max(0.0, float(val("control", "integral_limit", 5.0))),
            camera_timeout=max(0.01, float(val("safety", "camera_timeout", 0.5))),
            command_rate=max(1.0, float(val("safety", "command_rate", 10.0))),
            debug_rate=max(1.0, float(val("safety", "debug_rate", 10.0))),
        )


@dataclass(frozen=True)
class YellowComponent:
    x: int
    y: int
    width: int
    height: int
    area: int
    accepted: bool
    reason: str


@dataclass
class KreiselfahrtResult:
    state: str
    tracking_state: str
    confidence: float
    boundary_x_raw: Optional[float]
    boundary_x: Optional[float]
    error: float
    v: float
    omega: float
    yellow_age: Optional[float]
    birdseye: np.ndarray
    yellow_mask: np.ndarray
    filtered_mask: np.ndarray
    virtual_mask: np.ndarray
    row_right_edges: List[Tuple[int, int]]
    components: List[YellowComponent]


class KreiselfahrtFollower:
    """Tracks a virtual yellow left boundary and produces bounded drive commands."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.settings = KreiselfahrtSettings.from_parameters(parameters)
        self._matrix = self._make_homography(self.settings)
        self._last_boundary = None
        self._last_seen_sec = None
        self._tracking_state = STATE_SEARCH
        self._valid_frames = 0
        self._countersteer_armed = False
        self._countersteer_until_sec = None
        self._recovery_forward_start_sec = None
        self._recovery_forward_until_sec = None
        self._near_left_duckie_visible = False
        self._duckie_clearance_until_sec = None
        self._last_error = 0.0
        self._integral = 0.0
        self._pid_initialized = False

    @staticmethod
    def _make_homography(settings):
        last = float(settings.size - 1)
        destination = np.float32([[0, 0], [last, 0], [0, last], [last, last]])
        return cv2.getPerspectiveTransform(settings.source_points, destination)

    def update_parameters(self, parameters):
        previous_size = self.settings.size
        self.parameters = parameters
        self.settings = KreiselfahrtSettings.from_parameters(parameters)
        self._matrix = self._make_homography(self.settings)
        if previous_size != self.settings.size:
            self._last_boundary = None
            self._last_seen_sec = None
            self._tracking_state = STATE_SEARCH
            self._valid_frames = 0
            self._countersteer_armed = False
            self._countersteer_until_sec = None
            self._recovery_forward_start_sec = None
            self._recovery_forward_until_sec = None
            self._near_left_duckie_visible = False
            self._duckie_clearance_until_sec = None
        elif self._last_boundary is not None:
            self._last_boundary = float(clamp(self._last_boundary, 0.0, self.settings.size - 1.0))
        self.reset_control()

    def reset_control(self):
        self._last_error = 0.0
        self._integral = 0.0
        self._pid_initialized = False

    def birdseye(self, image):
        return cv2.warpPerspective(image, self._matrix, (self.settings.size, self.settings.size))

    def yellow_mask(self, birdseye):
        hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.settings.yellow_lower, self.settings.yellow_upper)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    def make_virtual_line(self, yellow_mask):
        settings = self.settings
        filtered = np.zeros_like(yellow_mask)
        virtual = np.zeros_like(yellow_mask)
        components = []

        count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow_mask, connectivity=8)
        for label in range(1, count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            intersects_roi = (
                x < settings.roi_right
                and x + width > settings.roi_left
                and y < settings.roi_bottom
                and y + height > settings.roi_top
            )
            if area < settings.min_component_area:
                accepted, reason = False, "too_small"
            elif not intersects_roi:
                accepted, reason = False, "outside_roi"
            else:
                center_y = y + height // 2
                duckie_like_shape = width >= max(
                    20, int(round(settings.near_field_min_aspect_ratio * height))
                )
                near_field_guard = center_y >= settings.near_field_top and duckie_like_shape
                if center_y < settings.lookahead_min_y:
                    accepted, reason = False, "too_far"
                elif center_y >= settings.near_field_top and not near_field_guard:
                    accepted, reason = False, "near_line_ignored"
                else:
                    accepted = True
                    reason = "near_guard" if near_field_guard else "accepted"

            if accepted:
                filtered[labels == label] = 255

                extra_padding = settings.near_field_extra_padding if near_field_guard else 0
                boundary_x = int(
                    clamp(
                        x + width - 1 + settings.right_padding + extra_padding,
                        settings.roi_left,
                        settings.roi_right - 1,
                    )
                )
                segment_height = max(height, settings.segment_length)
                y0 = int(clamp(center_y - segment_height // 2, 0, settings.size - 1))
                y1 = int(clamp(y0 + segment_height - 1, y0, settings.size - 1))
                half_width = settings.line_width // 2
                x0 = int(clamp(boundary_x - half_width, settings.roi_left, settings.roi_right - 1))
                x1 = int(clamp(boundary_x + half_width, x0, settings.roi_right - 1))
                cv2.rectangle(virtual, (x0, y0), (x1, y1), 255, -1)

            components.append(YellowComponent(x, y, width, height, area, accepted, reason))

        return filtered, virtual, components

    def measure_boundary(self, virtual_mask):
        settings = self.settings
        right_edges = []
        for y in range(settings.roi_top, settings.roi_bottom):
            cols = np.flatnonzero(virtual_mask[y, settings.roi_left : settings.roi_right])
            if cols.size:
                right_edges.append((y, int(cols[-1] + settings.roi_left)))

        roi_height = max(1, settings.roi_bottom - settings.roi_top)
        confidence = float(len(right_edges)) / float(roi_height)
        if not right_edges or confidence < settings.min_row_coverage:
            return None, confidence, right_edges

        boundary = float(np.percentile([x for _, x in right_edges], settings.boundary_percentile))
        return boundary, confidence, right_edges

    def _update_tracking(self, boundary_raw, confidence, now_sec):
        valid = boundary_raw is not None and confidence >= self.settings.min_row_coverage
        previous_state = self._tracking_state

        if valid:
            self._last_seen_sec = float(now_sec)
            alpha = self.settings.smoothing_alpha
            if self._last_boundary is None:
                self._last_boundary = float(boundary_raw)
            else:
                self._last_boundary = alpha * float(boundary_raw) + (1.0 - alpha) * self._last_boundary

            self._valid_frames += 1
            if previous_state == STATE_SEARCH and self._valid_frames < self.settings.reacquire_frames:
                self._tracking_state = STATE_SEARCH
            else:
                self._tracking_state = STATE_TRACK
        else:
            self._valid_frames = 0
            age = None if self._last_seen_sec is None else max(0.0, float(now_sec) - self._last_seen_sec)
            if self._last_boundary is not None and age is not None and age <= self.settings.hold_seconds:
                self._tracking_state = STATE_HOLD
            else:
                self._tracking_state = STATE_SEARCH

        countersteer_active = (
            self._countersteer_until_sec is not None
            and float(now_sec) < self._countersteer_until_sec
        )
        if (
            previous_state in (STATE_TRACK, STATE_HOLD)
            and self._tracking_state == STATE_SEARCH
            and not countersteer_active
        ):
            # Arm exactly once for this genuine loss/reacquisition cycle. This
            # deliberately does not arm on startup, when SEARCH is the initial state.
            self._countersteer_armed = True

        if previous_state == STATE_SEARCH and self._tracking_state == STATE_TRACK:
            self.reset_control()
            correction_omega = min(self.settings.search_omega, self.settings.max_omega)
            if self._countersteer_armed:
                maneuver_start = float(now_sec)
                if (
                    self.settings.reacquire_countersteer_angle > 0.0
                    and correction_omega > 0.0
                ):
                    duration = self.settings.reacquire_countersteer_angle / correction_omega
                    self._countersteer_until_sec = maneuver_start + duration
                    maneuver_start = self._countersteer_until_sec

                if self.settings.reacquire_forward_distance > 0.0:
                    duration = (
                        self.settings.reacquire_forward_distance
                        / self.settings.reacquire_forward_speed
                    )
                    self._recovery_forward_start_sec = maneuver_start
                    self._recovery_forward_until_sec = maneuver_start + duration
            self._countersteer_armed = False

        age = None if self._last_seen_sec is None else max(0.0, float(now_sec) - self._last_seen_sec)
        return self._tracking_state, age

    def _normalized_error(self):
        if self._last_boundary is None:
            return 0.0
        half_width = max(1.0, self.settings.size / 2.0)
        return float(clamp((self.settings.target_x - self._last_boundary) / half_width, -1.0, 1.0))

    def _pid(self, error):
        settings = self.settings
        if not self._pid_initialized:
            derivative = 0.0
            self._pid_initialized = True
        else:
            derivative = error - self._last_error

        if error != 0.0 and self._last_error != 0.0 and (error > 0.0) != (self._last_error > 0.0):
            self._integral = 0.0
        self._integral = clamp(self._integral + error, -settings.integral_limit, settings.integral_limit)
        omega = settings.kp * error + settings.ki * self._integral + settings.kd * derivative
        self._last_error = error
        return float(clamp(omega, -settings.max_omega, settings.max_omega))

    def _near_left_duckie(self, components):
        """Return whether a Duckie-like component is passing beside the bot.

        The existing near-field classifier supplies the object semantics. The
        left half of the tracking ROI distinguishes a Duckie alongside the bot
        from one that is still ahead and must remain visible to the tracker.
        """
        roi_midpoint = self.settings.roi_left + (
            self.settings.roi_right - self.settings.roi_left
        ) / 2.0
        return any(
            component.reason == "near_guard"
            and component.x + component.width / 2.0 <= roi_midpoint
            for component in components
        )

    def process_mask(self, yellow_mask, now_sec, control_enabled=True, birdseye=None):
        settings = self.settings
        if yellow_mask.shape != (settings.size, settings.size):
            yellow_mask = cv2.resize(yellow_mask, (settings.size, settings.size), interpolation=cv2.INTER_NEAREST)
        yellow_mask = np.asarray(yellow_mask, dtype=np.uint8)
        filtered, virtual, components = self.make_virtual_line(yellow_mask)
        boundary_raw, confidence, row_edges = self.measure_boundary(virtual)
        tracking_state, yellow_age = self._update_tracking(boundary_raw, confidence, now_sec)
        error = self._normalized_error()

        countersteer_active = (
            self._countersteer_until_sec is not None
            and float(now_sec) < self._countersteer_until_sec
        )
        if self._countersteer_until_sec is not None and not countersteer_active:
            self._countersteer_until_sec = None

        recovery_forward_active = (
            self._recovery_forward_start_sec is not None
            and self._recovery_forward_until_sec is not None
            and self._recovery_forward_start_sec <= float(now_sec) < self._recovery_forward_until_sec
        )
        if self._recovery_forward_until_sec is not None and float(now_sec) >= self._recovery_forward_until_sec:
            self._recovery_forward_start_sec = None
            self._recovery_forward_until_sec = None

        # Never continue the open-loop straight phase after yellow has been lost
        # again. SEARCH takes over immediately and arms a fresh recovery cycle.
        if recovery_forward_active and tracking_state == STATE_SEARCH:
            recovery_forward_active = False
            self._recovery_forward_start_sec = None
            self._recovery_forward_until_sec = None

        near_left_duckie_visible = self._near_left_duckie(components)
        had_near_left_duckie = self._near_left_duckie_visible
        self._near_left_duckie_visible = near_left_duckie_visible
        has_usable_component = any(component.accepted for component in components)

        duckie_clearance_active = (
            self._duckie_clearance_until_sec is not None
            and float(now_sec) < self._duckie_clearance_until_sec
        )
        if (
            control_enabled
            and had_near_left_duckie
            and not near_left_duckie_visible
            and not has_usable_component
            and settings.duckie_memory_seconds > 0.0
            and not countersteer_active
            and not recovery_forward_active
        ):
            self._duckie_clearance_until_sec = (
                float(now_sec) + settings.duckie_memory_seconds
            )
            duckie_clearance_active = True

        clearance_ended = False
        if duckie_clearance_active and has_usable_component:
            # Something useful is visible again; hand control back immediately.
            self._duckie_clearance_until_sec = None
            duckie_clearance_active = False
            clearance_ended = True
        elif (
            self._duckie_clearance_until_sec is not None
            and float(now_sec) >= self._duckie_clearance_until_sec
        ):
            self._duckie_clearance_until_sec = None
            duckie_clearance_active = False
            clearance_ended = True

        if duckie_clearance_active:
            # SEARCH has not physically started yet, so there is no left turn
            # that would need to be undone by COUNTERSTEER on reacquisition.
            self._countersteer_armed = False
        elif clearance_ended and tracking_state == STATE_SEARCH:
            # SEARCH starts physically now; a later reacquisition may countersteer.
            self._countersteer_armed = True

        if not control_enabled:
            self._countersteer_armed = False
            self._countersteer_until_sec = None
            self._recovery_forward_start_sec = None
            self._recovery_forward_until_sec = None
            self._duckie_clearance_until_sec = None
            self.reset_control()
            state, speed, omega = STATE_DISABLED, 0.0, 0.0
        elif countersteer_active:
            self.reset_control()
            state, speed = STATE_COUNTERSTEER, 0.0
            omega = -float(clamp(settings.search_omega, 0.0, settings.max_omega))
        elif recovery_forward_active:
            self.reset_control()
            state, speed, omega = STATE_RECOVERY_FORWARD, settings.reacquire_forward_speed, 0.0
        elif duckie_clearance_active:
            self.reset_control()
            state, speed, omega = STATE_DUCKIE_CLEARANCE, settings.reduced_speed, 0.0
        elif tracking_state == STATE_TRACK:
            state, speed, omega = STATE_TRACK, settings.nominal_speed, self._pid(error)
        elif tracking_state == STATE_HOLD:
            state, speed, omega = STATE_HOLD, settings.reduced_speed, self._pid(error)
        else:
            self.reset_control()
            state, speed = STATE_SEARCH, settings.search_speed
            omega = float(clamp(settings.search_omega, 0.0, settings.max_omega))

        if birdseye is None:
            birdseye = np.zeros((settings.size, settings.size, 3), dtype=np.uint8)

        return KreiselfahrtResult(
            state=state,
            tracking_state=tracking_state,
            confidence=confidence,
            boundary_x_raw=boundary_raw,
            boundary_x=self._last_boundary,
            error=error,
            v=float(speed),
            omega=float(omega),
            yellow_age=yellow_age,
            birdseye=birdseye,
            yellow_mask=yellow_mask,
            filtered_mask=filtered,
            virtual_mask=virtual,
            row_right_edges=row_edges,
            components=components,
        )

    def process_image(self, image, now_sec, control_enabled=True):
        birdseye = self.birdseye(image)
        mask = self.yellow_mask(birdseye)
        return self.process_mask(mask, now_sec, control_enabled=control_enabled, birdseye=birdseye)


def draw_camera_debug(image, settings):
    debug = image.copy()
    points = settings.source_points.astype(np.int32)
    polygon = np.array([points[0], points[1], points[3], points[2]], dtype=np.int32)
    cv2.polylines(debug, [polygon], True, (255, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(debug, "Bird's-Eye-Ausschnitt", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)
    return debug


def draw_yellow_debug(result):
    debug = result.birdseye.copy()
    overlay = np.zeros_like(debug)
    overlay[result.yellow_mask > 0] = (0, 220, 255)
    debug = cv2.addWeighted(debug, 0.55, overlay, 0.75, 0.0)
    for component in result.components:
        if component.reason == "near_guard":
            color = (255, 220, 0)
        elif component.reason in ("too_far", "near_line_ignored"):
            color = (0, 165, 255)
        else:
            color = (60, 220, 60) if component.accepted else (50, 50, 230)
        cv2.rectangle(
            debug,
            (component.x, component.y),
            (component.x + component.width - 1, component.y + component.height - 1),
            color,
            1,
        )
    cv2.putText(
        debug,
        "Gruen genutzt | Cyan Nahschutz | Orange Distanzfilter",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
    )
    return debug


def draw_virtual_debug(result, settings):
    debug = np.zeros((settings.size, settings.size, 3), dtype=np.uint8)
    debug[result.virtual_mask > 0] = (0, 255, 255)
    cv2.rectangle(debug, (settings.roi_left, settings.roi_top), (settings.roi_right - 1, settings.roi_bottom - 1), (255, 100, 0), 2)
    cv2.line(debug, (settings.roi_left, settings.lookahead_min_y), (settings.roi_right - 1, settings.lookahead_min_y), (0, 165, 255), 1)
    cv2.line(debug, (settings.roi_left, settings.near_field_top), (settings.roi_right - 1, settings.near_field_top), (0, 165, 255), 1)
    for y, x in result.row_right_edges[::5]:
        cv2.circle(debug, (x, y), 2, (255, 255, 255), -1)
    cv2.putText(debug, f"Zeilenabdeckung: {result.confidence * 100:.0f}%", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return debug


def draw_control_debug(result, settings, display_state=None, display_v=None, display_omega=None):
    debug = result.birdseye.copy()
    tint = np.zeros_like(debug)
    tint[result.virtual_mask > 0] = (0, 255, 255)
    debug = cv2.addWeighted(debug, 0.6, tint, 0.8, 0.0)
    cv2.rectangle(debug, (settings.roi_left, settings.roi_top), (settings.roi_right - 1, settings.roi_bottom - 1), (255, 100, 0), 2)
    cv2.line(debug, (settings.roi_left, settings.lookahead_min_y), (settings.roi_right - 1, settings.lookahead_min_y), (0, 165, 255), 1)
    cv2.line(debug, (settings.roi_left, settings.near_field_top), (settings.roi_right - 1, settings.near_field_top), (0, 165, 255), 1)
    cv2.line(debug, (int(settings.target_x), settings.roi_top), (int(settings.target_x), settings.roi_bottom - 1), (0, 255, 0), 2)
    if result.boundary_x_raw is not None:
        x = int(result.boundary_x_raw)
        cv2.line(debug, (x, settings.roi_top), (x, settings.roi_bottom - 1), (0, 150, 255), 1)
    if result.boundary_x is not None:
        x = int(result.boundary_x)
        cv2.line(debug, (x, settings.roi_top), (x, settings.roi_bottom - 1), (255, 0, 255), 2)

    state = display_state or result.state
    if state != result.tracking_state:
        state = f"{state} / {result.tracking_state}"
    speed = result.v if display_v is None else display_v
    omega = result.omega if display_omega is None else display_omega
    cv2.rectangle(debug, (0, 0), (settings.size, 55), (0, 0, 0), -1)
    cv2.putText(debug, f"{state}  Fehler={result.error:+.3f}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
    cv2.putText(debug, f"v={speed:.3f}  omega={omega:+.3f}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
    return debug
