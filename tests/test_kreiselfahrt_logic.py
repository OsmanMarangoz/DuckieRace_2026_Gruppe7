#!/usr/bin/env python3

import copy
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "packages" / "follow_lane" / "src"
CONFIG = ROOT / "src" / "packages" / "follow_lane" / "config" / "kreiselfahrt_node.json"
sys.path.insert(0, str(SRC))

from kreiselfahrt_logic import (  # noqa: E402
    STATE_COUNTERSTEER,
    STATE_DUCKIE_CLEARANCE,
    STATE_HOLD,
    STATE_RECOVERY_FORWARD,
    STATE_SEARCH,
    STATE_TRACK,
    KreiselfahrtFollower,
)
from kreiselfahrt_duckie_detector import DuckieYoloDetector  # noqa: E402


def load_parameters():
    with open(CONFIG, "r", encoding="utf-8") as handle:
        return json.load(handle)["parameters"]


class KreiselfahrtLogicTest(unittest.TestCase):
    def setUp(self):
        self.parameters = load_parameters()
        self.follower = KreiselfahrtFollower(copy.deepcopy(self.parameters))
        self.size = self.follower.settings.size

    def empty_mask(self):
        return np.zeros((self.size, self.size), dtype=np.uint8)

    def line_mask(self, x=40):
        mask = self.empty_mask()
        cv2.rectangle(mask, (x, 250), (x + 8, 270), 255, -1)
        cv2.rectangle(mask, (x, 305), (x + 8, 325), 255, -1)
        return mask

    def near_left_duckie_mask(self):
        mask = self.empty_mask()
        cv2.rectangle(mask, (0, 350), (60, 399), 255, -1)
        return mask

    def reach_track(self, mask, start=1.0):
        result = None
        for index in range(self.follower.settings.reacquire_frames):
            result = self.follower.process_mask(mask, start + index * 0.05, control_enabled=True)
        self.assertEqual(result.state, STATE_TRACK)
        return result

    def test_dashed_line_becomes_stable_virtual_boundary(self):
        result = self.reach_track(self.line_mask())

        self.assertGreaterEqual(result.confidence, self.follower.settings.min_row_coverage)
        self.assertIsNotNone(result.boundary_x)
        self.assertGreater(result.error, 0.0)
        self.assertGreater(result.omega, 0.0)
        self.assertEqual(sum(component.accepted for component in result.components), 2)

    def test_duckie_blob_moves_boundary_right_and_steers_right(self):
        self.reach_track(self.line_mask())
        mask = self.line_mask()
        cv2.rectangle(mask, (160, 270), (210, 325), 255, -1)
        result = self.follower.process_mask(mask, 1.2, control_enabled=True)

        self.assertEqual(result.state, STATE_TRACK)
        self.assertGreater(result.boundary_x_raw, self.follower.settings.target_x)
        self.assertLess(result.error, 0.0)
        self.assertLess(result.omega, 0.0)

    def test_ai_duckie_box_uses_the_same_virtual_line_logic(self):
        mask = DuckieYoloDetector.boxes_to_mask(
            (self.size, self.size, 3), [(160, 270, 210, 325)]
        )

        result = self.reach_track(mask)

        self.assertEqual(result.state, STATE_TRACK)
        self.assertEqual(len(result.components), 1)
        self.assertTrue(result.components[0].accepted)
        self.assertGreater(result.boundary_x_raw, self.follower.settings.target_x)
        self.assertLess(result.omega, 0.0)

    def test_ai_duckie_box_mask_clips_to_the_image(self):
        mask = DuckieYoloDetector.boxes_to_mask(
            (20, 30, 3), [(-10, -5, 8, 9), (40, 40, 50, 50)]
        )

        self.assertEqual(mask.shape, (20, 30))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[0, 0]), 255)
        self.assertEqual(int(mask[19, 29]), 0)

    def test_near_left_duckie_holds_back_left_turn(self):
        mask = self.empty_mask()
        cv2.rectangle(mask, (0, 359), (47, 399), 255, -1)

        result = self.reach_track(mask)

        near_component = next(component for component in result.components if component.accepted)
        self.assertEqual(near_component.reason, "near_guard")
        self.assertGreaterEqual(result.boundary_x_raw, self.follower.settings.target_x)
        self.assertLessEqual(result.error, 0.0)

    def test_far_duckie_does_not_override_current_lookahead(self):
        mask = self.line_mask(x=40)
        cv2.rectangle(mask, (170, 125), (240, 195), 255, -1)

        result = self.reach_track(mask)

        far_component = max(result.components, key=lambda component: component.x)
        self.assertEqual(far_component.reason, "too_far")
        self.assertLess(result.boundary_x_raw, self.follower.settings.target_x)
        self.assertGreater(result.omega, 0.0)

    def test_near_line_does_not_mask_left_lookahead(self):
        mask = self.empty_mask()
        cv2.rectangle(mask, (0, 226), (55, 311), 255, -1)
        cv2.rectangle(mask, (54, 323), (95, 388), 255, -1)

        result = self.reach_track(mask)

        ignored = [component for component in result.components if component.reason == "near_line_ignored"]
        self.assertEqual(len(ignored), 1)
        self.assertLess(result.boundary_x_raw, self.follower.settings.target_x)
        self.assertGreater(result.omega, 0.0)

    def test_small_and_outside_components_are_ignored(self):
        mask = self.empty_mask()
        mask[280:282, 20:22] = 255
        cv2.rectangle(mask, (330, 270), (360, 320), 255, -1)
        result = self.follower.process_mask(mask, 1.0, control_enabled=True)

        self.assertEqual(result.state, STATE_SEARCH)
        self.assertEqual(result.confidence, 0.0)
        self.assertFalse(any(component.accepted for component in result.components))
        self.assertEqual({component.reason for component in result.components}, {"too_small", "outside_roi"})

    def test_short_gap_holds_then_searches_left(self):
        self.reach_track(self.line_mask(), start=1.0)
        held = self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        searching = self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)

        self.assertEqual(held.state, STATE_HOLD)
        self.assertEqual(held.v, self.follower.settings.reduced_speed)
        self.assertEqual(searching.state, STATE_SEARCH)
        self.assertEqual(searching.v, self.follower.settings.search_speed)
        self.assertGreater(searching.omega, 0.0)

    def test_near_left_duckie_is_remembered_before_search(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["tracking"]["duckie_memory_seconds"]["default"] = 0.8
        self.follower = KreiselfahrtFollower(parameters)
        self.reach_track(self.near_left_duckie_mask(), start=1.0)

        just_disappeared = self.follower.process_mask(
            self.empty_mask(), 1.10, control_enabled=True
        )
        still_clearing = self.follower.process_mask(
            self.empty_mask(), 1.70, control_enabled=True
        )
        searching = self.follower.process_mask(self.empty_mask(), 1.91, control_enabled=True)

        self.assertEqual(just_disappeared.state, STATE_DUCKIE_CLEARANCE)
        self.assertEqual(still_clearing.state, STATE_DUCKIE_CLEARANCE)
        self.assertEqual(still_clearing.v, self.follower.settings.reduced_speed)
        self.assertEqual(still_clearing.omega, 0.0)
        self.assertEqual(searching.state, STATE_SEARCH)
        self.assertGreater(searching.omega, 0.0)

    def test_visible_yellow_ends_duckie_clearance_immediately(self):
        self.reach_track(self.near_left_duckie_mask(), start=1.0)
        clearing = self.follower.process_mask(self.empty_mask(), 1.10, control_enabled=True)
        visible_again = self.follower.process_mask(
            self.line_mask(), 1.15, control_enabled=True
        )

        self.assertEqual(clearing.state, STATE_DUCKIE_CLEARANCE)
        self.assertEqual(visible_again.state, STATE_TRACK)

    def test_regular_line_loss_does_not_trigger_duckie_clearance(self):
        self.reach_track(self.line_mask(), start=1.0)

        held = self.follower.process_mask(self.empty_mask(), 1.10, control_enabled=True)
        searching = self.follower.process_mask(self.empty_mask(), 1.70, control_enabled=True)

        self.assertEqual(held.state, STATE_HOLD)
        self.assertEqual(searching.state, STATE_SEARCH)

    def test_zero_duckie_memory_disables_clearance(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["tracking"]["duckie_memory_seconds"]["default"] = 0.0
        self.follower = KreiselfahrtFollower(parameters)
        self.reach_track(self.near_left_duckie_mask(), start=1.0)

        disappeared = self.follower.process_mask(self.empty_mask(), 1.10, control_enabled=True)

        self.assertEqual(disappeared.state, STATE_HOLD)

    def test_near_right_duckie_does_not_trigger_left_clearance(self):
        mask = self.empty_mask()
        cv2.rectangle(mask, (210, 350), (275, 399), 255, -1)
        self.reach_track(mask, start=1.0)

        disappeared = self.follower.process_mask(self.empty_mask(), 1.10, control_enabled=True)

        self.assertEqual(disappeared.state, STATE_HOLD)

    def test_reacquisition_resets_derivative_kick(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["p"]["default"] = 0.0
        parameters["control"]["i"]["default"] = 0.0
        parameters["control"]["d"]["default"] = 1.0
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.0
        parameters["control"]["reacquire_forward_distance"]["default"] = 0.0
        parameters["tracking"]["smoothing_alpha"]["default"] = 1.0
        follower = KreiselfahrtFollower(parameters)
        self.follower = follower

        self.reach_track(self.line_mask(), start=1.0)
        follower.process_mask(self.empty_mask(), 2.0, control_enabled=True)
        first = follower.process_mask(self.line_mask(x=150), 2.1, control_enabled=True)
        second = follower.process_mask(self.line_mask(x=150), 2.2, control_enabled=True)

        self.assertEqual(first.state, STATE_SEARCH)
        self.assertEqual(second.state, STATE_TRACK)
        self.assertAlmostEqual(second.omega, 0.0, places=6)

    def test_reacquisition_countersteers_once_then_resumes_track(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["search_omega"]["default"] = 1.0
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.2
        parameters["control"]["reacquire_forward_distance"]["default"] = 0.0
        self.follower = KreiselfahrtFollower(parameters)
        self.reach_track(self.line_mask(), start=1.0)
        self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        searching = self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)
        self.assertEqual(searching.state, STATE_SEARCH)

        first_valid = self.follower.process_mask(self.line_mask(), 1.60, control_enabled=True)
        correcting = self.follower.process_mask(self.line_mask(), 1.65, control_enabled=True)
        correction_omega = min(
            self.follower.settings.search_omega, self.follower.settings.max_omega
        )
        correction_duration = self.follower.settings.reacquire_countersteer_angle / correction_omega
        still_correcting = self.follower.process_mask(
            self.line_mask(), 1.65 + correction_duration * 0.5, control_enabled=True
        )
        resumed = self.follower.process_mask(
            self.line_mask(), 1.65 + correction_duration + 0.05, control_enabled=True
        )

        self.assertEqual(first_valid.state, STATE_SEARCH)
        self.assertEqual(correcting.state, STATE_COUNTERSTEER)
        self.assertEqual(correcting.v, 0.0)
        self.assertLess(correcting.omega, 0.0)
        self.assertAlmostEqual(
            correcting.omega,
            -min(self.follower.settings.search_omega, self.follower.settings.max_omega),
        )
        self.assertEqual(still_correcting.state, STATE_COUNTERSTEER)
        self.assertEqual(resumed.state, STATE_TRACK)
        self.assertEqual(resumed.v, self.follower.settings.nominal_speed)

    def test_reacquisition_drives_straight_before_pid_resumes(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["search_omega"]["default"] = 1.0
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.2
        parameters["control"]["reacquire_forward_distance"]["default"] = 0.08
        parameters["control"]["reacquire_forward_speed"]["default"] = 0.08
        self.follower = KreiselfahrtFollower(parameters)

        self.reach_track(self.line_mask(), start=1.0)
        self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)
        self.follower.process_mask(self.line_mask(), 1.60, control_enabled=True)
        countersteering = self.follower.process_mask(self.line_mask(), 1.65, control_enabled=True)
        driving_straight = self.follower.process_mask(
            self.line_mask(x=150), 1.90, control_enabled=True
        )
        resumed = self.follower.process_mask(self.line_mask(x=150), 2.90, control_enabled=True)

        self.assertEqual(countersteering.state, STATE_COUNTERSTEER)
        self.assertEqual(driving_straight.state, STATE_RECOVERY_FORWARD)
        self.assertEqual(driving_straight.tracking_state, STATE_TRACK)
        self.assertAlmostEqual(
            driving_straight.v, self.follower.settings.reacquire_forward_speed
        )
        self.assertEqual(driving_straight.omega, 0.0)
        self.assertEqual(resumed.state, STATE_TRACK)
        self.assertNotEqual(resumed.omega, 0.0)

    def test_lost_yellow_cancels_recovery_forward(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.0
        parameters["control"]["reacquire_forward_distance"]["default"] = 0.2
        parameters["control"]["reacquire_forward_speed"]["default"] = 0.1
        self.follower = KreiselfahrtFollower(parameters)

        self.reach_track(self.line_mask(), start=1.0)
        self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)
        self.follower.process_mask(self.line_mask(), 1.60, control_enabled=True)
        driving_straight = self.follower.process_mask(
            self.line_mask(), 1.65, control_enabled=True
        )
        yellow_lost = self.follower.process_mask(self.empty_mask(), 2.10, control_enabled=True)

        self.assertEqual(driving_straight.state, STATE_RECOVERY_FORWARD)
        self.assertEqual(yellow_lost.state, STATE_SEARCH)
        self.assertGreater(yellow_lost.omega, 0.0)

    def test_zero_countersteer_angle_disables_reacquisition_maneuver(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.0
        parameters["control"]["reacquire_forward_distance"]["default"] = 0.0
        self.follower = KreiselfahrtFollower(parameters)

        self.reach_track(self.line_mask(), start=1.0)
        self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)
        self.follower.process_mask(self.line_mask(), 1.60, control_enabled=True)
        reacquired = self.follower.process_mask(self.line_mask(), 1.65, control_enabled=True)

        self.assertEqual(reacquired.state, STATE_TRACK)

    def test_disabling_drive_cancels_active_countersteer(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["search_omega"]["default"] = 1.0
        parameters["control"]["reacquire_countersteer_angle"]["default"] = 0.2
        self.follower = KreiselfahrtFollower(parameters)
        self.reach_track(self.line_mask(), start=1.0)
        self.follower.process_mask(self.empty_mask(), 1.15, control_enabled=True)
        self.follower.process_mask(self.empty_mask(), 1.55, control_enabled=True)
        self.follower.process_mask(self.line_mask(), 1.60, control_enabled=True)
        correcting = self.follower.process_mask(self.line_mask(), 1.65, control_enabled=True)
        disabled = self.follower.process_mask(self.line_mask(), 1.70, control_enabled=False)
        reenabled = self.follower.process_mask(self.line_mask(), 1.75, control_enabled=True)

        self.assertEqual(correcting.state, STATE_COUNTERSTEER)
        self.assertEqual(disabled.v, 0.0)
        self.assertEqual(disabled.omega, 0.0)
        self.assertEqual(reenabled.state, STATE_TRACK)

    def test_pid_and_outputs_remain_bounded(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["p"]["default"] = 10.0
        parameters["control"]["i"]["default"] = 2.0
        parameters["control"]["max_omega"]["default"] = 0.3
        parameters["control"]["integral_limit"]["default"] = 0.2
        follower = KreiselfahrtFollower(parameters)
        self.follower = follower

        result = None
        for index in range(30):
            result = follower.process_mask(self.line_mask(x=0), 1.0 + index * 0.05, control_enabled=True)

        self.assertLessEqual(abs(result.error), 1.0)
        self.assertLessEqual(abs(result.omega), 0.3)
        self.assertLessEqual(abs(follower._integral), 0.2)
        self.assertGreaterEqual(result.v, 0.0)

    def test_live_parameter_update_keeps_valid_tracking_state(self):
        before = self.reach_track(self.line_mask(), start=1.0)
        parameters = copy.deepcopy(self.parameters)
        parameters["control"]["p"]["default"] = 4.0

        self.follower.update_parameters(parameters)
        after = self.follower.process_mask(self.line_mask(), 1.2, control_enabled=True)

        self.assertEqual(before.tracking_state, STATE_TRACK)
        self.assertEqual(after.state, STATE_TRACK)
        self.assertIsNotNone(after.boundary_x)


if __name__ == "__main__":
    unittest.main()
