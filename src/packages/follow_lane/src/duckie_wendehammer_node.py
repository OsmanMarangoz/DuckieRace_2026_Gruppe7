#!/usr/bin/env python3

import json
import os
from pathlib import Path

import cv2
import numpy as np
import rospy
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from duckie_wendehammer_planner import (
    DuckieModeGate,
    GapRecoveryController,
    GapTargetTracker,
    RawDuckieDetection,
    STATE_STOP,
    STATE_TRACK_TARGET,
    WendehammerGapPlanner,
    load_camera_calibration,
    numeric_bool,
    parameter_value,
    render_birdseye_debug,
    render_camera_debug,
    render_masks_debug,
)


def bool_param(name, default):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def resolve_workspace_path(path):
    if not path:
        return None
    configured = Path(path)
    if configured.exists():
        return str(configured)
    if str(configured).startswith("/workspace/"):
        try:
            repo_root = Path(__file__).resolve().parents[4]
            local = repo_root / str(configured)[len("/workspace/") :]
            if local.exists():
                return str(local)
        except IndexError:
            pass
    return str(configured)


class DuckieWendehammerNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self.node_name = node_name
        self.vehicle_name = os.environ["VEHICLE_NAME"]
        self.config = self.load_config()
        self.parameters = self.config["parameters"]
        self.drive_enabled = bool_param("~drive_enabled", False)
        self.debug_enabled = bool_param("~debug_enabled", True)
        self.publish_stop_when_disabled = bool_param("~publish_stop_when_disabled", False)

        self.planner = WendehammerGapPlanner(self.parameters)
        self.target_tracker = GapTargetTracker(self.planner.settings)
        self.duckie_mode_gate = DuckieModeGate(self.planner.settings)
        self.recovery_controller = GapRecoveryController(self.planner.settings)
        self.gap_mode_active = False
        self.near_duckie_trigger = False
        self.previous_target_x = None

        self.frame_counter = 0
        self.is_running = False
        self.latest_detections = []
        self.last_result = self.planner.empty_result(reason="startup")
        self.last_published_cmd = None
        self.last_command_published = False

        self.calibration = None
        self.calibration_status = "not_loaded"
        self.load_calibration()

        self.model = None
        self.detector_status = "not_loaded"
        self.load_detector()

        camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        cmd_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"
        update_topic = f"/{self.vehicle_name}/update_parameters"
        control_topic = f"/{self.vehicle_name}/duckie_wendehammer/control"
        debug_base = f"/{self.vehicle_name}/debug/duckie_wendehammer"

        self.sub_image = rospy.Subscriber(camera_topic, CompressedImage, self.cb_image, queue_size=1)
        self.sub_update = rospy.Subscriber(update_topic, String, self.cb_update_parameters, queue_size=1)
        self.pub_cmd = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)
        self.sub_control = rospy.Subscriber(control_topic, String, self.cb_control, queue_size=1)
        self.pub_debug = rospy.Publisher(f"{debug_base}/compressed", CompressedImage, queue_size=1)
        self.pub_debug_bev = rospy.Publisher(f"{debug_base}/bev/compressed", CompressedImage, queue_size=1)
        self.pub_debug_camera = rospy.Publisher(f"{debug_base}/camera/compressed", CompressedImage, queue_size=1)
        self.pub_debug_undistorted = rospy.Publisher(f"{debug_base}/undistorted/compressed", CompressedImage, queue_size=1)
        self.pub_debug_masks = rospy.Publisher(f"{debug_base}/masks/compressed", CompressedImage, queue_size=1)
        self.pub_state = rospy.Publisher(f"/{self.vehicle_name}/duckie_wendehammer/state", String, queue_size=1)
        self.pub_debug_json = rospy.Publisher(f"/{self.vehicle_name}/duckie_wendehammer/debug_json", String, queue_size=1)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "[%s] ready camera=%s drive_enabled=%s debug_enabled=%s calibration=%s detector=%s",
            self.node_name,
            camera_topic,
            self.drive_enabled,
            self.debug_enabled,
            self.calibration_status,
            self.detector_status,
        )

    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent / "config" / f"{self.node_name}.json"
        with open(config_path, "r") as handle:
            return json.load(handle)

    def load_calibration(self):
        configured_path = rospy.get_param("~camera_calibration_path", self.config.get("camera_calibration_path"))
        if configured_path:
            configured_path = str(configured_path).format(vehicle=self.vehicle_name)
        self.calibration_path = resolve_workspace_path(configured_path)
        self.calibration = None
        if not self.calibration_path:
            self.calibration_status = "missing_path"
            return
        try:
            self.calibration = load_camera_calibration(self.calibration_path)
        except Exception as exc:
            self.calibration_status = f"load_error:{exc}"
            rospy.logwarn("[%s] camera calibration load failed: %s", self.node_name, exc)
            return
        if self.calibration is None:
            self.calibration_status = "missing_file"
            rospy.logwarn("[%s] camera calibration file missing: %s", self.node_name, self.calibration_path)
        else:
            self.calibration_status = "loaded"
            rospy.loginfo("[%s] loaded camera calibration: %s", self.node_name, self.calibration_path)

    def load_detector(self):
        if YOLO is None:
            self.detector_status = "ultralytics_unavailable"
            rospy.logerr("[%s] ultralytics is not available; Wendehammer drive will stay safe-stopped", self.node_name)
            return

        model_path = rospy.get_param("~model_path", self.config.get("model_path"))
        model_path = resolve_workspace_path(model_path)
        if not model_path or not Path(model_path).exists():
            self.detector_status = "model_file_missing"
            rospy.logerr("[%s] YOLO model missing: %s", self.node_name, model_path)
            return
        try:
            self.model = YOLO(model_path)
            self.detector_status = "loaded"
            rospy.loginfo("[%s] loaded duckie detector: %s", self.node_name, model_path)
        except Exception as exc:
            self.detector_status = f"load_error:{exc}"
            rospy.logerr("[%s] YOLO model load failed: %s", self.node_name, exc)

    def cb_update_parameters(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        if payload.get("node") != self.node_name:
            return
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            return
        self.parameters = parameters
        self.planner.update(parameters)
        self.target_tracker.update_settings(self.planner.settings)
        self.duckie_mode_gate.update_settings(self.planner.settings)
        self.recovery_controller.update_settings(self.planner.settings)
        rospy.loginfo("[%s] parameters updated live", self.node_name)

    def cb_control(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            rospy.logwarn("[%s] ignored invalid control payload", self.node_name)
            return
        action = payload.get("action")
        if action == "set_drive_enabled":
            requested = bool(payload.get("enabled", False))
            if requested and (self.calibration is None or self.model is None):
                rospy.logerr("[%s] drive enable rejected: calibration or detector unavailable", self.node_name)
                requested = False
            self.drive_enabled = requested
            rospy.set_param("~drive_enabled", requested)
            self.target_tracker.reset()
            self.duckie_mode_gate.reset()
            self.recovery_controller.reset()
            self.previous_target_x = None
            if not requested:
                self.pub_cmd.publish(self.make_stop())
                self.last_published_cmd = self.make_stop()
                self.last_command_published = True
            rospy.logwarn("[%s] runtime drive_enabled=%s", self.node_name, requested)
        elif action == "reset_tracker":
            self.target_tracker.reset()
            self.duckie_mode_gate.reset()
            self.recovery_controller.reset()
            self.previous_target_x = None
            rospy.loginfo("[%s] target tracker reset", self.node_name)

    def require_calibration_for_drive(self):
        if rospy.has_param("~require_calibration_for_drive"):
            return bool_param("~require_calibration_for_drive", True)
        return numeric_bool(parameter_value(self.parameters, "calibration", "require_for_drive", 1))

    def use_raw_when_missing_calibration(self):
        return numeric_bool(parameter_value(self.parameters, "calibration", "use_raw_when_missing", 1))

    def undistort_image(self, image):
        if self.calibration is None:
            if self.use_raw_when_missing_calibration():
                return image, False, self.calibration_status
            return None, False, self.calibration_status
        if not self.calibration.usable_for(image):
            return image if self.use_raw_when_missing_calibration() else None, False, "image_size_mismatch"
        new_matrix = self.calibration.new_camera_matrix
        if new_matrix is None:
            new_matrix = self.calibration.camera_matrix
        undistorted = cv2.undistort(image, self.calibration.camera_matrix, self.calibration.dist_coeffs, None, new_matrix)
        return undistorted, True, "loaded"

    def detect_duckies(self, image, now_sec):
        if self.model is None:
            self.latest_detections = []
            return []

        self.frame_counter += 1
        stride = int(parameter_value(self.parameters, "model", "frame_stride", 3))
        # Run once on startup and then strictly at the configured cadence. The
        # previous empty-list shortcut ran YOLO on every frame on an empty road.
        should_detect = self.frame_counter == 1 or self.frame_counter % max(stride, 1) == 0
        if should_detect:
            conf = float(parameter_value(self.parameters, "model", "conf", 0.25))
            imgsz = int(parameter_value(self.parameters, "model", "imgsz", 640))
            try:
                results = self.model.predict(image, imgsz=imgsz, conf=conf, verbose=False)
                detections = []
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                        score = float(box.conf[0])
                        detections.append(RawDuckieDetection(x1, y1, x2, y2, score, now_sec))
                self.latest_detections = detections
                self.detector_status = "loaded"
            except Exception as exc:
                self.detector_status = f"predict_error:{exc}"
                rospy.logwarn_throttle(2.0, "[%s] duckie detector failed: %s", self.node_name, exc)

        timeout = float(parameter_value(self.parameters, "model", "track_timeout", 0.75))
        self.latest_detections = [
            detection
            for detection in self.latest_detections
            if now_sec - detection.stamp_sec <= timeout
        ]
        return list(self.latest_detections)

    def make_twist(self, result):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = float(result.v)
        msg.omega = float(result.omega)
        return msg

    def make_stop(self):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = 0.0
        msg.omega = 0.0
        return msg

    def publish_motion_command(self, msg):
        self.last_published_cmd = None
        self.last_command_published = False
        if self.drive_enabled:
            self.pub_cmd.publish(msg)
            self.last_published_cmd = msg
            self.last_command_published = True
            return
        if self.publish_stop_when_disabled:
            stop = self.make_stop()
            self.pub_cmd.publish(stop)
            self.last_published_cmd = stop
            self.last_command_published = True

    def safety_override(self, result, calibration_ok):
        if not self.drive_enabled:
            return result
        if self.model is None:
            return result.with_state(STATE_STOP, "duckie_detector_unavailable", speed_level="stop", v=0.0, omega=0.0, has_target=False)
        if self.detector_status.startswith("predict_error"):
            return result.with_state(STATE_STOP, "duckie_detector_predict_error", speed_level="stop", v=0.0, omega=0.0, has_target=False)
        if self.require_calibration_for_drive() and not calibration_ok:
            return result.with_state(STATE_STOP, "missing_or_invalid_camera_calibration", speed_level="stop", v=0.0, omega=0.0, has_target=False)
        return result

    def state_payload(self, result, calibration_ok, calibration_reason):
        payload = result.status_payload(include_candidates=True)
        payload.update(
            {
                "drive_enabled": bool(self.drive_enabled),
                "debug_enabled": bool(self.debug_enabled),
                "command_published": bool(self.last_command_published),
                "published_v": None if self.last_published_cmd is None else float(self.last_published_cmd.v),
                "published_omega": None if self.last_published_cmd is None else float(self.last_published_cmd.omega),
                "calibration_ok": bool(calibration_ok),
                "calibration_status": calibration_reason,
                "calibration_path": self.calibration_path,
                "detector_status": self.detector_status,
                "gap_mode_active": bool(self.gap_mode_active),
                "near_duckie_trigger": bool(self.near_duckie_trigger),
                "recovery_phase": self.recovery_controller.phase,
                "no_gap_frames": int(self.recovery_controller.no_gap_frames),
                "no_gap_confirm_frames": int(
                    self.planner.settings.recovery_no_gap_confirm_frames
                ),
            }
        )
        return payload

    def publish_state(self, result, calibration_ok, calibration_reason):
        payload = self.state_payload(result, calibration_ok, calibration_reason)
        self.pub_state.publish(String(data=json.dumps(payload)))
        if self.debug_enabled:
            self.pub_debug_json.publish(String(data=json.dumps(payload)))

    def wants_debug_image(self, publisher):
        return self.debug_enabled and publisher.get_num_connections() > 0

    def publish_debug_image_group(self, publishers, image):
        active_publishers = [publisher for publisher in publishers if self.wants_debug_image(publisher)]
        if not active_publishers:
            return
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(encoded).tobytes()
        for publisher in active_publishers:
            publisher.publish(msg)

    def publish_debug_images(self, raw_image, analysis_image, birdseye, white_mask, yellow_mask, raw_duckies, result, matrix, calibration_label):
        if self.wants_debug_image(self.pub_debug) or self.wants_debug_image(self.pub_debug_bev):
            debug_bev = render_birdseye_debug(
                birdseye,
                white_mask,
                yellow_mask,
                result,
                text_lines=[
                    f"{result.state} {result.reason}",
                    f"gap={result.chosen_gap_width_px} v={result.v:.3f} w={result.omega:.2f}",
                    f"duckies={len(result.duckies)} rejected={len(result.rejected_duckies)}",
                ],
            )
            self.publish_debug_image_group([self.pub_debug, self.pub_debug_bev], debug_bev)
        if self.wants_debug_image(self.pub_debug_camera) or self.wants_debug_image(self.pub_debug_undistorted):
            debug_camera = render_camera_debug(
                analysis_image,
                raw_duckies,
                result,
                matrix,
                calibration_label=calibration_label,
            )
            self.publish_debug_image_group(
                [self.pub_debug_camera, self.pub_debug_undistorted], debug_camera
            )
        if self.wants_debug_image(self.pub_debug_masks):
            self.publish_debug_image_group(
                [self.pub_debug_masks], render_masks_debug(white_mask, yellow_mask, result)
            )

    def cb_image(self, image_msg):
        if self.is_running:
            return
        self.is_running = True
        now = rospy.Time.now()
        now_sec = now.to_sec()
        try:
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if raw_image is None:
                result = self.planner.empty_result(state=STATE_STOP, reason="image_decode_failed")
                self.publish_motion_command(self.make_stop())
                self.publish_state(result, False, "image_decode_failed")
                return

            analysis_image, calibration_ok, calibration_reason = self.undistort_image(raw_image)
            if analysis_image is None:
                result = self.planner.empty_result(state=STATE_STOP, reason="camera_calibration_required")
                self.publish_motion_command(self.make_stop())
                self.publish_state(result, False, calibration_reason)
                return

            raw_duckies = self.detect_duckies(analysis_image, now_sec)
            self.gap_mode_active = self.duckie_mode_gate.update(raw_duckies)
            self.near_duckie_trigger = self.duckie_mode_gate.near_now
            result, birdseye, white_mask, yellow_mask, matrix = self.planner.plan_image(
                analysis_image,
                raw_duckies,
                now_sec,
                previous_target_x=self.previous_target_x,
                gap_mode_active=self.gap_mode_active,
            )
            result = self.target_tracker.update(result)
            result = self.recovery_controller.update(result, now_sec, self.gap_mode_active)
            result = self.safety_override(result, calibration_ok)

            if result.state == STATE_TRACK_TARGET and result.has_target:
                self.previous_target_x = result.target_x
            else:
                self.previous_target_x = None

            cmd = self.make_twist(result)
            self.publish_motion_command(cmd)
            self.last_result = result
            self.publish_state(result, calibration_ok, calibration_reason)
            debug_stride = max(
                1,
                int(parameter_value(self.parameters, "model", "debug_frame_stride", 2)),
            )
            if self.frame_counter % debug_stride == 0:
                self.publish_debug_images(
                    raw_image,
                    analysis_image,
                    birdseye,
                    white_mask,
                    yellow_mask,
                    raw_duckies,
                    result,
                    matrix,
                    f"calib={calibration_reason}",
                )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "[%s] failed: %s", self.node_name, exc)
            result = self.planner.empty_result(state=STATE_STOP, reason=f"exception:{exc}")
            self.last_result = result
            self.publish_motion_command(self.make_stop())
            self.publish_state(result, False, "exception")
        finally:
            self.is_running = False

    def shutdown(self):
        rospy.loginfo("[%s] shutdown drive_enabled=%s", self.node_name, self.drive_enabled)
        if self.drive_enabled or self.publish_stop_when_disabled:
            self.pub_cmd.publish(self.make_stop())

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckieWendehammerNode("duckie_wendehammer_node")
    node.run()
