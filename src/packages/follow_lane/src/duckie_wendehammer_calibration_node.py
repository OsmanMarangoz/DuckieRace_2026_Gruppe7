#!/usr/bin/env python3

import json
import os
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from duckie_wendehammer_planner import save_camera_calibration


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
            return str(repo_root / str(configured)[len("/workspace/") :])
        except IndexError:
            pass
    return str(configured)


class DuckieWendehammerCalibrationNode:
    def __init__(self):
        rospy.init_node("duckie_wendehammer_calibration_node")
        self.vehicle_name = os.environ["VEHICLE_NAME"]
        self.board_cols = int(rospy.get_param("~board_cols", 6))
        self.board_rows = int(rospy.get_param("~board_rows", 4))
        self.square_size = float(rospy.get_param("~square_size", 1.0))
        self.required_samples = int(rospy.get_param("~required_samples", 20))
        self.auto_capture = bool_param("~auto_capture", True)
        self.autosave = bool_param("~autosave", True)
        self.sample_interval_sec = float(rospy.get_param("~sample_interval_sec", 0.45))
        self.min_corner_shift_px = float(rospy.get_param("~min_corner_shift_px", 14.0))

        default_path = (
            f"/workspace/src/packages/follow_lane/data/camera_calibration_{self.vehicle_name}.json"
        )
        self.output_path = resolve_workspace_path(rospy.get_param("~output_path", default_path))

        self.obj_template = np.zeros((self.board_rows * self.board_cols, 3), np.float32)
        self.obj_template[:, :2] = np.mgrid[0 : self.board_cols, 0 : self.board_rows].T.reshape(-1, 2)
        self.obj_template *= self.square_size
        self.objpoints = []
        self.imgpoints = []
        self.image_size = None
        self.last_accepted_corners = None
        self.last_sample_time = rospy.Time(0)
        self.last_saved_sample_count = 0
        self.calibration_complete = False
        self.sample_metrics = []
        self.latest_status = {
            "state": "waiting_for_board",
            "samples": 0,
            "required_samples": self.required_samples,
        }

        base_topic = f"/{self.vehicle_name}/duckie_wendehammer"
        debug_topic = f"/{self.vehicle_name}/debug/duckie_wendehammer/calibration/compressed"
        camera_topic = f"/{self.vehicle_name}/camera_node/image/compressed"

        self.command_sub = rospy.Subscriber(f"{base_topic}/calibration_command", String, self.cb_command, queue_size=1)
        self.image_sub = rospy.Subscriber(camera_topic, CompressedImage, self.cb_image, queue_size=1)
        self.status_pub = rospy.Publisher(f"{base_topic}/calibration_status", String, queue_size=1)
        self.debug_pub = rospy.Publisher(debug_topic, CompressedImage, queue_size=1)

        rospy.on_shutdown(self.shutdown)
        rospy.loginfo(
            "[duckie_wendehammer_calibration_node] board=%dx%d required=%d output=%s auto_capture=%s",
            self.board_cols,
            self.board_rows,
            self.required_samples,
            self.output_path,
            self.auto_capture,
        )

    def cb_command(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = {"action": msg.data.strip()}
        action = payload.get("action")
        if action == "capture":
            self.auto_capture = False
            self.latest_status["manual_capture_requested"] = True
        elif action == "auto_on":
            self.auto_capture = True
        elif action == "auto_off":
            self.auto_capture = False
        elif action == "reset":
            self.reset_samples()
        elif action == "save":
            self.try_calibrate_and_save(force=True)

    def reset_samples(self):
        self.objpoints = []
        self.imgpoints = []
        self.image_size = None
        self.last_accepted_corners = None
        self.last_sample_time = rospy.Time(0)
        self.last_saved_sample_count = 0
        self.calibration_complete = False
        self.sample_metrics = []
        self.latest_status = {
            "state": "reset",
            "samples": 0,
            "required_samples": self.required_samples,
        }

    def should_accept_sample(self, corners, now):
        if self.calibration_complete:
            return False
        if self.latest_status.pop("manual_capture_requested", False):
            return True
        if not self.auto_capture:
            return False
        if (now - self.last_sample_time).to_sec() < self.sample_interval_sec:
            return False
        if self.last_accepted_corners is None:
            return True
        shift = float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - self.last_accepted_corners.reshape(-1, 2), axis=1)))
        return shift >= self.min_corner_shift_px

    def sample_metric(self, corners, image_size):
        pts = corners.reshape(-1, 2)
        width, height = image_size
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        board_width = max(float(max_x - min_x), 1.0)
        board_height = max(float(max_y - min_y), 1.0)
        top_width = float(np.linalg.norm(pts[self.board_cols - 1] - pts[0]))
        bottom_width = float(np.linalg.norm(pts[-1] - pts[-self.board_cols]))
        left_height = float(np.linalg.norm(pts[-self.board_cols] - pts[0]))
        right_height = float(np.linalg.norm(pts[-1] - pts[self.board_cols - 1]))
        perspective_tilt = abs(top_width - bottom_width) / max(top_width, bottom_width, 1.0)
        perspective_tilt += abs(left_height - right_height) / max(left_height, right_height, 1.0)
        return {
            "center_x": float(np.mean(pts[:, 0]) / max(float(width), 1.0)),
            "center_y": float(np.mean(pts[:, 1]) / max(float(height), 1.0)),
            "area_fraction": float((board_width * board_height) / max(float(width * height), 1.0)),
            "tilt": float(perspective_tilt),
        }

    def diversity_status(self):
        if not self.sample_metrics:
            return {
                "x_bins": 0,
                "y_bins": 0,
                "size_bins": 0,
                "tilt_bins": 0,
                "tilted_samples": 0,
                "diversity_ok": False,
            }

        def bins(values, edges):
            return len({int(np.digitize(value, edges)) for value in values})

        x_bins = bins([item["center_x"] for item in self.sample_metrics], [0.35, 0.50, 0.65])
        y_bins = bins([item["center_y"] for item in self.sample_metrics], [0.35, 0.50, 0.65])
        size_bins = bins([item["area_fraction"] for item in self.sample_metrics], [0.08, 0.16, 0.26])
        tilt_bins = bins([item["tilt"] for item in self.sample_metrics], [0.04, 0.10, 0.18])
        tilted_samples = sum(1 for item in self.sample_metrics if item["tilt"] >= 0.10)
        return {
            "x_bins": int(x_bins),
            "y_bins": int(y_bins),
            "size_bins": int(size_bins),
            "tilt_bins": int(tilt_bins),
            "tilted_samples": int(tilted_samples),
            "diversity_ok": bool(
                len(self.sample_metrics) >= self.required_samples
                and x_bins >= 3
                and y_bins >= 3
                and size_bins >= 3
                and tilt_bins >= 3
                and tilted_samples >= 6
            ),
        }

    def accept_sample(self, corners, image_size, now):
        self.objpoints.append(self.obj_template.copy())
        self.imgpoints.append(corners.copy())
        self.image_size = image_size
        self.last_accepted_corners = corners.copy()
        self.last_sample_time = now
        self.sample_metrics.append(self.sample_metric(corners, image_size))

    def try_calibrate_and_save(self, force=False):
        if len(self.objpoints) < self.required_samples and not force:
            return None
        if len(self.objpoints) < 3:
            self.latest_status["state"] = "need_more_samples"
            return None
        if self.image_size is None:
            self.latest_status["state"] = "missing_image_size"
            return None

        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            None,
            None,
        )
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            self.image_size,
            0,
            self.image_size,
        )
        payload = save_camera_calibration(
            self.output_path,
            self.image_size,
            camera_matrix,
            dist_coeffs,
            new_camera_matrix,
            roi,
            rms,
            self.board_cols,
            self.board_rows,
            self.square_size,
        )
        self.latest_status.update(
            {
                "state": "saved",
                "rms_error": float(rms),
                "output_path": self.output_path,
                "samples": len(self.objpoints),
                "diversity": self.diversity_status(),
                "calibration": payload,
            }
        )
        self.last_saved_sample_count = len(self.objpoints)
        self.calibration_complete = True
        self.auto_capture = False
        rospy.loginfo("[duckie_wendehammer_calibration_node] saved calibration rms=%.4f to %s", rms, self.output_path)
        return payload

    def cb_image(self, image_msg):
        now = rospy.Time.now()
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.latest_status["state"] = "image_decode_failed"
            self.publish_status()
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (int(gray.shape[1]), int(gray.shape[0]))
        pattern_size = (self.board_cols, self.board_rows)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        display = image.copy()

        accepted = False
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            if self.should_accept_sample(corners, now):
                self.accept_sample(corners, image_size, now)
                accepted = True
            state = "accepted_sample" if accepted else "board_found"
        else:
            state = "waiting_for_board"

        if self.calibration_complete:
            state = "saved"

        if len(self.objpoints) >= self.required_samples and self.autosave and len(self.objpoints) != self.last_saved_sample_count:
            self.try_calibrate_and_save()
        else:
            self.latest_status.update(
                {
                    "state": state,
                    "board_found": bool(found),
                    "accepted_sample": bool(accepted),
                    "samples": len(self.objpoints),
                    "required_samples": self.required_samples,
                    "diversity": self.diversity_status(),
                    "board_cols": self.board_cols,
                    "board_rows": self.board_rows,
                    "square_size": self.square_size,
                    "auto_capture": bool(self.auto_capture),
                    "autosave": bool(self.autosave),
                    "output_path": self.output_path,
                }
            )

        self.draw_status(display, found, accepted)
        self.publish_debug(display)
        self.publish_status()

    def draw_status(self, image, found, accepted):
        lines = [
            f"board {self.board_cols}x{self.board_rows} found={found} accepted={accepted}",
            f"samples {len(self.objpoints)}/{self.required_samples} auto={self.auto_capture} autosave={self.autosave}",
            "diversity x{x_bins} y{y_bins} size{size_bins} tilt{tilt_bins} ok={diversity_ok}".format(
                **self.diversity_status()
            ),
            f"out {self.output_path}",
        ]
        x, y = 8, 22
        for line in lines:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            y += 20

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.latest_status)))

    def publish_debug(self, image):
        if self.debug_pub.get_num_connections() <= 0:
            return
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(encoded).tobytes()
        self.debug_pub.publish(msg)

    def shutdown(self):
        rospy.loginfo("[duckie_wendehammer_calibration_node] shutdown")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = DuckieWendehammerCalibrationNode()
    node.run()
