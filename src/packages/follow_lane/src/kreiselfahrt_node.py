#!/usr/bin/env python3

"""ROS wrapper for the deliberately simple virtual-yellow-line controller."""

import copy
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospkg
import rospy
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse

# catkin executes this file through a devel-space wrapper. Prefer the real source
# directory so the adjacent pure-Python helper is imported as a module, not as a
# second executable wrapper.
_SOURCE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if _SOURCE_DIRECTORY not in sys.path:
    sys.path.insert(0, _SOURCE_DIRECTORY)

from kreiselfahrt_logic import (
    STATE_CAMERA_STALE,
    STATE_DISABLED,
    KreiselfahrtFollower,
    draw_camera_debug,
    draw_control_debug,
    draw_virtual_debug,
    draw_yellow_debug,
    parameter_value,
)
from kreiselfahrt_duckie_detector import DuckieYoloDetector


class KreiselfahrtNode:
    def __init__(self):
        rospy.init_node("kreiselfahrt_node")
        environment_vehicle = os.environ.get("VEHICLE_NAME", "").strip()
        parameter_vehicle = str(rospy.get_param("~vehicle_name", "")).strip()
        self._vehicle_name = environment_vehicle or parameter_vehicle
        if not self._vehicle_name:
            raise RuntimeError("VEHICLE_NAME oder der private Parameter ~vehicle_name fehlt")
        if environment_vehicle and parameter_vehicle and environment_vehicle != parameter_vehicle:
            rospy.logwarn(
                "[kreiselfahrt_node] Veralteten ~vehicle_name=%s ignoriert; VEHICLE_NAME=%s wird benutzt",
                parameter_vehicle,
                environment_vehicle,
            )

        package_path = rospkg.RosPack().get_path("follow_lane")
        self._config_path = rospy.get_param(
            "~config_path", os.path.join(package_path, "config", "kreiselfahrt_node.json")
        )
        self._config = self._load_config()
        self._parameters = self._config["parameters"]
        self._follower = KreiselfahrtFollower(self._parameters)

        configured_model_path = rospy.get_param(
            "~duckie_model_path",
            self._config.get(
                "duckie_model_path",
                os.path.join(package_path, "src", "model", "duckie_yolov8n_640.pt"),
            ),
        )
        if configured_model_path and not os.path.isfile(configured_model_path):
            local_model_path = os.path.join(
                package_path, "src", "model", os.path.basename(str(configured_model_path))
            )
            if os.path.isfile(local_model_path):
                configured_model_path = local_model_path
        self._duckie_detector = DuckieYoloDetector(configured_model_path)

        self._lock = threading.RLock()
        self._enabled = False
        self._ai_detection_enabled = False
        self._ai_duckie_count = 0
        self._last_result = None
        self._last_frame_monotonic = None
        self._last_debug_monotonic = 0.0
        self._last_fps_monotonic = None
        self._fps = 0.0
        self._shutdown_started = False

        default_camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        default_command_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        # Topic overrides use normal ROS remapping. Private topic parameters are
        # deliberately not read because ROS1 keeps them after a process exits and
        # can otherwise silently reconnect a later run to another Duckiebot.
        camera_topic = rospy.resolve_name(default_camera_topic)
        command_topic = rospy.resolve_name(default_command_topic)

        self._pub_command = rospy.Publisher(command_topic, Twist2DStamped, queue_size=1)
        self._pub_state = rospy.Publisher(
            f"/{self._vehicle_name}/kreiselfahrt/state", String, queue_size=1
        )
        debug_base = f"/{self._vehicle_name}/debug/kreiselfahrt"
        self._pub_debug_camera = rospy.Publisher(
            f"{debug_base}/camera/compressed", CompressedImage, queue_size=1
        )
        self._pub_debug_yellow = rospy.Publisher(
            f"{debug_base}/yellow_raw/compressed", CompressedImage, queue_size=1
        )
        self._pub_debug_virtual = rospy.Publisher(
            f"{debug_base}/yellow_virtual/compressed", CompressedImage, queue_size=1
        )
        self._pub_debug_control = rospy.Publisher(
            f"{debug_base}/control/compressed", CompressedImage, queue_size=1
        )

        self._sub_camera = rospy.Subscriber(
            camera_topic, CompressedImage, self._camera_callback, queue_size=1, buff_size=2**22
        )
        self._sub_parameters = rospy.Subscriber(
            f"/{self._vehicle_name}/update_parameters",
            String,
            self._parameter_callback,
            queue_size=1,
        )
        self._enable_service = rospy.Service(
            f"/{self._vehicle_name}/kreiselfahrt/set_enabled", SetBool, self._enable_callback
        )
        self._ai_enable_service = rospy.Service(
            f"/{self._vehicle_name}/kreiselfahrt/set_ai_detection",
            SetBool,
            self._ai_enable_callback,
        )

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "[kreiselfahrt_node] bereit: Kamera=%s, Kommando=%s, Fahrfreigabe AUS",
            camera_topic,
            command_topic,
        )

    def _load_config(self):
        with open(self._config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _parameter_callback(self, message):
        try:
            payload = json.loads(message.data)
            if payload.get("node") != "kreiselfahrt_node":
                return
            parameters = payload["parameters"]
            with self._lock:
                self._follower.update_parameters(copy.deepcopy(parameters))
                self._parameters = copy.deepcopy(parameters)
            rospy.loginfo("[kreiselfahrt_node] Live-Parameter aktualisiert")
        except (KeyError, TypeError, ValueError) as error:
            rospy.logwarn("[kreiselfahrt_node] Ungültiges Parameter-Update ignoriert: %s", error)

    def _enable_callback(self, request):
        with self._lock:
            self._enabled = bool(request.data)
            self._follower.reset_control()
        if not request.data:
            self._publish_command(0.0, 0.0)
        state = "EIN" if request.data else "AUS"
        rospy.logwarn("[kreiselfahrt_node] Fahrfreigabe %s", state)
        return SetBoolResponse(success=True, message=f"Fahrfreigabe {state}")

    def _ai_enable_callback(self, request):
        requested = bool(request.data)
        if requested:
            loaded, message = self._duckie_detector.ensure_loaded()
            if not loaded:
                rospy.logerr("[kreiselfahrt_node] KI-Modus abgelehnt: %s", message)
                return SetBoolResponse(success=False, message=message)

        with self._lock:
            self._ai_detection_enabled = requested
            self._ai_duckie_count = 0
            self._follower.reset_control()
        mode = "KI-Duckies + HSV-Linie" if requested else "HSV"
        rospy.logwarn("[kreiselfahrt_node] Erkennungsmodus: %s", mode)
        return SetBoolResponse(success=True, message=f"Erkennungsmodus: {mode}")

    @staticmethod
    def _decode_image(message):
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def _camera_callback(self, message):
        image = self._decode_image(message)
        if image is None:
            rospy.logwarn_throttle(2.0, "[kreiselfahrt_node] Kamerabild konnte nicht dekodiert werden")
            return

        now = time.monotonic()
        try:
            with self._lock:
                use_ai_detection = self._ai_detection_enabled
                ai_confidence = float(
                    parameter_value(self._parameters, "ai_detection", "confidence", 0.25)
                )

            duckie_camera_mask = None
            detected_duckies = 0
            if use_ai_detection:
                duckie_camera_mask, detected_duckies = self._duckie_detector.detect_mask(
                    image, ai_confidence
                )

            with self._lock:
                # A mode change may have happened while YOLO was running. In that
                # case, honor the newest mode and discard this frame's optional mask.
                use_ai_detection = use_ai_detection and self._ai_detection_enabled
                if use_ai_detection:
                    birdseye = self._follower.birdseye(image)
                    hsv_mask = self._follower.yellow_mask(birdseye)
                    duckie_birdseye_mask = self._follower.birdseye(duckie_camera_mask)
                    duckie_birdseye_mask = np.where(
                        duckie_birdseye_mask > 0, 255, 0
                    ).astype(np.uint8)
                    combined_mask = cv2.bitwise_or(hsv_mask, duckie_birdseye_mask)
                    result = self._follower.process_mask(
                        combined_mask,
                        now,
                        control_enabled=self._enabled,
                        birdseye=birdseye,
                    )
                    self._ai_duckie_count = detected_duckies
                else:
                    result = self._follower.process_image(
                        image, now, control_enabled=self._enabled
                    )
                    self._ai_duckie_count = 0
                self._last_result = result
                self._last_frame_monotonic = now
                settings = self._follower.settings
                publish_debug = now - self._last_debug_monotonic >= 1.0 / settings.debug_rate
                if publish_debug:
                    self._last_debug_monotonic = now

                if self._last_fps_monotonic is not None:
                    instant_fps = 1.0 / max(1e-6, now - self._last_fps_monotonic)
                    self._fps = instant_fps if self._fps == 0.0 else 0.2 * instant_fps + 0.8 * self._fps
                self._last_fps_monotonic = now

                if publish_debug:
                    debug_images = (
                        draw_camera_debug(image, settings),
                        draw_yellow_debug(result),
                        draw_virtual_debug(result, settings),
                        draw_control_debug(result, settings),
                    )
                else:
                    debug_images = None
        except Exception as error:  # keep the command watchdog alive on malformed frames/parameters
            rospy.logerr_throttle(1.0, "[kreiselfahrt_node] Bildverarbeitung fehlgeschlagen: %s", error)
            return

        if debug_images is not None:
            publishers = (
                self._pub_debug_camera,
                self._pub_debug_yellow,
                self._pub_debug_virtual,
                self._pub_debug_control,
            )
            for publisher, debug_image in zip(publishers, debug_images):
                if publisher.get_num_connections() > 0:
                    self._publish_debug_image(publisher, debug_image, message.header)

    @staticmethod
    def _publish_debug_image(publisher, image, source_header):
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            return
        message = CompressedImage()
        message.header = source_header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        publisher.publish(message)

    def _publish_command(self, speed, omega):
        message = Twist2DStamped()
        message.header.stamp = rospy.Time.now()
        message.v = float(speed)
        message.omega = float(omega)
        self._pub_command.publish(message)

    def _current_output(self, now):
        with self._lock:
            enabled = self._enabled
            result = self._last_result
            settings = self._follower.settings
            frame_age = (
                None
                if self._last_frame_monotonic is None
                else max(0.0, now - self._last_frame_monotonic)
            )

            if not enabled:
                state, speed, omega = STATE_DISABLED, 0.0, 0.0
                self._follower.reset_control()
            elif result is None or frame_age is None or frame_age > settings.camera_timeout:
                state, speed, omega = STATE_CAMERA_STALE, 0.0, 0.0
                self._follower.reset_control()
            else:
                state, speed, omega = result.state, result.v, result.omega

            status = {
                "enabled": enabled,
                "ai_detection_enabled": bool(self._ai_detection_enabled),
                "detection_mode": (
                    "AI_DUCKIES_PLUS_HSV" if self._ai_detection_enabled else "HSV"
                ),
                "ai_detector_status": self._duckie_detector.status,
                "ai_duckie_count": int(self._ai_duckie_count),
                "state": state,
                "tracking_state": None if result is None else result.tracking_state,
                "confidence": 0.0 if result is None else result.confidence,
                "boundary_x_raw": None if result is None else result.boundary_x_raw,
                "boundary_x": None if result is None else result.boundary_x,
                "error": 0.0 if result is None else result.error,
                "v": float(speed),
                "omega": float(omega),
                "yellow_age": None if result is None else result.yellow_age,
                "camera_age": frame_age,
                "fps": self._fps,
            }
            command_rate = settings.command_rate
        return speed, omega, status, command_rate

    def run(self):
        current_rate = None
        rate = None
        while not rospy.is_shutdown():
            speed, omega, status, desired_rate = self._current_output(time.monotonic())
            self._publish_command(speed, omega)
            self._pub_state.publish(String(data=json.dumps(status, separators=(",", ":"))))

            if rate is None or current_rate != desired_rate:
                current_rate = desired_rate
                rate = rospy.Rate(current_rate)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _shutdown(self):
        if self._shutdown_started:
            return
        self._shutdown_started = True
        with self._lock:
            self._enabled = False
            self._follower.reset_control()
        self._publish_command(0.0, 0.0)
        rospy.loginfo("[kreiselfahrt_node] beendet, Stoppkommando gesendet")


if __name__ == "__main__":
    node = KreiselfahrtNode()
    node.run()
