#!/usr/bin/env python3

import copy
import json
import os
import threading
import time
import unittest

import cv2
import numpy as np
import rospkg
import rospy
import rostest
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool


class KreiselfahrtRosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rospy.init_node("test_kreiselfahrt_ros", anonymous=True)
        cls._lock = threading.Lock()
        cls._last_command = None
        cls._last_state = {}
        cls._camera = rospy.Publisher(
            "/test/kreiselfahrt/camera/compressed", CompressedImage, queue_size=1
        )
        cls._parameters = rospy.Publisher("/testbot/update_parameters", String, queue_size=1)
        cls._command_sub = rospy.Subscriber(
            "/test/kreiselfahrt/cmd", Twist2DStamped, cls._command_callback, queue_size=1
        )
        cls._state_sub = rospy.Subscriber(
            "/testbot/kreiselfahrt/state", String, cls._state_callback, queue_size=1
        )
        rospy.wait_for_service("/testbot/kreiselfahrt/set_enabled", timeout=10.0)
        cls._enable = rospy.ServiceProxy("/testbot/kreiselfahrt/set_enabled", SetBool)

    @classmethod
    def _command_callback(cls, message):
        with cls._lock:
            cls._last_command = message

    @classmethod
    def _state_callback(cls, message):
        with cls._lock:
            cls._last_state = json.loads(message.data)

    @classmethod
    def _wait_for(cls, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not rospy.is_shutdown():
            with cls._lock:
                if predicate(cls._last_state, cls._last_command):
                    return
            rospy.sleep(0.03)
        with cls._lock:
            raise AssertionError(
                f"Bedingung nicht erreicht; state={cls._last_state}, command={cls._last_command}"
            )

    @classmethod
    def _publish_black_frame(cls):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        assert success
        message = CompressedImage()
        message.header.stamp = rospy.Time.now()
        message.format = "jpeg"
        message.data = encoded.tobytes()
        cls._camera.publish(message)

    def test_safe_lifecycle_and_live_parameters(self):
        self._wait_for(
            lambda state, command: state.get("state") == "DISABLED"
            and command is not None
            and command.v == 0.0
            and command.omega == 0.0
        )

        response = self._enable(True)
        self.assertTrue(response.success)
        for _ in range(4):
            self._publish_black_frame()
            rospy.sleep(0.04)
        self._wait_for(
            lambda state, command: state.get("state") == "SEARCH"
            and command is not None
            and command.v > 0.0
            and command.omega > 0.0
        )

        package = rospkg.RosPack().get_path("follow_lane")
        with open(os.path.join(package, "config", "kreiselfahrt_node.json"), "r") as handle:
            parameters = json.load(handle)["parameters"]
        parameters = copy.deepcopy(parameters)
        parameters["control"]["search_omega"]["default"] = 0.4
        parameters["safety"]["camera_timeout"]["default"] = 0.2
        payload = {"node": "kreiselfahrt_node", "parameters": parameters}
        self._parameters.publish(String(data=json.dumps(payload)))
        rospy.sleep(0.15)
        for _ in range(3):
            self._publish_black_frame()
            rospy.sleep(0.04)
        self._wait_for(
            lambda state, command: state.get("state") == "SEARCH"
            and command is not None
            and abs(command.omega - 0.4) < 0.05
        )

        self._wait_for(
            lambda state, command: state.get("state") == "CAMERA_STALE"
            and command is not None
            and command.v == 0.0
            and command.omega == 0.0,
            timeout=3.0,
        )

        response = self._enable(False)
        self.assertTrue(response.success)
        self._wait_for(
            lambda state, command: state.get("state") == "DISABLED"
            and command is not None
            and command.v == 0.0
            and command.omega == 0.0
        )


if __name__ == "__main__":
    rostest.rosrun("follow_lane", "kreiselfahrt_ros", KreiselfahrtRosTest)
