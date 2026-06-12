#!/usr/bin/env python3

import os
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs import msg
from std_msgs.msg import Int32, String
from switch_control_node import ControlType
import json
import util

# built from source: https://github.com/AprilRobotics/apriltag
import apriltag


class DetectAprilTagNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)

        self._vehicle_name = os.environ['VEHICLE_NAME']

        # Camera intrinsics — update these to match your actual camera calibration
        # These are rough defaults for the Duckiebot camera at 640x480
        self.fx = 336.0
        self.fy = 336.0
        self.cx = 320.0
        self.cy = 240.0
        self.tag_size = 0.065   # meters — physical size of the AprilTag

        self.is_running = False
        self.counter = 0
        self.display_image = None
        self._last_detected_id = -1  # Remember last detected tag ID
        self.tag_timeout = 3.0                       # NEU: default in Sekunden
        self._last_detection_time = rospy.Time(0)

        util.init_parameters(node_name, self.cbUpdateParameters)

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.sub_image = rospy.Subscriber(self._camera_topic, CompressedImage, self.cbDetectTag, queue_size=1)

        # Publishes the ID of the most recently detected tag (-1 if none)
        self.pub_tag_id = rospy.Publisher(f'/{self._vehicle_name}/apriltag/id', Int32, queue_size=1)

        # Publishes full detection info as JSON string: [{id, center_x, center_y}]
        self.pub_tag_info = rospy.Publisher(f'/{self._vehicle_name}/apriltag/detections', String, queue_size=1)

        # Debug image with tag overlays
        self.pub_debug = rospy.Publisher(f'/{self._vehicle_name}/apriltag/debug/compressed', CompressedImage, queue_size=1)

        # C extension API: apriltag.apriltag("familyname")
        self.detector = apriltag.apriltag("tagStandard52h13")

        self._current_action = ""
        self.sub_action = rospy.Subscriber(
            f'/{self._vehicle_name}/decision/action', String,
            self.cbAction, queue_size=1)

        self._current_mode = 1
        self.sub_mode = rospy.Subscriber(
            f'/{self._vehicle_name}/switch/control', Int32,
            self.cbMode, queue_size=1)

    def cbMode(self, msg):
        self._current_mode = msg.data

    def cbAction(self, msg):
        self._current_action = msg.data

    def cbUpdateParameters(self, parameters):
        try:
            self.fx = parameters["camera"]["fx"]["default"]
            self.fy = parameters["camera"]["fy"]["default"]
            self.cx = parameters["camera"]["cx"]["default"]
            self.cy = parameters["camera"]["cy"]["default"]
            self.tag_size = parameters["camera"]["tag_size"]["default"]
            self.tag_timeout = parameters["detection"]["tag_timeout"]["default"]
        except (KeyError, TypeError):
            pass  # use defaults set in __init__

    def cbDetectTag(self, image_msg):
        if self.counter <= 3:
            self.counter += 1
            return

        if self.is_running:
            return

        self.is_running = True
        self.counter = 0

        np_arr = np.frombuffer(image_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(gray)

        # The C extension returns a list of dicts with keys:
        # 'id', 'center', 'lb-rb-rt-lt' (corners), 'hamming', 'margin'
        results = []
        for det in detections:
            tag_id = det['id']
            cx = int(det['center'][0])
            cy = int(det['center'][1])

            results.append({
                "id": int(tag_id),
                "center_x": cx,
                "center_y": cy,
            })

            # Corners: lb=left-bottom, rb=right-bottom, rt=right-top, lt=left-top
            corners = np.array([
                det['lb-rb-rt-lt'][0],  # left-bottom
                det['lb-rb-rt-lt'][1],  # right-bottom
                det['lb-rb-rt-lt'][2],  # right-top
                det['lb-rb-rt-lt'][3],  # left-top
            ], dtype=int)

            cv2.polylines(cv_image, [corners.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(cv_image, f"ID:{tag_id}", (cx - 20, cy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            #rospy.loginfo(f"AprilTag detected: ID={tag_id}  center=({cx}, {cy})")

        #if not results:
        #    rospy.loginfo("No AprilTag detected")

        # Publish tag ID (keep last detected ID if nothing detected now)
        if results:
            self._last_detected_id = results[0]["id"]
            self._last_detection_time = rospy.Time.now()   # NEU
        else:
            # NEU: Timeout-Check
            elapsed = (rospy.Time.now() - self._last_detection_time).to_sec()
            if elapsed > self.tag_timeout:
                self._last_detected_id = -1

        msg_id = Int32()
        msg_id.data = self._last_detected_id
        self.pub_tag_id.publish(msg_id)

        # Publish full JSON info
        msg_info = String()
        msg_info.data = json.dumps(results)
        self.pub_tag_info.publish(msg_info)

        # Mode-Label (LANE / OBSTACLE) + Farbe
        if self._current_mode == ControlType.Lane.value:  # Lane
            mode_text = "MODE: LANE"
            mode_color = (0, 255, 0)      # grün
        else:                    # Obstacle
            mode_text = "MODE: OBSTACLE"
            mode_color = (0, 0, 255)      # rot

        cv2.putText(cv_image, mode_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)

        # Action (nur anzeigen wenn eine läuft)
        if self._current_action:
            cv2.putText(cv_image, f"ACTION: {self._current_action}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)   # gelb

        # Last Tag
        cv2.putText(cv_image, f"LAST TAG: {self._last_detected_id}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        # Publish debug image if anyone is subscribed
        if self.pub_debug.get_num_connections() > 0:
            debug_msg = CompressedImage()
            debug_msg.header.stamp = rospy.Time.now()
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tobytes()
            self.pub_debug.publish(debug_msg)

        #self.display_image = cv_image
        #cv_image = cv2.resize(cv_image, (1280, 960))  # 2x scale
        self.is_running = False

    def run(self):
        rate = rospy.Rate(10)
        # cv2.namedWindow('apriltag detection', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('apriltag detection', 1280, 960)  # Set initial size

        while not rospy.is_shutdown():
            if self.display_image is not None:
                # cv2.imshow('apriltag detection', self.display_image)
                pass
            # cv2.waitKey(1)
            rate.sleep()

        # cv2.destroyAllWindows()


if __name__ == '__main__':
    node = DetectAprilTagNode('detect_apriltag_node')
    node.run()
    rospy.spin()