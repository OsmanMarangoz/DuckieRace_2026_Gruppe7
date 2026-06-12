#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, Bool
from enum import Enum
import os

class ControlType(Enum):
    Lane = 1
    Obstacle = 2

class SwitchControlNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._control_mode = ControlType.Lane
        self._RED_PIXEL_THRESHOLD = 6000
        self._RED_COOLDOWN = 1.0
        self._last_red_stop_time = rospy.Time(0)

        self.sub_red_line = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/red_line", Float64,
            self.cbRedLine, queue_size=1)
        self.sub_done = rospy.Subscriber(
            f"/{self._vehicle_name}/decision/done", Bool,
            self.cbDecisionDone, queue_size=1)
        self.pub_control = rospy.Publisher(
            f"/{self._vehicle_name}/switch/control", Int32, queue_size=1)

    def cbRedLine(self, msg):
        now = rospy.Time.now()
        cooling_down = (now - self._last_red_stop_time).to_sec() < self._RED_COOLDOWN
        if (msg.data >= self._RED_PIXEL_THRESHOLD
                and self._control_mode == ControlType.Lane
                and not cooling_down):
            rospy.loginfo(f"Red line detected ({int(msg.data)} px) — switching to Obstacle")
            self._control_mode = ControlType.Obstacle
            self._last_red_stop_time = now

    def cbDecisionDone(self, msg):
        if msg.data and self._control_mode == ControlType.Obstacle:
            rospy.loginfo("Decision sequence done — resuming Lane mode")
            self._control_mode = ControlType.Lane
            # Cooldown ab JETZT, nicht ab Red-Line-Detection
            self._last_red_stop_time = rospy.Time.now()

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            msg_control = Int32()
            msg_control.data = self._control_mode.value
            self.pub_control.publish(msg_control)
            rate.sleep()

if __name__ == '__main__':
    node = SwitchControlNode(node_name='switch_control_node')
    node.run()
    rospy.spin()