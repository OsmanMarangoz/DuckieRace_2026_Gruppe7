#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, String

from duckietown_msgs.msg import Twist2DStamped
import os
from switch_control_node import ControlType
import yaml
import util

class ControlLaneNode:
    def __init__(self,node_name):
        rospy.init_node(node_name)
        self.enable = True
        self.lastError = 0
        self.integral = 0
        self.v = 0.0
        self.a = 0.0

        self.MAX_ERROR_JUMP = 0.4
        self._error_initialized = False

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size=1)

        detect_lane_topic = f"/{self._vehicle_name}/detect/lane"
        self.sub_lane = rospy.Subscriber(detect_lane_topic, Float64, self.cbFollowLane, queue_size=1)

        control_change_topic = f"/{self._vehicle_name}/switch/control"
        self.sub_control = rospy.Subscriber(control_change_topic, Int32, self.cbControl, queue_size=1)

        rospy.on_shutdown(self.fnShutDown)

    def cbControl(self, msg):
        if msg.data == ControlType.Lane.value:
            self.enable = True
        else:
            self.enable = False

    def cbUpdateParameters(self, parameters):
        self.kp = parameters["pid"]["p"]["default"]
        self.ki = parameters["pid"]["i"]["default"]
        self.kd = parameters["pid"]["d"]["default"]
        self.MAX_VEL = parameters["pid"]["max_vel"]["default"]

    # error between 1 and -1
    def cbFollowLane(self, error):
        error = error.data

        P = self.kp * error

        self.integral += error
        # Integral clamping (anti-windup)
        self.integral = max(-5.0, min(5.0, self.integral))
        # Integral reset on sign change
        if (error > 0) != (self.lastError > 0):
            self.integral = 0.0
        I = self.ki * self.integral

        derivative = error - self.lastError
        D = self.kd * derivative

        correction = P + I + D

        self.lastError = error

        # Adaptive speed: slow down in curves
        #self.v = self.MAX_VEL
        self.v = 0.2
        self.a = correction

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.enable:
                twist = Twist2DStamped()
                twist.header.stamp = rospy.Time.now()
                twist.v = self.v
                twist.omega = self.a
                self.pub_cmd_vel.publish(twist)
            else:
                self.integral = 0
                self.lastError = 0
            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = ControlLaneNode('control_lane_node')
    node.run()
    rospy.spin()