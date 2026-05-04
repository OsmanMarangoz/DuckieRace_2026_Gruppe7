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
        self.a = 0
        self.v = 0
        self.correction = 0

        # Red line stop state
        self.red_line_detected = False
        self.red_stop_time     = None
        self.red_stop_duration = rospy.Duration(2.0)  # stop for 2 seconds then resume
        self.RED_EDGE_THRESHOLD = 50

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size = 1)

        detect_lane_topic = f"/{self._vehicle_name}/detect/lane"
        self.sub_lane = rospy.Subscriber(detect_lane_topic, Float64, self.cbFollowLane, queue_size = 1)

        control_change_topic = f"/{self._vehicle_name}/switch/control"
        self.sub_control = rospy.Subscriber(control_change_topic, Int32, self.cbControl , queue_size = 1)

        red_line_topic = f"/{self._vehicle_name}/detect/red_line"
        self.sub_red_line = rospy.Subscriber(red_line_topic, Float64, self.cbRedLine, queue_size=1)


        rospy.on_shutdown(self.fnShutDown)

    def cbRedLine(self, msg):
        if msg.data >= self.RED_EDGE_THRESHOLD:
            if not self.red_line_detected:
                rospy.loginfo(f"Red line detected! Edge pixels: {msg.data}. Stopping.")
                self.red_line_detected = True
                self.red_stop_time     = rospy.Time.now()

    def isStoppedForRed(self):
        if not self.red_line_detected:
            return False
        elapsed = rospy.Time.now() - self.red_stop_time
        if elapsed < self.red_stop_duration:
            return True
        else:
            # Stop duration passed — resume and reset
            self.red_line_detected = False
            return False


    def cbControl(self,msg):
        if msg.data == ControlType.Lane.value:
            self.enable = True
        else:
            self.enable = False

    def cbUpdateParameters(self,parameters):
        self.kp = parameters["pid"]["p"]["default"]
        self.ki = parameters["pid"]["i"]["default"]
        self.kd = parameters["pid"]["d"]["default"]
        self.MAX_VEL = parameters["pid"]["max_vel"]["default"]

    # error between 1 and -1
    def cbFollowLane(self, error):
        print(f'received message. enabled : {self.enable}')
        error = error.data

        # PID
        P = self.kp * error

        self.integral += error
        I = self.ki * self.integral

        derivative = error - self.lastError
        D = self.kd * derivative

        correction = P + I + D
        self.lastError = error

        # Slow down when turning (larger error = more curve = slower)
        self.v = self.MAX_VEL * (1 - min(abs(error), 1.0))
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

                if self.isStoppedForRed():
                    twist.v = 0.0
                    twist.omega = 0.0
                else:
                    twist.v = self.v
                    twist.omega = self.a

                self.pub_cmd_vel.publish(twist)

            rate.sleep()


if __name__ == '__main__':
    # create the node
    node = ControlLaneNode('control_lane_node')
    node.run()
    rospy.spin()