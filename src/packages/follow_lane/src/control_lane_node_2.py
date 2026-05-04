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
        self.correction = 0

        # Red line stop state
        self._red_stop_active = False       # True while we are in the 2s stop pause
        self._red_stop_until = None         # rospy.Time when we may resume
        self._RED_PIXEL_THRESHOLD = 20000     # tune this to your track / lighting
        self._RED_COOLDOWN = 5.0            # seconds before a second stop is allowed
        self._last_red_stop_time = rospy.Time(0)  # avoid repeated triggers

        self._vehicle_name = os.environ['VEHICLE_NAME']
        util.init_parameters(node_name, self.cbUpdateParameters)

        twist_topic = f"/{self._vehicle_name}/car_cmd_switch_node/cmd"
        self.pub_cmd_vel = rospy.Publisher(twist_topic, Twist2DStamped, queue_size = 1)

        detect_lane_topic = f"/{self._vehicle_name}/detect/lane"
        self.sub_lane = rospy.Subscriber(detect_lane_topic, Float64, self.cbFollowLane, queue_size = 1)

        control_change_topic = f"/{self._vehicle_name}/switch/control"
        self.sub_control = rospy.Subscriber(control_change_topic, Int32, self.cbControl , queue_size = 1)

        red_line_topic = f"/{self._vehicle_name}/detect/red_line"
        self.sub_red_line = rospy.Subscriber(red_line_topic, Float64, self.cbRedLine, queue_size = 1)

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

    def cbRedLine(self, msg):
        """
        Called whenever detect_lane_node publishes the red pixel count.
        Triggers a 2-second stop if:
          - enough red pixels are visible
          - we are not already stopped
          - the cooldown since the last stop has elapsed
        """
        if not self.enable:
            return

        red_pixels = msg.data
        now = rospy.Time.now()

        already_cooling_down = (now - self._last_red_stop_time).to_sec() < self._RED_COOLDOWN

        if (red_pixels >= self._RED_PIXEL_THRESHOLD
                and not self._red_stop_active
                and not already_cooling_down):

            rospy.loginfo(f"Red line detected ({int(red_pixels)} px) — stopping for 2 s")
            self._red_stop_active = True
            self._red_stop_until = now + rospy.Duration(2.0)
            self._last_red_stop_time = now

            # Publish stop immediately so there is no lag waiting for run()
            self._publish_stop()

    # error between 1 and -1
    def cbFollowLane(self, error):
        #print(f'received message. enabled : {self.enable}')
        error = error.data

        P = self.kp * error

        self.integral += error

        I = self.ki * self.integral

        derivative = error - self.lastError
        D = self.kd * derivative

        correction = P + I + D

        self.lastError = error

        # Slow down when turning (larger error = more curve = slower)
        #self.v = self.MAX_VEL * (1 - min(abs(error), 1.0))
        self.v = self.MAX_VEL * (1 - min(abs(error), 1.0))

        print(f'error = {abs(error)}')
        print(f'self.v = { self.v}')
        self.a = correction

    def _publish_stop(self):
        twist = Twist2DStamped()
        twist.header.stamp = rospy.Time.now()
        twist.v = 0.0
        twist.omega = 0.0
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")
        twist = Twist2DStamped(v=0.0, omega=0.0)
        self.pub_cmd_vel.publish(twist)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.enable:
                now = rospy.Time.now()

                # Check if we are in a red-line stop
                if self._red_stop_active:
                    if now < self._red_stop_until:
                        # Still within the 2-second pause — keep publishing stop
                        rospy.sleep(2)
                        self._publish_stop()
                        rate.sleep()
                        continue
                    else:
                        # Pause over — resume normal driving
                        rospy.loginfo("Red line stop finished — resuming lane following")
                        self._red_stop_active = False
                        # Reset integral so the bot doesn't lurch after the pause
                        self.integral = 0
                        self.lastError = 0

                twist = Twist2DStamped()
                twist.header.stamp = now
                twist.v = self.v
                twist.omega = self.a
                self.pub_cmd_vel.publish(twist)

            rate.sleep()

if __name__ == '__main__':
    # create the node
    node = ControlLaneNode('control_lane_node')
    node.run()
    rospy.spin()