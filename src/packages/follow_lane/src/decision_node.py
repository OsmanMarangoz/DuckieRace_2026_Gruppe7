#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32
from duckietown_msgs.msg import Twist2DStamped
from enum import Enum
import os
import random

# Import ControlType from switch_control_node
from switch_control_node import ControlType


class DecisionNode:
    # Dictionary mapping AprilTag IDs to possible functions
    ID_FUNCTIONS = {
        1: ['turn_left', 'turn_right', 'move_forward'],
        2: ['turn_left', 'turn_right'],
        3: ['turn_left', 'move_forward'],
        4: ['turn_right', 'move_forward'],
    }

    def __init__(self, node_name):
        rospy.init_node(node_name)

        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._last_april_tag_id = -1
        self._last_control_mode = ControlType.Lane
        self._has_executed_action = False

        # Publisher for driving commands
        self.pub_cmd = rospy.Publisher(
            f'/{self._vehicle_name}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1
        )

        # Subscribe to AprilTag ID topic
        self.sub_tag_id = rospy.Subscriber(
            f'/{self._vehicle_name}/apriltag/id',
            Int32,
            self.cbAprilTagID,
            queue_size=1
        )

        # Subscribe to control mode topic
        self.sub_control = rospy.Subscriber(
            f'/{self._vehicle_name}/switch/control',
            Int32,
            self.cbControlMode,
            queue_size=1
        )

        rospy.loginfo(f"[{node_name}] Decision node ready")

    def cbAprilTagID(self, msg):
        self._last_april_tag_id = msg.data
        #if msg.data >= 1:
            #rospy.loginfo(f"AprilTag ID received: {msg.data}")

    def cbControlMode(self, msg):
        current_mode = msg.data

        if (current_mode == ControlType.Obstacle.value and
                self._last_control_mode != ControlType.Obstacle.value):
            rospy.loginfo(f"Red line stop detected. Executing action for AprilTag ID {self._last_april_tag_id}")
            self._has_executed_action = False
            self.execute_action_for_id(self._last_april_tag_id)

        elif current_mode == ControlType.Lane.value:
            self._has_executed_action = False

        self._last_control_mode = current_mode

    def execute_action_for_id(self, tag_id):
        if self._has_executed_action:
            return

        if tag_id not in self.ID_FUNCTIONS:
            rospy.logwarn(f"AprilTag ID {tag_id} not in decision map")
            return

        possible_functions = self.ID_FUNCTIONS[tag_id]
        chosen_function = random.choice(possible_functions)

        rospy.loginfo(f"Executing action '{chosen_function}' for AprilTag ID {tag_id}")

        if chosen_function == 'turn_left':
            self.turn_left()
        elif chosen_function == 'turn_right':
            self.turn_right()
        elif chosen_function == 'move_forward':
            self.move_forward()

        self._has_executed_action = True

    # ============ Helper ============

    def publish_cmd(self, v, omega, duration):
        """Publish a drive command for a given duration in seconds, then stop."""
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = omega

        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(10)
        while rospy.Time.now() < end_time:
            msg.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(msg)
            rate.sleep()

        # stop
        msg.v = 0.0
        msg.omega = 0.0
        msg.header.stamp = rospy.Time.now()
        self.pub_cmd.publish(msg)

    # ============ Actions ============

    def turn_left(self):
        rospy.loginfo("[Decision] Executing: turn_left()")
        self.publish_cmd(v=0.7, omega=2.1, duration=1.5)

    def turn_right(self):
        rospy.loginfo("[Decision] Executing: turn_right()")
        self.publish_cmd(v=0.25, omega=-3.4, duration=1.5)

    def move_forward(self):
        rospy.loginfo("[Decision] Executing: move_forward()")
        self.publish_cmd(v=0.55, omega=0.0, duration=2.0)

    # ============================================

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = DecisionNode('decision_node')
    node.run()