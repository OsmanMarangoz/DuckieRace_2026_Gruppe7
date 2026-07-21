#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64, Int32, Bool, String
from enum import Enum
import json
import os

class ControlType(Enum):
    Lane = 1
    Obstacle = 2
    Stop = 3

class SwitchControlNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._control_mode = ControlType.Lane
        self._RED_PIXEL_THRESHOLD = 5000
        self._RED_COOLDOWN = 1.0
        self._last_red_stop_time = rospy.Time(0)
        self._mapping_done = False  # gesetzt durch explorer state

        self.sub_red_line = rospy.Subscriber(
            f"/{self._vehicle_name}/detect/red_line", Float64,
            self.cbRedLine, queue_size=1)
        self.sub_done = rospy.Subscriber(
            f"/{self._vehicle_name}/decision/done", Bool,
            self.cbDecisionDone, queue_size=1)
        self.sub_explorer_halt = rospy.Subscriber(
            f"/{self._vehicle_name}/explore/halt", Bool,
            self.cbExplorerHalt, queue_size=1)
        self.sub_explorer_state = rospy.Subscriber(
            f"/{self._vehicle_name}/explore/state", String,
            self.cbExplorerState, queue_size=1)
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
            if self._mapping_done:
                rospy.loginfo("[switch] Mapping done — ignoriere Lane-Rückkehr")
                return
            rospy.loginfo("Decision sequence done — resuming Lane mode")
            self._control_mode = ControlType.Lane
            # Cooldown ab JETZT, nicht ab Red-Line-Detection
            self._last_red_stop_time = rospy.Time.now()

    def cbExplorerHalt(self, msg):
        """Sofortiger Halt — kommt vom Explorer, sobald Mapping fertig."""
        if msg.data:
            if self._control_mode != ControlType.Stop:
                rospy.loginfo("[switch] HALT — Explorer meldet Mapping fertig.")
                self._control_mode = ControlType.Stop
                self.pub_control.publish(Int32(data=ControlType.Stop.value))

    def cbExplorerState(self, msg):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        self._mapping_done = state.get("done", False)
        # Wenn Mapping fertig UND alle Tore zugeordnet: anhalten
        if self._mapping_done and state.get("gates_complete", False):
            if self._control_mode != ControlType.Stop:
                rospy.loginfo("[switch] Mapping+Gates complete — STOPPING.")
                self._control_mode = ControlType.Stop
                self.pub_control.publish(Int32(data=ControlType.Stop.value))

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