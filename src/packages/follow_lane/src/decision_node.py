#!/usr/bin/env python3

# ============================================================================
# CHALLENGE-4-ERWEITERUNG (minimal-invasiv):
#
# 1) direction_source (rosparam ~direction_source, default "random"):
#      "random"   -> EXAKT das alte Verhalten (Challenge 1-3 unveraendert).
#      "external" -> Richtung kommt vom explorer_node
#                    (/explore/suggested_action); ist die Empfehlung nicht
#                    in der erlaubten Liste des Kreuzungstags, Fallback random.
#
# 2) Kreuzungstag-Filter: fuer die ID_FUNCTIONS-Wahl zaehlen nur Tags 1-4.
#    Grund: In Challenge 4 stehen Tor-Tags (5-13) auf den Strecken. Ohne
#    Filter wuerde ein kurz vor der Kreuzung gesehenes Tor-Tag den
#    Kreuzungstag ueberschreiben -> decision wuerde "skip" fahren.
#    In Challenge 1-3 gibt es keine Tags 5-13 -> Filter aendert dort nichts.
#
# Alles andere (Stop-Mechanik, Manoever, done-Signal) ist unveraendert.
# ============================================================================

import rospy
from std_msgs.msg import Int32, Bool
from duckietown_msgs.msg import Twist2DStamped
import os
import random
from std_msgs.msg import String

from switch_control_node import ControlType


class DecisionNode:
    ID_FUNCTIONS = {
        1: ['turn_left', 'turn_right', 'move_forward'],
        2: ['turn_left', 'turn_right'],
        3: ['turn_left', 'move_forward'],
        4: ['turn_right', 'move_forward'],
    }
    STOP_DURATION = 2.0  # Sekunden v=0 bevor Action ausgeführt wird

    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._last_april_tag_id = -1
        self._last_intersection_tag_id = -1          # NEU: nur IDs 1-4
        self._last_control_mode = ControlType.Lane.value
        self._busy = False

        # NEU: Richtungsquelle ("random" = altes Verhalten)
        self._direction_source = rospy.get_param("~direction_source", "random")
        self._suggested_action = ""

        self.pub_cmd = rospy.Publisher(
            f'/{self._vehicle_name}/car_cmd_switch_node/cmd',
            Twist2DStamped, queue_size=1)
        self.pub_done = rospy.Publisher(
            f'/{self._vehicle_name}/decision/done', Bool, queue_size=1)

        self.sub_tag_id = rospy.Subscriber(
            f'/{self._vehicle_name}/apriltag/id', Int32,
            self.cbAprilTagID, queue_size=1)
        self.sub_control = rospy.Subscriber(
            f'/{self._vehicle_name}/switch/control', Int32,
            self.cbControlMode, queue_size=1)

        # NEU: Empfehlung vom Explorer (wird nur bei "external" benutzt)
        self.sub_suggest = rospy.Subscriber(
            f'/{self._vehicle_name}/explore/suggested_action', String,
            self.cbSuggestedAction, queue_size=1)

        self.pub_action = rospy.Publisher(
            f'/{self._vehicle_name}/decision/action', String, queue_size=1)

        rospy.loginfo(
            f"[{node_name}] Decision node ready"
            f" (direction_source={self._direction_source})")

    def cbAprilTagID(self, msg):
        self._last_april_tag_id = msg.data
        if msg.data in self.ID_FUNCTIONS:            # NEU: 1-4 separat merken
            self._last_intersection_tag_id = msg.data
        elif msg.data == -1:                         # Timeout vom Detector
            self._last_intersection_tag_id = -1

    def cbSuggestedAction(self, msg):                # NEU
        self._suggested_action = msg.data

    def cbControlMode(self, msg):
        current_mode = msg.data
        triggered = (current_mode == ControlType.Obstacle.value
                     and self._last_control_mode != ControlType.Obstacle.value
                     and not self._busy)
        # IMMER vor handle_obstacle updaten, damit weitere Obstacle-msgs
        # während des blockenden Calls nicht re-triggern
        self._last_control_mode = current_mode

        if triggered:
            self.handle_obstacle()

    def choose_action(self, allowed):                # NEU: Auswahl gekapselt
        """Waehlt das Manoever aus der erlaubten Liste des Kreuzungstags."""
        if self._direction_source == "external":
            suggestion = self._suggested_action
            if suggestion in allowed:
                rospy.loginfo(f"Externe Richtung uebernommen: '{suggestion}'")
                return suggestion
            rospy.logwarn(
                f"Externe Empfehlung '{suggestion}' nicht in {allowed}"
                f" — Fallback auf random.")
        return random.choice(allowed)

    def handle_obstacle(self):
        self._busy = True
        try:
            # NEU: Kreuzungstag (1-4) bevorzugen; Fallback altes Verhalten
            tag_id = self._last_intersection_tag_id
            if tag_id == -1:
                tag_id = self._last_april_tag_id
            rospy.loginfo(f"Obstacle phase started. Last tag ID: {tag_id}")

            # Phase 1: ECHTER Stop
            rospy.loginfo(f"Phase 1: stopping for {self.STOP_DURATION}s")
            self.pub_action.publish(String(data="stopping"))
            self.publish_cmd(v=0.0, omega=0.0, duration=self.STOP_DURATION)

            # Phase 2: Action (oder skip wenn keine valide Tag-ID)
            if tag_id in self.ID_FUNCTIONS:
                chosen = self.choose_action(self.ID_FUNCTIONS[tag_id])  # NEU
                rospy.loginfo(f"Phase 2: executing '{chosen}' for tag ID {tag_id}")
                self.pub_action.publish(String(data=chosen))
                if chosen == 'turn_left':
                    self.turn_left()
                elif chosen == 'turn_right':
                    self.turn_right()
                elif chosen == 'move_forward':
                    self.move_forward()
            else:
                rospy.loginfo(f"Phase 2: no valid tag (id={tag_id}) — skipping action")
                self.pub_action.publish(String(data="skip"))

            self.pub_action.publish(String(data=""))

            # Letztes v=0 publishen, damit der Roboter sauber steht bevor
            # control_lane_node übernimmt
            self.publish_cmd(v=0.0, omega=0.0, duration=0.1)

            # Phase 3: done signalisieren
            self.pub_done.publish(Bool(data=True))
            rospy.loginfo("Phase 3: done signal sent")
        finally:
            self._busy = False

    def publish_cmd(self, v, omega, duration):
        msg = Twist2DStamped()
        msg.v = v
        msg.omega = omega
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(10)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            msg.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(msg)
            rate.sleep()

    def turn_left(self):
        #self.publish_cmd(v=0.7, omega=2.1, duration=1.5)
        self.publish_cmd(v=0.2, omega=0.3, duration=1.4)
        self.publish_cmd(v=0.2, omega=2.5, duration=1.4)
        self.publish_cmd(v=0.2, omega=0.0, duration=0.3)

    def turn_right(self):
        self.publish_cmd(v=0.2, omega=-0.2, duration=1.2)
        self.publish_cmd(v=0.13, omega=-3.0, duration=0.4)

    def move_forward(self):
        self.publish_cmd(v=0.20, omega=0.0, duration=2.1)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = DecisionNode('decision_node')
    node.run()