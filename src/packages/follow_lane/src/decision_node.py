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
import json

from switch_control_node import ControlType
import time


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
        self._last_edge_id = ""
        self._halt_requested = False

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
        self.sub_halt = rospy.Subscriber(
            f'/{self._vehicle_name}/explore/halt', Bool,
            self.cbHalt, queue_size=1)

        # NEU: Timing fuer Kantengewichtung (wird vom Timing-Node gelesen)
        self._last_action_time = None
        self.pub_edge_time = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/edge_time', String, queue_size=1)

        # NEU: Geplante Route vom Planner
        self._planned_path = None
        self._planned_steps = None
        self._planner_done = False
        self.sub_planned_path = rospy.Subscriber(
            f'/{self._vehicle_name}/planned/path', String,
            self.cbPlannedPath, queue_size=1)

        # NEU: Lokalisierung abonnieren um Kante zu tracken (fuer Timing)
        self.sub_pose = rospy.Subscriber(
            f'/{self._vehicle_name}/mapping/pose', String,
            self.cbLastEdge, queue_size=1)

        self.pub_action = rospy.Publisher(
            f'/{self._vehicle_name}/decision/action', String, queue_size=1)
        self.pub_route_complete = rospy.Publisher(
            f'/{self._vehicle_name}/decision/route_complete', Bool,
            queue_size=1, latch=True)

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

    def cbHalt(self, msg):
        self._halt_requested = msg.data

    def cbPlannedPath(self, msg):                    # NEU: geplaante Route
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self._planned_path = data.get("actions")
        steps = data.get("steps")
        self._planned_steps = (
            [step for step in steps if step.get("kind") == "action"]
            if isinstance(steps, list) else None)
        if isinstance(steps, list) and steps:
            self._last_edge_id = steps[0].get("edge", self._last_edge_id)
        self._planner_done = data.get("done", False)
        if self._planned_path is not None:
            rospy.loginfo(f"[planner] Route empfangen: {len(self._planned_path)} Actions")

    def cbLastEdge(self, msg):                       # NEU: letzte Kante merken
        try:
            self._last_edge_id = json.loads(msg.data).get("edge_id", "")
        except (ValueError, TypeError):
            pass

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
        # GEPLANTE ROUTE zuerst pruefen
        if self._planned_path is not None and not self._planner_done:
            if len(self._planned_path) > 0:
                action = self._planned_path.pop(0)
                step = None
                if self._planned_steps:
                    step = self._planned_steps.pop(0)
                if step:
                    self._last_edge_id = step.get("edge", self._last_edge_id)
                    options = ", ".join(
                        f"{item['action']}->{item['edge']} "
                        f"({item['weight']:.2f}s)"
                        for item in step.get("alternatives", []))
                    rospy.loginfo(
                        f"[planner] Entscheidung: '{action}' -> {step['edge']} "
                        f"({step['weight']:.2f}s, {step['weight_source']}). "
                        f"Grund: {step['reason']}. Erlaubte Optionen: {options}")
                else:
                    rospy.loginfo(f"[planner] Plane Action: '{action}'")
                return action
            else:
                self._planner_done = True
                rospy.loginfo("[planner] Route erschopft — Fallback auf Explorer.")

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
            # Die soeben vollstaendig befahrene Kante sofort messen. Dadurch
            # geht auch die letzte Kante nicht verloren, wenn danach gestoppt
            # und die Gewichtsdatei gespeichert wird.
            self._publish_completed_edge_time()
            self.publish_cmd(v=0.0, omega=0.0, duration=self.STOP_DURATION)

            # Der Explorer meldet dies erst an der roten Linie am Ende der
            # letzten Mapping-Kante. Keine weitere Kreuzungsfahrt starten.
            if self._halt_requested:
                self.pub_action.publish(String(data=""))
                self.publish_cmd(v=0.0, omega=0.0, duration=0.1)
                rospy.loginfo(
                    f"[explorer] Letzte Kante '{self._last_edge_id}' "
                    f"vollstaendig abgefahren — STOP.")
                return

            # Keine Action mehr bedeutet: Die letzte geplante Kante ist jetzt
            # bis zu ihrer roten Linie vollstaendig abgefahren.
            if (self._planned_path is not None and not self._planner_done
                    and len(self._planned_path) == 0):
                self._planner_done = True
                self.pub_action.publish(String(data=""))
                self.pub_route_complete.publish(Bool(data=True))
                self.publish_cmd(v=0.0, omega=0.0, duration=0.1)
                rospy.loginfo(
                    f"[planner] Route beendet: letzte Kante "
                    f"'{self._last_edge_id}' vollstaendig abgefahren — STOP.")
                return

            # Phase 2: Action (oder skip wenn keine valide Tag-ID)
            if tag_id in self.ID_FUNCTIONS:
                chosen = self.choose_action(self.ID_FUNCTIONS[tag_id])  # NEU
                rospy.loginfo(f"Phase 2: executing '{chosen}' for tag ID {tag_id}")
                self._last_action_time = rospy.get_time()
                self.pub_action.publish(String(data=chosen))
                if chosen == 'turn_left':
                    self.turn_left()
                elif chosen == 'turn_right':
                    self.turn_right()
                elif chosen == 'move_forward':
                    self.move_forward()
            else:
                rospy.loginfo(f"Phase 2: no valid tag (id={tag_id}) — skipping action")
                self._last_action_time = rospy.get_time()
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

    def _publish_completed_edge_time(self):
        """Publiziere die Fahrzeit der gerade abgeschlossenen Kante."""
        now = rospy.get_time()
        if self._last_action_time is not None:
            delta = now - self._last_action_time
            payload = json.dumps({
                "edge_id": self._last_edge_id,
                "event": "edge_complete",
                "seconds": delta,
            })
            self.pub_edge_time.publish(String(data=payload))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = DecisionNode('decision_node')
    node.run()
