#!/usr/bin/env python3
"""
localization_node — Phase A (Challenge 4)

KOMPLETT PASSIV: liest nur mit (/decision/action, /apriltag/id) und
published die geglaubte Position als JSON.

Publisht:
  /<veh>/mapping/pose      (String, JSON)  — Dashboard: "wo denkt der Bot dass er ist?"
  /<veh>/mapping/complete  (Bool)          — alle Kanten besucht

Params (per launch setzbar):
  ~city_graph_file   Pfad zur Stadtgraph-JSON (default: ../config/city_graph.json)
  ~start_node        Startknoten, z.B. "A"
  ~start_exit_arm    Arm, ueber den der Bot den Startknoten verlaesst (1..4)
                     => definiert Startkante + Fahrtrichtung (frei waehlbar laut Aufgabe)
"""

import json
import os

import rospy
from std_msgs.msg import Bool, Int32, String

from city_graph import CityGraph, GraphTracker, INTERSECTION_TAG_IDS, ExpectedGatesMap


class LocalizationNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        default_graph = os.path.join(
            os.path.dirname(__file__), "../config/city_graph.json")
        graph_file = rospy.get_param("~city_graph_file", default_graph)
        start_node = str(rospy.get_param("~start_node", "A"))
        start_exit_arm = int(rospy.get_param("~start_exit_arm", 1))

        self.graph = CityGraph.from_json_file(graph_file)
        self.tracker = GraphTracker(self.graph, start_node, start_exit_arm)

        # expected_gates aus dem Graphen laden (falls vorhanden)
        self.expected_gates = None
        try:
            with open(graph_file) as f:
                data = json.load(f)
            eg = data.get("expected_gates")
            if eg:
                self.expected_gates = ExpectedGatesMap(eg)
        except (ValueError, KeyError, OSError):
            pass

        self._last_tag_id = -1
        self._complete_announced = False

        self.pub_pose = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/pose', String, queue_size=1)
        self.pub_complete = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/complete', Bool, queue_size=1)

        rospy.Subscriber(
            f'/{self._vehicle_name}/decision/action', String,
            self.cbAction, queue_size=5)
        rospy.Subscriber(
            f'/{self._vehicle_name}/apriltag/id', Int32,
            self.cbTagId, queue_size=1)

        rospy.loginfo(
            f"[{node_name}] Start: Kante {self.tracker.current_edge['edge_id']}"
            f" Richtung {self.tracker.current_edge['to_node']}"
            f" ({len(self.graph.all_edge_ids())} Strassen im Graph)")

    def cbTagId(self, msg):
        # nur Kreuzungstags (1-4) fuer den Soft-Check merken
        if msg.data in INTERSECTION_TAG_IDS:
            self._last_tag_id = msg.data

    def cbAction(self, msg):
        action = msg.data
        if action == "stopping":
            self.tracker.on_stopping(self.tracker.current_edge,
                                     last_tag_id=self._last_tag_id)
            p = self.tracker.current_edge
            rospy.loginfo(
                f"[localization] An Kreuzung {p['to_node']}"
                f" (Eingang Arm {p['to_arm']}, Tag {self._last_tag_id})")
        elif action in ("turn_left", "turn_right", "move_forward", "skip"):
            before = self.tracker.status
            self.tracker.on_action(action)
            if self.tracker.status == "LOST":
                rospy.logwarn(
                    f"[localization] DESYNC: '{action}' fuehrt laut Graph ins"
                    f" Nichts (war {before}). Position verloren.")
            else:
                e = self.tracker.current_edge
                rospy.loginfo(
                    f"[localization] '{action}' -> jetzt auf {e['edge_id']}"
                    f" Richtung {e['to_node']}"
                    f" [{self.tracker.pose_dict()['num_visited']}"
                    f"/{self.tracker.pose_dict()['num_total']} Strassen]")
        # action == "" (Reset) ignorieren wir bewusst
        self.publish_pose()

    def publish_pose(self):
        pose = self.tracker.pose_dict()
        self.pub_pose.publish(String(data=json.dumps(pose)))
        self.pub_complete.publish(Bool(data=pose["coverage_complete"]))
        if pose["coverage_complete"] and not self._complete_announced:
            self._complete_announced = True
            rospy.loginfo("[localization] ALLE STRASSEN BESUCHT — Abdeckung komplett.")

    def run(self):
        # Pose auch periodisch senden (fuer Dashboard/late joiner)
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            self.publish_pose()
            rate.sleep()


if __name__ == '__main__':
    node = LocalizationNode('localization_node')
    node.run()
