#!/usr/bin/env python3
"""
planner_node — Plaene die Route durch eine Gate-Sequenz.

Liest city_graph.json + Map-Datei, berechnet mit Dijkstra die
schnellste Route und publiziert sie als String (JSON) auf
/<veh>/planned/path.

Params:
  ~start_plan_node       Startknoten, z.B. "A"
  ~start_plan_exit_arm   Arm, ueber den der Bot den Startknoten verlaesst
  ~gate_sequence         Space-separated Gate-IDs, z.B. "8 9 7 10 6"
  ~map_output            Pfad zur Map-Datei (default: /tmp/duckie_city_map.json)
"""

import json
import os

import rospy
from std_msgs.msg import String
from duckietown_msgs.msg import Twist2DStamped

from city_graph import CityGraph
from planner import Planner
from switch_control_node import ControlType


class PlannerNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._start_node = str(rospy.get_param("~start_plan_node", "A"))
        self._start_arm = int(rospy.get_param("~start_plan_exit_arm", 1))
        self._gate_sequence_str = rospy.get_param("~gate_sequence", "8 9 7 10 6")
        self._map_output = rospy.get_param("~map_output", "/tmp/duckie_city_map.json")

        # CityGraph laden
        graph_file = os.path.join(
            os.path.dirname(__file__), "../config/city_graph.json")
        self.graph = CityGraph.from_json_file(graph_file)

        # Map-Datei laden (gates)
        self.gate_to_edge = {}
        self.edge_weights = {}
        self._load_map()
        self._load_weights()

        # Planner initialisieren
        self.planner = Planner(self.graph, self.gate_to_edge, self.edge_weights)

        # Publisher
        self.pub_planned_path = rospy.Publisher(
            f'/{self._vehicle_name}/planned/path', String, queue_size=1)
        self.pub_switch = rospy.Publisher(
            f'/{self._vehicle_name}/car_cmd_switch_node/cmd',
            Twist2DStamped, queue_size=1)

        # Subscriber fuer Param-Updates
        rospy.Subscriber(
            f'/{self._vehicle_name}/planner/gate_sequence', String,
            self.cbGateSequence, queue_size=1)

        # Erstes Planning
        self._plan_and_publish()

        rospy.loginfo(f"[{node_name}] Planner bereit -> {self._map_output}")

    def _load_map(self):
        """Lade Map-Datei fuer gates."""
        try:
            with open(self._map_output, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            rospy.logwarn(f"[planner] Map-Datei nicht lesbar: {e}")
            rospy.logwarn("[planner] Verwende expected_gates aus city_graph.json")
            self._load_expected_gates()
            return

        # Gates laden (validierte Tore)
        validated = data.get("validated_gates", {})
        if validated:
            self.gate_to_edge = {
                int(k): v for k, v in validated.items()
            }
        else:
            gates = data.get("gates", {})
            self.gate_to_edge = {
                int(k): v for v, k in gates.items()
            }

        if not self.gate_to_edge:
            rospy.logwarn("[planner] Keine Gates in Map-Datei — lade expected_gates")
            self._load_expected_gates()

    def _load_weights(self):
        """Lade Kantengewichte aus separater Datei (timing_node)."""
        weights_file = self._map_output.replace(".json", "_weights.json")
        try:
            with open(weights_file, "r") as f:
                self.edge_weights = json.load(f)
            rospy.loginfo(
                f"[planner] Kantengewichte geladen: {len(self.edge_weights)} Kanten")
        except (OSError, json.JSONDecodeError):
            rospy.logwarn("[planner] Keine Kantengewichte vorhanden — verwende Hop-Distanz")

    def _load_expected_gates(self):
        """Falls Map-Datei fehlt: expected_gates aus city_graph.json."""
        graph_file = os.path.join(
            os.path.dirname(__file__), "../config/city_graph.json")
        try:
            with open(graph_file) as f:
                data = json.load(f)
            eg = data.get("expected_gates", {})
            self.gate_to_edge = {
                int(k): v for k, v in eg.items()
            }
        except (OSError, json.JSONDecodeError):
            rospy.logerr("[planner] Kann expected_gates nicht laden!")

    def _plan_and_publish(self):
        """Planning durchfuehren und publizieren."""
        try:
            gate_sequence = [int(g) for g in self._gate_sequence_str.split()]
        except ValueError:
            rospy.logerr(f"[planner] Ungueltige gate_sequence: {self._gate_sequence_str}")
            return

        # Startkante berechnen und validieren
        start_edge = self.planner.compute_start_edge(self._start_node, self._start_arm)
        if gate_sequence:
            first_gate = gate_sequence[0]
            first_gate_edge = self.gate_to_edge.get(first_gate)
            if first_gate_edge:
                self.planner.validate_start_edge(start_edge, first_gate_edge)
            else:
                rospy.logwarn(
                    f"[planner] Erstes Gate {first_gate} nicht in gate_to_edge")

        # Pfad planen
        try:
            path = self.planner.plan_path(gate_sequence)
        except ValueError as e:
            rospy.logerr(f"[planner] Planning fehlgeschlagen: {e}")
            self._publish_error(str(e))
            return

        if path is None:
            rospy.logerr("[planner] Kein Pfad gefunden — Graph ist unzusammenhaengend")
            self._publish_error("No path found")
            return

        # Kanten in Actions umwandeln
        all_edges = []
        current_edge = None
        for gate_id, target_edge in path:
            if current_edge is not None:
                seg = self.planner._dijkstra(current_edge, target_edge)
                if seg:
                    all_edges.extend(seg)
            all_edges.append(target_edge)
            current_edge = target_edge

        actions, _ = self.planner.edges_to_actions(all_edges)
        action_list = [a for a, _ in actions if a is not None]

        # Publizieren
        payload = {
            "gates": [gid for gid, _ in path],
            "edges": [eid for _, eid in path],
            "actions": action_list,
            "total_actions": len(action_list),
            "done": False,
            "error": None,
        }
        self.pub_planned_path.publish(String(data=json.dumps(payload)))
        rospy.loginfo(
            f"[planner] Route geplant: {len(action_list)} Actions, "
            f"Gates {gate_sequence}")

        # Auf Lane-Modus schalten (switch_control_node)
        lane_msg = Twist2DStamped()
        lane_msg.data = ControlType.Lane.value
        self.pub_switch.publish(lane_msg)
        rospy.loginfo(f"[planner] Auf Lane-Modus geschaltet")

        # Kurze Pause, damit switch_control_node die Aenderung verarbeitet
        rospy.sleep(0.5)

    def _publish_error(self, error_msg):
        """Publiziere Fehler-Message."""
        payload = {
            "gates": [],
            "edges": [],
            "actions": [],
            "total_actions": 0,
            "done": True,
            "error": error_msg,
        }
        self.pub_planned_path.publish(String(data=json.dumps(payload)))

    def cbGateSequence(self, msg):
        """Param-Update: neue Gate-Sequenz."""
        self._gate_sequence_str = msg.data
        rospy.loginfo(f"[planner] Neue Gate-Sequenz: {msg.data}")
        self._plan_and_publish()

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    node = PlannerNode('planner_node')
    node.run()
