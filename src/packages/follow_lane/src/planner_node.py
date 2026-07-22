#!/usr/bin/env python3
"""
planner_node — Plaene die Route durch eine Gate-Sequenz.

Liest city_graph.json + Map-Datei, berechnet mit Dijkstra die
schnellste Route und publiziert sie als String (JSON) auf
/<veh>/planned/path.

Params:
  ~start_plan_node       Knoten der aktuellen Startkante, z.B. "A"
  ~start_plan_exit_arm   Arm, ueber den der Bot von diesem Knoten wegfaehrt
  ~gate_sequence         Space-separated Gate-IDs, z.B. "8 9 7 10 6"
  ~map_output            Pfad zur Map-Datei (default: /tmp/duckie_city_map.json)
"""

import json
import os

import rospy
from std_msgs.msg import String

from city_graph import CityGraph
from planner import Planner


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
            f'/{self._vehicle_name}/planned/path', String, queue_size=1,
            latch=True)
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

        # Der Gate-Mapper speichert zwei absichtlich redundante Formate:
        #   gates        : {edge_id: gate_id}
        #   gates_by_tag : {gate_id: edge_id}
        # Die Live-Meldung "validated_gates" hat ebenfalls das erste Format.
        # Akzeptiere alle drei Formate, aber leite sie immer zu gate -> edge um.
        self.gate_to_edge = {}
        for field in ("gates_by_tag", "validated_gates", "gates"):
            self._add_gate_entries(data.get(field, {}))

        if not self.gate_to_edge:
            rospy.logwarn("[planner] Keine Gates in Map-Datei — lade expected_gates")
            self._load_expected_gates()

    def _add_gate_entries(self, entries):
        """Fuege Gate-Eintraege im Format gate->edge oder edge->gate hinzu."""
        if not isinstance(entries, dict):
            return
        for key, value in entries.items():
            # gate -> edge (gates_by_tag oder eine externe Map-Datei)
            try:
                gate_id = int(key)
                edge_id = str(value)
            except (TypeError, ValueError):
                # edge -> gate (gate_mapper_node)
                try:
                    gate_id = int(value)
                    edge_id = str(key)
                except (TypeError, ValueError):
                    rospy.logwarn(
                        f"[planner] Ungueltiger Gate-Map-Eintrag: {key!r}: {value!r}")
                    continue
            self.gate_to_edge.setdefault(gate_id, edge_id)

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

        # Pfad planen
        try:
            route = self.planner.plan_route(
                gate_sequence, self._start_node, self._start_arm)
        except ValueError as e:
            rospy.logerr(f"[planner] Planning fehlgeschlagen: {e}")
            self._publish_error(str(e))
            return

        self._log_weights()

        # Publizieren
        payload = {
            "gates": route["gates"],
            "gate_edges": route["gate_edges"],
            "edges": route["edges"],
            "actions": route["actions"],
            "steps": route["steps"],
            "total_actions": len(route["actions"]),
            "total_weight": route["total_weight"],
            "done": False,
            "error": None,
        }
        self.pub_planned_path.publish(String(data=json.dumps(payload)))
        rospy.loginfo(
            f"[planner] Route geplant: {len(route['actions'])} Actions, "
            f"{len(route['edges'])} Kanten, Gewicht {route['total_weight']:.2f}, "
            f"Gates {gate_sequence}")
        self._log_route(route)

    @staticmethod
    def _log_route(route):
        """Gibt den Plan lesbar auf der ROS-Konsole aus."""
        rospy.loginfo("[planner] ===== Geplanter Weg =====")
        for number, step in enumerate(route["steps"]):
            if step["kind"] == "start":
                rospy.loginfo(
                    f"[planner] Start: {step['from_node']} Arm {step['exit_arm']} "
                    f"-- {step['edge']} --> {step['to_node']} Arm {step['to_arm']} "
                    f"| Gewicht {step['weight']:.2f}s ({step['weight_source']}) "
                    f"| {step['reason']}")
                continue
            alternatives = ", ".join(
                f"{item['action_label']} -> {item['edge']} "
                f"({item['weight']:.2f}s)"
                for item in step["alternatives"])
            rospy.loginfo(
                f"[planner] {number}. bei {step['at_node']} (Eingang "
                f"Arm {step['entry_arm']}): {step['action_label']} ueber "
                f"Arm {step['exit_arm']} -> {step['edge']} -> "
                f"{step['to_node']} Arm {step['to_arm']} | "
                f"Gewicht {step['weight']:.2f}s ({step['weight_source']}), "
                f"Summe {step['cumulative_weight']:.2f}s | {step['reason']} | "
                f"Erlaubte Optionen: {alternatives}")
        rospy.loginfo("[planner] =========================")

    def _log_weights(self):
        """Gibt alle vom Planner verwendeten Kantengewichte aus."""
        rospy.loginfo("[planner] ===== Kantengewichte =====")
        for edge_id in sorted(self.graph.all_edge_ids()):
            rospy.loginfo(
                f"[planner] {edge_id}: {self.planner._weight(edge_id):.2f}s "
                f"({self.planner._weight_source(edge_id)})")
        rospy.loginfo("[planner] ==========================")

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
