#!/usr/bin/env python3
"""
explorer_node — Phase A (Challenge 4)

Berechnet aus der geglaubten Position die Abbiege-Empfehlung fuer die
Kantenabdeckung. Mehrfach-Sweep: wenn nach voller Abdeckung weniger Tore
gefunden wurden als erwartet (~expected_gates), werden die torlosen
Strassen gezielt erneut abgefahren.
Die Empfehlung wird kontinuierlich gepublisht; decision_node liest sie
NUR mit direction_source=external. Ohne den Param: altes Verhalten.

Publisht:
  /<veh>/explore/suggested_action  (String: turn_left|turn_right|move_forward)
  /<veh>/explore/state             (String, JSON: sweep, done, ...)

Params:
  ~city_graph_file   Stadtgraph-JSON
  ~expected_gates    Anzahl Tore auf der Strecke (-1 = unbekannt -> 1 Sweep)
  ~max_sweeps        max. Anzahl Sweeps (default 3)
"""

import json
import os
import atexit
from collections import deque

import rospy
from std_msgs.msg import String

from city_graph import CityGraph, ExplorerPolicy, ExpectedGatesMap


class ExplorerNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        default_graph = os.path.join(
            os.path.dirname(__file__), "../config/city_graph.json")
        graph_file = rospy.get_param("~city_graph_file", default_graph)
        expected = int(rospy.get_param("~expected_gates", 5))
        max_sweeps = int(rospy.get_param("~max_sweeps", 3))

        self.graph = CityGraph.from_json_file(graph_file)
        # expected_gates aus Graphen laden (None wenn nicht vorhanden)
        try:
            with open(graph_file) as f:
                data = json.load(f)
            eg = data.get("expected_gates")
            self.expected_gates_map = ExpectedGatesMap(eg) if eg else None
        except (ValueError, KeyError, OSError):
            self.expected_gates_map = None

        self.policy = ExplorerPolicy(
            self.graph,
            expected_gates=(None if expected < 0 else self.expected_gates_map),
            max_sweeps=max_sweeps)

        self._suggestion = ""
        self._done = False
        self._gates = {}
        self._validated_gates = {}  # nur validierte Tore (match zu expected)
        self._last_edge = None
        self._target_edge = ""  # welche Kante der Explorer empfehlen will

        self.pub_suggest = rospy.Publisher(
            f'/{self._vehicle_name}/explore/suggested_action',
            String, queue_size=1)
        self.pub_state = rospy.Publisher(
            f'/{self._vehicle_name}/explore/state', String, queue_size=1)

        rospy.Subscriber(f'/{self._vehicle_name}/mapping/pose', String,
                         self.cbPose, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/mapping/gates', String,
                         self.cbGates, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/mapping/validated_gates', String,
                         self.cbValidatedGates, queue_size=1)

        atexit.register(self.shutdown)

        rospy.loginfo(
            f"[{node_name}] Explorer bereit"
            f" (expected_gates={expected}, max_sweeps={max_sweeps})")

    def cbGates(self, msg):
        try:
            self._gates = json.loads(msg.data).get("gates", {})
        except ValueError:
            pass

    def cbValidatedGates(self, msg):
        try:
            self._validated_gates = json.loads(msg.data).get("gates", {})
        except ValueError:
            pass

    def _recover(self, node, entry_arm, visited):
        """Versuche, aus LOST-Zustand zurückzukommen: BFS zu einer
        unbesuchten Kante und empfehle den ersten Schritt dorthin."""
        try:
            all_ids = self.graph.all_edge_ids()
        except Exception:
            return ""

        # 1) Gibt es an diesem Knoten eine unbesuchte Kante?
        for arm in self.graph.arms_of(node):
            action = self._graph_exit_arm_to_turn(entry_arm, arm)
            if action is None:
                continue
            edge = self.graph.edge_from(node, arm)
            if edge and edge["edge_id"] not in visited:
                rospy.logwarn_once(
                    f"[explorer] Recovery: unbesuchte Kante {edge['edge_id']}"
                    f" von {node} Arm {arm} ({action})")
                return action

        # 2) BFS zum nächsten Knoten mit unbesuchter Kante
        start = (node, entry_arm)
        parent = {start: None}
        q = deque([start])
        goal = None
        while q:
            cn, ce = q.popleft()
            has_unvisited = any(
                self.graph.edge_from(cn, a)
                and self.graph.edge_from(cn, a)["edge_id"] not in visited
                for a in self.graph.arms_of(cn)
            )
            if has_unvisited and (cn, ce) != start:
                goal = (cn, ce)
                break
            for a in self.graph.arms_of(cn):
                act = self._graph_exit_arm_to_turn(ce, a)
                if act is None:
                    continue
                edge = self.graph.edge_from(cn, a)
                if edge:
                    nxt = (edge["to_node"], edge["to_arm"])
                    if nxt not in parent:
                        parent[nxt] = (cn, ce, a)
                        q.append(nxt)
        if goal is None:
            return ""
        # Ersten Schritt aus BFS-Baum rekonstruieren
        state = goal
        while parent[state][:2] != start:
            state = parent[state][:2]
        first_arm = parent[state][2]
        action = self._graph_exit_arm_to_turn(entry_arm, first_arm)
        rospy.logwarn(
            f"[explorer] Recovery: BFS-Pfad zu {goal[0]}, "
            f"erster Schritt Arm {first_arm} ({action})")
        return action

    @staticmethod
    def _graph_exit_arm_to_turn(entry_arm, exit_arm):
        """Übersetzung entry+exit -> action (ohne Abhängigkeit von city_graph)."""
        entry = ((int(entry_arm) - 1) % 4) + 1
        exit_ = ((int(exit_arm) - 1) % 4) + 1
        if exit_ == entry:
            return None
        # CLOCKWISE_NUMBERING = True
        sign = 1
        if (exit_ - entry) % 4 == sign:
            return "turn_left"
        if (exit_ - entry) % 4 == (4 - sign):
            return "turn_right"
        if (exit_ - entry) % 4 == 2:
            return "move_forward"
        return None

    def cbPose(self, msg):
        try:
            pose = json.loads(msg.data)
        except ValueError:
            return
        if pose.get("status") == "LOST":
            # Statt Vorschlag zu löschen: versuche Recovery
            visited = set(pose.get("visited", []))
            recovered = self._recover(
                pose.get("to_node", ""), pose.get("to_arm", ""), visited)
            if recovered:
                self._suggestion = recovered
                rospy.logwarn_once(
                    f"[explorer] LOST — Recovery-Vorschlag: '{recovered}'")
                return
            # Keine Recovery möglich -> Vorschlag löschen (Roboter steht)
            self._suggestion = ""
            return

        # jede befahrene Strasse dem aktuellen Sweep gutschreiben
        eid = pose.get("edge_id")
        if eid and eid != self._last_edge:
            self._last_edge = eid
            self.policy.note_edge(eid)

        # Empfehlung fuer die NAECHSTE Kreuzung (to_node, Ankunft to_arm)
        exit_arm, action, reason = self.policy.decide(
            pose["to_node"], pose["to_arm"],
            set(pose.get("visited", [])), self._validated_gates)

        if action is None:
            if not self._done:
                self._done = True
                rospy.loginfo(
                    f"[explorer] MAPPING FERTIG — {len(self._gates)} Tore,"
                    f" {self.policy.sweep} Sweep(s).")
            self._suggestion = ""
            return

        self._done = False
        if action != self._suggestion:
            rospy.loginfo(
                f"[explorer] Naechste Kreuzung {pose['to_node']}"
                f" (Eingang Arm {pose['to_arm']}): '{action}'"
                f" -> Arm {exit_arm} ({reason})")
        self._suggestion = action

    def run(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if self._suggestion:
                self.pub_suggest.publish(String(data=self._suggestion))
            # gates_complete: alle expected gates zugewiesen -> DuckieBot darf stoppen
            if self.expected_gates_map:
                eg = getattr(self.expected_gates_map, 'expected', None)
                gates_complete = (eg is not None
                                  and all(edge in self._validated_gates
                                          for edge in eg))
            else:
                gates_complete = True  # kein expected_gates -> keine Bedingung
            if gates_complete and not self._done:
                self._done = True
                rospy.loginfo(
                    f"[explorer] ALLE TORE RICHTIG ZUGEORDNET ({len(self._validated_gates)})"
                    f" — stoppe Erkundung.")
            self.pub_state.publish(String(data=json.dumps({
                "sweep": self.policy.sweep,
                "done": self._done,
                "num_gates": len(self._gates),
                "gates_complete": gates_complete,
                "suggestion": self._suggestion,
            })))
            rate.sleep()

    def shutdown(self):
        self.save_map()


if __name__ == '__main__':
    node = ExplorerNode('explorer_node')
    node.run()
