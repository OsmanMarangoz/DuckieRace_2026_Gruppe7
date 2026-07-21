#!/usr/bin/env python3
"""
gate_mapper_node — Phase A (Challenge 4)

Ordnet Tor-AprilTags (IDs 5-13) der Strasse zu, auf der der Bot gerade
faehrt (laut localization_node). PASSIV — kein Einfluss aufs Fahren.

Regeln:
  - Zuordnung nur im Lane-Modus (auf der Strecke; an Kreuzungen wird
    nicht zugeordnet, da dort schon das naechste/vorherige Tor sichtbar
    sein koennte).
  - Max. 1 Tor pro Kante (laut Orga) -> Konflikte werden geloggt, die
    ERSTE Zuordnung gewinnt (haeufigste Sichtung = fruehe Sichtung).
  - Ungerichteter Graph: gleiches Tor von beiden Spuren -> gleiche ID
    auf gleicher Kante -> Dedup ist trivial.
  - VALIDIERUNG: ein Tor wird NUR dann einer Kante zugeordnet, wenn es
    im expected_gates (aus city_graph.json) auf dieser Kante steht.
    Tore ohne match werden verworfen und geloggt.

Publisht:
  /<veh>/mapping/gates         (String, JSON)  — Dashboard: die Karte
  /<veh>/mapping/validated_gates (String, JSON) — nur validierte Tore

Speichert bei Abdeckungs-Komplettierung (und bei Shutdown) die fertige
Karte als JSON — Input fuer Phase B.

Params:
  ~map_output       Zieldatei (default: /tmp/duckie_city_map.json)
  ~expected_gates   Anzahl Tore die mindestens gefunden werden muessen
                    (default: 5). Der Mapper wartet bis sowohl alle
                    Kanten besucht ALS auch diese Anzahl Tore gefunden
                    wurden.
  ~min_gate_area    Mindestflaeche eines Tor-Tags im Bild (default: 390).
                    Nur Tags mit mindestens dieser Flaeche werden als
                    erkannt gewertet (verhindert, dass sehr ferne Tore
                    falsch zugeordnet werden).
"""

import json
import os

import rospy
from std_msgs.msg import Bool, Int32, String

from city_graph import GATE_TAG_IDS, ExpectedGatesMap
from switch_control_node import ControlType


class GateMapperNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._map_output = rospy.get_param("~map_output", "/tmp/duckie_city_map.json")
        self._expected_gates = int(rospy.get_param("~expected_gates", 5))
        self._min_gate_area = int(rospy.get_param("~min_gate_area", 400))
        # expected_gates aus city_graph.json laden
        graph_file = os.path.join(os.path.dirname(__file__), "../config/city_graph.json")
        self.expected_gates = None
        try:
            with open(graph_file) as f:
                data = json.load(f)
            eg = data.get("expected_gates")
            if eg:
                self.expected_gates = ExpectedGatesMap(eg)
        except (ValueError, KeyError, OSError):
            pass

        self.gates = {}            # edge_id -> gate_tag_id
        self.validated_gates = {}  # edge_id -> gate_tag_id (nur wenn match zu expected)
        self.sightings = {}        # gate_tag_id -> {edge_id: count} (Diagnose)
        self._pose = None
        self._mode = ControlType.Lane.value
        self._coverage_complete = False
        self._gates_complete = False
        self._last_gate_area = {}        # tag_id -> letzte Flaeche
        self._started_detecting = False  # erst nach erstem Stopp aktivieren

        self.pub_gates = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/gates', String, queue_size=1)
        self.pub_validated = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/validated_gates', String, queue_size=1)

        rospy.Subscriber(f'/{self._vehicle_name}/apriltag/id', Int32,
                         self.cbTagId, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/apriltag/detections', String,
                         self.cbDetections, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/mapping/pose', String,
                         self.cbPose, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/switch/control', Int32,
                         self.cbMode, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/mapping/complete', Bool,
                         self.cbComplete, queue_size=1)
        rospy.Subscriber(f'/{self._vehicle_name}/decision/action', String,
                         self.cbAction, queue_size=1)

        rospy.on_shutdown(self.save_map)
        rospy.loginfo(f"[{node_name}] Gate-Mapper bereit -> {self._map_output}")
        self._publish_validated()

    def cbMode(self, msg):
        self._mode = msg.data

    def cbPose(self, msg):
        try:
            self._pose = json.loads(msg.data)
        except ValueError:
            pass

    def cbDetections(self, msg):
        """Empfaengt alle erkannten Tags und merkt sich die Flaeche der Tore."""
        if not self._started_detecting:
            return
        try:
            detections = json.loads(msg.data)
            for det in detections:
                tag_id = det.get("id")
                area = det.get("area", 0)
                if tag_id in GATE_TAG_IDS and area > 0:
                    self._last_gate_area[tag_id] = area
        except (ValueError, TypeError):
            pass

    def cbTagId(self, msg):
        tag = msg.data
        if tag not in GATE_TAG_IDS:
            return
        if not self._started_detecting:
            return  # noch nicht beim ersten Stopp angelangt
        if self._mode != ControlType.Lane.value:
            return  # an der Kreuzung nicht zuordnen
        if self._pose is None or self._pose.get("status") != "ON_EDGE":
            return
        edge = self._pose["edge_id"]

        # NUR annehmen wenn Tag gross genug (also nah genug)
        area = self._last_gate_area.get(tag, 0)
        if area < self._min_gate_area:
            rospy.loginfo(
                f"[gate_mapper] Tor {tag} auf {edge} -> IGNORIERT (area={area}"
                f" < min_gate_area={self._min_gate_area})")
            return

        self.sightings.setdefault(tag, {})
        self.sightings[tag][edge] = self.sightings[tag].get(edge, 0) + 1

        existing = self.gates.get(edge)
        if existing is None:
            # steht dieses Tor schon auf einer anderen Kante?
            other = [e for e, g in self.gates.items() if g == tag]
            if other:
                rospy.logwarn(
                    f"[gate_mapper] Tor {tag} auf {edge} gesehen, aber schon"
                    f" {other[0]} zugeordnet — ignoriere (Sichtung geloggt).")
                return

            # VALIDIERUNG: passt das Tor zur erwarteten Position?
            if self.expected_gates and not self.expected_gates.validate_gate(edge, tag):
                rospy.loginfo(
                    f"[gate_mapper] Tor {tag} auf {edge} gesehen, aber"
                    f" nicht erwartet — ignoriere (kein match in expected_gates)."
                    f" (expected_gates={self.expected_gates.expected})")
                return

            self.gates[edge] = tag
            self.validated_gates[edge] = tag
            rospy.loginfo(
                f"[gate_mapper] TOR {tag} -> Strasse {edge} (area={area})")
            self.publish_gates()
            self._publish_validated()
            self._maybe_save()   # neu: auch wenn noch nicht alle Kanten besucht
        elif existing != tag:
            rospy.logwarn(
                f"[gate_mapper] Konflikt auf {edge}: {existing} zugeordnet,"
                f" jetzt {tag} gesehen — behalte {existing} (max 1 Tor/Kante).")

    def _maybe_save(self):
        """Speichert Karte wenn ALLE Bedingungen erfellt: Kanten+Tore."""
        if (self._coverage_complete and not self._gates_complete
                and len(self.gates) >= self._expected_gates):
            self._gates_complete = True
            rospy.loginfo(
                f"[gate_mapper] GENUG TORE GEFUNDEN ({len(self.gates)}/{self._expected_gates})"
                f" — speichere Karte.")
            self.save_map()

    def cbComplete(self, msg):
        if msg.data and not self._coverage_complete:
            self._coverage_complete = True
            self._maybe_save()

    def cbAction(self, msg):
        # Erster Stopp = Start der Tor-Detektion
        if msg.data == "stopping" and not self._started_detecting:
            self._started_detecting = True
            rospy.loginfo("[gate_mapper] Tor-Detektion aktiviert (erster Stopp)")

    def publish_gates(self):
        payload = {
            "gates": self.gates,                       # edge_id -> tag
            "gates_by_tag": {str(t): e for e, t in self.gates.items()},
            "num_gates": len(self.gates),
            "sightings": self.sightings,
        }
        self.pub_gates.publish(String(data=json.dumps(payload)))

    def save_map(self):
        data = {
            "gates": self.gates,
            "gates_by_tag": {str(t): e for e, t in self.gates.items()},
            "pose_at_save": self._pose,
        }
        try:
            with open(self._map_output, "w") as f:
                json.dump(data, f, indent=2)
            rospy.loginfo(
                f"[gate_mapper] Karte gespeichert: {self._map_output}"
                f" ({len(self.gates)} Tore)")
        except OSError as e:
            rospy.logerr(f"[gate_mapper] Speichern fehlgeschlagen: {e}")

    def _publish_validated(self):
        """Publiziert die validierten Tore (nur wenn match zu expected_gates)."""
        payload = {
            "gates": self.validated_gates,
            "num_validated": len(self.validated_gates),
        }
        self.pub_validated.publish(String(data=json.dumps(payload)))

    def run(self):
        rate = rospy.Rate(1)
        _last_diagnostic = -1
        while not rospy.is_shutdown():
            self.publish_gates()
            self._publish_validated()
            now = rospy.get_time()
            if int(now / 30) != _last_diagnostic:
                _last_diagnostic = int(now / 30)
                rospy.loginfo_once(
                    f"[gate_mapper] Start: expected_gates={self._expected_gates}"
                    f" min_gate_area={self._min_gate_area}"
                    f" detection_active={self._started_detecting}"
                    f" expected_gates_loaded={self.expected_gates is not None}")
            rate.sleep()


if __name__ == '__main__':
    node = GateMapperNode('gate_mapper_node')
    node.run()
