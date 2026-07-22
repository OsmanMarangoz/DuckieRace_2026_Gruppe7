#!/usr/bin/env python3
"""
timing_node — Misst die Zeit pro Kante und speichert sie in der Map.

WAHREND DES MAPPINGS:
  decision_node publiziert bei jeder Aktion die Zeit seit der letzten
  Aktion zusammen mit der aktuellen Kante. timing_node sammelt diese
  Werte und schreibt sie (beim Herunterfahren) in eine SEPARATE
  Datei 'duckie_city_map_weights.json' — die Map-Datei bleibt
  unangetastet (mit den Gates von gate_mapper).

Publisht:
  /<veh>/mapping/edge_stats (String, JSON) — Statistik der Kantengewichte
"""

import json
import os
import atexit

import rospy
from std_msgs.msg import String, Bool


class TimingNode:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._map_output = rospy.get_param("~map_output", "/tmp/duckie_city_map.json")
        self._weights_file = rospy.get_param("~weights_output", "/tmp/duckie_city_map_weights.json")
        self._weights = {}  # edge_id -> [times, ...]
        self._last_msg_time = 0.0
        self._flush_interval = rospy.get_param("~flush_interval", 30.0)
        self._mapping_complete = False

        self.pub_stats = rospy.Publisher(
            f'/{self._vehicle_name}/mapping/edge_stats', String, queue_size=1)

        rospy.Subscriber(
            f'/{self._vehicle_name}/mapping/edge_time', String,
            self.cb_edge_time, queue_size=1)

        # Halt vom Explorer (nicht mapping/complete — gate_mapper
        # schreibt danach noch in die Map und wuerde die Gewichte
        # ueberschreiben). Der Explorer haltet NACHdem gate_mapper
        # alle Tore zugewiesen hat.
        rospy.Subscriber(
            f'/{self._vehicle_name}/explore/halt', Bool,
            self.cb_mapping_complete, queue_size=1)

        atexit.register(self._save)
        rospy.on_shutdown(self._save)

        rospy.loginfo(f"[{node_name}] Timing-Node bereit -> {self._map_output}")

    def cb_edge_time(self, msg):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return

        edge_id = data.get("edge_id")
        seconds = data.get("seconds")

        if not edge_id or seconds is None:
            return

        self._weights.setdefault(edge_id, [])
        self._weights[edge_id].append(seconds)
        rospy.loginfo(
            f"[timing_node] Kante {edge_id}: {seconds:.2f}s "
            f"(Messung {len(self._weights[edge_id])})")

        # Eine eventuell knapp nach dem Halt eintreffende letzte Messung muss
        # die bereits geschriebene Datei noch aktualisieren.
        if self._mapping_complete:
            self._save()

    def cb_mapping_complete(self, msg):
        if msg.data and not self._mapping_complete:
            self._mapping_complete = True
            rospy.loginfo("[timing_node] Mapping abgeschlossen — speichere Gewichte")
            self._save()

    def _save(self):
        """Schreibe Kantengewichte in SEPARATE Datei — zerstört NICHT die
        Map-Datei mit den Gates."""
        # Kantengewichte berechnen (Mittelwert)
        weights = {}
        for edge_id, times in self._weights.items():
            weights[edge_id] = sum(times) / len(times)

        try:
            with open(self._weights_file, "w") as f:
                json.dump(weights, f, indent=2)
            rospy.loginfo(
                f"[timing_node] Kantengewichte gespeichert: "
                f"{len(weights)} Kanten -> {self._weights_file}")
            for edge_id in sorted(weights):
                rospy.loginfo(
                    f"[timing_node] Gewicht {edge_id}: "
                    f"{weights[edge_id]:.2f}s "
                    f"({len(self._weights[edge_id])} Messung(en))")
        except OSError as e:
            rospy.logerr(f"[timing_node] Speichern fehlgeschlagen: {e}")

        # Statistik publizieren
        self._publish_stats()

    def _publish_stats(self):
        if not self._weights:
            payload = {"count": 0}
        else:
            all_times = [t for times in self._weights.values() for t in times]
            payload = {
                "edges": len(self._weights),
                "total_measurements": len(all_times),
                "mean": sum(all_times) / len(all_times) if all_times else 0,
                "min": min(all_times) if all_times else 0,
                "max": max(all_times) if all_times else 0,
            }

        self.pub_stats.publish(String(data=json.dumps(payload)))

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self._publish_stats()
            rate.sleep()


if __name__ == '__main__':
    node = TimingNode('timing_node')
    node.run()
