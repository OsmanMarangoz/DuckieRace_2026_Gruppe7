#!/usr/bin/env python3
"""
mapping_debug_overlay — Publiziert ein Debug-Bild mit Mapping-Info als
overlay auf einem CompressedImage-Topic (fuer RQT).

Publisht: /<veh>/mapping/debug/compressed  (CompressedImage)

Benutzung:
    rosrun follow_lane mapping_debug_overlay.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import rospy
from std_msgs.msg import Bool, String
from sensor_msgs.msg import CompressedImage


class MappingDebugOverlay:
    def __init__(self, node_name):
        rospy.init_node(node_name)
        vehicle = os.environ.get("VEHICLE_NAME", "<unknown>")
        self._prefix = f"/{vehicle}"

        self._pose = {}
        self._complete = False
        self._gates = {}
        self._gates_by_tag = {}
        self._suggested_action = ""
        self._explore_state = {}

        # Subscriber
        rospy.Subscriber(
            f"{self._prefix}/mapping/pose", String,
            self._cb_pose, queue_size=1)
        rospy.Subscriber(
            f"{self._prefix}/mapping/complete", Bool,
            self._cb_complete, queue_size=1)
        rospy.Subscriber(
            f"{self._prefix}/mapping/gates", String,
            self._cb_gates, queue_size=1)
        rospy.Subscriber(
            f"{self._prefix}/explore/suggested_action", String,
            self._cb_suggested, queue_size=1)
        rospy.Subscriber(
            f"{self._prefix}/explore/state", String,
            self._cb_explore_state, queue_size=1)

        # Publisher — eigenes Debug-Topic
        self.pub_debug = rospy.Publisher(
            f'{self._prefix}/mapping/debug/compressed',
            CompressedImage, queue_size=1)

        # Leeres Bild als Basis (wird mit Text ueberschrieben)
        self._img = np.zeros((480, 640, 3), dtype=np.uint8)

        self._last_update = time.time()

        rospy.loginfo("[mapping-overlay] Publiziert auf "
                      f"{self._prefix}/mapping/debug/compressed")

    def _cb_pose(self, msg):
        try:
            self._pose = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _cb_complete(self, msg):
        self._complete = bool(msg.data)

    def _cb_gates(self, msg):
        try:
            data = json.loads(msg.data)
            self._gates = data.get("gates", {})
            self._gates_by_tag = data.get("gates_by_tag", {})
        except (ValueError, TypeError):
            pass

    def _cb_suggested(self, msg):
        self._suggested_action = msg.data

    def _cb_explore_state(self, msg):
        try:
            self._explore_state = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _draw(self):
        now = time.time()
        if now - self._last_update < 0.3:  # ~3 Hz
            return
        self._last_update = now

        img = self._img.copy()
        vehicle = os.environ.get("VEHICLE_NAME", "?")

        y = 18
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        # Titel
        cv2.putText(img, f"MAPPING DEBUG — {vehicle}", (10, y),
                    font, scale, (255, 255, 255), thickness)
        y += 24

        # Trennlinie
        cv2.line(img, (10, y), (630, y), (80, 80, 80), 1)
        y += 10

        # ── Status ──
        status = self._pose.get("status", "???")
        status_color = (255, 255, 255)
        if status == "ON_EDGE":
            status_color = (0, 255, 0)
        elif status == "AT_INTERSECTION":
            status_color = (255, 255, 0)
        elif status == "LOST":
            status_color = (0, 0, 255)

        cv2.putText(img, f"Status:     {status}", (15, y), font, scale,
                    status_color, thickness)
        y += 18

        edge = self._pose.get("edge_id", "—")
        to_node = self._pose.get("to_node", "—")
        to_arm = self._pose.get("to_arm", "—")
        cv2.putText(img, f"Auf Kante:  {edge}", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 18
        cv2.putText(img, f"Richtung →: {to_node} (Arm {to_arm})", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 18

        num_visited = self._pose.get("num_visited", "?")
        num_total = self._pose.get("num_total", "?")
        cv2.putText(img, f"Besucht:    {num_visited}/{num_total}", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 28

        # ── Exploration ──
        cv2.line(img, (10, y), (630, y), (80, 80, 80), 1)
        y += 10

        sweep = self._explore_state.get("sweep", "?")
        explorer_done = self._explore_state.get("done", False)
        explorer_gates = self._explore_state.get("num_gates", "?")
        suggestion = self._suggested_action

        cv2.putText(img, f"Sweep:      {sweep}", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 18
        cv2.putText(img, f"Explorer:   {'DONE' if explorer_done else 'aktiv'}", (15, y),
                    font, scale, (0, 255, 0) if explorer_done else (255, 255, 255),
                    thickness)
        y += 18
        cv2.putText(img, f"Gef. Tore:  {explorer_gates}", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 18
        sug_color = (255, 255, 255) if suggestion else (80, 80, 80)
        cv2.putText(img, f"Nächste:    {suggestion}", (15, y), font, scale,
                    sug_color, thickness)
        y += 28

        # ── Tore ──
        cv2.line(img, (10, y), (630, y), (80, 80, 80), 1)
        y += 10

        cv2.putText(img, "GEFUNDENE TORE:", (15, y), font, scale,
                    (255, 255, 255), thickness)
        y += 18

        if self._gates:
            cv2.putText(img, f"{'Kante':<20} {'Tor-ID':>6}  {'Tag':>6}",
                        (15, y), font, 0.45, (200, 200, 200), 1)
            y += 14
            for eid, tid in sorted(self._gates.items()):
                tag_id = self._gates_by_tag.get(str(tid), "—")
                cv2.putText(img, f"{eid:<20} {tid:>6}  {tag_id:>6}",
                            (15, y), font, 0.45, (200, 255, 200), 1)
                y += 13
        else:
            cv2.putText(img, "(noch keine Tore)", (15, y), font, 0.45,
                        (100, 100, 100), 1)
        y += 18

        # ── Abdeckung ──
        cv2.line(img, (10, y), (630, y), (80, 80, 80), 1)
        y += 10

        cov_color = (0, 255, 0) if self._complete else (0, 0, 255)
        cov_text = "ALLE BESUCHT!" if self._complete else "Noch nicht fertig."
        cv2.putText(img, f"Abdeckung:  {cov_text}", (15, y), font, scale,
                    cov_color, thickness)

        # Publish
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
        self.pub_debug.publish(msg)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self._draw()
            rate.sleep()


if __name__ == "__main__":
    MappingDebugOverlay("mapping_debug_overlay").run()
