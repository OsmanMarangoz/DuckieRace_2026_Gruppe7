#!/usr/bin/env python3
"""
planner.py — Weighted Shortest-Path-Planer fuer Challenge 4 (Phase B).

BEWUSST OHNE ROS-IMPORTS: pure Python, offline testbar.

Funktionsweise:
  1. Gate-zu-Kanten-Zuordnung und Kantengewichte (Zeit) laden
  2. Fuer jede aufeinanderfolgende Gate-Paar Dijkstra auf Kanten
     anwenden (gewichtet, nicht nur Hops!)
  3. Gesamtpfad = Kette aller Zwischenschritte

Die Gewichte kommen aus der Map-Datei (Timing zwischen Aktionen
wahrend des Mappings). Kanten ohne Gewicht bekommen den Mittelwert.
"""

import heapq
import json
import math

from city_graph import CityGraph, wrap_arm, exit_arm_to_turn, TURN_ACTIONS


class Planner:
    """Plaene den schnellsten Pfad durch eine Sequenz von Toren."""

    def __init__(self, graph, gate_to_edge, edge_weights=None):
        self.graph = graph
        self.gate_to_edge = gate_to_edge  # {gate_id: edge_id}
        self.edge_weights = edge_weights or {}  # {edge_id: weight}

        # Precompute: build adjacency for fast lookup
        self._edge_map = {}  # (from_node, exit_arm) -> edge_id
        for node in self.graph.nodes():
            for arm in self.graph.arms_of(node):
                e = self.graph.edge_from(node, arm)
                if e:
                    self._edge_map[(node, arm)] = e

    def _weight(self, edge_id, entry_arm):
        """Gewicht einer Kante. Falls kein Gewicht bekannt -> Mittelwert."""
        w = self.edge_weights.get(edge_id)
        if w is not None:
            return w
        mean = self._mean_weight()
        if mean is not None:
            return mean
        return 1.0

    def _mean_weight(self):
        """Mittelwert aller bekannten Gewichte."""
        if not self.edge_weights:
            return None
        vals = [v for v in self.edge_weights.values() if v > 0]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _edge_to_node(edge_id):
        """Letzte(n) Knoten einer Kante extrahieren.

        edge_id ist kanonisch, z.B. 'A1-B1' => Knoten 'A'.
        Der Arm-Suffix (Ziffern nach dem Buchstaben) wird entfernt.
        """
        idx = edge_id.rfind("-")
        if idx == -1:
            return edge_id
        # Knotenname: alles vor dem Bindestrich, Arm-Suffix abschneiden
        node_part = edge_id[:idx]
        # Z.B. "A1" -> "A", "B3" -> "B"
        return ''.join(c for c in node_part if not c.isdigit())

    def _dijkstra(self, start_edge, target_edge):
        """Dijkstra-Kuerzester-Weg auf Kanten (gewichtet).

        Suchzustand = (knoten, eingangs-arm), damit Wenden ausgeschlossen
        bleiben. Distanz = Summe der Kantengewichte (Zeit).

        Ziel: die Kante target_edge traversieren.

        Rueckgabe: Liste von edge_ids oder None wenn unerklich.
        """
        if start_edge == target_edge:
            return []

        # Finde Startzustand: welchen Arm muss ich nehmen, um start_edge zu
        # betreten? start_edge ist kanonisch ("Xn-Ym").
        start_state = self._find_entry_state(start_edge)
        if start_state is None:
            return None

        # Dijkstra
        dist = {start_state: 0.0}
        parent = {start_state: None}
        heap = [(0.0, start_state)]
        visited = set()

        while heap:
            d, (node, entry_arm) = heapq.heappop(heap)
            if (node, entry_arm) in visited:
                continue
            visited.add((node, entry_arm))

            # Expandiere alle gueltigen Ausgaenge
            for exit_arm in self.graph.arms_of(node):
                if exit_arm == entry_arm:
                    continue  # Wende verboten
                e_id = self._edge_map.get((node, exit_arm))
                if e_id is None:
                    continue

                # Ziel erreicht? (Kante == target_edge)
                if e_id == target_edge:
                    # Pfad zurueckverfolgen
                    state = (node, entry_arm, exit_arm)
                    path = [e_id]
                    while parent.get(state) is not None:
                        pn, pe, prev_exit = parent[state]
                        prev_e = self._edge_map[(pn, prev_exit)]
                        path.append(prev_e)
                        state = (pn, pe)
                    return list(reversed(path))

                weight = self._weight(e_id, exit_arm)
                to_node = self._edge_to_node(e_id)
                # Der Eingang am Naechstknoten ist der Ausgangsarm
                # (weil der Graph ungerichtet ist: exit_arm am Knoten A
                #  ist entry_arm am Knoten B)
                to_entry_arm = exit_arm
                neighbor = (to_node, to_entry_arm)
                new_dist = d + weight
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    parent[neighbor] = (node, entry_arm, exit_arm)
                    heapq.heappush(heap, (new_dist, neighbor))

        return None  # unerklich

    def _find_entry_state(self, edge_id):
        """Finde (knoten, arm), um die Kante edge_id zu betreten."""
        for (node, arm), eid in self._edge_map.items():
            if eid == edge_id:
                return (node, arm)
        return None

    # --- Pfadplanung -------------------------------------------------------

    def plan_path(self, gate_sequence):
        """Kuerzester Pfad durch Gate-Sequenz.

        Rueckgabe: Liste von (gate_id, edge_id) Paaren oder None.
        """
        if not gate_sequence:
            return []

        result = []
        current_edge = None

        for gate_id in gate_sequence:
            target_edge = self.gate_to_edge.get(gate_id)
            if target_edge is None:
                raise ValueError(
                    f"Gate {gate_id} nicht in gate_to_edge. "
                    f"Fehlendes Gate oder falsche Map geladen?")

            if current_edge is not None:
                path = self._dijkstra(current_edge, target_edge)
                if path is None:
                    raise ValueError(
                        f"Gate {gate_id} ist nicht erreichbar von Gate "
                        f"{result[-1][0]}. Graph ist unzusammenhaengend.")

            current_edge = target_edge
            result.append((gate_id, target_edge))

        return result

    def edges_to_actions(self, edge_sequence):
        """Kanten-Liste in Action-Liste uebersetzen.

        Benoetigt entry_arm-Information aus der vorherigen Kante.
        Rueckgabe: [(action, edge_id), ...] wobei action None sein kann
        wenn entry_arm unbekannt.
        """
        actions = []
        prev_to_arm = None

        for edge_id in edge_sequence:
            for (node, arm), eid in self._edge_map.items():
                if eid == edge_id:
                    if prev_to_arm is not None:
                        action = exit_arm_to_turn(prev_to_arm, arm)
                        actions.append((action, edge_id))
                    else:
                        actions.append((None, edge_id))
                    prev_to_arm = arm
                    break
            else:
                actions.append((None, edge_id))

        return actions

    # --- Start-Validierung -------------------------------------------------

    def compute_start_edge(self, start_node, start_exit_arm):
        """Kante, die man abfaehrt, wenn man start_node ueber
        start_exit_arm verlaesst."""
        e = self.graph.edge_from(start_node, start_exit_arm)
        if e is None:
            raise ValueError(
                f"Start ungueltig: {start_node} hat keinen Arm {start_exit_arm}")
        return e["edge_id"]

    def validate_start_edge(self, start_edge, first_gate_edge):
        """Pruefen ob die Startkante mit dem ersten Gate uebereinstimmt."""
        if start_edge != first_gate_edge:
            gate_label = self._gate_for_edge(first_gate_edge)
            raise ValueError(
                f"Start-Kante '{start_edge}' stimmt nicht mit erstem Gate "
                f"'{first_gate_edge}' (Gate {gate_label}) ueberein.\n"
                f"Bitte den Roboter auf die Kante {first_gate_edge} stellen.")

    def _gate_for_edge(self, edge_id):
        for gid, eid in self.gate_to_edge.items():
            if eid == edge_id:
                return gid
        return "?"

    # --- Stats -------------------------------------------------------------

    def edge_stats(self):
        """Statistiken zu den Kantengewichten."""
        if not self.edge_weights:
            return "Keine Kantengewichte vorhanden."
        return {
            "edges": len(self.edge_weights),
            "mean": self._mean_weight(),
            "min": min(self.edge_weights.values()) if self.edge_weights else 0,
            "max": max(self.edge_weights.values()) if self.edge_weights else 0,
        }
