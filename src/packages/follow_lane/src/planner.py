#!/usr/bin/env python3
"""Gewichteter Routenplaner fuer die Gate-Reihenfolge (Challenge 4, Phase B).

Dieses Modul hat bewusst keine ROS-Abhaengigkeit, damit der Graph und die
Routen offline testbar bleiben.  Ein Planungszustand ist immer
``(Kreuzung, Eingangsarm)``.  Das ist wichtig: Eine Strassen-ID wie
``A1-B1`` ist ungerichtet und verraet allein nicht, an welcher Kreuzung der
Roboter danach steht.
"""

import heapq

from city_graph import exit_arm_to_turn


ACTION_LABELS = {
    "turn_left": "links abbiegen",
    "turn_right": "rechts abbiegen",
    "move_forward": "geradeaus fahren",
}


class Planner:
    """Berechnet die schnellste erlaubte Route durch eine feste Gate-Folge."""

    def __init__(self, graph, gate_to_edge, edge_weights=None):
        self.graph = graph
        self.gate_to_edge = {
            int(gate_id): str(edge_id)
            for gate_id, edge_id in gate_to_edge.items()
        }
        self.edge_weights = edge_weights or {}

        # Kompatibilitaet zu den bisherigen Offline-Helfern: Nur IDs, keine
        # edge_from()-Dictionaries speichern.
        self._edge_map = {}
        self._edge_ids = set()
        for node in self.graph.nodes():
            for arm in self.graph.arms_of(node):
                edge = self.graph.edge_from(node, arm)
                self._edge_map[(node, arm)] = edge["edge_id"]
                self._edge_ids.add(edge["edge_id"])

    # ---------------------------------------------------------------- Gewichte

    def _weight(self, edge_id):
        """Zeitgewicht einer Kante; unbekannte Kanten erhalten den Mittelwert."""
        try:
            weight = float(self.edge_weights.get(edge_id))
            if weight > 0:
                return weight
        except (TypeError, ValueError):
            pass

        mean = self._mean_weight()
        return mean if mean is not None else 1.0

    def _mean_weight(self):
        """Mittelwert aller brauchbaren gemessenen Kantengewichte."""
        values = []
        for value in self.edge_weights.values():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        return sum(values) / len(values) if values else None

    def _weight_source(self, edge_id):
        """Quelle des verwendeten Gewichts fuer die Diagnose."""
        try:
            if float(self.edge_weights.get(edge_id)) > 0:
                return "gemessen"
        except (TypeError, ValueError):
            pass
        return "Mittelwert" if self._mean_weight() is not None else "Hop-Fallback"

    # ----------------------------------------------------------- Graph-Helfer

    def _transitions(self, state, skip_edge=None):
        """Alle fahrbaren Uebergaenge vom Planungszustand aus liefern."""
        node, entry_arm = state
        for exit_arm in self.graph.arms_of(node):
            action = exit_arm_to_turn(entry_arm, exit_arm)
            if action is None:
                continue
            edge = self.graph.edge_from(node, exit_arm)
            if edge["edge_id"] == skip_edge:
                continue
            next_state = (edge["to_node"], edge["to_arm"])
            yield edge, next_state, action

    def _transition_for_edge(self, state, edge_id):
        """Den fahrbaren Uebergang fuer ``edge_id`` aus ``state`` finden."""
        for edge, next_state, action in self._transitions(state):
            if edge["edge_id"] == edge_id:
                return edge, next_state, action
        return None

    @staticmethod
    def _reconstruct(parent, state):
        """Kantenfolge vom Dijkstra-Start bis ``state`` rekonstruieren."""
        path = []
        while parent[state] is not None:
            previous_state, edge_id = parent[state]
            path.append(edge_id)
            state = previous_state
        return list(reversed(path))

    def _paths_to_edge(self, start_state, target_edge):
        """Kuerzeste Pfade, die ``target_edge`` als letzte Kante befahren.

        Rueckgabe ist ``{end_state: (kosten, kanten)}``.  Es bleiben beide
        moeglichen Fahrtrichtungen durch das Zieltor erhalten.  Dadurch kann
        die anschliessende Gate-Etappe bei der Gesamtplanung beruecksichtigt
        werden, statt an einem Tor vorschnell die falsche Richtung zu waehlen.
        """
        if target_edge not in self._edge_ids:
            return {}
        if start_state is None:
            return {}

        try:
            self.graph.arms_of(start_state[0])
        except (KeyError, TypeError):
            return {}

        # Zielkante waehrend der Suche auslassen.  Sie wird erst als letzter
        # Schritt angehaengt; mit positiven Gewichten kann das keinen besten
        # Pfad ausschliessen.
        dist = {start_state: 0.0}
        parent = {start_state: None}
        heap = [(0.0, start_state)]

        while heap:
            distance, state = heapq.heappop(heap)
            if distance != dist.get(state):
                continue

            for edge, next_state, _ in self._transitions(state, skip_edge=target_edge):
                new_distance = distance + self._weight(edge["edge_id"])
                if new_distance < dist.get(next_state, float("inf")):
                    dist[next_state] = new_distance
                    parent[next_state] = (state, edge["edge_id"])
                    heapq.heappush(heap, (new_distance, next_state))

        # Die Zielkante kann von beiden Seiten und mit unterschiedlichen
        # Eingangsarmen erreicht werden.  Fuer jeden resultierenden Zustand
        # behalten wir nur die beste Variante.
        result = {}
        for state, distance in dist.items():
            transition = self._transition_for_edge(state, target_edge)
            if transition is None:
                continue
            edge, end_state, _ = transition
            candidate = (
                distance + self._weight(edge["edge_id"]),
                self._reconstruct(parent, state) + [target_edge],
            )
            previous = result.get(end_state)
            if previous is None or candidate[0] < previous[0]:
                result[end_state] = candidate
        return result

    def _dijkstra(self, start_state, target_edge):
        """Kuerzeste Kantenfolge von ``start_state`` durch ``target_edge``.

        Diese kleine Wrapper-Methode bleibt fuer die bisherigen Offline-Tests
        erhalten.  Die Gesamtplanung nutzt intern beide Endrichtungen des
        Zieltors via :meth:`_paths_to_edge`.
        """
        paths = self._paths_to_edge(start_state, target_edge)
        if not paths:
            return None
        return min(paths.values(), key=lambda item: item[0])[1]

    # -------------------------------------------------------------- Planung

    def plan_route(self, gate_sequence, start_node, start_exit_arm):
        """Plane eine Route ab der bekannten, bereits befahrenen Startkante.

        ``start_node`` und ``start_exit_arm`` beschreiben wie beim Mapper die
        aktuelle Fahrtrichtung: Der Bot befindet sich auf der Kante, die
        ``start_node`` ueber diesen Arm verlaesst.  Diese unvermeidbare
        Startkante wird in ``edges`` aufgefuehrt, braucht aber keine Action,
        weil die erste Entscheidung erst an ihrer Zielkreuzung faellt.

        Die Gate-Reihenfolge ist fest.  Dynamische Programmierung waehlt aber
        an jedem Gate die Richtung, die zusammen mit den folgenden Gates die
        kleinste Gesamtzeit ergibt.
        """
        gate_sequence = [int(gate_id) for gate_id in gate_sequence]
        if not gate_sequence:
            return {
                "gates": [], "gate_edges": [], "edges": [], "actions": [],
                "steps": [],
                "total_weight": 0.0,
            }

        start_edge_id = self.compute_start_edge(start_node, start_exit_arm)
        start_edge = self.graph.edge_from(start_node, start_exit_arm)
        start_state = (start_edge["to_node"], start_edge["to_arm"])

        gate_edges = []
        for gate_id in gate_sequence:
            target_edge = self.gate_to_edge.get(gate_id)
            if target_edge is None:
                raise ValueError(
                    f"Gate {gate_id} nicht in gate_to_edge. "
                    "Fehlendes Gate oder falsche Map geladen?")
            if target_edge not in self._edge_ids:
                raise ValueError(
                    f"Gate {gate_id} verweist auf unbekannte Kante "
                    f"'{target_edge}'.")
            gate_edges.append(target_edge)

        # state -> (bisherige Zeit, befahrene Kanten inkl. Startkante)
        candidates = {
            start_state: (self._weight(start_edge_id), [start_edge_id])
        }

        for index, (gate_id, target_edge) in enumerate(zip(gate_sequence, gate_edges)):
            # Das erste Tor kann auf der Kante stehen, auf der der Bot bereits
            # unterwegs ist.  Es ist beim Erreichen der ersten Kreuzung schon
            # passiert und erzeugt deshalb keine Kreuzungs-Action.
            if index == 0 and target_edge == start_edge_id:
                continue

            next_candidates = {}
            for state, (base_cost, base_edges) in candidates.items():
                for end_state, (leg_cost, leg_edges) in self._paths_to_edge(
                        state, target_edge).items():
                    candidate = (base_cost + leg_cost, base_edges + leg_edges)
                    previous = next_candidates.get(end_state)
                    if previous is None or candidate[0] < previous[0]:
                        next_candidates[end_state] = candidate

            if not next_candidates:
                if index == 0:
                    raise ValueError(
                        f"Erstes Gate {gate_id} ist nicht erreichbar vom "
                        f"Start ({start_node}/{start_exit_arm}).")
                raise ValueError(
                    f"Gate {gate_id} ist nicht erreichbar von Gate "
                    f"{gate_sequence[index - 1]}. Graph ist unzusammenhaengend.")
            candidates = next_candidates

        final_state, (total_weight, edge_path) = min(
            candidates.items(), key=lambda item: item[1][0])

        # Die Startkante wird schon gefahren.  Alle weiteren Kanten werden
        # an Kreuzungen als konkrete Actions an den decision_node gegeben.
        actions = []
        first_gate_on_start_edge = gate_edges[0] == start_edge_id
        steps = [{
            "kind": "start",
            "from_node": start_node,
            "exit_arm": int(start_exit_arm),
            "edge": start_edge_id,
            "to_node": start_edge["to_node"],
            "to_arm": start_edge["to_arm"],
            "weight": self._weight(start_edge_id),
            "weight_source": self._weight_source(start_edge_id),
            "cumulative_weight": self._weight(start_edge_id),
            "reason": (
                f"Tor {gate_sequence[0]} liegt auf der Startkante"
                if first_gate_on_start_edge
                else f"Startkante; danach Weg zu Tor {gate_sequence[0]}"
            ),
        }]
        next_gate_index = 1 if first_gate_on_start_edge else 0
        state = start_state
        cumulative_weight = self._weight(start_edge_id)
        for edge_id in edge_path[1:]:
            transition = self._transition_for_edge(state, edge_id)
            if transition is None:
                raise RuntimeError(
                    f"Interner Planungsfehler: {edge_id} ist von {state} "
                    "nicht fahrbar.")
            edge, next_state, action = transition
            actions.append(action)
            reaches_gate = (
                next_gate_index < len(gate_edges)
                and edge_id == gate_edges[next_gate_index]
            )
            if reaches_gate:
                reason = f"Tor {gate_sequence[next_gate_index]} erreichen"
                next_gate_index += 1
            else:
                reason = f"Weg zu Tor {gate_sequence[next_gate_index]}"
            edge_weight = self._weight(edge_id)
            cumulative_weight += edge_weight
            alternatives = [{
                "action": candidate_action,
                "action_label": ACTION_LABELS[candidate_action],
                "edge": candidate_edge["edge_id"],
                "weight": self._weight(candidate_edge["edge_id"]),
                "weight_source": self._weight_source(candidate_edge["edge_id"]),
            } for candidate_edge, _, candidate_action in self._transitions(state)]
            steps.append({
                "kind": "action",
                "at_node": state[0],
                "entry_arm": state[1],
                "action": action,
                "action_label": ACTION_LABELS[action],
                "exit_arm": edge["from_arm"],
                "edge": edge_id,
                "to_node": next_state[0],
                "to_arm": next_state[1],
                "weight": edge_weight,
                "weight_source": self._weight_source(edge_id),
                "cumulative_weight": cumulative_weight,
                "alternatives": alternatives,
                "reason": reason,
            })
            state = next_state

        if state != final_state:
            raise RuntimeError("Interner Planungsfehler: Endzustand stimmt nicht.")

        return {
            "gates": gate_sequence,
            "gate_edges": gate_edges,
            "edges": edge_path,
            "actions": actions,
            "steps": steps,
            "total_weight": total_weight,
        }

    def plan_path(self, gate_sequence, start_node, start_exit_arm):
        """Abwaertskompatible Kurzform: nur die verlangten Gate-Kanten."""
        route = self.plan_route(gate_sequence, start_node, start_exit_arm)
        return list(zip(route["gates"], route["gate_edges"]))

    def edges_to_actions(self, edge_sequence):
        """Alter Diagnose-Helfer fuer eine zusammenhaengende Kantenfolge.

        Fuer die echte Planung wird :meth:`plan_route` verwendet, weil dort
        der Startzustand bekannt ist.  Die erste Kante hat hier absichtlich
        keine Action, da ihr Eingangsarm nicht uebergeben wurde.
        """
        if not edge_sequence:
            return []

        actions = [(None, edge_sequence[0])]
        start_state = self._find_entry_state(edge_sequence[0])
        if start_state is None:
            return actions + [(None, edge_id) for edge_id in edge_sequence[1:]]

        first = self._transition_for_edge(start_state, edge_sequence[0])
        if first is None:
            return actions + [(None, edge_id) for edge_id in edge_sequence[1:]]
        _, state, _ = first

        for edge_id in edge_sequence[1:]:
            transition = self._transition_for_edge(state, edge_id)
            if transition is None:
                actions.append((None, edge_id))
                continue
            _, state, action = transition
            actions.append((action, edge_id))
        return actions

    def _find_entry_state(self, edge_id):
        """Einen Ausgangszustand fuer die erste Kante eines Diagnosepfads."""
        for node in self.graph.nodes():
            for entry_arm in self.graph.arms_of(node):
                if self._transition_for_edge((node, entry_arm), edge_id):
                    return node, entry_arm
        return None

    # ----------------------------------------------------------- Start/Stats

    def compute_start_edge(self, start_node, start_exit_arm):
        """Kante, auf der der Bot laut Startparametern bereits unterwegs ist."""
        edge = self.graph.edge_from(start_node, start_exit_arm)
        if edge is None:
            raise ValueError(
                f"Start ungueltig: {start_node} hat keinen Arm {start_exit_arm}")
        return edge["edge_id"]

    def validate_start_edge(self, start_edge, first_gate_edge):
        """Optionaler Diagnose-Helfer fuer den alten Start-auf-Gate-Fall."""
        if start_edge != first_gate_edge:
            gate_label = self._gate_for_edge(first_gate_edge)
            raise ValueError(
                f"Start-Kante '{start_edge}' stimmt nicht mit erstem Gate "
                f"'{first_gate_edge}' (Gate {gate_label}) ueberein.")

    def _gate_for_edge(self, edge_id):
        for gate_id, known_edge_id in self.gate_to_edge.items():
            if known_edge_id == edge_id:
                return gate_id
        return "?"

    def edge_stats(self):
        """Statistiken zu den geladenen Kantengewichten."""
        values = []
        for value in self.edge_weights.values():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        if not values:
            return "Keine Kantengewichte vorhanden."
        return {
            "edges": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
