#!/usr/bin/env python3
"""
city_graph.py — Stadtgraph-Logik fuer Challenge 4 (Phase A: Mapping).

BEWUSST OHNE ROS-IMPORTS: alles hier ist pure Python und damit offline
testbar (siehe test_city_graph.py). Die ROS-Nodes (localization_node,
explorer_node, gate_mapper_node) sind nur duenne Wrapper um dieses Modul.

Graph-Format (wie Folie 10 des Einfuehrungsfoliensatzes):
    { "A": { 1: ["B", 1], 2: ["C", 2], 3: ["C", 1], 4: ["B", 2] },
      "B": { 1: ["A", 1], 2: ["A", 4], 3: ["C", 4] },
      "C": { 1: ["A", 3], 2: ["A", 2], 4: ["B", 3] } }
Bedeutung: node -> { arm_nummer: (nachbar, dessen_arm) }.
Der Graph ist UNGERICHTET: jede Strasse steht von beiden Seiten drin.

Arm-Konvention (aus der Aufgabenstellung):
    "1 und 3 sind gegenueber, 2 ist immer rechts von 1, 4 links von 1"
    => Nummerierung im Uhrzeigersinn in der Draufsicht: 1,2,3,4.
Faehrt man ueber Arm e IN die Kreuzung hinein (Blick Richtung Zentrum):
    geradeaus -> Arm e+2   |   links -> Arm e+1   |   rechts -> Arm e-1
(jeweils mod 4, 1-basiert).

!!! FALLS auf der echten Strecke links/rechts vertauscht wirken, nur
CLOCKWISE_NUMBERING unten auf False setzen — sonst nichts anfassen. !!!
"""

import json
from collections import deque

# True  = Arme im Uhrzeigersinn nummeriert (Standard-Lesart der Aufgabe).
# False = gegen den Uhrzeigersinn (kippt links/rechts).
CLOCKWISE_NUMBERING = True

ACTION_FORWARD = "move_forward"
ACTION_LEFT = "turn_left"
ACTION_RIGHT = "turn_right"
TURN_ACTIONS = (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT)

GATE_TAG_IDS = set(range(5, 15))          # Tore: AprilTag 5..14
INTERSECTION_TAG_IDS = set(range(1, 5))   # Kreuzungen: AprilTag 1..4

# Erlaubte Actions fuer jeden Kreuzungstag (ID 1..4).
ID_FUNCTIONS = {
    1: ['turn_left', 'turn_right', 'move_forward'],
    2: ['turn_left', 'turn_right'],
    3: ['turn_left', 'move_forward'],
    4: ['turn_right', 'move_forward'],
}


# ---------------------------------------------------------------- Arm-Mathe

def wrap_arm(a):
    """Arm-Nummer auf 1..4 normieren."""
    return ((int(a) - 1) % 4) + 1


def turn_to_exit_arm(entry_arm, action):
    """Eingangs-Arm + Manoever -> Ausgangs-Arm."""
    e = wrap_arm(entry_arm)
    sign = 1 if CLOCKWISE_NUMBERING else -1
    if action == ACTION_FORWARD:
        return wrap_arm(e + 2 * sign)
    if action == ACTION_LEFT:
        return e              # exit_arm = entry_arm (identitaet)
    if action == ACTION_RIGHT:
        return wrap_arm(e + sign)
    raise ValueError(f"unbekannte Action: {action!r}")


def exit_arm_to_turn(entry_arm, exit_arm):
    """Eingangs- und Ausgangs-Arm -> Manoever. None = Wende (nicht fahrbar)."""
    e, x = wrap_arm(entry_arm), wrap_arm(exit_arm)
    for action in TURN_ACTIONS:
        if turn_to_exit_arm(e, action) == x:
            return action
    return None  # x == e waere eine Wende


# ---------------------------------------------------------------- CityGraph

def edge_id(u, arm_u, v, arm_v):
    """Kanonische, richtungsunabhaengige Strassen-ID, z.B. 'A1-B1'."""
    a, b = sorted([(str(u), wrap_arm(arm_u)), (str(v), wrap_arm(arm_v))])
    return f"{a[0]}{a[1]}-{b[0]}{b[1]}"


class CityGraph:
    """Gegebener Stadtgraph + Hilfsfunktionen fuer Mapping und Planung."""

    def __init__(self, city_dict):
        # Keys koennen aus JSON als Strings kommen -> normalisieren
        self.city = {
            str(node): {wrap_arm(arm): (str(nb), wrap_arm(nb_arm))
                        for arm, (nb, nb_arm) in arms.items()}
            for node, arms in city_dict.items()
        }
        self._check_consistency()

    @classmethod
    def from_json_file(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(data["city"] if "city" in data else data)

    def _check_consistency(self):
        problems = []
        for node, arms in self.city.items():
            for arm, (nb, nb_arm) in arms.items():
                back = self.city.get(nb, {}).get(nb_arm)
                if back != (node, arm):
                    problems.append(f"{node}{arm}->{nb}{nb_arm} ohne Rueckweg")
        if problems:
            raise ValueError("Stadtgraph inkonsistent: " + "; ".join(problems))

    # --- Nachschlagen -----------------------------------------------------

    def nodes(self):
        return list(self.city.keys())

    def arms_of(self, node):
        return sorted(self.city[node].keys())

    def neighbor(self, node, arm):
        """(nachbar, dessen_arm) oder None, wenn an diesem Arm keine Strasse."""
        return self.city.get(node, {}).get(wrap_arm(arm))

    def all_edge_ids(self):
        ids = set()
        for node, arms in self.city.items():
            for arm, (nb, nb_arm) in arms.items():
                ids.add(edge_id(node, arm, nb, nb_arm))
        return ids

    def edge_from(self, node, exit_arm):
        """Strasse, die man nimmt, wenn man node ueber exit_arm verlaesst.
        Rueckgabe: dict mit allem, was die Lokalisierung braucht, oder None."""
        nb = self.neighbor(node, exit_arm)
        if nb is None:
            return None
        to_node, to_arm = nb
        return {
            "edge_id": edge_id(node, exit_arm, to_node, to_arm),
            "from_node": node, "from_arm": wrap_arm(exit_arm),
            "to_node": to_node, "to_arm": to_arm,
        }

    # --- Explorer-Politik (Phase A, nicht auf Zeit -> simpel & vollstaendig)

    def choose_exit_arm(self, node, entry_arm, visited_edge_ids):
        """Waehlt am Knoten `node` (angekommen ueber `entry_arm`) den
        Ausgangs-Arm fuer die Kantenabdeckung.

        Prioritaet:
          1) direkt anliegende UNBESUCHTE Strasse (bei mehreren: die, die
             einem echten Manoever entspricht; Wende ausgeschlossen)
          2) sonst: erster Schritt des kuerzesten Wegs (BFS, Hops) zum
             naechsten Knoten mit unbesuchter anliegender Strasse
          3) alles besucht -> None (Mapping fertig)

        WICHTIG: KEINE Tag-Filterung! Der Explorer entscheidet nur nach
        Kantenabdeckung. Der decision_node filtert spaeter nach Tag.

        Rueckgabe: (exit_arm, action, reason) oder (None, None, 'complete').
        """
        entry_arm = wrap_arm(entry_arm)
        candidates = []  # (exit_arm, action, edge)
        for arm in self.arms_of(node):
            action = exit_arm_to_turn(entry_arm, arm)
            if action is None:          # Wende: fahren wir nicht
                continue
            edge = self.edge_from(node, arm)
            candidates.append((arm, action, edge))

        if not candidates:
            return None, None, "dead_end"

        # 1) unbesuchte direkte Strasse?
        unvisited = [c for c in candidates
                     if c[2]["edge_id"] not in visited_edge_ids]
        if unvisited:
            arm, action, edge = unvisited[0]
            return arm, action, f"unvisited:{edge['edge_id']}"

        # 2) fertig?
        if self.all_edge_ids() <= set(visited_edge_ids):
            return None, None, "complete"

        # 3) BFS zum naechsten Knoten mit unbesuchter Strasse.
        #    Zustand = (knoten, ankunfts_arm), damit Wenden ausgeschlossen bleiben.
        start = (node, entry_arm)
        parent = {start: None}
        q = deque([start])
        goal_state = None
        while q:
            cur_node, cur_entry = q.popleft()
            has_unvisited = any(
                self.edge_from(cur_node, a)["edge_id"] not in visited_edge_ids
                for a in self.arms_of(cur_node)
            )
            if has_unvisited and (cur_node, cur_entry) != start:
                goal_state = (cur_node, cur_entry)
                break
            for a in self.arms_of(cur_node):
                action = exit_arm_to_turn(cur_entry, a)
                if action is None:
                    continue
                edge = self.edge_from(cur_node, a)
                nxt = (edge["to_node"], edge["to_arm"])
                if nxt not in parent:
                    parent[nxt] = (cur_node, cur_entry, a)
                    q.append(nxt)

        if goal_state is None:
            return None, None, "unreachable"

        # ersten Schritt aus dem BFS-Baum rekonstruieren
        state = goal_state
        while parent[state][:2] != start:
            state = parent[state][:2]
        first_exit_arm = parent[state][2]
        action = exit_arm_to_turn(entry_arm, first_exit_arm)
        return first_exit_arm, action, f"route_to:{goal_state[0]}"

    # --- Kuerzeste Wege (fuer Explorer-Routing & spaeter Phase B) ----------

    def shortest_path_edges(self, from_node, from_entry_arm, to_node):
        """Kuerzester Weg (in Hops) von (from_node, angekommen ueber Arm)
        zu to_node, ohne Wenden. Rueckgabe: Liste von edge_ids oder None."""
        start = (from_node, wrap_arm(from_entry_arm))
        parent = {start: None}
        q = deque([start])
        goal = None
        while q:
            cur_node, cur_entry = q.popleft()
            if cur_node == to_node and (cur_node, cur_entry) != start:
                goal = (cur_node, cur_entry)
                break
            for a in self.arms_of(cur_node):
                if exit_arm_to_turn(cur_entry, a) is None:
                    continue
                edge = self.edge_from(cur_node, a)
                nxt = (edge["to_node"], edge["to_arm"])
                if nxt not in parent:
                    parent[nxt] = (cur_node, cur_entry, a)
                    q.append(nxt)
        if goal is None:
            return None
        path = []
        state = goal
        while parent[state] is not None:
            pn, pe, arm = parent[state]
            path.append(self.edge_from(pn, arm)["edge_id"])
            state = (pn, pe)
        return list(reversed(path))


# ---------------------------------------------------------------- ExpectedGatesMap

class ExpectedGatesMap:
    """Zuordnung: gate_tag_id -> edge_id (welches Tor wo sitzt).

    Ein Tor ist beidseitig sichtbar -> es reicht, wenn das Tor auf der
    angegebenen Kante im expected_gates steht (egal von welcher Seite
    man kommt). Wenn die Kante nicht im expected_gates steht, ist es
    kein match.
    """

    def __init__(self, expected_dict):
        self.expected = {}
        for k, v in expected_dict.items():
            canon = _canonical_edge_id(str(v))
            self.expected[int(k)] = canon

    def __bool__(self):
        return bool(self.expected)

    def validate_gate(self, edge_id, tag_id):
        # edge_id vom Tracker ist immer kanonisch, aber zur Sicherheit
        # auch den Parameter kanalisieren (z.B. von Testcode).
        return self.expected.get(tag_id) == _canonical_edge_id(edge_id)

    def iter_unassigned(self, assigned):
        """Alle Tore, die noch keiner Kante zugeordnet wurden.
        assigned: set von edge_ids mit bereits zugewiesenen Toren."""
        for tag, edge in self.expected.items():
            if edge not in assigned:
                yield tag, edge


def _canonical_edge_id(eid):
    """Kanalisiere eine Kanten-ID wie 'C4-B3' -> 'B3-C4'.

    Analog zu edge_id(): sortiert die beiden Endpunkte lexikographisch
    nach (node, arm).
    """
    idx = eid.rfind("-")
    if idx == -1:
        return eid
    left = eid[:idx]
    right = eid[idx + 1:]
    a, b = sorted([left, right])
    return a + "-" + b


# ---------------------------------------------------------------- Explorer

class ExplorerPolicy:
    """Mehrfach-Sweep-Erkundung.

    Problem: 'alle Kanten einmal befahren' garantiert NICHT 'alle Tore
    gesehen' — die Tag-Erkennung trifft pro Vorbeifahrt nur ~90%. Darum:

      Sweep 1: klassische Kantenabdeckung (jede Strasse mindestens 1x).
      Sweep 2+: nur falls weniger Tore gefunden als erwartet
                (expected_gates): gezielt die Strassen OHNE zugeordnetes
                Tor erneut abfahren. Wiederholen bis Anzahl stimmt oder
                max_sweeps erreicht (0.9-Erkennung => nach 3 Paessen ist
                die Verpass-Wahrscheinlichkeit pro Tor nur noch 0.1%).

    expected_gates=None => unbekannte Toranzahl => nur Sweep 1.
    """

    def __init__(self, graph, expected_gates=None, max_sweeps=3):
        self.graph = graph
        self.expected_gates = expected_gates
        self.max_sweeps = max(1, int(max_sweeps))
        self.sweep = 1
        self.sweep_visited = set()   # in DIESEM Sweep befahrene Strassen

    def note_edge(self, eid):
        """Von aussen aufrufen, sobald eine Strasse befahren wird."""
        self.sweep_visited.add(eid)

    def _need_more_gates(self, gates):
        """True wenn Tore fehlen und noch Sweeps übrig."""
        if self.expected_gates is None:
            return False
        if isinstance(self.expected_gates, int):
            return len(gates) < self.expected_gates
        eg = getattr(self.expected_gates, 'expected', None)
        return (eg is not None
                and len(gates) < len(eg))

    def _todo(self, visited_edges, gates):
        """Noch abzufahrende Strassen im aktuellen Sweep.

        Sweep 1: alle unbesuchten Kanten (volle Kantenabdeckung).
        Sweep 2+: nur Kanten ohne zugeordnetes Tor (zurueckfahren
                  und nochmal prüfen).

        Wenn alle Kanten besucht sind, wird automatisch naechster Sweep
        gestartet (solange noch Sweeps uebrig und Tore fehlen).
        Rueckgabe: leere Menge wenn Mapping fertig.
        """
        all_ids = self.graph.all_edge_ids()
        while True:
            if self.sweep == 1:
                todo = all_ids - set(visited_edges)
            else:
                todo = (all_ids - set(gates.keys())) - self.sweep_visited
            if todo:
                return todo
            # aktueller Sweep fertig -> naechster noetig?
            if self._need_more_gates(gates) and self.sweep < self.max_sweeps:
                self.sweep += 1
                self.sweep_visited = set()
                continue
            return set()

    def decide(self, node, entry_arm, visited_edges, gates):
        """Wie CityGraph.choose_exit_arm, aber ueber die Sweep-Todo-Menge.
        Entscheidet NUR nach Kantenabdeckung (ohne Tag-Filter).
        Rueckgabe: (exit_arm, action, reason); action=None => fertig."""
        todo = self._todo(visited_edges, gates)
        if not todo:
            return None, None, "complete"
        synthetic_visited = self.graph.all_edge_ids() - todo
        exit_arm, action, reason = self.graph.choose_exit_arm(
            node, entry_arm, synthetic_visited)
        if action is None and reason == "complete":
            # kann passieren, wenn todo gerade leer wurde
            return None, None, "complete"
        return exit_arm, action, f"sweep{self.sweep}:{reason}"


# ---------------------------------------------------------------- Tracker

class GraphTracker:
    """Dead-Reckoning auf dem Graphen.

    Zustand: aktuelle Strasse + aktuelle Kreuzung.
    Der entry_arm wird AUS DER STRASSE abgeleitet, nicht aus dem Tag.
    Der Tag (1-4) identifiziert nur die Kreuzung, nicht den Eingangs-Arm.
    """

    def __init__(self, graph, start_node, start_exit_arm):
        self.graph = graph
        edge = graph.edge_from(start_node, start_exit_arm)
        if edge is None:
            raise ValueError(
                f"Start ungueltig: {start_node} hat keinen Arm {start_exit_arm}")
        self.current_edge = edge
        self.current_node = start_node
        self.entry_arm = None
        self.visited = {edge["edge_id"]}
        self.status = "ON_EDGE"
        self.last_intersection_tag = -1
        self._gates = {}

    @property
    def gates(self):
        return self._gates

    def set_gate(self, edge_id, tag_id):
        self._gates[edge_id] = tag_id

    def on_stopping(self, edge, last_tag_id=-1):
        """Rote Linie: entry_arm = to_arm der Strasse, auf der wir waren."""
        if self.status == "LOST":
            return
        self.status = "AT_INTERSECTION"
        self.entry_arm = edge["to_arm"]
        self.current_node = edge["to_node"]
        self.last_intersection_tag = last_tag_id

    def on_action(self, action):
        if self.status != "AT_INTERSECTION":
            return
        node = self.current_node
        if action == "skip":
            action = ACTION_FORWARD
        if action not in TURN_ACTIONS:
            return
        exit_arm = turn_to_exit_arm(self.entry_arm, action)
        edge = self.graph.edge_from(node, exit_arm)
        if edge is None:
            self.status = "LOST"
            return
        self.current_edge = edge
        self.visited.add(edge["edge_id"])
        self.status = "ON_EDGE"
        self.entry_arm = None

    # -- Abfragen ------------------------------------------------------------

    def coverage_complete(self):
        return self.graph.all_edge_ids() <= self.visited

    def pose_dict(self):
        """Serialisierbarer Zustand fuers Dashboard ('wo denkt der Bot dass
        er ist?') und fuer explorer/gate_mapper."""
        e = self.current_edge
        return {
            "status": self.status,
            "edge_id": e["edge_id"],
            "from_node": e["from_node"], "from_arm": e["from_arm"],
            "to_node": e["to_node"], "to_arm": e["to_arm"],
            "entry_arm": self.entry_arm,
            "visited": sorted(self.visited),
            "num_visited": len(self.visited),
            "num_total": len(self.graph.all_edge_ids()),
            "coverage_complete": self.coverage_complete(),
            "last_intersection_tag": self.last_intersection_tag,
        }
