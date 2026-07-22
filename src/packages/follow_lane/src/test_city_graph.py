#!/usr/bin/env python3
"""
Offline-Tests fuer city_graph.py — laufen OHNE ROS ueberall:
    python3 test_city_graph.py

Enthaelt eine komplette Fahr-SIMULATION: ein simulierter Duckiebot faehrt
auf dem Folie-10-Graphen, gesteuert von der Explorer-Politik, waehrend
GraphTracker (Lokalisierung) und ein simulierter Gate-Mapper mitlaufen.
Verglichen wird permanent gegen die Ground Truth der Simulation.
"""

import itertools
import random
import sys

from city_graph import (
    ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT,
    CityGraph, ExplorerPolicy, GraphTracker,
    ExpectedGatesMap,
    edge_id, exit_arm_to_turn, turn_to_exit_arm, wrap_arm,
)

CITY = {
    "A": {1: ("B", 1), 2: ("C", 2), 3: ("C", 1), 4: ("B", 2)},
    "B": {1: ("A", 1), 2: ("A", 4), 3: ("C", 4)},
    "C": {1: ("A", 3), 2: ("A", 2), 4: ("B", 3)},
}

PASS = 0
FAIL = 0


def check(cond, name):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name}")


# ------------------------------------------------------------ 1. Arm-Mathe

def test_arm_math():
    print("[1] Arm-Mathe")
    check(wrap_arm(5) == 1 and wrap_arm(0) == 4 and wrap_arm(-1) == 3,
          "wrap_arm normiert auf 1..4")
    # Geradeaus ist immer der gegenueberliegende Arm (1<->3, 2<->4)
    for e in (1, 2, 3, 4):
        check(turn_to_exit_arm(e, ACTION_FORWARD) == wrap_arm(e + 2),
              f"forward von Arm {e} = gegenueber")
    # links = identitaet (entry_arm == exit_arm)
    for e in (1, 2, 3, 4):
        check(turn_to_exit_arm(e, ACTION_LEFT) == e,
              f"links von Arm {e} = {e} (identitaet)")
    # rechts ist der benachbarte Arm
    for e in (1, 2, 3, 4):
        r = turn_to_exit_arm(e, ACTION_RIGHT)
        check(r != e and r != wrap_arm(e + 2),
              f"rechts von Arm {e} = {r} (benachbart, nicht gegenueber)")
    # Invers: exit_arm_to_turn(turn_to_exit_arm(e, a), e) == a
    # und: turn_to_exit_arm(e, exit_arm_to_turn(e, x)) == x (wenn != None)
    for e, a in itertools.product((1, 2, 3, 4),
                                  (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT)):
        x = turn_to_exit_arm(e, a)
        check(exit_arm_to_turn(x, e) == a,
              f"inv1 e={e} a={a} x={x}")
        inv = exit_arm_to_turn(e, x)
        if inv is not None:
            check(turn_to_exit_arm(e, inv) == x,
                  f"inv2 e={e} x={x} inv={inv}")
    # Wende (gegenueber) -> move_forward
    for e in (1, 2, 3, 4):
        opp = wrap_arm(e + 2)
        check(exit_arm_to_turn(e, opp) == ACTION_FORWARD,
              f"opposite e={e} -> opp={opp} = move_forward")
    # Gleicher Arm -> LEFT (Identity)
    for e in (1, 2, 3, 4):
        check(exit_arm_to_turn(e, e) == ACTION_LEFT,
              f"same arm e={e} -> {e} = turn_left")


# ------------------------------------------------------------ 2. Graph

def test_graph():
    print("[2] Graph-Aufbau")
    g = CityGraph(CITY)
    ids = g.all_edge_ids()
    check(ids == {"A1-B1", "A4-B2", "A2-C2", "A3-C1", "B3-C4"},
          f"5 kanonische Strassen (bekam: {sorted(ids)})")
    check(edge_id("B", 1, "A", 1) == edge_id("A", 1, "B", 1) == "A1-B1",
          "edge_id richtungsunabhaengig")
    check(g.neighbor("B", 4) is None, "B hat keinen Arm 4")
    e = g.edge_from("A", 2)
    check(e["to_node"] == "C" and e["to_arm"] == 2 and e["edge_id"] == "A2-C2",
          "edge_from A ueber Arm 2 -> C Arm 2")
    # Inkonsistenter Graph muss abgelehnt werden
    bad = {"A": {1: ("B", 1)}, "B": {1: ("A", 2)}}
    try:
        CityGraph(bad)
        check(False, "inkonsistenter Graph faelschlich akzeptiert")
    except ValueError:
        check(True, "inkonsistenter Graph abgelehnt")


# ------------------------------------------------------------ 3. Simulation

class SimBot:
    """Ground-Truth-Bot: faehrt physikalisch korrekt auf dem Graphen und
    emuliert die /decision/action-Events, die die echten Nodes sehen."""

    def __init__(self, graph, start_node, start_exit_arm):
        self.graph = graph
        self.edge = graph.edge_from(start_node, start_exit_arm)

    def arrive(self):
        """Rote Linie: steht jetzt an edge.to_node, Eingang edge.to_arm."""
        return self.edge["to_node"], self.edge["to_arm"]

    def execute(self, action):
        """Fuehrt das Manoever physisch aus. True = ok, False = Crash
        (Manoever fuehrt auf einen Arm ohne Strasse)."""
        node, entry = self.arrive()
        exit_arm = turn_to_exit_arm(entry, action)
        nxt = self.graph.edge_from(node, exit_arm)
        if nxt is None:
            return False
        self.edge = nxt
        return True


def run_mapping_simulation(start_node, start_exit_arm, gate_edges, seed,
                           verbose=False, max_intersections=200):
    """Eine komplette Phase-A-Fahrt. gate_edges: {edge_id: tag}.
    Rueckgabe: (ok, gefundene_gates, anzahl_kreuzungen, fehlermeldung)."""
    random.seed(seed)
    graph = CityGraph(CITY)
    bot = SimBot(graph, start_node, start_exit_arm)          # Ground Truth
    tracker = GraphTracker(graph, start_node, start_exit_arm)  # Lokalisierung
    expected_gates_map = ExpectedGatesMap({v: k for k, v in gate_edges.items()})
    policy = ExplorerPolicy(graph, expected_gates=expected_gates_map, max_sweeps=4)
    found_gates = {}                                          # Gate-Mapper
    policy.note_edge(tracker.pose_dict()["edge_id"])

    def scan_gates():
        # Bot faehrt ueber die aktuelle Kante -> sieht deren Tor (90% robust
        # laut User; wir simulieren gelegentliches Verpassen einer Sichtung,
        # aber die Kante wird ggf. erneut befahren)
        eid = bot.edge["edge_id"]
        if eid in gate_edges and random.random() < 0.9:
            tag = gate_edges[eid]
            tracked_edge = tracker.pose_dict()["edge_id"]
            if tracked_edge not in found_gates:
                found_gates[tracked_edge] = tag

    scan_gates()  # Startkante
    for step in range(max_intersections):
        # --- an der Kreuzung ankommen -----------------------------------
        node, entry = bot.arrive()
        # Erster Stopp initialisiert den Tracker; folgende stoppen setzen
        # AT_INTERSECTION.
        if tracker.status == "WAITING_FIRST_STOP":
            tracker.on_stopping(bot.edge)
        else:
            tracker.on_stopping(bot.edge, last_tag_id=entry)

        # Lokalisierung gegen Ground Truth pruefen
        p = tracker.pose_dict()
        if (p["to_node"], p["to_arm"]) != (node, entry):
            return False, found_gates, step, (
                f"Tracker-Desync an Schritt {step}: glaubt"
                f" {p['to_node']}/{p['to_arm']}, ist {node}/{entry}")

        # --- Explorer waehlt (Mehrfach-Sweep) ----------------------------
        exit_arm, action, reason = policy.decide(
            node, entry, tracker.visited, found_gates)
        if action is None:
            if reason == "complete":
                break
            return False, found_gates, step, f"Explorer: {reason}"

        # --- Manoever ausfuehren (Ground Truth) + Tracker folgt ----------
        if not bot.execute(action):
            return False, found_gates, step, (
                f"CRASH: '{action}' an {node} (Eingang {entry}) fuehrt ins Nichts")
        tracker.on_action(action)
        if tracker.status == "LOST":
            return False, found_gates, step, "Tracker LOST"
        policy.note_edge(tracker.pose_dict()["edge_id"])
        scan_gates()

        if verbose:
            print(f"    {step:2d}: {node} Eingang {entry} -> '{action}'"
                  f" -> {tracker.pose_dict()['edge_id']} ({reason})")
    else:
        return False, found_gates, max_intersections, "nicht terminiert"

    # Erfolgskriterien
    if not tracker.coverage_complete():
        return False, found_gates, step, "Abdeckung unvollstaendig"
    # Alle Tore muessen gefunden UND der richtigen Kante zugeordnet sein.
    # Verpasste Einzel-Sichtungen (10%) muss die Sweep-Politik durch
    # erneutes Befahren torloser Kanten ausgleichen.
    missing = {e: t for e, t in gate_edges.items() if found_gates.get(e) != t}
    if missing:
        return False, found_gates, step, f"Tore fehlen/falsch: {missing}"
    return True, found_gates, step, ""


def test_simulation():
    print("[3] Fahr-Simulation (Explorer + Tracker + Gate-Mapper)")
    graph = CityGraph(CITY)
    all_edges = sorted(graph.all_edge_ids())

    # alle gueltigen Startpositionen
    starts = [(n, a) for n in graph.nodes() for a in graph.arms_of(n)]
    check(len(starts) == 10, f"10 Startpositionen ({len(starts)})")

    runs = fails = 0
    total_steps = []
    rng = random.Random(42)
    for start_node, start_arm in starts:
        for trial in range(30):
            # zufaellige Tor-Belegung: 1-5 Tore, max 1 pro Kante, IDs aus 5..13
            k = rng.randint(1, 5)
            edges = rng.sample(all_edges, k)
            tags = rng.sample(range(5, 14), k)
            gate_edges = dict(zip(edges, tags))
            seed = rng.randint(0, 10**9)

            ok, found, steps, err = run_mapping_simulation(
                start_node, start_arm, gate_edges, seed)
            runs += 1
            total_steps.append(steps)
            if not ok:
                fails += 1
                print(f"  FAIL Start {start_node}/Arm{start_arm}"
                      f" gates={gate_edges}: {err}")
    check(fails == 0, f"{fails}/{runs} Simulationen fehlgeschlagen")
    if fails == 0:
        print(f"  OK: {runs} Fahrten, alle Kanten + alle Tore korrekt."
              f" Kreuzungen bis Abdeckung: min={min(total_steps)}"
              f" avg={sum(total_steps)/len(total_steps):.1f}"
              f" max={max(total_steps)}")

    # eine Beispiel-Fahrt ausfuehrlich zeigen
    print("  Beispiel-Fahrt (Start A ueber Arm 1, Tore auf A2-C2=7, B3-C4=5):")
    ok, found, steps, err = run_mapping_simulation(
        "A", 1, {"A2-C2": 7, "B3-C4": 5}, seed=1, verbose=True)
    check(ok, f"Beispiel-Fahrt ({err})")
    print(f"    -> gefundene Tore: {found}")


# ------------------------------------------------------------ 4. Robustheit

def test_desync_detection():
    print("[4] Desync-Erkennung")
    graph = CityGraph(CITY)
    t = GraphTracker(graph, "A", 1)
    check(t.status == "WAITING_FIRST_STOP", "Tracker startet in WAITING_FIRST_STOP")
    t.on_stopping(graph.edge_from("A", 1))         # auf A1-B1, an B
    check(t.status == "ON_EDGE", "erster Stopp -> ON_EDGE")
    t.on_action(ACTION_RIGHT)                       # B Eingang 1, rechts -> Arm 4
    check(t.status == "LOST", "unmoegliches Manoever -> LOST")

    t2 = GraphTracker(graph, "A", 1)
    t2.on_stopping(graph.edge_from("A", 1), last_tag_id=3)
    # turn_right(1) mit CLOCKWISE_NUMBERING=False -> Arm 4 (B->A, A4-B2)
    t2.on_action(ACTION_RIGHT)
    check(t2.pose_dict()["edge_id"] == "A4-B2" and t2.status == "ON_EDGE",
          "Tracking laeuft trotz Soft-Mismatch weiter")


# ------------------------------------------------------------ 5. ExpectedGatesMap

def test_expected_gates():
    print("[5] ExpectedGatesMap + Validation")
    eg = ExpectedGatesMap({10: "A1-B2", 7: "A1-B1", 8: "A3-C1"})
    check(bool(eg), "nicht-leeres Map ist True")
    check(not ExpectedGatesMap({}), "leeres Map ist False")

    # match
    check(eg.validate_gate("A1-B2", 10), "Tor 10 auf A1-B2: match")
    check(eg.validate_gate("A1-B1", 7), "Tor 7 auf A1-B1: match")

    # bidirektional: gleiche Kante, andere Richtung
    check(eg.validate_gate("B2-A1", 10), "Tor 10 auf B2-A1 (umgekehrt): match")
    check(eg.validate_gate("B1-A1", 7), "Tor 7 auf B1-A1 (umgekehrt): match")

    # no match
    check(not eg.validate_gate("A1-B2", 7), "Tor 7 auf A1-B2: kein match")
    check(not eg.validate_gate("A2-C2", 10), "Tor 10 auf A2-C2: kein match (Kante nicht erwartet)")
    check(not eg.validate_gate("A1-B1", 10), "Tor 10 auf A1-B1: kein match (falsches Tor)")

    # iter_unassigned
    assigned = {"A1-B1"}  # Tor 7 wurde bereits gefunden
    unassigned = list(eg.iter_unassigned(assigned))
    check(len(unassigned) == 2, f"2 Tore unvergeben (saw {len(unassigned)})")
    tags = {t for t, _ in unassigned}
    check(10 in tags and 8 in tags, f"unassigned: {tags}")


if __name__ == "__main__":
    test_arm_math()
    test_graph()
    test_simulation()
    test_desync_detection()
    test_expected_gates()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
