#!/usr/bin/env python3
"""
Tests fuer planner.py — LAEUFT OHNE ROS!

Starten mit: python3 src/packages/follow_lane/src/test_planner.py
"""

import sys
import os
import json

# Sicherstellen, dass das Paket gefunden wird
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from city_graph import CityGraph
from planner import Planner


# ================================================================ Test-Helfer
GRAPH_DICT = {
    "A": {"1": ["B", 1], "2": ["C", 2], "3": ["C", 1], "4": ["B", 2]},
    "B": {"1": ["A", 1], "2": ["A", 4], "3": ["C", 4]},
    "C": {"1": ["A", 3], "2": ["A", 2], "4": ["B", 3]},
}

GATES = {
    10: "A4-B2",  # A-B
    9: "A3-C1",   # A-C
    7: "A1-B1",   # A-B
    8: "B3-C4",   # B-C
    6: "A2-C2",   # A-C
}

WEIGHTS = {
    "A1-B1": 2.5,
    "A2-C2": 4.0,
    "A3-C1": 3.0,
    "A4-B2": 3.5,
    "B3-C4": 5.0,
}


def test_single_gate():
    """Ein Gate -> leerer Pfad."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, {8: "B3-C4"}, WEIGHTS)
    path = p.plan_path([8])
    assert path == [(8, "B3-C4")], f"Erwartet [(8, B3-C4)], got {path}"
    print("OK  test_single_gate")


def test_two_direct_gates():
    """Zwei Gates, die direkt benachbart sind."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, {7: "A1-B1", 8: "B3-C4"}, WEIGHTS)
    path = p.plan_path([7, 8])
    assert path is not None, "Pfad sollte gefunden werden"
    assert len(path) == 2, f"2 Gates erwartet, got {len(path)}"
    print("OK  test_two_direct_gates")


def test_multi_gate_path():
    """Mehrfache Gates durch den ganzen Graphen."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    path = p.plan_path([8, 9, 7, 10, 6])
    assert path is not None, "Pfad sollte gefunden werden"
    assert len(path) == 5, f"5 Gates erwartet, got {len(path)}"
    # Alle Gates sollten im Pfad sein
    gate_ids = [gid for gid, _ in path]
    assert sorted(gate_ids) == sorted([8, 9, 7, 10, 6]), f"Gate-Reihenfolge stimmt nicht: {gate_ids}"
    print("OK  test_multi_gate_path")


def test_start_validation_match():
    """Startkante stimmt mit erstem Gate ueberein."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    # Gate 7 = A1-B1, Start A mit Arm 1 -> A1-B1
    start_edge = p.compute_start_edge("A", 1)
    p.validate_start_edge(start_edge, GATES[7])
    print("OK  test_start_validation_match")


def test_start_validation_mismatch():
    """Startkante stimmt NICHT mit erstem Gate ueberein."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    start_edge = p.compute_start_edge("A", 1)  # A1-B1
    try:
        p.validate_start_edge(start_edge, GATES[8])  # B3-C4
        assert False, "Sollte ValueError geworfen haben"
    except ValueError:
        pass
    print("OK  test_start_validation_mismatch")


def test_missing_gate():
    """Nicht-kenntliches Gate."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, {7: "A1-B1"}, WEIGHTS)
    try:
        p.plan_path([7, 99])
        assert False, "Sollte ValueError geworfen haben"
    except ValueError as e:
        assert "99" in str(e), f"Fehlermeldung sollte Gate 99 enthalten: {e}"
    print("OK  test_missing_gate")


def test_empty_sequence():
    """Leere Gate-Sequenz."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    path = p.plan_path([])
    assert path == [], f"Leere Liste erwartet, got {path}"
    print("OK  test_empty_sequence")


def test_missing_weight_fallback():
    """Fehlende Kantengewichte -> Mittelwert."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, {7: "A1-B1", 8: "B3-C4"}, {})  # keine Gewichte
    path = p.plan_path([7, 8])
    assert path is not None, "Pfad sollte mit Fallback gefunden werden"
    print("OK  test_missing_weight_fallback")


def test_weight_mean():
    """Mittelwert wird korrekt berechnet."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    mean = p._mean_weight()
    expected = sum(WEIGHTS.values()) / len(WEIGHTS)
    assert abs(mean - expected) < 1e-10, f"Mittelwert {mean} != {expected}"
    print("OK  test_weight_mean")


def test_dijkstra_returns_edges():
    """Dijkstra gibt Liste von edge_ids zurueck."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    path = p._dijkstra("A1-B1", "B3-C4")
    assert path is not None, "Pfad sollte gefunden werden"
    assert all(isinstance(e, str) for e in path), "Alle Elemente sollten Strings sein"
    print("OK  test_dijkstra_returns_edges")


def test_edge_to_actions():
    """Kanten-Liste wird in Actions umgewandelt."""
    g = CityGraph(GRAPH_DICT)
    p = Planner(g, GATES, WEIGHTS)
    actions = p.edges_to_actions(["A1-B1", "B3-C4"])
    # Erster Entry_Arm ist None, rest sollte Actions sein
    assert len(actions) == 2, f"2 Actions erwartet, got {len(actions)}"
    assert actions[0][0] is None, "Erste Action sollte None sein (unklarer Entry_Arm)"
    assert actions[1][0] in ("turn_left", "turn_right", "move_forward"), \
        f"Zweite Action sollte gueltig sein: {actions[1][0]}"
    print("OK  test_edge_to_actions")


def test_no_ros_import():
    """planner.py darf keine ROS-Imports haben."""
    with open(os.path.join(os.path.dirname(__file__), "planner.py")) as f:
        source = f.read()
    assert "import rospy" not in source, "planner.py sollte keine ROS-Imports haben"
    assert "import ros" not in source, "planner.py sollte keine ROS-Imports haben"
    print("OK  test_no_ros_import")


if __name__ == '__main__':
    tests = [
        test_single_gate,
        test_two_direct_gates,
        test_multi_gate_path,
        test_start_validation_match,
        test_start_validation_mismatch,
        test_missing_gate,
        test_empty_sequence,
        test_missing_weight_fallback,
        test_weight_mean,
        test_dijkstra_returns_edges,
        test_edge_to_actions,
        test_no_ros_import,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1

    if failed:
        print(f"\n{failed}/{len(tests)} Tests fehlgeschlagen!")
        sys.exit(1)
    else:
        print(f"\nAlle {len(tests)} Tests bestanden!")
        sys.exit(0)
