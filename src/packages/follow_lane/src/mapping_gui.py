#!/usr/bin/env python3
"""
mapping_gui — Interaktive GUI fuer die Mapping-Phase.

Zeigt den Stadtgraphen mit aktueller Position, besuchten Kanten und
gefundenen Toren. Links Graph, rechts Tor-Liste.
"""

import json
import os
import tkinter as tk
from tkinter import font as tkfont

import rospy
from std_msgs.msg import Bool, String


# ============================================================ Stadtgraph-Layout
def load_graph_from_config():
    """Laedt den Stadtgraphen aus city_graph.json und berechnet Kanten."""
    config_path = os.path.join(os.path.dirname(__file__), f"../config/city_graph.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    city = config["city"]
    expected_gates = config.get("expected_gates", {})
    nodes = set(city.keys())

    edges_by_pair = {}
    for node, arms in city.items():
        for arm, (neighbor, neighbor_arm) in arms.items():
            arm = str(arm)
            neighbor_arm = str(neighbor_arm)
            pair = tuple(sorted([node, neighbor]))
            if pair not in edges_by_pair:
                edges_by_pair[pair] = []

            edge_id = f"{node}{arm}-{neighbor}{neighbor_arm}"

            for edge_id_check, _ in expected_gates.items():
                parts = edge_id_check.split('-')
                if (node + arm == parts[0] and neighbor + neighbor_arm == parts[1]) or \
                   (neighbor + neighbor_arm == parts[0] and node + arm == parts[1]):
                    edge_id = edge_id_check
                    break

            if edge_id not in edges_by_pair[pair]:
                edges_by_pair[pair].append(edge_id)

    edges = []
    for (n1, n2), edge_ids in edges_by_pair.items():
        for edge_id in edge_ids:
            edges.append((n1, n2, edge_id))

    return nodes, edges, expected_gates


def auto_layout_nodes(nodes):
    """Fixe Positionen fuer den 9-Knoten Stadtgraph (D-A H-E-B I-F-C G Layout)."""
    # Layout:
    #   D   A
    # H  E  B
    # I  F  C
    #   G
    positions = {
        "A": (550, 100),
        "B": (550, 220),
        "C": (550, 380),
        "D": (300, 100),
        "E": (300, 220),
        "F": (300, 380),
        "G": (300, 500),
        "H": (100, 220),
        "I": (100, 380),
    }
    return positions


def init_graph():
    nodes, edges, expected_gates = load_graph_from_config()
    positions = auto_layout_nodes(nodes)
    return nodes, edges, positions, expected_gates


NODES, EDGES, NODE_POSITIONS, EXPECTED_GATES = init_graph()


# ============================================================ ROS-Integration
class MappingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DuckieRace — Mapping Debug")
        self.root.geometry("1024x720")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        vehicle = os.environ.get("VEHICLE_NAME", "unknown")
        self.prefix = f"/{vehicle}"

        self.pose = {}
        self.gates = {}
        self.explore_state = {}
        self.suggested_action = ""
        self.complete = False

        self._weights = {}

        rospy.Subscriber(f"{self.prefix}/mapping/pose", String, self._cb_pose)
        rospy.Subscriber(f"{self.prefix}/mapping/gates", String, self._cb_gates)
        rospy.Subscriber(f"{self.prefix}/explore/state", String, self._cb_explore)
        rospy.Subscriber(f"{self.prefix}/explore/suggested_action", String, self._cb_suggested)
        rospy.Subscriber(f"{self.prefix}/mapping/complete", Bool, self._cb_complete)
        rospy.Subscriber(f"{self.prefix}/mapping/edge_stats", String, self._cb_edge_stats)

        self._setup_ui()
        self._weights = self._load_weights()
        rospy.loginfo("[mapping_gui] Mapping GUI started")

    def _load_weights(self):
        """Ladet Kantengewichte aus der weights-Datei des timing_node."""
        weights_file = os.environ.get("MAP_WEIGHTS_FILE", "/tmp/duckie_city_map_weights.json")
        try:
            with open(weights_file, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _cb_edge_stats(self, msg):
        """Ladet die aktuellen Gewichte neu, wenn neue Stats empfangen werden."""
        self._weights = self._load_weights()
        self._update_gate_list()

    def _setup_ui(self):
        """Erstellt das UI-Layout."""
        label_font = tkfont.Font(family="Monospace", size=14)
        value_font = tkfont.Font(family="Monospace", size=14, weight="bold")
        header_font = tkfont.Font(family="Helvetica", size=16, weight="bold")

        # ── Hauptcontainer ──
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # ── Linke Seite: Info + Graph ──
        left_frame = tk.Frame(main_frame, bg="#1a1a2e")
        left_frame.pack(side="left", fill="both", expand=True)

        # Info Panel (oben)
        info_frame = tk.Frame(left_frame, bg="#1a1a2e", pady=5)
        info_frame.pack(fill="x")

        status_row = tk.Frame(info_frame, bg="#1a1a2e")
        status_row.pack(fill="x")
        tk.Label(status_row, text="Status:", font=label_font, fg="#aaa", bg="#1a1a2e").pack(side="left")
        self.status_val = tk.Label(status_row, text="???", font=value_font, fg="#fff", bg="#1a1a2e")
        self.status_val.pack(side="left", padx=10)

        info_row = tk.Frame(info_frame, bg="#1a1a2e")
        info_row.pack(fill="x", pady=(5, 0))
        labels = [("Besucht:", "visited_val"), ("Sweep:", "sweep_val"), ("Tore:", "gates_val")]
        for i, (lbl, tag) in enumerate(labels):
            tk.Label(info_row, text=lbl, font=label_font, fg="#aaa", bg="#1a1a2e").pack(side="left", padx=(20 if i > 0 else 0, 5))
            setattr(self, tag, tk.Label(info_row, text="?", font=value_font, fg="#0f0", bg="#1a1a2e"))
            getattr(self, tag).pack(side="left")

        action_row = tk.Frame(info_frame, bg="#1a1a2e")
        action_row.pack(fill="x", pady=(5, 0))
        tk.Label(action_row, text="Naechste:", font=label_font, fg="#aaa", bg="#1a1a2e").pack(side="left")
        self.action_val = tk.Label(action_row, text="—", font=value_font, fg="#ff0", bg="#1a1a2e")
        self.action_val.pack(side="left", padx=10)

        # Graph Canvas
        self.canvas = tk.Canvas(left_frame, width=700, height=630, bg="#16213e", highlightthickness=0)
        self.canvas.pack(padx=0, pady=10)

        # ── Rechte Seite: Tor-Liste ──
        right_frame = tk.Frame(main_frame, bg="#1a1a2e", width=300)
        right_frame.pack(side="right", fill="both", padx=(15, 0))
        right_frame.pack_propagate(False)

        # Tor-Liste Header
        tk.Label(
            right_frame, text="Gefundene Tore", font=header_font,
            fg="#ffc800", bg="#1a1a2e"
        ).pack(pady=(0, 10))

        # Container fuer dynamische Tor-Eintraege
        self.gates_container = tk.Frame(right_frame, bg="#1a1a2e")
        self.gates_container.pack(fill="both", expand=True, pady=5)

        # Keine Tore Placeholder
        self.no_gates_label = tk.Label(
            self.gates_container, text="Noch keine Tore\ngesichtet...",
            font=label_font, fg="#555", bg="#1a1a2e", justify="left"
        )
        self.no_gates_label.pack(pady=20)

        # ── Gate-Eintraege (werden dynamisch erstellt) ──
        self.gate_labels = {}

    def _update_gate_list(self):
        """Aktualisiert die Tor-Liste rechts."""
        # Alte Eintraege loeschen
        for widget in self.gates_container.winfo_children():
            widget.destroy()
        self.gate_labels.clear()

        if not self.gates:
            self.no_gates_label = tk.Label(
                self.gates_container, text="Noch keine Tore\ngesichtet...",
                font=tkfont.Font(family="Monospace", size=14), fg="#555",
                bg="#1a1a2e", justify="left"
            )
            self.no_gates_label.pack(pady=20)
            return

        # Fuer jedes gefundene Tor einen Eintrag erstellen
        for edge_id, tag_id in sorted(self.gates.items(), key=lambda x: x[1]):
            entry_frame = tk.Frame(self.gates_container, bg="#1a1a2e", pady=3)
            entry_frame.pack(fill="x", pady=2)

            # Tor-Icon
            tk.Label(
                entry_frame, text=f"T{tag_id}",
                font=tkfont.Font(family="Monospace", size=16, weight="bold"),
                fg="#ffc800", bg="#1a1a2e", width=4
            ).pack(side="left", padx=(0, 10))

            # Kante
            tk.Label(
                entry_frame, text=edge_id,
                font=tkfont.Font(family="Monospace", size=14),
                fg="#ddd", bg="#1a1a2e"
            ).pack(side="left", padx=(0, 10))

            # Kantengewicht (falls vorhanden)
            weight = self._weights.get(edge_id)
            if weight is not None:
                tk.Label(
                    entry_frame, text=f"{float(weight):.1f}s",
                    font=tkfont.Font(family="Monospace", size=12),
                    fg="#888", bg="#1a1a2e"
                ).pack(side="left")

    def _draw_graph(self):
        """Zeichnet den Graphen neu."""
        self.canvas.delete("all")

        visited = set(self.pose.get("visited", []))
        current_node = self.pose.get("to_node", "")
        current_arm = self.pose.get("to_arm", 0)

        # Gruppiere Kanten nach Knotenpaaren
        edges_by_pair = {}
        for n1, n2, edge_id in EDGES:
            pair = tuple(sorted([n1, n2]))
            if pair not in edges_by_pair:
                edges_by_pair[pair] = []
            edges_by_pair[pair].append((n1, n2, edge_id))

        # Kanten zeichnen
        for pair, edges in edges_by_pair.items():
            n_edges = len(edges)
            x1, y1 = NODE_POSITIONS[pair[0]]
            x2, y2 = NODE_POSITIONS[pair[1]]

            dx = x2 - x1
            dy = y2 - y1
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                nx, ny = -dy / length, dx / length
            else:
                nx, ny = 0, 0

            for i, (n1, n2, edge_id) in enumerate(edges):
                if n_edges > 1:
                    offset_idx = i - (n_edges - 1) / 2
                    offset_dist = 25 * offset_idx  # Mehr Abstand
                    ox, oy = nx * offset_dist, ny * offset_dist
                else:
                    ox, oy = 0, 0

                if edge_id in self.gates:
                    color = "#ffc800"
                elif edge_id in visited:
                    color = "#00cc44"
                else:
                    color = "#444"

                width = 4 if edge_id in visited else 3

                self.canvas.create_line(
                    x1 + ox, y1 + oy, x2 + ox, y2 + oy,
                    fill=color, width=width, smooth=True
                )

                # Kanten-ID
                mx = (x1 + x2) / 2 + ox * 1.2
                my = (y1 + y2) / 2 + oy * 1.2 - 8
                self.canvas.create_text(mx, my, text=edge_id, fill="#888", font=("Monospace", 11))

                # Kantengewicht (falls vorhanden)
                weight = self._weights.get(edge_id)
                if weight is not None:
                    wt = float(weight)
                    # Farbskala: blau (schnell) -> gruen -> rot (langsam)
                    max_t = max(wt, 5.0)  # Skala bis ~5s
                    ratio = min(wt / max_t, 1.0)
                    r = int(100 + 155 * ratio)
                    g = int(200 - 100 * ratio)
                    b = int(255 - 200 * ratio)
                    wt_color = f"#{r:02x}{g:02x}{b:02x}"
                    self.canvas.create_text(mx, my + 16, text=f"{wt:.1f}s", fill=wt_color, font=("Monospace", 10))

        # Knoten zeichnen (grosser)
        node_radius = 35
        for node, (x, y) in NODE_POSITIONS.items():
            is_current = (node == current_node)
            r = int(node_radius * 1.15) if is_current else node_radius

            fill = "#00c8ff" if is_current else "#3a3a5c"
            outline = "#00c8ff" if is_current else "#5a5a7c"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=outline, width=4)

            text_color = "#000" if is_current else "#ddd"
            self.canvas.create_text(x, y, text=node, fill=text_color, font=("Helvetica", 26, "bold"))

        # Legende unten
        self.canvas.create_text(350, 610, text="Kantengewichte: blau=schnell → rot=langsam", fill="#666", font=("Monospace", 10))

    def _update(self):
        """Aktualisiert alle Anzeigen."""
        status = self.pose.get("status", "???")
        self.status_val.config(text=status)

        status_colors = {
            "ON_EDGE": "#0f0",
            "AT_INTERSECTION": "#ff0",
            "LOST": "#f44",
            "WAITING_FIRST_STOP": "#888",
        }
        self.status_val.config(fg=status_colors.get(status, "#fff"))

        num_vis = self.pose.get("num_visited", 0)
        num_tot = self.pose.get("num_total", "?")
        self.visited_val.config(text=f"{num_vis}/{num_tot}")
        self.sweep_val.config(text=str(self.explore_state.get("sweep", "?")))
        self.gates_val.config(text=str(self.explore_state.get("num_gates", 0)))
        self.action_val.config(text=self.suggested_action or "—")

        self._update_gate_list()
        self._draw_graph()

    # ── ROS Callbacks ──
    def _cb_pose(self, msg):
        try:
            self.pose = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _cb_gates(self, msg):
        try:
            data = json.loads(msg.data)
            self.gates = data.get("gates", {})
        except (ValueError, TypeError):
            pass

    def _cb_explore(self, msg):
        try:
            self.explore_state = json.loads(msg.data)
        except (ValueError, TypeError):
            pass

    def _cb_suggested(self, msg):
        self.suggested_action = msg.data

    def _cb_complete(self, msg):
        self.complete = bool(msg.data)


# ============================================================ Main
def main():
    rospy.init_node("mapping_gui")

    root = tk.Tk()
    gui = MappingGUI(root)

    def update_loop():
        gui._update()
        root.after(100, update_loop)

    root.after(100, update_loop)
    root.mainloop()


if __name__ == "__main__":
    main()
