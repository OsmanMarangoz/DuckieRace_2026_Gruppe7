#!/usr/bin/env python3

import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


PARAMETER_TABS = [
    (
        "Fahren",
        [
            ("behavior", "lane_follow_speed", "Linienfahrt-Tempo", "m/s", "Vorwärtsgeschwindigkeit, solange noch keine nahen Duckies den Gap-Modus aktivieren."),
            ("behavior", "lane_fallback_speed", "Tempo ohne sichtbare Linien", "m/s", "Konservatives Geradeaus-Tempo, wenn beide Fahrbahnlinien fehlen."),
            ("behavior", "lane_omega_gain", "Linien-Lenkverstärkung", "", "Lenkverstärkung der integrierten alten Linienregelung."),
            ("behavior", "lane_max_omega", "Maximale Linien-Drehung", "rad/s", "Begrenzt die Winkelgeschwindigkeit während der Linienfahrt."),
            ("behavior", "lane_turn_slowdown", "Abbremsen in Kurven", "", "Reduziert das Linienfahrt-Tempo bei großen Lenkausschlägen."),
            ("control", "crawl_speed", "Minimaltempo", "m/s", "Tempo bei einer gerade noch ausreichend breiten Lücke."),
            ("control", "max_speed", "Maximaltempo", "m/s", "Absolute obere Geschwindigkeitsgrenze des Wendehammer-Planners."),
            ("control", "gap_pass_speed", "Lücke durchfahren", "m/s", "Vorwärtstempo, sobald eine nahe und zentrierte Lücke erreicht wurde."),
            ("control", "gap_pass_frames", "Durchfahrdauer", "Frames", "So viele Planner-Frames wird die erreichte Lücke vorwärts durchfahren."),
            ("control", "search_omega", "Suchdrehung", "rad/s", "Drehgeschwindigkeit, wenn kein gültiges Ziel vorhanden ist."),
            ("control", "omega_gain", "Lenkverstärkung", "", "Wie stark ein seitlicher Zielfehler in eine Drehung umgesetzt wird."),
            ("control", "max_omega", "Maximale Drehung", "rad/s", "Absolute Begrenzung der geplanten Winkelgeschwindigkeit."),
            ("control", "target_smoothing", "Zielglättung", "", "Höher = ruhigeres Ziel, aber langsamere Reaktion auf Änderungen."),
            ("control", "target_reached_error_px", "Zentriert-Toleranz", "px", "Maximaler seitlicher Fehler, damit ein Ziel als zentriert zählt."),
            ("control", "target_centered_frames", "Stabile Frames", "Frames", "So viele zentrierte Frames sind für 'Ziel erreicht' nötig."),
            ("control", "target_reached_y", "Nah-Schwelle", "BEV-px", "Größer = Ziel muss im Bird's-Eye-Bild näher am Bot sein."),
            ("control", "target_complete_search_frames", "Suchpause danach", "Frames", "Suchframes nach einer abgeschlossenen Lücke."),
        ],
    ),
    (
        "Enten & Lücken",
        [
            ("model", "conf", "YOLO-Konfidenz", "", "Erkennungen unter diesem Vertrauenswert werden ignoriert."),
            ("model", "frame_stride", "YOLO nur jedes n-te Bild", "Frames", "Höher entlastet die CPU, reagiert aber langsamer."),
            ("model", "debug_frame_stride", "Dashboard nur jedes n-te Bild", "Frames", "Reduziert JPEG-Last und Latenz, ohne die Fahrentscheidung auszudünnen."),
            ("model", "track_timeout", "Erkennung behalten", "s", "Wie lange die letzte YOLO-Box zwischen Inferenzframes gültig bleibt."),
            ("safety", "min_box_height_px", "Minimale Boxhöhe", "Bild-px", "Kleinere beziehungsweise weiter entfernte Enten werden ignoriert."),
            ("safety", "min_box_area_px", "Minimale Boxfläche", "px²", "Zusätzlicher Größenfilter gegen kleine Fehl-Erkennungen."),
            ("safety", "min_duckie_y", "Reagieren ab", "BEV-Y", "Größer = weiter entfernte Enten später berücksichtigen."),
            ("safety", "max_duckie_y", "Enten gültig bis", "BEV-Y", "Enten außerhalb des nahen Bird's-Eye-Bereichs ignorieren."),
            ("safety", "min_gap_px", "Minimale Lückenbreite", "BEV-px", "Kleinere Abstände zwischen zwei Hindernissen werden nicht angefahren."),
            ("safety", "duckie_margin_x_px", "Seitlicher Sicherheitsrand", "BEV-px", "Virtueller Sicherheitsabstand links und rechts jeder Ente."),
            ("safety", "duckie_margin_y_px", "Längs-Sicherheitsrand", "BEV-px", "Virtueller Sicherheitsabstand vor und hinter jeder Ente."),
            ("safety", "max_box_age_sec", "Maximales Boxalter", "s", "Ältere Erkennungen werden verworfen."),
            ("safety", "stale_margin_gain_px", "Zusatzrand bei alter Box", "BEV-px", "Vergrößert den Sicherheitsrand, solange eine Box altert."),
            ("behavior", "gap_activation_min_box_height_px", "Gap-Modus ab Boxhöhe", "Bild-px", "Erst ab dieser Boxhöhe gilt mindestens eine Ente als direkt vor Daisy."),
            ("behavior", "gap_activation_min_box_area_px", "Gap-Modus ab Boxfläche", "px²", "Zusätzliche Mindestfläche für den Wechsel von Linienfahrt zu Gap-Suche."),
            ("behavior", "gap_activation_frames", "Aktivierung bestätigen", "Frames", "Nahe Enten müssen so viele Frames stabil erkannt werden."),
            ("behavior", "gap_release_frames", "Gap-Modus halten", "Frames", "Verhindert Rücksprünge zur Linienfahrt bei kurz flackernder Erkennung."),
        ],
    ),
    (
        "Rücksetzen & Suchen",
        [
            ("recovery", "stop_seconds", "Stopp vor Rückwärtsfahrt", "s", "Kurzer Stillstand, bevor ein blockierter Wendehammer zurückgesetzt wird."),
            ("recovery", "no_gap_confirm_frames", "Keine Lücke bestätigen", "Frames", "Erst nach so vielen aufeinanderfolgenden Bildern ohne Alternative darf Daisy zurücksetzen."),
            ("recovery", "reverse_speed", "Rückwärts-Tempo", "m/s", "Betrag der Rückwärtsgeschwindigkeit; ausreichend hoch für die Motorreibung wählen."),
            ("recovery", "reverse_seconds", "Rückwärts-Dauer", "s", "Wie lange Daisy bei fehlender sicherer Lücke zurücksetzt."),
            ("recovery", "turn_omega", "Neuorientierungs-Drehung", "rad/s", "Drehgeschwindigkeit nach dem Zurücksetzen."),
            ("recovery", "turn_seconds", "Neuorientierungs-Dauer", "s", "Dauer der Winkelanpassung vor der erneuten Gap-Suche."),
            ("recovery", "settle_seconds", "Beruhigungszeit", "s", "Stillstand nach der Drehung vor der nächsten Bildentscheidung."),
        ],
    ),
    (
        "Linien",
        [
            ("lane", "lookahead_y", "Linien-Lookahead", "BEV-Y", "Zielzeile für die vorausschauende Linienregelung; kleiner schaut weiter voraus."),
            ("lane", "boundary_clearance_px", "Linien-Sicherheitsabstand", "BEV-px", "Ab diesem seitlichen Abstand wird die Randvermeidung zunehmend verstärkt."),
            ("lane", "boundary_avoid_omega", "Notlenkung an Linie", "rad/s", "Mindestlenkung, wenn Weiß oder Gelb in den Sicherheitskorridor eindringt."),
            ("lane", "emergency_min_y", "Notlenkung ab Tiefe", "BEV-Y", "Ignoriert weiter entfernte Maskenstörungen unterhalb dieser BEV-Zeile."),
            ("lane", "min_turn_speed", "Mindesttempo in Kurve", "m/s", "Verhindert, dass die Motoren bei starker Linienkorrektur unter ihre wirksame Leistung fallen."),
            ("planner", "slice_near_y", "Naher Messschnitt", "BEV-Y", "Unterste Zeile für die Korridor- und Lückensuche."),
            ("planner", "slice_far_y", "Ferner Messschnitt", "BEV-Y", "Oberste Zeile für die Korridor- und Lückensuche."),
            ("planner", "slice_count", "Anzahl Messschnitte", "", "Mehr Schnitte liefern mehr Kandidaten, kosten aber Rechenzeit."),
            ("planner", "line_window_half_height", "Linien-Suchhöhe", "px", "Vertikaler Suchbereich um jeden Messschnitt."),
            ("planner", "min_line_points", "Minimale Linienpixel", "px", "Unterhalb dieser Pixelzahl gilt eine Linie als nicht gefunden."),
            ("planner", "line_guard_px", "Abstand zur Linie", "BEV-px", "Sicherheitsrand nach innen an weißer und gelber Linie."),
            ("planner", "corridor_margin_px", "Rand des BEV", "BEV-px", "Verhindert Ziele direkt am Rand des Bird's-Eye-Bildes."),
            ("planner", "yellow_right_percentile", "Gelb-Perzentil", "%", "Welche Seite der gelben Pixel als innere Grenze verwendet wird."),
            ("planner", "white_left_percentile", "Weiß-Perzentil", "%", "Welche Seite der weißen Pixel als innere Grenze verwendet wird."),
            ("planner", "require_duckie_for_target", "Ente an Lücke erforderlich", "", "Verhindert Vorwärtsfahrt nur aufgrund von Linien."),
            ("planner", "allow_virtual_corridor_edges", "Offenen Wendehammer-Rand nutzen", "", "Erlaubt eine breite Lücke zwischen Duckie und konservativem Bildrand, wenn dort keine Linie sichtbar ist."),
        ],
    ),
    (
        "Perspektive",
        [
            ("birdseye", "top_left_x", "Oben links X", "Bild-px", "Linker oberer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "top_left_y", "Oben links Y", "Bild-px", "Linker oberer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "top_right_x", "Oben rechts X", "Bild-px", "Rechter oberer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "top_right_y", "Oben rechts Y", "Bild-px", "Rechter oberer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "bottom_left_x", "Unten links X", "Bild-px", "Linker unterer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "bottom_left_y", "Unten links Y", "Bild-px", "Linker unterer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "bottom_right_x", "Unten rechts X", "Bild-px", "Rechter unterer Quellpunkt der Bird's-Eye-Transformation."),
            ("birdseye", "bottom_right_y", "Unten rechts Y", "Bild-px", "Rechter unterer Quellpunkt der Bird's-Eye-Transformation."),
        ],
    ),
    (
        "Farben (HSV)",
        [
            ("yellow", "hl", "Gelb H min", "", "Untere HSV-Farbgrenze für Gelb."),
            ("yellow", "hh", "Gelb H max", "", "Obere HSV-Farbgrenze für Gelb."),
            ("yellow", "sl", "Gelb S min", "", "Minimale Sättigung für Gelb."),
            ("yellow", "vl", "Gelb V min", "", "Minimale Helligkeit für Gelb."),
            ("white", "sl", "Weiß S min", "", "Untere Sättigungsgrenze für Weiß."),
            ("white", "sh", "Weiß S max", "", "Maximale Sättigung für Weiß."),
            ("white", "vl", "Weiß V min", "", "Minimale Helligkeit für Weiß."),
            ("white", "vh", "Weiß V max", "", "Maximale Helligkeit für Weiß."),
        ],
    ),
]


class ImagePanel(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600; padding: 2px;")
        self.image = QLabel("Warte auf Bild …")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(320, 220)
        self.image.setStyleSheet("background: #16191d; color: #aeb4bc;")
        layout.addWidget(self.title)
        layout.addWidget(self.image, 1)
        self.current_qimage = None

    def set_cv_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        self.current_qimage = QImage(
            rgb.data, width, height, width * 3, QImage.Format_RGB888
        ).copy()
        self.refresh_pixmap()

    def refresh_pixmap(self):
        if self.current_qimage is None:
            return
        pixmap = QPixmap.fromImage(self.current_qimage).scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_pixmap()


class ParameterControl(QWidget):
    def __init__(self, metadata, value, callback, tooltip=""):
        super().__init__()
        self.metadata = metadata
        self.callback = callback
        self._updating = False
        self.is_bool = (
            metadata["min"] == 0
            and metadata["max"] == 1
            and isinstance(metadata["default"], int)
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if self.is_bool:
            self.checkbox = QCheckBox("aktiv")
            self.checkbox.setChecked(bool(value))
            self.checkbox.stateChanged.connect(self._checkbox_changed)
            layout.addWidget(self.checkbox)
            layout.addStretch(1)
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 1000)
            self.slider.valueChanged.connect(self._slider_changed)
            is_float = any(isinstance(metadata[key], float) for key in ("default", "min", "max"))
            if is_float:
                self.spin = QDoubleSpinBox()
                span = abs(float(metadata["max"]) - float(metadata["min"]))
                decimals = 3 if span <= 1.0 else (2 if span <= 10.0 else 1)
                self.spin.setDecimals(decimals)
                self.spin.setSingleStep(max(span / 200.0, 10 ** (-decimals)))
            else:
                self.spin = QSpinBox()
                self.spin.setSingleStep(1)
            self.spin.setRange(metadata["min"], metadata["max"])
            self.spin.valueChanged.connect(self._spin_changed)
            self.spin.setFixedWidth(92)
            layout.addWidget(self.slider, 1)
            layout.addWidget(self.spin)
            self.set_value(value, emit=False)
        self.setToolTip(tooltip)

    def _to_slider(self, value):
        low, high = float(self.metadata["min"]), float(self.metadata["max"])
        if high <= low:
            return 0
        return int(round((float(value) - low) / (high - low) * 1000.0))

    def _from_slider(self, position):
        low, high = float(self.metadata["min"]), float(self.metadata["max"])
        value = low + (high - low) * float(position) / 1000.0
        if isinstance(self.metadata["default"], int):
            return int(round(value))
        return float(value)

    def _checkbox_changed(self, state):
        if not self._updating:
            self.callback(1 if state == Qt.Checked else 0)

    def _slider_changed(self, position):
        if self._updating:
            return
        value = self._from_slider(position)
        self._updating = True
        self.spin.setValue(value)
        self._updating = False
        self.callback(value)

    def _spin_changed(self, value):
        if self._updating:
            return
        if isinstance(self.metadata["default"], int):
            value = int(value)
        else:
            value = float(value)
        self._updating = True
        self.slider.setValue(self._to_slider(value))
        self._updating = False
        self.callback(value)

    def set_value(self, value, emit=False):
        self._updating = not emit
        if self.is_bool:
            self.checkbox.setChecked(bool(value))
        else:
            self.slider.setValue(self._to_slider(value))
            self.spin.setValue(value)
        self._updating = False


class WendehammerDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        rospy.init_node("duckie_wendehammer_dashboard", disable_signals=True)
        self.vehicle = os.environ.get("VEHICLE_NAME", "daisy")
        self.node_name = "duckie_wendehammer_node"
        self.config_path = Path(__file__).resolve().parent.parent / "config" / f"{self.node_name}.json"
        self.config = self.load_config()
        self.parameters = self.config["parameters"]
        self.controls = {}
        self.data_lock = threading.Lock()
        self.latest_images = {}
        self.image_versions = {}
        self.displayed_versions = {}
        self.latest_state = None
        self.state_version = 0
        self.displayed_state_version = -1

        self.update_pub = rospy.Publisher(
            f"/{self.vehicle}/update_parameters", String, queue_size=1, latch=True
        )
        self.control_pub = rospy.Publisher(
            f"/{self.vehicle}/duckie_wendehammer/control", String, queue_size=1
        )

        topics = {
            "camera": f"/{self.vehicle}/debug/duckie_wendehammer/camera/compressed",
            "bev": f"/{self.vehicle}/debug/duckie_wendehammer/bev/compressed",
            "masks": f"/{self.vehicle}/debug/duckie_wendehammer/masks/compressed",
            "undistorted": f"/{self.vehicle}/debug/duckie_wendehammer/undistorted/compressed",
        }
        self.subscribers = [
            rospy.Subscriber(topic, CompressedImage, self.image_callback, callback_args=key, queue_size=1)
            for key, topic in topics.items()
        ]
        self.state_sub = rospy.Subscriber(
            f"/{self.vehicle}/duckie_wendehammer/state", String, self.state_callback, queue_size=1
        )

        self.setWindowTitle(f"Wendehammer Dashboard — {self.vehicle}")
        self.resize(1760, 1020)
        self.setMinimumSize(1200, 760)
        self.build_ui()

        self.publish_timer = QTimer(self)
        self.publish_timer.setSingleShot(True)
        self.publish_timer.timeout.connect(self.publish_parameters)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_ui)
        self.refresh_timer.start(100)
        QTimer.singleShot(800, self.publish_parameters)
        QTimer.singleShot(2200, self.publish_parameters)

    def load_config(self):
        with open(self.config_path, "r") as handle:
            return json.load(handle)

    def build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.addWidget(self.build_status_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.build_image_area())
        splitter.addWidget(self.build_control_area())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1180, 520])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def build_status_bar(self):
        box = QFrame()
        box.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(box)
        self.drive_badge = QLabel("SICHER · FAHREN AUS")
        self.drive_badge.setAlignment(Qt.AlignCenter)
        self.drive_badge.setMinimumWidth(190)
        self.state_label = QLabel("Status: warte auf Node …")
        self.perception_label = QLabel("Kalibrierung: —   YOLO: —")
        self.motion_label = QLabel("Plan: v=—  ω=—")
        for label in (self.state_label, self.perception_label, self.motion_label):
            label.setStyleSheet("font-size: 13px; padding: 5px;")
        layout.addWidget(self.drive_badge)
        layout.addWidget(self.state_label, 2)
        layout.addWidget(self.perception_label, 2)
        layout.addWidget(self.motion_label, 2)
        self.set_drive_badge(False)
        return box

    def build_image_area(self):
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 4, 0)
        self.panels = {
            "camera": ImagePanel("Kamera · Erkennungen und Ziel"),
            "bev": ImagePanel("Bird's-Eye · Korridor, Enten und gewählte Lücke"),
            "masks": ImagePanel("Masken · Weiß/Gelb nach Enten-Unterdrückung"),
            "undistorted": ImagePanel("Entzerrtes Kamerabild"),
        }
        grid.addWidget(self.panels["camera"], 0, 0)
        grid.addWidget(self.panels["bev"], 0, 1)
        grid.addWidget(self.panels["masks"], 1, 0)
        grid.addWidget(self.panels["undistorted"], 1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return widget

    def build_control_area(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 0, 0, 0)

        safety = QGroupBox("Fahrfreigabe")
        safety_layout = QGridLayout(safety)
        self.stop_button = QPushButton("■  NOT-STOPP / FAHREN AUS")
        self.stop_button.setMinimumHeight(48)
        self.stop_button.setStyleSheet(
            "QPushButton { background:#b52025; color:white; font-weight:bold; font-size:15px; }"
            "QPushButton:hover { background:#d22b31; }"
        )
        self.stop_button.clicked.connect(self.disable_drive)
        self.enable_button = QPushButton("Fahren bewusst freigeben …")
        self.enable_button.setMinimumHeight(38)
        self.enable_button.clicked.connect(self.confirm_enable_drive)
        self.reset_tracker_button = QPushButton("Ziel-Tracker zurücksetzen")
        self.reset_tracker_button.clicked.connect(self.reset_tracker)
        safety_layout.addWidget(self.stop_button, 0, 0, 1, 2)
        safety_layout.addWidget(self.enable_button, 1, 0)
        safety_layout.addWidget(self.reset_tracker_button, 1, 1)
        layout.addWidget(safety)

        note = QLabel(
            "Abstände sind derzeit Bild-/Bird's-Eye-Pixel, nicht Zentimeter. "
            "Änderungen werden live angewendet, aber erst mit „Speichern“ dauerhaft."
        )
        note.setWordWrap(True)
        note.setStyleSheet("background:#28313a; color:#e4e8ec; padding:8px; border-radius:4px;")
        layout.addWidget(note)

        tabs = QTabWidget()
        for tab_name, specs in PARAMETER_TABS:
            tabs.addTab(self.build_parameter_tab(specs), tab_name)
        layout.addWidget(tabs, 1)

        actions = QGridLayout()
        self.save_button = QPushButton("Auf Platte speichern")
        self.save_button.clicked.connect(self.save_config)
        self.reload_button = QPushButton("Gespeicherte Werte laden")
        self.reload_button.clicked.connect(self.reload_config)
        self.slow_button = QPushButton("Preset: langsam + Motorreserve")
        self.slow_button.clicked.connect(self.apply_slow_preset)
        self.apply_button = QPushButton("Alle Werte jetzt senden")
        self.apply_button.clicked.connect(self.publish_parameters)
        actions.addWidget(self.save_button, 0, 0)
        actions.addWidget(self.reload_button, 0, 1)
        actions.addWidget(self.slow_button, 1, 0)
        actions.addWidget(self.apply_button, 1, 1)
        layout.addLayout(actions)
        self.action_status = QLabel("Live-Verbindung wird aufgebaut …")
        self.action_status.setStyleSheet("color:#8f9aa6; padding:4px;")
        layout.addWidget(self.action_status)
        return widget

    def build_parameter_tab(self, specs):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        for group, name, label, unit, tooltip in specs:
            metadata = self.parameters.get(group, {}).get(name)
            if not isinstance(metadata, dict) or not all(k in metadata for k in ("default", "min", "max")):
                continue
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 3, 0, 5)
            title = QLabel(f"{label}{'  [' + unit + ']' if unit else ''}")
            title.setToolTip(tooltip)
            title.setStyleSheet("font-weight:600;")
            control = ParameterControl(
                metadata,
                metadata["default"],
                lambda value, g=group, n=name: self.parameter_changed(g, n, value),
                tooltip,
            )
            self.controls[(group, name)] = control
            row_layout.addWidget(title)
            row_layout.addWidget(control)
            layout.addWidget(row)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def image_callback(self, msg, key):
        data = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return
        with self.data_lock:
            self.latest_images[key] = image
            self.image_versions[key] = self.image_versions.get(key, 0) + 1

    def state_callback(self, msg):
        try:
            state = json.loads(msg.data)
        except ValueError:
            return
        with self.data_lock:
            self.latest_state = state
            self.state_version += 1

    def refresh_ui(self):
        with self.data_lock:
            images = {
                key: (self.image_versions.get(key, 0), image.copy())
                for key, image in self.latest_images.items()
            }
            state = None if self.latest_state is None else dict(self.latest_state)
            state_version = self.state_version
        for key, (version, image) in images.items():
            if self.displayed_versions.get(key) != version:
                self.panels[key].set_cv_image(image)
                self.displayed_versions[key] = version
        if state is not None and state_version != self.displayed_state_version:
            self.update_status(state)
            self.displayed_state_version = state_version

    def update_status(self, state):
        drive_enabled = bool(state.get("drive_enabled", False))
        self.set_drive_badge(drive_enabled)
        reason = state.get("reason", "—")
        duckies = len(state.get("duckies", []))
        rejected = len(state.get("rejected_duckies", []))
        gap = state.get("chosen_gap_width_px")
        gap_text = "—" if gap is None else f"{gap:.1f}px"
        mode = "GAP" if state.get("gap_mode_active") else "LINIE"
        recovery = state.get("recovery_phase", "idle")
        no_gap = int(state.get("no_gap_frames", 0))
        no_gap_required = int(state.get("no_gap_confirm_frames", 0))
        self.state_label.setText(
            f"Status: {state.get('state', '—')} · {reason} · Modus {mode} · Recovery {recovery} · "
            f"No-Gap {no_gap}/{no_gap_required} · Enten {duckies} (+{rejected} ignoriert) · Lücke {gap_text}"
        )
        calibration = "OK" if state.get("calibration_ok") else state.get("calibration_status", "FEHLT")
        self.perception_label.setText(
            f"Kalibrierung: {calibration} · YOLO: {state.get('detector_status', '—')}"
        )
        v = float(state.get("v", 0.0))
        omega = float(state.get("omega", 0.0))
        published = bool(state.get("command_published", False))
        qualifier = "PUBLIZIERT" if published else "nur geplant"
        self.motion_label.setText(f"Plan: v={v:.3f} m/s · ω={omega:.3f} rad/s · {qualifier}")

    def set_drive_badge(self, enabled):
        if enabled:
            self.drive_badge.setText("ACHTUNG · FAHREN AN")
            self.drive_badge.setStyleSheet(
                "background:#d17b00; color:white; font-weight:bold; font-size:14px; padding:9px; border-radius:4px;"
            )
        else:
            self.drive_badge.setText("SICHER · FAHREN AUS")
            self.drive_badge.setStyleSheet(
                "background:#227447; color:white; font-weight:bold; font-size:14px; padding:9px; border-radius:4px;"
            )

    def parameter_changed(self, group, name, value):
        metadata = self.parameters[group][name]
        metadata["default"] = int(value) if isinstance(metadata["default"], int) else float(value)
        self.action_status.setText(f"Live geändert: {group}.{name} = {metadata['default']}")
        self.publish_timer.start(120)

    def publish_parameters(self):
        payload = {"node": self.node_name, "parameters": self.parameters}
        self.update_pub.publish(String(data=json.dumps(payload)))
        self.action_status.setText("Alle Werte live an den Planner gesendet.")

    def save_config(self):
        self.config["parameters"] = self.parameters
        temporary = self.config_path.with_suffix(".json.tmp")
        with open(temporary, "w") as handle:
            json.dump(self.config, handle, indent=2)
            handle.write("\n")
        os.replace(str(temporary), str(self.config_path))
        self.action_status.setText(f"Dauerhaft gespeichert: {self.config_path}")

    def reload_config(self):
        self.config = self.load_config()
        self.parameters = self.config["parameters"]
        for (group, name), control in self.controls.items():
            control.metadata = self.parameters[group][name]
            control.set_value(self.parameters[group][name]["default"], emit=False)
        self.publish_parameters()
        self.action_status.setText("Gespeicherte Werte geladen und live gesendet.")

    def apply_slow_preset(self):
        values = {
            ("behavior", "lane_follow_speed"): 0.12,
            ("behavior", "lane_fallback_speed"): 0.10,
            ("control", "crawl_speed"): 0.07,
            ("control", "max_speed"): 0.11,
            ("control", "gap_pass_speed"): 0.10,
            ("control", "search_omega"): 0.20,
            ("control", "omega_gain"): 0.85,
            ("control", "max_omega"): 0.75,
            ("recovery", "reverse_speed"): 0.11,
        }
        for key, value in values.items():
            group, name = key
            self.parameters[group][name]["default"] = value
            if key in self.controls:
                self.controls[key].set_value(value, emit=False)
        self.publish_parameters()
        self.action_status.setText("Preset „langsam + Motorreserve“ ist live aktiv (noch nicht gespeichert).")

    def send_control(self, enabled):
        payload = {"action": "set_drive_enabled", "enabled": bool(enabled)}
        self.control_pub.publish(String(data=json.dumps(payload)))

    def disable_drive(self):
        self.send_control(False)
        self.set_drive_badge(False)
        self.action_status.setText("NOT-STOPP gesendet; Fahrfreigabe deaktiviert.")

    def confirm_enable_drive(self):
        answer = QMessageBox.warning(
            self,
            "Fahren wirklich freigeben?",
            "Daisy kann sich unmittelbar bewegen. Strecke freihalten und NOT-STOPP bereithalten.\n\n"
            "Soll die Fahrfreigabe jetzt aktiviert werden?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer == QMessageBox.Yes:
            self.send_control(True)
            self.action_status.setText("Fahrfreigabe angefordert — Live-Status kontrollieren.")

    def reset_tracker(self):
        self.control_pub.publish(String(data=json.dumps({"action": "reset_tracker"})))
        self.action_status.setText("Ziel-Tracker zurückgesetzt.")

    def closeEvent(self, event):
        self.send_control(False)
        rospy.signal_shutdown("Dashboard closed")
        event.accept()


def main():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    dashboard = WendehammerDashboard()
    dashboard.showMaximized()
    application.aboutToQuit.connect(lambda: rospy.signal_shutdown("Dashboard closed"))
    application.exec_()


if __name__ == "__main__":
    main()
