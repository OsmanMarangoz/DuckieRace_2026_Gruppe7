#!/usr/bin/env python3

"""Qt dashboard for observing and tuning the simple Kreiselfahrt controller."""

import copy
import json
import os
import stat
import sys
import tempfile
import threading

import cv2
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import SetBool

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


PARAMETER_GROUPS = [
    (
        "Bild & Farbe",
        [
            (
                "birdseye",
                "Perspektive",
                [
                    ("top_left_x", "Oben links X", "px", "X-Position der linken oberen Ecke des hellblauen Vierecks im Kamerabild. Größer = Ecke nach rechts."),
                    ("top_left_y", "Oben links Y", "px", "Y-Position der linken oberen Ecke des hellblauen Vierecks im Kamerabild. Größer = Ecke nach unten und der ferne Bildbereich wird kleiner."),
                    ("top_right_x", "Oben rechts X", "px", "X-Position der rechten oberen Ecke des hellblauen Vierecks. Größer = Ecke nach rechts."),
                    ("top_right_y", "Oben rechts Y", "px", "Y-Position der rechten oberen Ecke des hellblauen Vierecks. Größer = Ecke nach unten."),
                    ("bottom_left_x", "Unten links X", "px", "X-Position der linken unteren Ecke des hellblauen Vierecks. Größer = Ecke nach rechts."),
                    ("bottom_left_y", "Unten links Y", "px", "Y-Position der linken unteren Ecke des hellblauen Vierecks. Größer = Ecke weiter nach unten."),
                    ("bottom_right_x", "Unten rechts X", "px", "X-Position der rechten unteren Ecke des hellblauen Vierecks. Größer = Ecke nach rechts."),
                    ("bottom_right_y", "Unten rechts Y", "px", "Y-Position der rechten unteren Ecke des hellblauen Vierecks. Größer = Ecke weiter nach unten."),
                ],
            ),
            (
                "yellow",
                "Gelbe HSV-Maske",
                [
                    ("hl", "H min", "", "Untere Grenze für den Gelb-Farbton. Erhöhe sie, wenn orange/rote Flächen fälschlich gelb werden. Das erkannte Gelb erscheint im Bild „Rohe Gelbmaske“ gelb eingefärbt."),
                    ("hh", "H max", "", "Obere Grenze für den Gelb-Farbton. Verringere sie, wenn grünliche Flächen fälschlich gelb werden."),
                    ("sl", "S min", "", "Mindest-Sättigung. Erhöhen blendet blasse, graue oder weiße Flächen aus; zu hoch kann schlecht beleuchtetes Gelb verlieren."),
                    ("sh", "S max", "", "Höchste erlaubte Sättigung. Meist auf 255 lassen; kleiner schließt sehr kräftige Farben aus."),
                    ("vl", "V min", "", "Mindest-Helligkeit. Erhöhen entfernt dunkle Schatten; zu hoch blendet Gelb bei wenig Licht aus."),
                    ("vh", "V max", "", "Höchste erlaubte Helligkeit. Meist auf 255 lassen; kleiner kann stark überbelichtete Stellen ausblenden."),
                ],
            ),
            (
                "ai_detection",
                "Optionale Duckie-KI",
                [
                    ("confidence", "Duckie-Schwellwert", "", "Mindest-Konfidenz des alten Wendehammer-Modells. Der Regler wirkt nur bei aktivierter Duckie-KI. Größer = weniger, aber sicherere Erkennungen; kleiner = mehr mögliche Duckies und mehr Fehlalarme."),
                ],
            ),
        ],
    ),
    (
        "Virtuelle Linie",
        [
            (
                "virtual_line",
                "Duckies und Liniensegmente",
                [
                    ("min_component_area", "Minimale Fläche", "px²", "Kleinste gelbe Fläche, die benutzt wird. Erhöhen ignoriert kleine gelbe Flecken und Bildrauschen. Benutzte Flächen haben im Bild „Rohe Gelbmaske“ einen grünen Rahmen, verworfene einen roten oder orangefarbenen Rahmen."),
                    ("segment_length", "Segmentlänge", "px", "Mindestlänge der erzeugten gelben virtuellen Linie. Größer verbindet kurze oder unterbrochene gelbe Stücke über eine längere Strecke."),
                    ("line_width", "Linienbreite", "px", "Dicke der gelben virtuellen Linie im Bild „Virtuelle Gelblinie“. Beeinflusst vor allem, in wie vielen Bildzeilen eine Messung gefunden wird."),
                    ("right_padding", "Abstand rechts", "px", "Setzt die gelbe virtuelle Grenze weiter rechts neben eine erkannte Fläche. Größer lässt Daisy mehr Abstand rechts vom Duckie und lenkt früher nach rechts."),
                    ("lookahead_min_y", "Vorschau beginnt ab Y", "px", "Obere orange Linie im virtuellen Bild. Gelb oberhalb dieser Linie ist weit weg und wird nicht zur Lenkung benutzt. Erhöhen = mehr weit entferntes Gelb ignorieren; verringern = früher nach vorn schauen."),
                    ("near_field_top", "Nahschutz ab Y", "px", "Untere orange Linie im virtuellen Bild. Unterhalb davon werden schmale Linien ignoriert; nur breite Duckie-Flächen bleiben als Nahschutz aktiv. Größer verschiebt die Linie nach unten und verkleinert den Nahbereich."),
                    ("near_field_min_aspect_ratio", "Min. Duckie-Breite/Höhe", "", "Bestimmt, wie breit eine gelbe Fläche im Verhältnis zu ihrer Höhe sein muss, damit sie unten im Bild als Duckie gilt. Größer = nur deutlich breitere Flächen werden als Duckie-Nahschutz erkannt."),
                    ("near_field_extra_padding", "Zusätzlicher Nahabstand", "px", "Zusätzlicher Sicherheitsabstand rechts neben einem nahen Duckie. Größer hält Daisy länger rechts und verhindert ein zu frühes Einschlagen nach links. Nahschutz-Flächen sind cyan markiert."),
                ],
            )
        ],
    ),
    (
        "Tracking",
        [
            (
                "tracking",
                "Messbereich und Stabilität",
                [
                    ("roi_left", "Messbereich links", "px", "Linker Rand des blauen Messrahmens im virtuellen Bild. Gelb links außerhalb des Rahmens wird ignoriert. Größer = mehr vom linken Bereich ausblenden."),
                    ("roi_right", "Messbereich rechts", "px", "Rechter Rand des blauen Messrahmens. Gelb und Duckies rechts außerhalb des Rahmens werden ignoriert. Verkleinern, wenn ein weit rechts stehendes Duckie nicht als Linie gelten soll."),
                    ("roi_top", "Messbereich oben", "px", "Oberer Rand des blauen Messrahmens. Erhöhen blendet mehr vom weit entfernten oberen Bildbereich aus."),
                    ("roi_bottom", "Messbereich unten", "px", "Unterer Rand des blauen Messrahmens. Verkleinern blendet mehr vom sehr nahen unteren Bildbereich aus."),
                    ("target_x", "Gewünschte Gelbposition", "px", "Grüne senkrechte Linie in der Regelansicht. Daisy lenkt so, dass die geglättete magentafarbene Grenze auf diese grüne Ziellinie kommt. Größer verschiebt das Ziel nach rechts."),
                    ("min_row_coverage", "Minimale Zeilenabdeckung", "", "Wie viel vom Messbereich eine gelbe virtuelle Linie mindestens abdecken muss. Beispiel: 0,15 bedeutet 15 %. Größer verhindert TRACK bei kurzen Flecken, kann Gelb aber später erkennen."),
                    ("boundary_percentile", "Kantenperzentil", "%", "Bestimmt aus den weißen Messpunkten die orange Rohgrenze. Größer bevorzugt Punkte weiter rechts und reagiert stärker auf rechts liegende Duckies; kleiner folgt eher der normalen gelben Linie."),
                    ("smoothing_alpha", "Glättung", "", "Wie stark der neueste orange Messwert die magentafarbene geglättete Grenze verändert. Nahe 1 = schnell, aber unruhiger. Nahe 0 = ruhig, aber langsamer."),
                    ("hold_seconds", "Lücke überbrücken", "s", "Zeit im Zustand HOLD, wenn Gelb kurz verschwindet. Daisy benutzt dabei die letzte magentafarbene Grenze. Größer überbrückt längere Lücken, kann aber länger einer alten Richtung folgen."),
                    ("duckie_memory_seconds", "Duckie links merken", "s", "Wie lange Daisy nach dem Verschwinden eines nahen Duckies unten links mit HOLD-Tempo geradeaus fährt, wenn kein weiteres nutzbares Gelb sichtbar ist. Danach beginnt SEARCH. 0 deaktiviert die Funktion."),
                    ("reacquire_frames", "Wiedererkennungsframes", "", "So viele gültige Kamerabilder hintereinander müssen Gelb zeigen, bevor SEARCH endet. Kleiner = schnellere Reaktion; größer = weniger Reaktion auf einzelne falsche gelbe Flecken."),
                ],
            )
        ],
    ),
    (
        "Fahren & Sicherheit",
        [
            (
                "control",
                "Regler",
                [
                    ("p", "P", "", "Direkte Lenkstärke für den aktuellen Abstand zwischen grüner Ziellinie und magentafarbener Grenze. Größer = Daisy lenkt sofort stärker, kann aber pendeln."),
                    ("i", "I", "", "Korrigiert einen kleinen Fehler, der längere Zeit bestehen bleibt. Nur vorsichtig erhöhen: zu groß führt zu langsamem Aufschaukeln und Überschwingen."),
                    ("d", "D", "", "Bremst schnelle Änderungen des Regelfehlers. Größer kann Pendeln dämpfen, reagiert aber stärker auf unruhige Bildmessungen."),
                    ("nominal_speed", "TRACK-Tempo", "m/s", "Vorwärtsgeschwindigkeit im grünen Zustand TRACK, wenn Gelb sicher erkannt wird. Für erste Tests niedrig einstellen."),
                    ("reduced_speed", "HOLD-Tempo", "m/s", "Langsamere Vorwärtsgeschwindigkeit im orangefarbenen Zustand HOLD, wenn Gelb nur kurz fehlt."),
                    ("search_speed", "SEARCH-Tempo", "m/s", "Vorwärtsgeschwindigkeit im orangefarbenen Zustand SEARCH. Zusammen mit SEARCH-Drehung bestimmt sie die Größe des Linksbogens: weniger Tempo ergibt bei gleicher Drehung einen engeren Bogen."),
                    ("search_omega", "SEARCH-Drehung", "rad/s", "Drehgeschwindigkeit nach links in SEARCH. Größer = engerer Linksbogen. Derselbe Betrag wird während COUNTERSTEER nach rechts verwendet."),
                    ("reacquire_countersteer_angle", "Gegenlenkwinkel nach SEARCH", "rad", "Violette Phase COUNTERSTEER nach bestätigter Gelb-Wiedererkennung: Daisy hält an und dreht um diesen Winkel nach rechts. 0 deaktiviert die Korrektur; 0,20 rad sind ungefähr 11,5 Grad."),
                    ("reacquire_forward_distance", "Geradeausweg nach Gegenlenken", "m", "Geschätzte Strecke, die Daisy nach COUNTERSTEER mit gerader Lenkung fährt, bevor der TRACK-Regler wieder übernimmt. Die Strecke wird aus Tempo und Zeit geschätzt; 0 deaktiviert die Phase."),
                    ("reacquire_forward_speed", "Geradeaustempo nach Gegenlenken", "m/s", "Konstantes Tempo in RECOVERY_FORWARD. Zusammen mit dem Geradeausweg bestimmt es die Dauer der Stabilisierungsphase. Niedrig beginnen und aufgebockt testen."),
                    ("max_omega", "Maximale Drehung", "rad/s", "Sicherheitsgrenze für die Drehgeschwindigkeit des PID-Reglers und der Suchbewegungen. Kleinere Werte begrenzen alle Lenkbewegungen stärker."),
                    ("integral_limit", "Integralgrenze", "", "Begrenzt, wie viel sich der I-Anteil merken darf. Kleiner verhindert starkes Nachlenken nach einem länger anhaltenden Fehler."),
                ],
            ),
            (
                "safety",
                "Sicherheit und Last",
                [
                    ("camera_timeout", "Kamera-Timeout", "s", "Maximal erlaubtes Alter des letzten Kamerabildes. Ist das Bild länger weg, wird CAMERA_STALE rot angezeigt und Daisy bekommt ein Stoppkommando."),
                    ("command_rate", "Kommandorate", "Hz", "Wie oft pro Sekunde Fahr- oder Stoppbefehle gesendet werden. Normalerweise nicht ändern; höher belastet ROS stärker."),
                    ("debug_rate", "Dashboard-Bildrate", "Hz", "Höchste Bildrate der vier Dashboard-Ansichten. Kleiner entlastet Rechner und Netzwerk, verändert aber nicht die eigentliche Kameraverarbeitung."),
                ],
            ),
        ],
    ),
]


IMAGE_PANELS = [
    ("camera", "Kamera und Bird's-Eye-Ausschnitt", "camera"),
    ("yellow_raw", "Erkennungsmaske", "yellow_raw"),
    ("yellow_virtual", "Virtuelle Gelblinie", "yellow_virtual"),
    ("control", "Regelansicht", "control"),
]

IMAGE_PANEL_TOOLTIPS = {
    "camera": (
        "Originalbild von Daisy. Das hellblaue Viereck zeigt den Bereich, der für die "
        "Bird's-Eye-Ansicht geradegezogen wird. Seine Ecken stellst du unter „Perspektive“ ein."
    ),
    "yellow_raw": (
        "Gelb eingefärbt = aktuelle Maske: standardmäßig HSV, im KI-Modus zusätzlich "
        "die farbunabhängig erkannten Duckie-Boxen. Grüner Rahmen = wird benutzt; "
        "cyanfarbener Rahmen = nahes Duckie mit Sicherheitsabstand; orange = wegen Entfernung "
        "oder Nahbereich ignoriert; rot = zu klein oder außerhalb des Messbereichs."
    ),
    "yellow_virtual": (
        "Gelb = erzeugte virtuelle Grenze; blaues Rechteck = ausgewerteter Messbereich; "
        "orange waagerecht = Distanz- und Nahbereichsgrenzen; weiße Punkte = tatsächlich "
        "gemessene rechte Kante."
    ),
    "control": (
        "Grün senkrecht = gewünschte Gelbposition; orange senkrecht = aktueller Rohwert; "
        "magenta senkrecht = geglättete Grenze des Reglers; blaues Rechteck und orange "
        "waagerechte Linien = Mess- und Entfernungsbereiche."
    ),
}


class KreiselfahrtDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        environment_vehicle = os.environ.get("VEHICLE_NAME", "").strip()
        parameter_vehicle = str(rospy.get_param("~vehicle_name", "")).strip()
        self._vehicle_name = environment_vehicle or parameter_vehicle
        if not self._vehicle_name:
            raise RuntimeError("VEHICLE_NAME oder der private Parameter ~vehicle_name fehlt")
        if environment_vehicle and parameter_vehicle and environment_vehicle != parameter_vehicle:
            rospy.logwarn(
                "[kreiselfahrt_dashboard] Veralteten ~vehicle_name=%s ignoriert; VEHICLE_NAME=%s wird benutzt",
                parameter_vehicle,
                environment_vehicle,
            )

        package_path = rospkg.RosPack().get_path("follow_lane")
        self._config_path = rospy.get_param(
            "~config_path", os.path.join(package_path, "config", "kreiselfahrt_node.json")
        )
        self._config = self._load_config()
        self._parameters = self._config["parameters"]
        self._dirty = False
        self._drive_enabled = False
        self._ai_detection_enabled = False
        self._closing = False
        self._data_lock = threading.Lock()
        self._pending_images = {}
        self._latest_state = {}
        self._parameter_controls = {}

        self._update_publisher = rospy.Publisher(
            f"/{self._vehicle_name}/update_parameters", String, queue_size=1
        )
        self._enable_service_name = f"/{self._vehicle_name}/kreiselfahrt/set_enabled"
        self._enable_service = rospy.ServiceProxy(self._enable_service_name, SetBool)
        self._ai_enable_service_name = (
            f"/{self._vehicle_name}/kreiselfahrt/set_ai_detection"
        )
        self._ai_enable_service = rospy.ServiceProxy(self._ai_enable_service_name, SetBool)

        debug_base = f"/{self._vehicle_name}/debug/kreiselfahrt"
        self._subscribers = [
            rospy.Subscriber(
                f"{debug_base}/{topic}/compressed",
                CompressedImage,
                lambda message, key=key: self._image_callback(key, message),
                queue_size=1,
            )
            for key, _, topic in IMAGE_PANELS
        ]
        self._subscribers.append(
            rospy.Subscriber(
                f"/{self._vehicle_name}/kreiselfahrt/state",
                String,
                self._state_callback,
                queue_size=1,
            )
        )

        self.setWindowTitle(f"Kreiselfahrt – {self._vehicle_name}")
        self.resize(1500, 930)
        self._build_ui()

        self._publish_timer = QTimer(self)
        self._publish_timer.setSingleShot(True)
        self._publish_timer.setInterval(100)
        self._publish_timer.timeout.connect(self._publish_parameters)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(100)
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()

    def _load_config(self):
        with open(self._config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_ui(self):
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        image_frame = QFrame()
        image_grid = QGridLayout(image_frame)
        image_grid.setContentsMargins(0, 0, 0, 0)
        image_grid.setSpacing(8)
        self._image_labels = {}
        for index, (key, title, _) in enumerate(IMAGE_PANELS):
            group = QGroupBox(title)
            group.setToolTip(IMAGE_PANEL_TOOLTIPS[key])
            layout = QVBoxLayout(group)
            label = QLabel("Warte auf Debugbild …")
            label.setToolTip(IMAGE_PANEL_TOOLTIPS[key])
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(360, 300)
            label.setStyleSheet("QLabel { background: #171717; color: #aaaaaa; border: 1px solid #444; }")
            layout.addWidget(label)
            image_grid.addWidget(group, index // 2, index % 2)
            self._image_labels[key] = label
        root.addWidget(image_frame, 3)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(455)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        status_group = QGroupBox("Zustand")
        status_layout = QGridLayout(status_group)
        status_names = [
            ("state", "Fahrstatus"),
            ("detection_mode", "Erkennung"),
            ("tracking_state", "Gelb-Tracking"),
            ("confidence", "Gelb-Konfidenz"),
            ("yellow_age", "Gelb zuletzt gesehen"),
            ("boundary_x", "Grenze X"),
            ("error", "Regelfehler"),
            ("motion", "Fahrbefehl"),
            ("fps", "Bildrate"),
            ("camera_age", "Kameraalter"),
        ]
        self._status_labels = {}
        for row, (key, title) in enumerate(status_names):
            status_layout.addWidget(QLabel(title), row, 0)
            value = QLabel("—")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_layout.addWidget(value, row, 1)
            self._status_labels[key] = value
        control_layout.addWidget(status_group)

        self._ai_button = QPushButton("Duckie-KI aktivieren")
        self._ai_button.setMinimumHeight(42)
        self._ai_button.setToolTip(
            "Schaltet nur die zusätzliche, farbunabhängige Duckie-Erkennung um. "
            "Die gelbe Linie und der Standardmodus bleiben HSV-basiert."
        )
        self._ai_button.clicked.connect(self._toggle_ai_detection)
        control_layout.addWidget(self._ai_button)
        self._update_ai_button(False)

        self._drive_button = QPushButton("Fahrt starten")
        self._drive_button.setMinimumHeight(52)
        self._drive_button.clicked.connect(self._toggle_drive)
        control_layout.addWidget(self._drive_button)
        self._update_drive_button(False)

        self._tabs = QTabWidget()
        self._build_parameter_tabs()
        control_layout.addWidget(self._tabs, 1)

        button_row = QHBoxLayout()
        self._save_button = QPushButton("Parameter speichern")
        self._save_button.setToolTip("Alle aktuell wirksamen Werte dauerhaft in kreiselfahrt_node.json speichern.")
        self._save_button.clicked.connect(self._save_config)
        reload_button = QPushButton("Neu laden")
        reload_button.clicked.connect(self._reload_config)
        button_row.addWidget(self._save_button)
        button_row.addWidget(reload_button)
        control_layout.addLayout(button_row)

        self._dirty_label = QLabel("Alle Änderungen gespeichert")
        self._dirty_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self._dirty_label)
        control_scroll.setWidget(control_widget)
        root.addWidget(control_scroll, 1)
        self.setCentralWidget(central)

    def _build_parameter_tabs(self):
        self._tabs.clear()
        self._parameter_controls = {}
        for tab_title, groups in PARAMETER_GROUPS:
            tab_body = QWidget()
            tab_layout = QVBoxLayout(tab_body)
            for group_name, group_title, specifications in groups:
                group_box = QGroupBox(group_title)
                group_layout = QVBoxLayout(group_box)
                for name, title, unit, description in specifications:
                    definition = self._parameters.get(group_name, {}).get(name)
                    if not self._is_tunable(definition):
                        continue
                    group_layout.addWidget(
                        self._make_parameter_row(group_name, name, title, unit, description, definition)
                    )
                tab_layout.addWidget(group_box)
            tab_layout.addStretch(1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(tab_body)
            self._tabs.addTab(scroll, tab_title)

    @staticmethod
    def _is_tunable(definition):
        return isinstance(definition, dict) and all(key in definition for key in ("default", "min", "max"))

    def _make_parameter_row(self, group, name, title, unit, description, definition):
        row = QFrame()
        row.setToolTip(description)
        layout = QGridLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)

        text = title if not unit else f"{title} [{unit}]"
        label = QLabel(text)
        label.setToolTip(description)
        layout.addWidget(label, 0, 0, 1, 2)

        is_float = any(isinstance(definition[key], float) for key in ("default", "min", "max"))
        scale = 100.0 if is_float else 1.0
        slider = QSlider(Qt.Horizontal)
        slider.setToolTip(description)
        slider.setRange(int(round(definition["min"] * scale)), int(round(definition["max"] * scale)))
        slider.setValue(int(round(definition["default"] * scale)))
        layout.addWidget(slider, 1, 0)

        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setSingleStep(0.01)
        else:
            spin = QSpinBox()
            spin.setSingleStep(1)
        spin.setRange(definition["min"], definition["max"])
        spin.setValue(definition["default"])
        spin.setMinimumWidth(92)
        spin.setToolTip(description)
        layout.addWidget(spin, 1, 1)

        slider.valueChanged.connect(lambda value, widget=spin, factor=scale: widget.setValue(value / factor))
        spin.valueChanged.connect(
            lambda value, g=group, n=name, s=slider, factor=scale, floating=is_float: self._parameter_changed(
                g, n, value, s, factor, floating
            )
        )
        self._parameter_controls[(group, name)] = (slider, spin)
        return row

    def _parameter_changed(self, group, name, value, slider, scale, is_float):
        slider_value = int(round(float(value) * scale))
        if slider.value() != slider_value:
            slider.setValue(slider_value)
        self._parameters[group][name]["default"] = float(value) if is_float else int(value)
        self._set_dirty(True)
        self._publish_timer.start()

    def _set_dirty(self, dirty):
        self._dirty = dirty
        self._dirty_label.setText(
            "Ungespeicherte Live-Änderungen" if dirty else "Alle Änderungen gespeichert"
        )
        self._dirty_label.setStyleSheet("color: #d88400;" if dirty else "color: #408040;")

    def _publish_parameters(self):
        payload = {"node": "kreiselfahrt_node", "parameters": self._parameters}
        self._update_publisher.publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _save_config(self):
        answer = QMessageBox.question(
            self,
            "Kreiselfahrt speichern",
            "Die aktuell wirksamen Werte dauerhaft in kreiselfahrt_node.json speichern?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        temporary_path = None
        try:
            config = copy.deepcopy(self._config)
            config["parameters"] = copy.deepcopy(self._parameters)
            directory = os.path.dirname(self._config_path)
            mode = stat.S_IMODE(os.stat(self._config_path).st_mode)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, prefix=".kreiselfahrt_", suffix=".json", delete=False
            ) as handle:
                temporary_path = handle.name
                json.dump(config, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, self._config_path)
            temporary_path = None
            self._config = config
            self._set_dirty(False)
        except OSError as error:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(error))
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _reload_config(self):
        if self._dirty:
            answer = QMessageBox.question(
                self,
                "Live-Änderungen verwerfen",
                "Ungespeicherte Änderungen verwerfen und die JSON-Datei neu laden?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        try:
            self._config = self._load_config()
            self._parameters = self._config["parameters"]
            self._build_parameter_tabs()
            self._set_dirty(False)
            self._publish_parameters()
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.critical(self, "Neu laden fehlgeschlagen", str(error))

    def _image_callback(self, key, message):
        with self._data_lock:
            self._pending_images[key] = bytes(message.data)

    def _state_callback(self, message):
        try:
            state = json.loads(message.data)
        except ValueError:
            return
        with self._data_lock:
            self._latest_state = state

    def _refresh_ui(self):
        if rospy.is_shutdown():
            QApplication.instance().quit()
            return
        with self._data_lock:
            images = self._pending_images
            self._pending_images = {}
            state = dict(self._latest_state)

        for key, encoded in images.items():
            array = np.frombuffer(encoded, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
            label = self._image_labels[key]
            pixmap = QPixmap.fromImage(qimage).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label.setPixmap(pixmap)

        if state:
            self._drive_enabled = bool(state.get("enabled", False))
            self._update_drive_button(self._drive_enabled)
            self._ai_detection_enabled = bool(state.get("ai_detection_enabled", False))
            self._update_ai_button(self._ai_detection_enabled)
            mode = state.get("state") or "—"
            tracking_mode = state.get("tracking_state") or "—"
            self._status_labels["state"].setText(str(mode))
            if self._ai_detection_enabled:
                detection_text = "KI-Duckies + HSV-Linie"
                detection_text += f" ({int(state.get('ai_duckie_count', 0))})"
            else:
                detection_text = "HSV (Standard)"
            detector_status = state.get("ai_detector_status")
            if detector_status and detector_status not in ("not_loaded", "loaded"):
                detection_text += f" · {detector_status}"
            self._status_labels["detection_mode"].setText(detection_text)
            self._status_labels["tracking_state"].setText(str(tracking_mode))
            self._status_labels["confidence"].setText(f"{float(state.get('confidence', 0.0)) * 100:.0f} %")
            self._status_labels["yellow_age"].setText(self._seconds_text(state.get("yellow_age")))
            self._status_labels["boundary_x"].setText(self._number_text(state.get("boundary_x"), " px"))
            self._status_labels["error"].setText(self._number_text(state.get("error"), "", signed=True))
            self._status_labels["motion"].setText(
                f"v={float(state.get('v', 0.0)):.3f}  ω={float(state.get('omega', 0.0)):+.3f}"
            )
            self._status_labels["fps"].setText(f"{float(state.get('fps', 0.0)):.1f} Hz")
            self._status_labels["camera_age"].setText(self._seconds_text(state.get("camera_age")))
            color = {
                "TRACK": "#228844",
                "HOLD": "#b47a00",
                "SEARCH": "#d06000",
                "COUNTERSTEER": "#7256b8",
                "RECOVERY_FORWARD": "#287f9e",
                "DUCKIE_CLEARANCE": "#1c8b83",
                "CAMERA_STALE": "#c03030",
                "DISABLED": "#777777",
            }.get(mode, "#777777")
            self._status_labels["state"].setStyleSheet(f"font-weight: bold; color: {color};")
            tracking_color = {
                "TRACK": "#228844",
                "HOLD": "#b47a00",
                "SEARCH": "#d06000",
            }.get(tracking_mode, "#777777")
            self._status_labels["tracking_state"].setStyleSheet(
                f"font-weight: bold; color: {tracking_color};"
            )

    @staticmethod
    def _number_text(value, suffix="", signed=False):
        if value is None:
            return "—"
        pattern = "+.3f" if signed else ".1f"
        return f"{float(value):{pattern}}{suffix}"

    @staticmethod
    def _seconds_text(value):
        return "—" if value is None else f"{float(value):.2f} s"

    def _toggle_drive(self):
        target = not self._drive_enabled
        if target:
            answer = QMessageBox.warning(
                self,
                "Fahrt wirklich starten?",
                "Die Kreiselfahrt-Node wird echte Fahrbefehle senden. Sind Masken und Regler geprüft und ist der Stopp erreichbar?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self._set_drive(target)

    def _toggle_ai_detection(self):
        target = not self._ai_detection_enabled
        try:
            rospy.wait_for_service(self._ai_enable_service_name, timeout=1.0)
            response = self._ai_enable_service(bool(target))
            if not response.success:
                raise RuntimeError(response.message)
            self._ai_detection_enabled = target
            self._update_ai_button(target)
        except Exception as error:
            QMessageBox.critical(self, "Duckie-KI konnte nicht umgeschaltet werden", str(error))

    def _update_ai_button(self, enabled):
        if enabled:
            self._ai_button.setText("Duckie-KI AKTIV – nur HSV verwenden")
            self._ai_button.setStyleSheet(
                "QPushButton { background: #276a8f; color: white; font-weight: bold; }"
            )
        else:
            self._ai_button.setText("Duckie-KI aktivieren")
            self._ai_button.setStyleSheet("")

    def _set_drive(self, enabled, show_errors=True):
        try:
            rospy.wait_for_service(self._enable_service_name, timeout=1.0)
            response = self._enable_service(bool(enabled))
            if not response.success:
                raise RuntimeError(response.message)
            self._drive_enabled = bool(enabled)
            self._update_drive_button(self._drive_enabled)
            return True
        except Exception as error:
            if show_errors:
                QMessageBox.critical(self, "Fahrfreigabe fehlgeschlagen", str(error))
            return False

    def _update_drive_button(self, enabled):
        if enabled:
            self._drive_button.setText("STOPP – Fahrt aktiv")
            self._drive_button.setStyleSheet(
                "QPushButton { background: #b02020; color: white; font-size: 17px; font-weight: bold; }"
            )
        else:
            self._drive_button.setText("Fahrt starten")
            self._drive_button.setStyleSheet(
                "QPushButton { background: #277a3a; color: white; font-size: 17px; font-weight: bold; }"
            )

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return
        self._closing = True
        self._set_drive(False, show_errors=False)
        rospy.signal_shutdown("Kreiselfahrt-Dashboard geschlossen")
        event.accept()


def main():
    rospy.init_node("kreiselfahrt_dashboard", disable_signals=True)
    application = QApplication(sys.argv)
    application.setApplicationName("Kreiselfahrt Dashboard")
    dashboard = KreiselfahrtDashboard()
    dashboard.show()
    try:
        exit_code = application.exec_()
    except KeyboardInterrupt:
        exit_code = 0
    if not rospy.is_shutdown():
        rospy.signal_shutdown("Kreiselfahrt-Dashboard beendet")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
