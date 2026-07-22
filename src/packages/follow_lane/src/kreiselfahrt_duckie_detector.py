#!/usr/bin/env python3

"""Optional YOLO adapter for the Kreiselfahrt perception input.

The driving logic deliberately knows nothing about YOLO. This adapter turns
Duckie bounding boxes into a binary mask that can be merged with the unchanged
yellow HSV mask.
"""

import importlib
import os

import cv2
import numpy as np


class DuckieYoloDetector:
    """Lazily loads the old Wendehammer Duckie detector."""

    def __init__(self, model_path, image_size=640):
        self.model_path = str(model_path or "")
        self.image_size = int(image_size)
        self.model = None
        self.status = "not_loaded"

    def ensure_loaded(self):
        if self.model is not None:
            return True, "Duckie-KI ist geladen"
        if not self.model_path or not os.path.isfile(self.model_path):
            self.status = "model_file_missing"
            return False, f"Duckie-Modell fehlt: {self.model_path}"

        try:
            yolo_class = importlib.import_module("ultralytics").YOLO
            self.model = yolo_class(self.model_path)
            self.status = "loaded"
            return True, "Duckie-KI ist geladen"
        except Exception as error:
            self.model = None
            self.status = f"load_error:{error}"
            return False, f"Duckie-Modell konnte nicht geladen werden: {error}"

    @staticmethod
    def boxes_to_mask(image_shape, boxes):
        """Return a filled uint8 mask for ``(x1, y1, x2, y2)`` boxes."""
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            left = max(0, min(width - 1, int(round(x1))))
            top = max(0, min(height - 1, int(round(y1))))
            right = max(0, min(width - 1, int(round(x2))))
            bottom = max(0, min(height - 1, int(round(y2))))
            if right <= left or bottom <= top:
                continue
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
        return mask

    def detect_mask(self, image, confidence):
        if self.model is None:
            raise RuntimeError("Duckie-KI ist nicht geladen")

        threshold = max(0.0, min(1.0, float(confidence)))
        try:
            results = self.model.predict(
                image,
                imgsz=self.image_size,
                conf=threshold,
                verbose=False,
            )
            boxes = []
            if results:
                for box in results[0].boxes:
                    score = float(box.conf[0])
                    if score >= threshold:
                        boxes.append(tuple(float(value) for value in box.xyxy[0].tolist()))
            self.status = "loaded"
            return self.boxes_to_mask(image.shape, boxes), len(boxes)
        except Exception as error:
            self.status = f"predict_error:{error}"
            raise RuntimeError(f"Duckie-Erkennung fehlgeschlagen: {error}")
