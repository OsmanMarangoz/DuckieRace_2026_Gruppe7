#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from duckie_wendehammer_planner import (
    GapTargetTracker,
    RawDuckieDetection,
    WendehammerGapPlanner,
    load_camera_calibration,
    parameter_value,
    render_birdseye_debug,
    render_camera_debug,
    render_masks_debug,
)


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "duckie_wendehammer_node.json"
    with open(config_path, "r") as handle:
        return json.load(handle)


def resolve_workspace_path(path):
    configured = Path(path)
    if configured.exists():
        return str(configured)
    if str(configured).startswith("/workspace/"):
        repo_root = Path(__file__).resolve().parents[4]
        local = repo_root / str(configured)[len("/workspace/") :]
        if local.exists():
            return str(local)
    return str(configured)


def load_frames(input_path, limit=0):
    path = Path(input_path)
    frames = []
    if path.is_dir():
        for item in sorted(path.iterdir()):
            if item.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            image = cv2.imread(str(item))
            if image is not None:
                frames.append((item.name, image))
                if limit and len(frames) >= limit:
                    break
        return frames

    if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        image = cv2.imread(str(path))
        return [(path.name, image)] if image is not None else []

    cap = cv2.VideoCapture(str(path))
    index = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((f"frame_{index:05d}.jpg", frame))
        index += 1
        if limit and len(frames) >= limit:
            break
    cap.release()
    return frames


def load_optional_calibration(path):
    if not path:
        return None
    resolved = resolve_workspace_path(path)
    try:
        return load_camera_calibration(resolved)
    except Exception:
        return None


def undistort_if_possible(image, calibration):
    if calibration is None or not calibration.usable_for(image):
        return image, False
    new_matrix = calibration.new_camera_matrix
    if new_matrix is None:
        new_matrix = calibration.camera_matrix
    return cv2.undistort(image, calibration.camera_matrix, calibration.dist_coeffs, None, new_matrix), True


def resolve_model_path(config):
    configured = config.get("model_path")
    if configured:
        resolved = resolve_workspace_path(configured)
        if Path(resolved).exists():
            return resolved
    local_path = Path(__file__).resolve().parent / "model" / "duckie_yolov8n_640.pt"
    return str(local_path) if local_path.exists() else configured


def detect_duckies(model, image, parameters, now_sec):
    if model is None:
        return []
    conf = float(parameter_value(parameters, "model", "conf", 0.25))
    imgsz = int(parameter_value(parameters, "model", "imgsz", 640))
    results = model.predict(image, imgsz=imgsz, conf=conf, verbose=False)
    detections = []
    if results and len(results) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0])
            detections.append(RawDuckieDetection(x1, y1, x2, y2, score, now_sec))
    return detections


def main():
    parser = argparse.ArgumentParser(description="Replay saved frames/videos through the Wendehammer gap planner.")
    parser.add_argument("input", help="Frame directory, image, or video file")
    parser.add_argument("--output-dir", default="/tmp/duckie_wendehammer_replay", help="Directory for debug images")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of frames to process")
    parser.add_argument("--use-yolo", action="store_true", help="Run duckie YOLO during replay")
    parser.add_argument("--calibration", default=None, help="Optional camera_calibration.json path")
    args = parser.parse_args()

    config = load_config()
    parameters = config["parameters"]
    calibration_path = args.calibration or config.get("camera_calibration_path")
    if calibration_path:
        calibration_path = str(calibration_path).format(
            vehicle=os.environ.get("VEHICLE_NAME", "daisy")
        )
    calibration = load_optional_calibration(calibration_path)
    planner = WendehammerGapPlanner(parameters)
    tracker = GapTargetTracker(planner.settings)

    model = None
    if args.use_yolo:
        if YOLO is None:
            raise RuntimeError("ultralytics is not available")
        model = YOLO(resolve_model_path(config))

    frames = load_frames(args.input, args.limit)
    if not frames:
        raise RuntimeError(f"No frames found in {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_target = None
    summaries = []
    for index, (name, image) in enumerate(frames):
        now_sec = index / 10.0
        analysis_image, calibration_ok = undistort_if_possible(image, calibration)
        raw_duckies = detect_duckies(model, analysis_image, parameters, now_sec)
        result, birdseye, white_mask, yellow_mask, matrix = planner.plan_image(
            analysis_image,
            raw_duckies,
            now_sec,
            previous_target_x=previous_target,
        )
        result = tracker.update(result)
        previous_target = result.target_x if result.has_target else None

        debug_bev = render_birdseye_debug(
            birdseye,
            white_mask,
            yellow_mask,
            result,
            text_lines=[
                f"{name} {result.state} {result.reason}",
                f"gap={result.chosen_gap_width_px} v={result.v:.3f} w={result.omega:.2f}",
                f"duckies={len(result.duckies)} calib={calibration_ok}",
            ],
        )
        debug_camera = render_camera_debug(
            analysis_image,
            raw_duckies,
            result,
            matrix,
            calibration_label=f"calib={calibration_ok}",
        )
        debug_masks = render_masks_debug(white_mask, yellow_mask, result)

        stem = Path(name).stem
        cv2.imwrite(str(output_dir / f"{stem}_bev.jpg"), debug_bev)
        cv2.imwrite(str(output_dir / f"{stem}_camera.jpg"), debug_camera)
        cv2.imwrite(str(output_dir / f"{stem}_masks.jpg"), debug_masks)
        summaries.append(
            {
                "frame": name,
                "state": result.state,
                "reason": result.reason,
                "speed_level": result.speed_level,
                "chosen_gap_width_px": result.chosen_gap_width_px,
                "target": result.status_payload(include_candidates=False)["target"],
                "duckies": len(result.duckies),
                "rejected_duckies": len(result.rejected_duckies),
                "v": result.v,
                "omega": result.omega,
                "calibration_ok": calibration_ok,
            }
        )

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    print(f"Processed {len(frames)} frames. Debug output: {output_dir}")


if __name__ == "__main__":
    main()
