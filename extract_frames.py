#!/usr/bin/env python3
"""
Extract frames from an MP4 video file.
Usage: python3 extract_frames.py <video_file> [output_dir]
"""

import cv2
import sys
import os
from pathlib import Path

def extract_frames(video_path, output_dir="frames", num_frames=100):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the MP4 video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return False

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if total_frames < num_frames:
        print(f"Warning: Video has only {total_frames} frames, extracting all available")
        num_frames = total_frames

    # Calculate frame interval
    frame_interval = total_frames // num_frames

    frame_count = 0
    extracted = 0

    print(f"Extracting {num_frames} frames to '{output_dir}'...")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract every frame_interval-th frame
        if frame_count % frame_interval == 0 and extracted < num_frames:
            filename = os.path.join(output_dir, f"frame_{extracted:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            extracted += 1

        frame_count += 1

    cap.release()

    print(f"Successfully extracted {extracted} frames to '{output_dir}'")
    return True

if __name__ == "__main__":
    # Configuration
    video_file = "/workspace/src/videos/lane_20260609_113028.mp4"
    output_directory = "frames"
    num_frames_to_extract = 100

    extract_frames(video_file, output_directory, num_frames_to_extract)
