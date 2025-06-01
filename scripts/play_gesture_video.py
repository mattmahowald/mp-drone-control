#!/usr/bin/env python3
"""
Play a demo video (or a sequence of images) for a given gesture (e.g. "takeoff") from the raw data folder.
Usage:
    poetry run python scripts/play_gesture_video.py --gesture takeoff
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path

# --- demo (or "gesture") folder ---
# (Replace "takeoff" with your desired gesture (e.g. "land", "hover", "forward", "backward", "left", "right", "rotate_left", "rotate_right", "stop") if needed.)
GESTURE_FOLDER = "mp-drone-control-main/data/raw/takeoff"

# --- demo (or "gesture") filename ---
# (In our case, the "takeoff" folder contains a single .npy (numpy) file (20250515_151012_takeoff.npy).)
DEMO_FILENAME = "20250515_151012_takeoff.npy"

# --- main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Play a demo video (or a sequence of images) for a given gesture (e.g. "takeoff") from the raw data folder.")
    ap.add_argument("--gesture", type=str, default="takeoff", help="Gesture (demo) folder name (default: "takeoff").")
    args = ap.parse_args()

    # --- demo (or "gesture") folder ---
    demo_folder = Path(GESTURE_FOLDER).parent / args.gesture
    if not demo_folder.exists():
         print("Demo folder \"{}\" does not exist. Exiting.".format(demo_folder))
         exit(1)

    # --- demo (or "gesture") file ---
    demo_file = demo_folder / DEMO_FILENAME
    if not demo_file.exists():
         print("Demo file \"{}\" does not exist. Exiting.".format(demo_file))
         exit(1)

    # --- read (and "play") demo (or "gesture") ---
    # (In our case, the demo file is a .npy (numpy) file (20250515_151012_takeoff.npy).)
    # (We read it using numpy (np.load) and then "play" (display) the demo (or a sequence of images) using OpenCV.)
    demo = np.load(demo_file)
    print("Demo (or \"gesture\") \"{}\" (shape: {}) loaded.".format(args.gesture, demo.shape))

    # --- "play" (display) demo (or "gesture") ---
    # (In our case, demo is a numpy array (for example, a sequence of images (or frames) (e.g. (N, H, W, 3) for RGB images).)
    # (We "play" (display) each frame (using cv2.imshow) and wait (using cv2.waitKey) for a key (for example, "q") to quit.)
    for i, frame in enumerate(demo):
         cv2.imshow("Demo (or \"gesture\") \"{}\" (frame {})".format(args.gesture, i), frame)
         if cv2.waitKey(1) & 0xFF == ord("q"):
             break

    cv2.destroyAllWindows() 