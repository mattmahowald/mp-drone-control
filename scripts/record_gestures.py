#!/usr/bin/env python3
"""
Record hand-gesture landmarks to .npy files.

Example:
  poetry run python scripts/record_gestures.py --label takeoff --seconds 30
"""
import argparse, time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

def main(args):
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frame_count = int(args.seconds * args.fps)
    pbar = tqdm(total=frame_count, desc=f"Recording '{args.label}'")

    landmarks_list = []

    while pbar.n < frame_count:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            points = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            landmarks_list.append(points)

        if args.display:
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # Esc to abort
                break
        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    if landmarks_list:
        ts = time.strftime("%Y%m%d_%H%M%S")
        outfile = out_dir / f"{ts}_{args.label}.npy"
        np.save(outfile, np.stack(landmarks_list))
        print(f"Saved {len(landmarks_list)} frames → {outfile}")
    else:
        print("No landmarks captured—nothing saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="gesture class name")
    ap.add_argument("--seconds", type=float, default=30, help="recording length")
    ap.add_argument("--fps", type=int, default=30, help="target frames")
    ap.add_argument("--out-dir", default="data/raw", help="root output directory")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--display", action="store_true", help="show live feed")
    args = ap.parse_args()
    main(args)
