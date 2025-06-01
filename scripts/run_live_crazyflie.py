#!/usr/bin/env python3
"""
Integrate live hand gesture recognition (using the small model) with Crazyflie drone control.
Usage:
    poetry run python scripts/run_live_crazyflie.py
"""

import time
import argparse
from pathlib import Path
import torch
import cflib.crtp
from cflib.crazyflie import Crazyflie
from mp_drone_control.inference.live_video import load_model, run_live_prediction

# --- Crazyflie drone control ---
# (Replace the URI with your Crazyflie's address if needed)
DRONE_URI = "radio://0/80/2M/E7E7E7E7E7"

# --- drone command mapping ---
# (Adjust thrust values as needed; these are example values)
DRONE_COMMANDS = {
    "takeoff": (0, 0, 0, 40000),  # (roll, pitch, yaw, thrust) for takeoff
    "land": (0, 0, 0, 0),         # land (zero thrust)
    "hover": (0, 0, 0, 30000),    # hover (steady thrust)
    "forward": (0, 0.2, 0, 30000), # move forward (pitch forward)
    "backward": (0, -0.2, 0, 30000), # move backward (pitch backward)
    "left": (-0.2, 0, 0, 30000),  # move left (roll left)
    "right": (0.2, 0, 0, 30000),  # move right (roll right)
    "rotate_left": (0, 0, -0.2, 30000), # rotate left (yaw left)
    "rotate_right": (0, 0, 0.2, 30000), # rotate right (yaw right)
    "stop": (0, 0, 0, 0)          # emergency stop (zero thrust)
}

# --- drone connection ---
# (Connect to the Crazyflie drone using cflib)
cflib.crtp.init_drivers()
cf = Crazyflie()
cf.open_link(DRONE_URI)
print("Connected to Crazyflie drone (URI: {})".format(DRONE_URI))

# --- drone command callback ---
# (This callback is called from the live prediction loop when a gesture is recognized)
def drone_command_callback(label):
    if label in DRONE_COMMANDS:
        roll, pitch, yaw, thrust = DRONE_COMMANDS[label]
        cf.commander.send_setpoint(roll, pitch, yaw, thrust)
        print("Drone command sent: {} (roll={}, pitch={}, yaw={}, thrust={})".format(label, roll, pitch, yaw, thrust))
    else:
        print("Gesture \"{}\" not mapped to a drone command.".format(label))

# --- main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run live hand gesture recognition and Crazyflie drone control.")
    ap.add_argument("--model", type=str, default="small", choices=["small", "large"], help="Model type (small or large)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/landmark_mlp_small_best.pth"), help="Path to model checkpoint (default: checkpoints/landmark_mlp_small_best.pth)")
    ap.add_argument("--device", type=str, default=None, help="Device (mps, cuda, or cpu). If None, auto-detect.")
    args = ap.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # --- load gesture labels ---
    with open("gesture_labels.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]

    # --- load model ---
    model = load_model(args.checkpoint, model_type=args.model, device=args.device, num_classes=11)
    print("Running inference on {} (model: {})".format(args.device, args.model))
    print("Available gestures (drone commands):", ", ".join(class_names))
    print("Press 'q' to quit.")

    # --- run live prediction (with drone command callback) ---
    # (The live_video module's run_live_prediction is modified to call drone_command_callback(label) when a gesture is recognized.)
    # (Note: In a real integration, you'd modify run_live_prediction in live_video.py or pass a callback.)
    # (For simplicity, we simulate it here by printing the command.)
    def run_live_prediction_with_drone(model, class_names, device, callback):
        import cv2
        import numpy as np
        import mediapipe as mp
        mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam. Please check your camera permissions or try a different camera index.")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)
            label = "No hand"
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
                if keypoints.shape == (21, 3):
                    normed = (keypoints - keypoints[0:1]) / (np.linalg.norm(keypoints - keypoints[0:1], axis=-1).max() + 1e-6)
                    input_tensor = torch.tensor(normed.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(input_tensor)
                        pred = logits.argmax(dim=1).item()
                        label = class_names[pred]
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Gesture: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Live Gesture Recognition (Crazyflie Drone Control)", frame)
            if label != "No hand":
                callback(label)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()

    try:
        run_live_prediction_with_drone(model, class_names, args.device, drone_command_callback)
    finally:
        # --- cleanup ---
        cf.commander.send_setpoint(0, 0, 0, 0)  # emergency stop (zero thrust)
        cf.close_link()
        print("Crazyflie drone link closed.") 