#!/usr/bin/env python3
"""
Execute pre-programmed flight plans based on hand gesture recognition.
Usage:
    poetry run python scripts/run_flight_plans.py
"""

import time
import argparse
from pathlib import Path
import torch
import cflib.crtp
from cflib.crazyflie import Crazyflie
from mp_drone_control.inference.live_video import load_model, run_live_prediction

# --- Crazyflie drone control ---
DRONE_URI = "radio://0/80/2M/E7E7E7E7E7"

# --- Flight plan definitions ---
# Each plan is a list of (roll, pitch, yaw, thrust, duration) tuples
# duration is in seconds
FLIGHT_PLANS = {
    "hover": [
        (0, 0, 0, 40000, 0.1),  # Just hover in place with constant thrust
    ],
    "land": [
        (0, 0, 0, 0, 0.5),      # Emergency land
    ]
}

# --- drone connection ---
cflib.crtp.init_drivers()
cf = Crazyflie()
cf.open_link(DRONE_URI)
print("Connected to Crazyflie drone (URI: {})".format(DRONE_URI))

# --- execute flight plan ---
def execute_flight_plan(plan_name):
    if plan_name not in FLIGHT_PLANS:
        print(f"Flight plan '{plan_name}' not found.")
        return
    
    print(f"Executing flight plan: {plan_name}")
    plan = FLIGHT_PLANS[plan_name]
    
    try:
        # For hover, we want to keep sending the command
        if plan_name == "hover":
            roll, pitch, yaw, thrust, _ = plan[0]
            while True:  # Keep hovering until interrupted
                cf.commander.send_setpoint(roll, pitch, yaw, thrust)
                time.sleep(0.1)  # Send command every 100ms
        else:
            # For other commands (like land), execute once
            for roll, pitch, yaw, thrust, duration in plan:
                print(f"Command: roll={roll}, pitch={pitch}, yaw={yaw}, thrust={thrust}, duration={duration}s")
                cf.commander.send_setpoint(roll, pitch, yaw, thrust)
                time.sleep(duration)
    except KeyboardInterrupt:
        print("\nStopping hover...")
        cf.commander.send_setpoint(0, 0, 0, 0)  # Emergency stop
    except Exception as e:
        print(f"Error during flight plan execution: {e}")
        cf.commander.send_setpoint(0, 0, 0, 0)  # Emergency stop
    finally:
        if plan_name != "hover":  # Don't send hover command after landing
            cf.commander.send_setpoint(0, 0, 0, 30000)

# --- gesture callback ---
def gesture_callback(label):
    if label in FLIGHT_PLANS:
        execute_flight_plan(label)
    else:
        print(f"Gesture '{label}' not mapped to a flight plan.")

# --- main ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run flight plans based on hand gesture recognition.")
    ap.add_argument("--model", type=str, default="small", choices=["small", "large"], help="Model type (small or large)")
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/landmark_mlp_small_best.pth"), help="Path to model checkpoint")
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
    print("Available commands: hover, land")
    print("Press 'q' to quit.")

    # --- run live prediction with flight plan callback ---
    def run_live_prediction_with_plans(model, class_names, device, callback):
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
            cv2.imshow("Hover Control", frame)
            
            # Only trigger hover or land commands
            if label in ["hover", "land"]:
                callback(label)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        mp_hands.close()

    try:
        run_live_prediction_with_plans(model, class_names, args.device, gesture_callback)
    finally:
        # --- cleanup ---
        cf.commander.send_setpoint(0, 0, 0, 0)  # emergency stop
        cf.close_link()
        print("Crazyflie drone link closed.") 