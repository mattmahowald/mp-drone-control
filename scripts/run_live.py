#!/usr/bin/env python3
"""
Run live hand gesture recognition using the trained small model.

Usage:
    poetry run python scripts/run_live.py
"""

from pathlib import Path
import torch
from mp_drone_control.inference.live_video import load_model, run_live_prediction

def main():
    # Load gesture labels
    with open("gesture_labels.txt") as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # Load the small model
    model_path = Path("checkpoints/landmark_mlp_small_best.pth")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, model_type="small", device=device, num_classes=11)
    
    # Run live prediction
    print(f"Running inference on {device}")
    print("Available gestures:", ", ".join(class_names))
    print("Press 'q' to quit")
    run_live_prediction(model, class_names, device=device)

if __name__ == "__main__":
    main() 