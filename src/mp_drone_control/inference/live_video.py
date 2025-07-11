import cv2
import numpy as np
import torch
from pathlib import Path
from mp_drone_control.models.mobilenet import (
    LandmarkClassifier,
    LargeLandmarkClassifier,
)
import mediapipe as mp
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()


# Load model
def load_model(checkpoint_path: Path, model_type: str = "small", device: str = "cpu", num_classes: int = 10):
    if model_type == "large":
        model = LargeLandmarkClassifier(input_dim=63, num_classes=num_classes)
    else:
        model = LandmarkClassifier(input_dim=63, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded {model_type} model from {checkpoint_path} and set to {device}")
    return model


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    wrist = landmarks[0:1]
    translated = landmarks - wrist
    max_dist = np.linalg.norm(translated, axis=-1).max()
    return translated / max_dist


def run_live_prediction(model, class_names: list[str], device: str = "cpu"):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
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
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
            )

            if keypoints.shape == (21, 3):
                normed = normalize_landmarks(keypoints)
                input_tensor = (
                    torch.tensor(normed.flatten(), dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    logits = model(input_tensor)
                    pred = logits.argmax(dim=1).item()
                    label = class_names[pred]

            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            f"Gesture: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Live Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()


def run_live_comparison(
    small_model, large_model, class_names: list[str], device: str = "cpu"
):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)

        label_small = "No hand"
        label_large = "No hand"
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            keypoints = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
            )

            if keypoints.shape == (21, 3):
                normed = normalize_landmarks(keypoints)
                input_tensor = (
                    torch.tensor(normed.flatten(), dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    logits_small = small_model(input_tensor)
                    pred_small = logits_small.argmax(dim=1).item()
                    label_small = class_names[pred_small]

                    logits_large = large_model(input_tensor)
                    pred_large = logits_large.argmax(dim=1).item()
                    label_large = class_names[pred_large]

            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

        # Draw both predictions on the frame
        cv2.putText(
            frame,
            f"Small: {label_small}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Large: {label_large}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Live Gesture Comparison (Small vs Large)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
