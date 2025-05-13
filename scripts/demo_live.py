from pathlib import Path
from mp_drone_control.inference.live_video import load_model, run_live_prediction

if __name__ == "__main__":
    model_path = Path("models/landmark_classifier.pt")
    class_names = [str(i) for i in range(10)]  # digits 0–9
    model = load_model(model_path)
    run_live_prediction(model, class_names)
