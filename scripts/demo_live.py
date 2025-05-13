from pathlib import Path
from mp_drone_control.inference.live_video import load_model, run_live_comparison

if __name__ == "__main__":
    small_model_path = Path("models/small_model.pth")
    large_model_path = Path("models/large_model.pth")
    class_names = [str(i) for i in range(10)]  # digits 0–9
    small_model = load_model(small_model_path, model_type="small")
    large_model = load_model(large_model_path, model_type="large")
    run_live_comparison(small_model, large_model, class_names)
