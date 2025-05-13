from pathlib import Path
from mp_drone_control.models.trainer import train

if __name__ == "__main__":
    train(
        data_dir=Path("data/processed/asl_digits"),
        num_epochs=20,
        batch_size=64,
        lr=1e-3,
        save_path=Path("models/landmark_classifier.pt"),
    )
