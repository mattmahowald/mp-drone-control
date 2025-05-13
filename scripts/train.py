from pathlib import Path
from mp_drone_control.models.trainer import train
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

if __name__ == "__main__":
    logger.info("Starting model training...")
    train(
        data_dir=Path("data/processed/asl_digits"),
        num_epochs=20,
        batch_size=64,
        lr=1e-3,
        save_path=Path("models/landmark_classifier.pt"),
    )
    logger.info("Training complete!")
