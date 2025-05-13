from pathlib import Path
from mp_drone_control.models.trainer import train_and_save_all_models
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

if __name__ == "__main__":
    logger.info("Starting training for both small and large models...")
    train_and_save_all_models(
        data_dir=Path("data/processed/asl_digits"),
        num_epochs=10,
        batch_size=64,
        lr=1e-3,
        save_dir=Path("models/"),
    )
    logger.info("Training for both models complete!")
