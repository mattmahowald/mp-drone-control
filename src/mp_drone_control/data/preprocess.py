from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size must be less than 1")


def split_data(
    landmarks: np.ndarray,
    labels: np.ndarray,
    config: PreprocessingConfig,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train, validation, and test sets using stratified sampling.

    Args:
        landmarks: Array of shape (N, ...)
        labels: Array of shape (N,)
        config: Preprocessing configuration

    Returns:
        Dictionary mapping split names to (X, y) tuples
    """
    config.validate()

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        landmarks,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    # Second split: separate validation set from remaining data
    val_ratio = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=config.random_state,
        stratify=y_temp,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    return splits


def save_splits(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: Path
) -> None:
    """
    Save the split datasets to NumPy files.

    Args:
        splits: Dictionary mapping split names to (X, y) tuples
        output_dir: Directory to save processed data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (X, y) in splits.items():
        np.save(output_dir / f"{split_name}_X.npy", X)
        np.save(output_dir / f"{split_name}_y.npy", y)
