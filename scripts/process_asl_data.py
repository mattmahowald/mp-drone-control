#!/usr/bin/env python3
"""
Script to process ASL dataset:
1. Extract landmarks from images using MediaPipe
2. Split data into train/val/test sets
3. Save processed splits
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from mp_drone_control.utils.logging_config import setup_logging
from mp_drone_control.data.preprocess import (
    PreprocessingConfig,
    split_data,
    save_splits,
)

# Set up logging for the entire project
logger = setup_logging()


def extract_keypoints_from_image(image_path: Path) -> np.ndarray | None:
    """
    Run MediaPipe on an image and return 21 (x, y, z) landmarks, or None.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Failed to read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands_config.process(img_rgb)

    if not results.multi_hand_landmarks:
        logger.debug(f"No hand landmarks detected in: {image_path}")
        return None

    landmarks = results.multi_hand_landmarks[0]
    keypoints = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    )
    return keypoints  # shape (21, 3)


def extract_landmarks(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk through ASL digit dataset folders and extract landmarks from each image.

    Returns:
        Tuple of (landmarks, labels) arrays
    """
    X = []
    y = []
    total_processed = 0
    total_skipped = 0

    # Get all label directories first
    label_dirs = [d for d in sorted((dataset_dir / "Dataset").iterdir()) if d.is_dir()]

    logger.info(f"Found {len(label_dirs)} classes to process")

    # Main progress bar for classes
    with tqdm(label_dirs, desc="Processing classes", position=0) as class_pbar:
        for label_dir in class_pbar:
            class_label = int(label_dir.name)
            image_paths = list(label_dir.glob("*.JPG"))

            class_pbar.set_description(f"Processing class '{class_label}'")
            logger.info(
                f"Processing class '{class_label}' with {len(image_paths)} images"
            )

            # Progress bar for images within each class
            with tqdm(
                image_paths, desc="Processing images", position=1, leave=False
            ) as img_pbar:
                for image_path in img_pbar:
                    keypoints = extract_keypoints_from_image(image_path)
                    if keypoints is not None and keypoints.shape == (21, 3):
                        X.append(keypoints)
                        y.append(class_label)
                        total_processed += 1
                    else:
                        total_skipped += 1

                    img_pbar.set_postfix(
                        {"processed": total_processed, "skipped": total_skipped}
                    )

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    logger.info(f"Extraction complete!")
    logger.info(f"Total samples processed: {len(X)}")
    logger.info(f"Total samples skipped: {total_skipped}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def process_pipeline(
    raw_data_dir: Path,
    processed_data_dir: Path,
    config: PreprocessingConfig,
) -> None:
    """
    Complete processing pipeline:
    1. Extract landmarks from images
    2. Split into train/val/test sets
    3. Save processed splits

    Args:
        raw_data_dir: Directory containing raw ASL dataset
        processed_data_dir: Directory to save processed data
        config: Preprocessing configuration
    """
    logger.info("Starting ASL dataset processing pipeline")
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")

    # Extract landmarks
    landmarks, labels = extract_landmarks(raw_data_dir)

    # Split data
    splits = split_data(landmarks, labels, config)

    # Save splits
    save_splits(splits, processed_data_dir)

    logger.info("Processing pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ASL dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/asl_digits"),
        help="Directory containing raw ASL dataset",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/asl_digits"),
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_hands_config = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )

    # Create config
    config = PreprocessingConfig(
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    try:
        process_pipeline(
            raw_data_dir=args.raw_dir,
            processed_data_dir=args.processed_dir,
            config=config,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise
