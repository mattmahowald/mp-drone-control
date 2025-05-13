import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_hands_config = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)


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


def process_asl_dataset(dataset_dir: Path, output_dir: Path):
    """
    Walk through ASL digit dataset folders and extract landmarks from each image.
    Save X.npy and y.npy into output_dir.
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

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_X.npy", X)
    np.save(output_dir / "train_y.npy", y)

    logger.info(f"Processing complete!")
    logger.info(f"Total samples processed: {len(X)}")
    logger.info(f"Total samples skipped: {total_skipped}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")


if __name__ == "__main__":
    dataset_path = Path("data/raw/asl_digits")
    output_path = Path("data/processed")

    logger.info(f"Starting ASL dataset processing")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output path: {output_path}")

    process_asl_dataset(dataset_path, output_path)
