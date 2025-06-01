#!/usr/bin/env python3
"""Extract hand landmarks from JPG images using MediaPipe.

This script processes JPG images in a directory structure like:
    input_dir/
        0/
            IMG_*.JPG
        1/
            IMG_*.JPG
        ...
    And saves landmarks as .npy files in:
    output_dir/
        0/
            IMG_*.npy
        1/
            IMG_*.npy
        ...

Each .npy file contains a sequence of T frames (default 10) of hand landmarks,
where each frame is duplicated from the single detected frame to create a temporal
sequence. This is done to match the expected input format of the processing script.

Example:
    poetry run python scripts/extract_landmarks.py \
        --input-dir data/raw/asl_digits/Dataset \
        --output-dir data/raw/asl_digits/landmarks \
        --num-frames 10
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def extract_landmarks(image_path: Path, mp_hands: mp.solutions.hands.Hands) -> Optional[np.ndarray]:
    """Extract hand landmarks from an image.
    
    Args:
        image_path: Path to the image file
        mp_hands: MediaPipe Hands instance
        
    Returns:
        numpy array of shape (21, 3) containing hand landmarks, or None if no hand detected
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Could not read image {image_path}")
        return None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = mp_hands.process(image_rgb)
    
    # Extract landmarks if hand detected
    if results.multi_hand_landmarks:
        # Get landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Convert to numpy array (21 landmarks, each with x,y,z coordinates)
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        return landmarks_array
    
    logger.warning(f"No hand detected in {image_path}")
    return None

def create_temporal_sequence(landmarks: np.ndarray, num_frames: int) -> np.ndarray:
    """Create a temporal sequence by duplicating the landmarks frame.
    
    Args:
        landmarks: numpy array of shape (21, 3) containing hand landmarks
        num_frames: number of frames to create in the sequence
        
    Returns:
        numpy array of shape (num_frames, 21, 3) containing the duplicated landmarks
    """
    # Add small random noise to each frame to create variation
    noise = np.random.normal(0, 0.001, (num_frames, 21, 3))
    sequence = np.tile(landmarks, (num_frames, 1, 1)) + noise
    return sequence

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract hand landmarks from JPG images using MediaPipe")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing gesture subdirectories")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for landmark files")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to create in each sequence")
    args = parser.parse_args()
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each gesture directory
    for gesture_dir in sorted(args.input_dir.iterdir()):
        if not gesture_dir.is_dir():
            continue
            
        gesture_label = gesture_dir.name
        logger.info(f"Processing {gesture_label}...")
        
        # Create output subdirectory
        output_gesture_dir = args.output_dir / gesture_label
        output_gesture_dir.mkdir(exist_ok=True)
        
        # Process each image
        image_files = sorted(gesture_dir.glob("*.JPG"))
        for image_file in tqdm(image_files, desc=gesture_label):
            # Extract landmarks
            landmarks = extract_landmarks(image_file, mp_hands)
            if landmarks is not None:
                # Create temporal sequence
                sequence = create_temporal_sequence(landmarks, args.num_frames)
                
                # Save as .npy file
                output_file = output_gesture_dir / f"{image_file.stem}.npy"
                np.save(output_file, sequence)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 