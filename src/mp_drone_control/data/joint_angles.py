"""
Joint angle extraction from MediaPipe hand landmarks.
Converts 3D hand landmarks to biomechanically meaningful joint angles.
"""

import numpy as np
import math
from typing import Tuple, List
from pathlib import Path

# MediaPipe hand landmark indices
# See: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
LANDMARK_NAMES = [
    'WRIST',                    # 0
    'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',          # 1-4
    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',  # 5-8
    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',  # 9-12
    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',     # 13-16
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'         # 17-20
]

# Define finger joint chains
FINGER_CHAINS = {
    'thumb': [0, 1, 2, 3, 4],      # wrist -> cmc -> mcp -> ip -> tip
    'index': [0, 5, 6, 7, 8],      # wrist -> mcp -> pip -> dip -> tip
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'pinky': [0, 17, 18, 19, 20]
}

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians."""
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    
    # Calculate angle using dot product
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return math.acos(cos_angle)

def calculate_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate joint angle at point p2, formed by vectors p1->p2 and p2->p3.
    Returns angle in radians.
    """
    # Vectors from joint to adjacent points
    v1 = p1 - p2  # Vector from joint to previous point
    v2 = p3 - p2  # Vector from joint to next point
    
    return vector_angle(v1, v2)

def extract_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract joint angles from MediaPipe hand landmarks.
    
    Args:
        landmarks: np.ndarray of shape (21, 3) - hand landmarks
        
    Returns:
        np.ndarray of joint angles - much smaller dimension than 63D
    """
    if landmarks.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")
    
    angles = []
    
    # For each finger, calculate joint angles
    for finger_name, chain in FINGER_CHAINS.items():
        # Skip first and last points (no joint angle at endpoints)
        for i in range(1, len(chain) - 1):
            p1 = landmarks[chain[i-1]]  # Previous point
            p2 = landmarks[chain[i]]    # Joint point
            p3 = landmarks[chain[i+1]]  # Next point
            
            angle = calculate_joint_angle(p1, p2, p3)
            angles.append(angle)
    
    # Add finger spread angles (angles between adjacent fingers)
    finger_tips = [4, 8, 12, 16, 20]  # Tip landmarks
    finger_bases = [2, 5, 9, 13, 17]  # Base landmarks
    
    # Calculate spread angles between adjacent fingers
    for i in range(len(finger_bases) - 1):
        # Vector from wrist to finger base
        v1 = landmarks[finger_bases[i]] - landmarks[0]    # Current finger
        v2 = landmarks[finger_bases[i+1]] - landmarks[0]  # Next finger
        
        spread_angle = vector_angle(v1, v2)
        angles.append(spread_angle)
    
    # Add thumb abduction angle (thumb vs index finger)
    thumb_vec = landmarks[4] - landmarks[0]  # Wrist to thumb tip
    index_vec = landmarks[8] - landmarks[0]  # Wrist to index tip
    thumb_abduction = vector_angle(thumb_vec, index_vec)
    angles.append(thumb_abduction)
    
    return np.array(angles, dtype=np.float32)

def get_joint_angle_feature_names() -> List[str]:
    """Get descriptive names for each joint angle feature."""
    names = []
    
    # Joint angles for each finger
    for finger_name, chain in FINGER_CHAINS.items():
        for i in range(1, len(chain) - 1):
            joint_name = LANDMARK_NAMES[chain[i]].replace('_', ' ').title()
            names.append(f"{finger_name.title()} {joint_name} Angle")
    
    # Finger spread angles
    finger_names = ['Thumb-Index', 'Index-Middle', 'Middle-Ring', 'Ring-Pinky']
    for name in finger_names:
        names.append(f"{name} Spread")
    
    # Thumb abduction
    names.append("Thumb Abduction")
    
    return names

def convert_landmarks_to_joint_angles(landmarks_path: Path, output_path: Path):
    """
    Convert a landmarks .npy file to joint angles representation.
    
    Args:
        landmarks_path: Path to input landmarks file (N, 21, 3) or (N, 63)
        output_path: Path to save joint angles (N, num_angles)
    """
    # Load landmarks
    landmarks = np.load(landmarks_path)
    
    # Handle both (N, 21, 3) and (N, 63) formats
    if landmarks.ndim == 2 and landmarks.shape[1] == 63:
        landmarks = landmarks.reshape(-1, 21, 3)
    elif landmarks.ndim == 3 and landmarks.shape[1:] == (21, 3):
        pass  # Already correct format
    else:
        raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")
    
    # Extract joint angles for each frame
    joint_angles_list = []
    for frame_landmarks in landmarks:
        angles = extract_joint_angles(frame_landmarks)
        joint_angles_list.append(angles)
    
    joint_angles = np.array(joint_angles_list)
    
    # Save joint angles
    np.save(output_path, joint_angles)
    
    print(f"Converted {landmarks.shape[0]} frames from {landmarks.shape[1:]} landmarks "
          f"to {joint_angles.shape[1]} joint angles")
    print(f"Feature names: {get_joint_angle_feature_names()}")
    
    return joint_angles

def process_dataset_to_joint_angles(data_dir: Path, output_dir: Path):
    """
    Convert an entire dataset from landmarks to joint angles.
    
    Args:
        data_dir: Directory containing train_X.npy, val_X.npy, test_X.npy
        output_dir: Directory to save joint angle versions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        landmarks_file = data_dir / f"{split}_X.npy"
        labels_file = data_dir / f"{split}_y.npy"
        
        if landmarks_file.exists():
            # Convert landmarks to joint angles
            joint_angles_file = output_dir / f"{split}_X.npy"
            convert_landmarks_to_joint_angles(landmarks_file, joint_angles_file)
            
            # Copy labels unchanged
            if labels_file.exists():
                labels_output = output_dir / f"{split}_y.npy"
                labels = np.load(labels_file)
                np.save(labels_output, labels)
                print(f"Copied {len(labels)} labels for {split} split")
        else:
            print(f"Warning: {landmarks_file} not found, skipping {split} split")
    
    # Save feature information
    feature_names = get_joint_angle_feature_names()
    feature_info = {
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'description': 'Joint angles extracted from MediaPipe hand landmarks'
    }
    
    import json
    with open(output_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Dataset conversion complete! Joint angle features: {len(feature_names)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert landmark dataset to joint angles")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Input directory with landmark data")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for joint angle data")
    
    args = parser.parse_args()
    
    process_dataset_to_joint_angles(args.input_dir, args.output_dir) 