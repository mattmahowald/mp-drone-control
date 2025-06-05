from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .augment import geom_aug, jitter
import math

class HandLandmarkDataset(Dataset):
    """
    Loads hand features from a NumPy file or directory.
    Handles both landmarks (N, 63) and joint angles (N, ~20).
    """

    def __init__(self, X_path: Path, y_path: Path, normalize=False, augment=True):
        self.landmarks = np.load(X_path)        # (N, feature_dim)
        self.labels    = np.load(y_path)        # array of strings
        self.augment   = augment
        
        # Detect if this is landmark data or joint angle data
        self.feature_dim = self.landmarks.shape[1]
        self.is_landmarks = self.feature_dim == 63
        self.is_joint_angles = self.feature_dim < 63  # Joint angles have fewer features

        # 🔑 build mapping once
        uniques            = sorted(set(self.labels))
        self.label2idx     = {lbl: i for i, lbl in enumerate(uniques)}
        self.idx2label     = uniques            # optional, handy for debug
        self.labels_int    = np.array([self.label2idx[l] for l in self.labels],
                                      dtype=np.int64)

        if normalize and self.is_landmarks:
            self.landmarks = self._normalize_landmarks(self.landmarks)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to have wrist at origin and unit scale."""
        # Only works for landmark data, not joint angles
        if not self.is_landmarks:
            return landmarks
            
        # Make a copy to avoid modifying the original data
        landmarks = landmarks.copy()

        # Reshape to (N, 21, 3) for normalization
        landmarks_3d = landmarks.reshape(-1, 21, 3)

        # Set wrist as origin
        wrist = landmarks_3d[:, 0:1, :]  # shape (N, 1, 3)
        landmarks_3d = landmarks_3d - wrist

        # Scale to unit distance
        max_dist = np.linalg.norm(landmarks_3d, axis=-1).max(axis=1, keepdims=True)
        landmarks_3d = landmarks_3d / (
            max_dist[..., np.newaxis] + 1e-8
        )  # Add small epsilon to avoid division by zero

        return landmarks_3d.reshape(-1, 63)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        y = int(self.labels_int[idx])           # ✅ always an int now
        
        if self.augment and self.is_landmarks:
            # Only apply landmark-specific augmentation to landmark data
            x = x.reshape(-1).contiguous()          # (63,)
            # Gaussian noise
            x += torch.randn_like(x) * 0.01          # ~1 % jitter
            # In-plane rotation
            theta = torch.randn(1).item() * 0.15     # ±8.5°
            R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                            [math.sin(theta),  math.cos(theta), 0],
                            [0,                0,               1]])
            x = (R @ x.view(21, 3).T).T.flatten()
        elif self.augment and self.is_joint_angles:
            # Simple augmentation for joint angles - just add small noise
            x += torch.randn_like(x) * 0.01
        else:
            x = x.reshape(-1).contiguous()

        return x, y


def get_dataloader(
    data_dir: Path,
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    normalize: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """
    Utility to load a torch DataLoader for training or evaluation.
    Assumes data_dir contains `{split}_X.npy` and `{split}_y.npy`.

    Args:
        data_dir: Directory containing the data files
        split: Which split to load ('train', 'val', or 'test')
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        normalize: Whether to normalize the landmarks
        num_workers: Number of worker processes for data loading

    Returns:
        DataLoader instance
    """
    X_path = data_dir / f"{split}_X.npy"
    y_path = data_dir / f"{split}_y.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Data files not found in {data_dir}. " f"Expected {X_path} and {y_path}"
        )

    dataset = HandLandmarkDataset(X_path, y_path, normalize=normalize)

    # Only enable pin_memory for CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,  # Only enable for CUDA devices
    )
