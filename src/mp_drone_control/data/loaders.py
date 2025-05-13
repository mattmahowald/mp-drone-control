from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HandLandmarkDataset(Dataset):
    """
    Loads 21-point MediaPipe hand landmarks from a NumPy file or directory.
    Assumes shape (N, 21, 3) for landmarks and (N,) for integer labels.
    """

    def __init__(self, data_path: Path, label_path: Path, normalize: bool = True):
        self.landmarks = np.load(data_path)  # shape: (N, 21, 3)
        self.labels = np.load(label_path)  # shape: (N,)
        self.normalize = normalize

        if normalize:
            self.landmarks = self._normalize_landmarks(self.landmarks)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        # Normalize so wrist is origin and scale is unit distance
        wrist = landmarks[:, 0:1, :]  # shape (N, 1, 3)
        translated = landmarks - wrist
        max_dist = np.linalg.norm(translated, axis=-1).max(axis=1, keepdims=True)
        return translated / max_dist[..., np.newaxis]

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x = torch.tensor(self.landmarks[idx], dtype=torch.float32)  # (21, 3)
        y = int(self.labels[idx])
        return x.view(-1), y  # Flatten to (63,) vector


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
    """
    X_path = data_dir / f"{split}_X.npy"
    y_path = data_dir / f"{split}_y.npy"

    dataset = HandLandmarkDataset(X_path, y_path, normalize=normalize)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
