import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class LandmarkSequenceDataset(Dataset):
    """
    Dataset for temporal sequences that maintains temporal relationships.
    """
    def __init__(self, data_dir, split, seq_len=16, stride=8):
        # Load the frame-level data
        X_path = Path(data_dir) / f"{split}_X.npy"
        y_path = Path(data_dir) / f"{split}_y.npy"
        
        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"Data files not found: {X_path}, {y_path}")
        
        landmarks = np.load(X_path)  # (N, 63)
        labels = np.load(y_path)     # (N,) - string labels
        
        # Convert string labels to integers
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels_int = np.array([self.label2idx[label] for label in labels])
        
        # Create sequences using sliding window over ALL data
        # Use the LAST frame's label as the sequence label
        self.samples = []
        for i in range(0, len(landmarks) - seq_len + 1, stride):
            sequence = landmarks[i:i + seq_len]  # (seq_len, 63)
            # Use the label of the last frame in the sequence
            sequence_label = labels_int[i + seq_len - 1]
            self.samples.append((sequence.astype(np.float32), sequence_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

class GRUGesture(nn.Module):
    def __init__(self, input_dim=63, hidden=128, num_layers=2, num_classes=21):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, num_classes))
    def forward(self, x):          # x: (B, L, 63)
        out, _ = self.rnn(x)
        feat = out[:, -1]          # last time-step
        return self.classifier(feat)

class TemporalConvGesture(nn.Module):
    def __init__(self, input_dim=63, num_classes=21, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hid, 3, padding=1, groups=1),
            nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 3, padding=1, groups=hid), # depthwise
            nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Conv1d(hid, hid, 3, padding=1),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(hid, num_classes)
    def forward(self, x):          # (B, L, 63)
        x = x.transpose(1,2)       # -> (B, 63, L)
        feat = self.net(x).squeeze(-1)
        return self.fc(feat)

def get_seq_dataloaders(data_dir, seq_len=16, batch_size=32, augment=False):
    """Get sequence dataloaders for train, val, test splits."""
    train_dataset = LandmarkSequenceDataset(data_dir, "train", seq_len=seq_len)
    val_dataset = LandmarkSequenceDataset(data_dir, "val", seq_len=seq_len)
    test_dataset = LandmarkSequenceDataset(data_dir, "test", seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
