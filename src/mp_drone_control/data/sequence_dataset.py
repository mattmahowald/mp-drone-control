import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import math

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
        
        landmarks = np.load(X_path)  # (N, feature_dim) - could be 63 or 20
        labels = np.load(y_path)     # (N,) - string labels
        
        # Store input dimension for model creation
        self.input_dim = landmarks.shape[1]
        
        # Convert string labels to integers
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels_int = np.array([self.label2idx[label] for label in labels])
        
        # Create sequences using sliding window over ALL data
        # Use the LAST frame's label as the sequence label
        self.samples = []
        for i in range(0, len(landmarks) - seq_len + 1, stride):
            sequence = landmarks[i:i + seq_len]  # (seq_len, feature_dim)
            # Use the label of the last frame in the sequence
            sequence_label = labels_int[i + seq_len - 1]
            self.samples.append((sequence.astype(np.float32), sequence_label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)

class GRUGesture(nn.Module):
    def __init__(self, input_dim=None, hidden=128, num_layers=2, num_classes=21):
        super().__init__()
        # Auto-detect input dimension if not provided
        if input_dim is None:
            input_dim = 63  # Default for backward compatibility
        
        self.input_dim = input_dim
        self.rnn = nn.GRU(input_dim, hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, num_classes))
    
    def forward(self, x):          # x: (B, L, input_dim)
        out, _ = self.rnn(x)
        feat = out[:, -1]          # last time-step
        return self.classifier(feat)

class TemporalConvGesture(nn.Module):
    def __init__(self, input_dim=None, num_classes=21, hid=64):
        super().__init__()
        # Auto-detect input dimension if not provided
        if input_dim is None:
            input_dim = 63  # Default for backward compatibility
            
        self.input_dim = input_dim
        self.net = nn.Sequential(
            # Simpler architecture
            nn.Conv1d(input_dim, hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid), 
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(hid, hid*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid*2), 
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(hid*2, hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid), 
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(hid, num_classes)
        
    def forward(self, x):          # (B, L, input_dim)
        x = x.transpose(1,2)       # -> (B, input_dim, L)
        feat = self.net(x).squeeze(-1)
        return self.fc(feat)

class LSTMGesture(nn.Module):
    def __init__(self, input_dim=None, hidden=128, num_layers=2, num_classes=21):
        super().__init__()
        if input_dim is None:
            input_dim = 63
        
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, num_classes))
    
    def forward(self, x):          # x: (B, L, input_dim)
        out, _ = self.lstm(x)
        feat = out[:, -1]          # last time-step
        return self.classifier(feat)

class TransformerGesture(nn.Module):
    def __init__(self, input_dim=None, d_model=128, nhead=8, num_layers=4, num_classes=21, max_seq_len=32):
        super().__init__()
        if input_dim is None:
            input_dim = 63
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        return pe
    
    def forward(self, x):  # x: (B, L, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (B, L, d_model)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)  # (B, L, d_model)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification
        return self.classifier(x)

def get_seq_dataloaders(data_dir, seq_len=16, batch_size=32, augment=False):
    """Get sequence dataloaders for train, val, test splits."""
    train_dataset = LandmarkSequenceDataset(data_dir, "train", seq_len=seq_len)
    val_dataset = LandmarkSequenceDataset(data_dir, "val", seq_len=seq_len)
    test_dataset = LandmarkSequenceDataset(data_dir, "test", seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
