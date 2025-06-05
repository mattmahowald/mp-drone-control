#!/usr/bin/env python3
"""Evaluate our top 3 best models individually."""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import LSTMGesture, get_seq_dataloaders
from mp_drone_control.models.landmark_mlp import LandmarkMLPLarge
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

def evaluate_mlp():
    """Evaluate MLP Large model."""
    data_dir = Path("data/processed/custom")
    test_loader = get_dataloader(data_dir, split="test", batch_size=64, normalize=False)
    
    model = LandmarkMLPLarge(n_classes=11)
    model.load_state_dict(torch.load("checkpoints/landmark_mlp_large_best.pth", map_location='cpu'))
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.numpy())
            targets.extend(batch_y.numpy())
    
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    
    logger.info(f"🏆 MLP Large Results:")
    logger.info(f"   Accuracy: {acc:.4f}")
    logger.info(f"   Macro F1: {f1:.4f}")
    
    return acc, f1

def evaluate_lstm():
    """Evaluate LSTM model."""
    data_dir = Path("data/processed/custom")
    _, _, test_loader = get_seq_dataloaders(data_dir, seq_len=16, batch_size=64)
    
    # Auto-detect input dimension
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch[0].shape[-1]
    
    model = LSTMGesture(input_dim=input_dim, num_classes=11)
    model.load_state_dict(torch.load("checkpoints/lstm_gesture_model.pth", map_location='cpu'))
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.numpy())
            targets.extend(batch_y.numpy())
    
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='macro')
    
    logger.info(f"🥇 LSTM Results:")
    logger.info(f"   Accuracy: {acc:.4f}")
    logger.info(f"   Macro F1: {f1:.4f}")
    
    return acc, f1

def main():
    logger.info("Evaluating Best Models...")
    
    try:
        mlp_acc, mlp_f1 = evaluate_mlp()
    except Exception as e:
        logger.error(f"MLP evaluation failed: {e}")
        mlp_acc, mlp_f1 = 0, 0
    
    try:
        lstm_acc, lstm_f1 = evaluate_lstm()
    except Exception as e:
        logger.error(f"LSTM evaluation failed: {e}")
        lstm_acc, lstm_f1 = 0, 0
    
    logger.info("\n" + "="*50)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("="*50)
    logger.info(f"MLP Large:  {mlp_acc:.4f} acc, {mlp_f1:.4f} F1")
    logger.info(f"LSTM:       {lstm_acc:.4f} acc, {lstm_f1:.4f} F1")
    
    if lstm_acc > mlp_acc:
        logger.info("🏆 WINNER: LSTM (Best Overall)")
    else:
        logger.info("🏆 WINNER: MLP Large (Most Efficient)")

if __name__ == "__main__":
    main() 