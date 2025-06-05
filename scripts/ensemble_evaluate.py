#!/usr/bin/env python3
"""Fixed ensemble evaluation that aligns MLP and temporal model predictions."""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, LSTMGesture, get_seq_dataloaders
from mp_drone_control.models.landmark_mlp import LandmarkMLPLarge
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

def get_mlp_sequence_predictions(checkpoint_path, data_dir, n_classes, seq_len=16, stride=8):
    """
    Get MLP predictions aligned to sequence structure.
    Groups frame-level MLP predictions into sequences to match temporal models.
    """
    # Load test data
    test_loader = get_dataloader(Path(data_dir), split="test", batch_size=1, normalize=False)
    
    # Load MLP model
    model = LandmarkMLPLarge(n_classes=n_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    # Get all frame predictions
    all_frame_preds = []
    all_frame_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            all_frame_preds.append(probs.numpy()[0])  # Remove batch dimension
            all_frame_targets.append(batch_y.numpy()[0])
    
    all_frame_preds = np.array(all_frame_preds)
    all_frame_targets = np.array(all_frame_targets)
    
    # Group frames into sequences (same logic as LandmarkSequenceDataset)
    sequence_preds = []
    sequence_targets = []
    
    # Group by class to maintain class consistency within sequences
    unique_classes = np.unique(all_frame_targets)
    
    for cls in unique_classes:
        cls_indices = np.where(all_frame_targets == cls)[0]
        cls_preds = all_frame_preds[cls_indices]
        
        # Create sequences with sliding window
        for start_idx in range(0, len(cls_preds) - seq_len + 1, stride):
            end_idx = start_idx + seq_len
            
            # Average predictions over the sequence
            seq_pred = np.mean(cls_preds[start_idx:end_idx], axis=0)
            sequence_preds.append(seq_pred)
            sequence_targets.append(cls)  # All frames in sequence have same label
    
    return np.array(sequence_preds), np.array(sequence_targets)

def get_temporal_predictions(checkpoint_path, data_dir, n_classes, model_type="lstm", seq_len=16):
    """Get temporal model predictions."""
    _, _, test_loader = get_seq_dataloaders(Path(data_dir), seq_len=seq_len, batch_size=64)
    
    # Auto-detect input dimension
    sample_batch = next(iter(test_loader))
    input_dim = sample_batch[0].shape[-1]
    
    # Load model
    if model_type == "lstm":
        model = LSTMGesture(input_dim=input_dim, num_classes=n_classes)
    elif model_type == "gru":
        model = GRUGesture(input_dim=input_dim, num_classes=n_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load with error handling
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    except RuntimeError as e:
        logger.error(f"Error loading {model_type}: {e}")
        return None, None
    
    model.eval()
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.numpy())
            targets.append(batch_y.numpy())
    
    return np.vstack(predictions), np.concatenate(targets)

def align_predictions(*pred_target_pairs):
    """
    Align predictions from different models to same samples.
    Takes minimum number of samples across all models.
    """
    min_samples = min(len(targets) for _, targets in pred_target_pairs if targets is not None)
    
    aligned_preds = []
    aligned_targets = None
    
    for preds, targets in pred_target_pairs:
        if preds is not None and targets is not None:
            # Take first min_samples
            aligned_preds.append(preds[:min_samples])
            if aligned_targets is None:
                aligned_targets = targets[:min_samples]
    
    return aligned_preds, aligned_targets

def main():
    data_dir = "data/processed/custom"
    n_classes = 11
    seq_len = 16
    
    logger.info("Loading aligned model predictions...")
    
    # Get MLP predictions aligned to sequences
    try:
        mlp_preds, mlp_targets = get_mlp_sequence_predictions(
            "checkpoints/landmark_mlp_large_best.pth", 
            data_dir, n_classes, seq_len
        )
        logger.info(f"✅ MLP Large: {mlp_preds.shape[0]} sequence predictions")
        mlp_success = True
    except Exception as e:
        logger.error(f"❌ MLP Large failed: {e}")
        mlp_preds, mlp_targets = None, None
        mlp_success = False
    
    # Get LSTM predictions
    try:
        lstm_preds, lstm_targets = get_temporal_predictions(
            "checkpoints/lstm_gesture_model.pth", 
            data_dir, n_classes, "lstm", seq_len
        )
        logger.info(f"✅ LSTM: {lstm_preds.shape[0]} sequence predictions")
        lstm_success = True
    except Exception as e:
        logger.error(f"❌ LSTM failed: {e}")
        lstm_preds, lstm_targets = None, None
        lstm_success = False
    
    # Get GRU predictions (might fail)
    try:
        gru_preds, gru_targets = get_temporal_predictions(
            "checkpoints/gru_gesture_model.pth", 
            data_dir, n_classes, "gru", seq_len
        )
        logger.info(f"✅ GRU: {gru_preds.shape[0]} sequence predictions")
        gru_success = True
    except Exception as e:
        logger.error(f"❌ GRU failed: {e}")
        gru_preds, gru_targets = None, None
        gru_success = False
    
    # Align predictions
    pred_target_pairs = [
        (mlp_preds, mlp_targets),
        (lstm_preds, lstm_targets), 
        (gru_preds, gru_targets)
    ]
    
    aligned_preds, targets = align_predictions(*pred_target_pairs)
    
    if len(aligned_preds) == 0:
        logger.error("❌ No models loaded successfully!")
        return
    
    model_names = []
    if mlp_success and mlp_preds is not None:
        model_names.append("MLP_Large")
    if lstm_success and lstm_preds is not None:
        model_names.append("LSTM")
    if gru_success and gru_preds is not None:
        model_names.append("GRU")
    
    logger.info(f"Ensemble using: {', '.join(model_names)}")
    logger.info(f"Aligned predictions shape: {len(aligned_preds)} models × {aligned_preds[0].shape[0]} samples")
    
    # Create ensemble
    ensemble_probs = np.mean(aligned_preds, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    # Evaluate ensemble
    ensemble_acc = accuracy_score(targets, ensemble_preds)
    ensemble_f1 = f1_score(targets, ensemble_preds, average='macro')
    
    logger.info(f"\n🎯 ENSEMBLE RESULTS:")
    logger.info(f"   Accuracy: {ensemble_acc:.4f}")
    logger.info(f"   Macro F1: {ensemble_f1:.4f}")
    logger.info(f"   Models: {', '.join(model_names)}")
    
    # Individual model comparison
    logger.info(f"\n📊 INDIVIDUAL MODEL COMPARISON:")
    for i, (preds, name) in enumerate(zip(aligned_preds, model_names)):
        individual_preds = np.argmax(preds, axis=1)
        individual_acc = accuracy_score(targets, individual_preds)
        individual_f1 = f1_score(targets, individual_preds, average='macro')
        logger.info(f"   {name:10}: {individual_acc:.4f} acc, {individual_f1:.4f} F1")
    
    # Check if ensemble beats best individual
    best_individual_acc = max([
        accuracy_score(targets, np.argmax(preds, axis=1)) 
        for preds in aligned_preds
    ])
    
    if ensemble_acc > best_individual_acc:
        logger.info(f"🎉 Ensemble IMPROVES over best individual!")
    else:
        logger.info(f"📉 Ensemble does not improve over best individual")

if __name__ == "__main__":
    main() 