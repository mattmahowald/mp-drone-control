#!/usr/bin/env python3
"""evaluate.py  ‑‑  Comprehensive evaluation for all model types with extended metrics"""

from pathlib import Path
import argparse, json
import time

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, precision_score, 
    recall_score, balanced_accuracy_score, top_k_accuracy_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from mp_drone_control.models.landmark_mlp import LandmarkMLPSmall, LandmarkMLPLarge
from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, TemporalConvGesture, LSTMGesture, TransformerGesture, get_seq_dataloaders
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

MODEL_MAP = {
    "small": LandmarkMLPSmall,
    "large": LandmarkMLPLarge,
    "gru": GRUGesture,
    "tcn": TemporalConvGesture,
    "lstm": LSTMGesture,
    "transformer": TransformerGesture,
}

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_top_k_accuracy(y_true, y_prob, k):
    """Calculate top-k accuracy."""
    if len(y_prob.shape) == 1:  # If we only have predictions, not probabilities
        return None
    return top_k_accuracy_score(y_true, y_prob, k=k)

def load_model(model_name: str, ckpt_path: Path, num_classes: int, device="cpu"):
    model_cls = MODEL_MAP[model_name]
    
    if model_name in ["gru", "tcn"]:
        model = model_cls(num_classes=num_classes)
    else:
        model = model_cls(num_classes)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model

def comprehensive_evaluate(model, dataloader, device="cpu"):
    """Comprehensive evaluation with multiple metrics."""
    y_true, y_pred, y_prob = [], [], []
    inference_times = []
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            
            # Measure inference time
            start_time = time.time()
            logits = model(xb)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(yb.numpy())
            y_prob.extend(probs.cpu().numpy())
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    
    # Calculate all metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average="macro")
    metrics['micro_f1'] = f1_score(y_true, y_pred, average="micro")
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average="weighted")
    
    # Precision and Recall
    metrics['macro_precision'] = precision_score(y_true, y_pred, average="macro")
    metrics['macro_recall'] = recall_score(y_true, y_pred, average="macro")
    
    # Top-k accuracy
    metrics['top_2_accuracy'] = get_top_k_accuracy(y_true, y_prob, k=2)
    metrics['top_3_accuracy'] = get_top_k_accuracy(y_true, y_prob, k=3)
    
    # Performance metrics
    metrics['avg_inference_time'] = np.mean(inference_times)
    metrics['std_inference_time'] = np.std(inference_times)
    metrics['total_samples'] = len(y_true)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    per_class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics, cm, y_true, y_pred, per_class_report

def save_comprehensive_results(metrics, cm, per_class_report, labels, save_dir, model_name):
    """Save comprehensive results including plots and detailed metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Save per-class performance plot
    class_names = [str(i) for i in range(len(labels))]
    class_f1_scores = [per_class_report[cls]['f1-score'] for cls in class_names]
    class_support = [per_class_report[cls]['support'] for cls in class_names]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # F1 scores per class
    bars1 = ax1.bar(range(len(class_names)), class_f1_scores, color='skyblue')
    ax1.set_xlabel('Gesture Class')
    ax1.set_ylabel('F1 Score')
    ax1.set_title(f'Per-Class F1 Scores - {model_name.upper()}')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels([labels[i] if i < len(labels) else f'Class_{i}' for i in range(len(class_names))], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, class_f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Support per class
    bars2 = ax2.bar(range(len(class_names)), class_support, color='lightcoral')
    ax2.set_xlabel('Gesture Class')
    ax2.set_ylabel('Number of Test Samples')
    ax2.set_title(f'Test Set Support per Class - {model_name.upper()}')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels([labels[i] if i < len(labels) else f'Class_{i}' for i in range(len(class_names))], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, support in zip(bars2, class_support):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{support}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{model_name}_per_class_analysis.png", dpi=300)
    plt.close()
    
    # Save comprehensive metrics to JSON
    comprehensive_metrics = {
        'model_name': model_name,
        'overall_metrics': metrics,
        'per_class_metrics': per_class_report
    }
    
    with open(save_dir / f"{model_name}_comprehensive_metrics.json", "w") as f:
        json.dump(comprehensive_metrics, f, indent=2, default=str)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="data/processed/custom")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the .pth checkpoint to evaluate")
    ap.add_argument("--model", choices=["small", "large", "gru", "tcn", "lstm", "transformer"], required=True,
                    help="Model type to evaluate")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=16,
                    help="Sequence length for temporal models")
    ap.add_argument("--save-dir", type=str, default="evaluation_results",
                    help="Directory to save comprehensive evaluation results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)

    # Load dataset & model
    data_root = Path(args.data_dir)
    
    if args.model in ["gru", "tcn", "lstm", "transformer"]:
        # Temporal models
        _, _, test_loader = get_seq_dataloaders(
            data_root, seq_len=args.seq_len, batch_size=args.batch_size
        )
        
        # Auto-detect input dimension from the dataset
        sample_data, sample_target = next(iter(test_loader))  # Unpack the tuple properly
        input_dim = sample_data.shape[-1]  # Last dimension
        logger.info(f"Auto-detected input dimension: {input_dim}")
        
        # Get number of classes from the original data
        train_y = np.load(data_root / "train_y.npy")
        num_classes = len(set(train_y))
        unique_labels = sorted(set(train_y))
        labels = unique_labels
        
        # Create model with correct input dimension
        if args.model == "gru":
            model = GRUGesture(input_dim=input_dim, num_classes=num_classes).to(device)
        elif args.model == "tcn":
            model = TemporalConvGesture(input_dim=input_dim, num_classes=num_classes).to(device)
        elif args.model == "lstm":
            model = LSTMGesture(input_dim=input_dim, num_classes=num_classes).to(device)
        elif args.model == "transformer":
            model = TransformerGesture(input_dim=input_dim, num_classes=num_classes).to(device)
        
        # Load the checkpoint
        model.load_state_dict(torch.load(Path(args.checkpoint), map_location=device))
        model.eval()
        
    else:
        # MLP models
        test_loader = get_dataloader(data_root, split="test",
                                    batch_size=args.batch_size, normalize=False)
        num_classes = len(test_loader.dataset.label2idx)
        labels = list(test_loader.dataset.label2idx.keys())
        
        # Load MLP model (unchanged)
        model = load_model(args.model, Path(args.checkpoint), num_classes, device)

    # Count model parameters
    param_count = count_parameters(model)
    logger.info(f"Model has {param_count:,} trainable parameters")

    # Comprehensive evaluation
    logger.info(f"Running comprehensive evaluation for {args.model} model...")
    metrics, cm, y_true, y_pred, per_class_report = comprehensive_evaluate(model, test_loader, device)
    
    # Add parameter count to metrics
    metrics['model_parameters'] = param_count
    metrics['model_type'] = args.model
    if args.model in ["gru", "tcn"]:
        metrics['sequence_length'] = args.seq_len

    # Log key results
    logger.info("="*60)
    logger.info(f"COMPREHENSIVE EVALUATION RESULTS - {args.model.upper()}")
    logger.info("="*60)
    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"Macro F1:          {metrics['macro_f1']:.4f}")
    logger.info(f"Micro F1:          {metrics['micro_f1']:.4f}")
    logger.info(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    if metrics['top_2_accuracy']:
        logger.info(f"Top-2 Accuracy:    {metrics['top_2_accuracy']:.4f}")
        logger.info(f"Top-3 Accuracy:    {metrics['top_3_accuracy']:.4f}")
    logger.info(f"Macro Precision:   {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall:      {metrics['macro_recall']:.4f}")
    logger.info(f"Avg Inference:     {metrics['avg_inference_time']*1000:.2f}ms per batch")
    logger.info(f"Model Parameters:  {metrics['model_parameters']:,}")
    logger.info("="*60)

    # Save comprehensive results
    save_comprehensive_results(metrics, cm, per_class_report, labels, save_dir, args.model)
    logger.info(f"Comprehensive evaluation results saved to {save_dir}")

