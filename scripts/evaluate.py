#!/usr/bin/env python3
"""evaluate.py  ‑‑  Evaluate trained MLP or temporal model checkpoints on the custom dataset

Usage (PowerShell):

    # Evaluate MLP models
    poetry run python scripts/evaluate.py \
            --data-dir  data/processed/custom \
            --checkpoint checkpoints/landmark_mlp_small_best.pth \
            --model small \
            --save-cm   report/baseline_cm.png \
            --save-json report/baseline_metrics.json

    # Evaluate temporal models
    poetry run python scripts/evaluate.py \
            --data-dir  data/processed/custom \
            --checkpoint checkpoints/gru_gesture_model.pth \
            --model gru \
            --seq-len 16 \
            --save-cm   report/gru_cm.png \
            --save-json report/gru_metrics.json
"""

from pathlib import Path
import argparse, json

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from mp_drone_control.models.landmark_mlp import LandmarkMLPSmall, LandmarkMLPLarge
from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, TemporalConvGesture, get_seq_dataloaders
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

MODEL_MAP = {
    "small": LandmarkMLPSmall,
    "large": LandmarkMLPLarge,
    "gru": GRUGesture,
    "tcn": TemporalConvGesture,
}

# ---------------------------------------------------------------------------

def load_model(model_name: str, ckpt_path: Path, num_classes: int, device="cpu"):
    model_cls = MODEL_MAP[model_name]
    
    if model_name in ["gru", "tcn"]:
        # Temporal models
        model = model_cls(num_classes=num_classes)
    else:
        # MLP models
        model = model_cls(num_classes)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model


def evaluate(model, dataloader, device="cpu"):
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(yb.numpy())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, macro_f1, cm, y_true, y_pred


def save_confusion_matrix(cm: np.ndarray, out_path: Path, labels):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  default="data/processed/custom")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the .pth checkpoint to evaluate")
    ap.add_argument("--model", choices=["small", "large", "gru", "tcn"], required=True,
                    help="Model type to evaluate")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=16,
                    help="Sequence length for temporal models (ignored for MLP models)")
    ap.add_argument("--save-cm",   type=str, default=None,
                    help="PNG path to save confusion matrix")
    ap.add_argument("--save-json", type=str, default=None,
                    help="JSON file to dump accuracy/F1 numbers")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load dataset & model
    data_root = Path(args.data_dir)
    
    if args.model in ["gru", "tcn"]:
        # Temporal models
        _, _, test_loader = get_seq_dataloaders(
            data_root, seq_len=args.seq_len, batch_size=args.batch_size
        )
        # Get number of classes from the original data
        train_y = np.load(data_root / "train_y.npy")
        num_classes = len(set(train_y))
        # For temporal models, we'll need to create our own label mapping
        unique_labels = sorted(set(train_y))
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels = unique_labels
    else:
        # MLP models
        test_loader = get_dataloader(data_root, split="test",
                                    batch_size=args.batch_size, normalize=False)
        num_classes = len(test_loader.dataset.label2idx)
        labels = list(test_loader.dataset.label2idx.keys())

    model = load_model(args.model, Path(args.checkpoint), num_classes, device)

    # ------------------------------------------------------------------
    # Evaluate
    logger.info(f"Evaluating {args.model} model on test data...")
    acc, macro_f1, cm, y_true, y_pred = evaluate(model, test_loader, device)
    logger.info(f"Test Accuracy: {acc:.4f}  |  Macro‑F1: {macro_f1:.4f}")

    # ------------------------------------------------------------------
    # Optional outputs
    if args.save_cm:
        save_confusion_matrix(cm, Path(args.save_cm), labels)
        logger.info(f"Confusion matrix saved to {args.save_cm}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "model_type": args.model,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "num_test_samples": len(y_true),
            "num_classes": num_classes
        }
        if args.model in ["gru", "tcn"]:
            metrics["seq_len"] = args.seq_len
            
        with open(args.save_json, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics JSON saved to {args.save_json}")

