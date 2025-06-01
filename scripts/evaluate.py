#!/usr/bin/env python3
"""evaluate.py  ‑‑  Evaluate trained MLP checkpoints on the custom dataset

Usage (PowerShell):

    poetry run python scripts/evaluate.py \
            --data-dir  data/processed/custom \
            --checkpoint checkpoints/landmark_mlp_small_best.pth \
            --save-cm   report/baseline_cm.png \
            --save-json report/baseline_metrics.json
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
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

MODEL_MAP = {
    "small": LandmarkMLPSmall,
    "large": LandmarkMLPLarge,
}

# ---------------------------------------------------------------------------

def load_model(model_name: str, ckpt_path: Path, num_classes: int, device="cpu"):
    model_cls = MODEL_MAP[model_name]
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
    ap.add_argument("--model", choices=["small", "large"], default="small")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--save-cm",   type=str, default=None,
                    help="PNG path to save confusion matrix")
    ap.add_argument("--save-json", type=str, default=None,
                    help="JSON file to dump accuracy/F1 numbers")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load dataset & model
    data_root = Path(args.data_dir)
    test_loader = get_dataloader(data_root, split="test",
                                batch_size=args.batch_size, normalize=False)

    num_classes = len(test_loader.dataset.label2idx)
    model = load_model(args.model, Path(args.checkpoint), num_classes, device)

    # ------------------------------------------------------------------
    # Evaluate
    acc, macro_f1, cm, y_true, y_pred = evaluate(model, test_loader, device)
    logger.info(f"Test Accuracy: {acc:.4f}  |  Macro‑F1: {macro_f1:.4f}")

    # ------------------------------------------------------------------
    # Optional outputs
    if args.save_cm:
        # Recover label names for ticks (if available)
        labels = list(test_loader.dataset.label2idx.keys())
        save_confusion_matrix(cm, Path(args.save_cm), labels)
        logger.info(f"Confusion matrix saved to {args.save_cm}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump({"accuracy": acc, "macro_f1": macro_f1}, f, indent=2)
        logger.info(f"Metrics JSON saved to {args.save_json}")

