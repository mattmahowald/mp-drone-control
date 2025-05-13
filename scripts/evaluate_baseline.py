#!/usr/bin/env python3
"""
Script to evaluate our model against the MediaPipe baseline.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "3"

import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.models.mobilenet import LandmarkClassifier
from mp_drone_control.evaluation.baseline import MediaPipeBaseline
from mp_drone_control.utils.logging_config import setup_logging

# Set up logging for the entire project
logger = setup_logging()


def print_classification_table(report):
    # Only print per-class rows (skip avg/accuracy rows)
    headers = f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}"
    logger.info("\n📊 Classification Report Table:")
    logger.info(headers)
    logger.info("-" * len(headers))
    for cls, metrics in report.items():
        if isinstance(metrics, dict) and all(
            k in metrics for k in ["precision", "recall", "f1-score", "support"]
        ):
            logger.info(
                f"{str(cls):<10}{metrics['precision']:<12.2f}{metrics['recall']:<12.2f}{metrics['f1-score']:<12.2f}{metrics['support']:<10}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate model against baseline")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/asl_digits"),
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/landmark_classifier.pt"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu/cuda/mps)",
    )

    args = parser.parse_args()

    logger.info(
        f"\n==============================\nEvaluating model against baseline with data from {args.data_dir}\n=============================="
    )

    # Load test data
    test_loader = get_dataloader(args.data_dir, split="test")
    X_test = []
    y_test = []
    for X, y in test_loader:
        X_test.append(X.numpy())
        y_test.append(y.numpy())
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    logger.info("Loaded test data successfully.")

    # Load our model
    model = LandmarkClassifier(input_dim=63, num_classes=10)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()

    logger.info(f"Loaded model from {args.model_path} and set to {args.device}.")

    # Evaluate our model
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, device=args.device)
        logits = model(X_tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    our_accuracy = (predictions == y_test).mean()
    logger.info(f"Our model accuracy: {our_accuracy:.3f}")

    # Evaluate baseline
    baseline = MediaPipeBaseline()
    logger.info(f"X_test shape: {X_test.shape}")
    baseline_results = baseline.evaluate(X_test, y_test)
    baseline_accuracy = baseline_results["accuracy"]
    logger.info(f"Baseline accuracy: {baseline_accuracy:.3f}")

    # Pretty print classification report table
    print_classification_table(baseline_results["classification_report"])

    # Compare results
    logger.info(
        "\n==============================\nComparison\n=============================="
    )
    logger.info(f"Our model accuracy: {our_accuracy:.3f}")
    logger.info(f"Baseline accuracy: {baseline_accuracy:.3f}")
    logger.info(f"Improvement: {our_accuracy - baseline_accuracy:.3f}")


if __name__ == "__main__":
    main()
