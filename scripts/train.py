#!/usr/bin/env python3
"""Unified training entry‑point for MLP baselines.

Examples
--------
# train the small network for 30 epochs on the custom dataset
poetry run python scripts/train.py --model small --epochs 30

# train both small and large and save to checkpoints/
poetry run python scripts/train.py --model both --epochs 20
"""

from pathlib import Path
import argparse

from mp_drone_control.utils.logging_config import setup_logging
from mp_drone_control.models.trainer import (
    train,
    train_and_save_all_models,
)

logger = setup_logging()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/custom",
                    help="Root folder that contains train_X.npy etc.")
    ap.add_argument("--model", choices=["small", "large", "both"], default="small",
                    help="Which baseline to train (or 'both' to loop over both).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save-dir", default="checkpoints")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "both":
        logger.info("Training BOTH small and large MLPs …")
        train_and_save_all_models(
            data_dir=data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=save_dir,
        )
    else:
        logger.info(f"Training {args.model} MLP for {args.epochs} epochs …")
        ckpt = save_dir / f"landmark_mlp_{args.model}_best.pth"
        train(
            data_dir=data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=ckpt,
            model_name=args.model,
        )
        logger.info(f"✓ Model saved to {ckpt}")

