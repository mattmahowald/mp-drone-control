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
import numpy as np
import torch

from mp_drone_control.utils.logging_config import setup_logging
from mp_drone_control.models.trainer import (
    train,
    train_and_save_all_models,
)
from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, TemporalConvGesture, get_seq_dataloaders

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
    ap.add_argument("--augment", action="store_true")
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--temporal-model', choices=['gru','tcn'], default=None)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get number of classes from data
    train_y = np.load(data_dir / "train_y.npy")
    n_cls = len(set(train_y))
    logger.info(f"Number of classes: {n_cls}")

    if args.temporal_model:
        logger.info(f"Training {args.temporal_model} temporal model...")
        train_loader, val_loader, test_loader = get_seq_dataloaders(
            args.data_dir, args.seq_len, batch_size=args.batch_size, augment=args.augment)
        
        if args.temporal_model == 'gru':
            model = GRUGesture(num_classes=n_cls).to(device)
        elif args.temporal_model == 'tcn':
            model = TemporalConvGesture(num_classes=n_cls).to(device)
        
        # For now, let's implement a simple training loop for temporal models
        # You might want to extend this with the full trainer functionality later
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        logger.info(f"Starting training for {args.epochs} epochs...")
        model.train()
        
        for epoch in range(args.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f'Epoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}')
        
        # Save the model
        model_save_path = save_dir / f"{args.temporal_model}_gesture_model.pth"
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"✓ Temporal model saved to {model_save_path}")
        
    else:
        # Regular MLP training
        train_loader, val_loader, test_loader = get_dataloader(
            args.data_dir, split="train", batch_size=args.batch_size, normalize=False
        ), get_dataloader(
            args.data_dir, split="val", batch_size=args.batch_size, normalize=False
        ), get_dataloader(
            args.data_dir, split="test", batch_size=args.batch_size, normalize=False
        )
        
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

