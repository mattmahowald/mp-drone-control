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
from sklearn.metrics import accuracy_score

from mp_drone_control.utils.logging_config import setup_logging
from mp_drone_control.models.trainer import (
    train,
    train_and_save_all_models,
)
from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, TemporalConvGesture, LSTMGesture, TransformerGesture, get_seq_dataloaders

logger = setup_logging()

def validate_temporal_model(model, val_loader, criterion, device):
    """Validate temporal model and return loss and accuracy."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_targets, all_preds)
    
    return avg_val_loss, val_accuracy

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
    ap.add_argument('--temporal-model', choices=['gru','tcn','lstm','transformer'], default=None)
    ap.add_argument('--patience', type=int, default=50, 
                    help="Early stopping patience for temporal models")
    ap.add_argument('--val-every', type=int, default=10,
                    help="Validate every N epochs for temporal models")

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
        
        # Auto-detect input dimension from the dataset
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[-1]  # Last dimension
        logger.info(f"Auto-detected input dimension: {input_dim}")
        
        if args.temporal_model == 'gru':
            model = GRUGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'tcn':
            model = TemporalConvGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'lstm':
            model = LSTMGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'transformer':
            model = TransformerGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-7
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = save_dir / f"{args.temporal_model}_gesture_model_best.pth"
        
        logger.info(f"Starting training for {args.epochs} epochs...")
        logger.info(f"Validation every {args.val_every} epochs, early stopping patience: {args.patience}")
        
        for epoch in range(args.epochs):
            # Training phase
            model.train()
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
            
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Validation phase
            if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
                val_loss, val_acc = validate_temporal_model(model, val_loader, criterion, device)
                
                logger.info(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, '
                           f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Current learning rate: {current_lr:.2e}')
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f'✓ New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                else:
                    patience_counter += args.val_every
                
                # Early stopping check
                if patience_counter >= args.patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs. '
                               f'Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}')
                    break
                    
                # Stop if learning rate becomes too small
                if current_lr < 1e-7:
                    logger.info(f'Learning rate too small ({current_lr:.2e}), stopping training.')
                    break
            else:
                logger.info(f'Epoch {epoch+1}/{args.epochs} completed. Average Train Loss: {avg_train_loss:.4f}')
        
        # Load best model for final save
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
        
        # Save final model (for compatibility)
        final_model_path = save_dir / f"{args.temporal_model}_gesture_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✓ Final temporal model saved to {final_model_path}")
        
    else:
        # Regular MLP training
        train_loader, val_loader, test_loader = get_dataloader(
            data_dir, split="train", batch_size=args.batch_size, normalize=False
        ), get_dataloader(
            data_dir, split="val", batch_size=args.batch_size, normalize=False
        ), get_dataloader(
            data_dir, split="test", batch_size=args.batch_size, normalize=False
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

