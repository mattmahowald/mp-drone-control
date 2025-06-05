#!/usr/bin/env python3
"""Enhanced training script with curve tracking for paper figures."""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score

from mp_drone_control.utils.logging_config import setup_logging
from mp_drone_control.models.trainer import (
    train,
    train_and_save_all_models,
)
from mp_drone_control.data.loaders import get_dataloader
from mp_drone_control.data.sequence_dataset import GRUGesture, TemporalConvGesture, LSTMGesture, TransformerGesture, get_seq_dataloaders

logger = setup_logging()

class TrainingTracker:
    """Track training metrics and create curves."""
    
    def __init__(self, model_name, save_dir):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metric storage
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
        self.learning_rates = []
        
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Update metrics for current epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
        
        metrics_file = self.save_dir / f"{self.model_name}_training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_file}")
        
    def create_training_curves(self):
        """Create and save training curve plots."""
        if not self.epochs:
            logger.warning("No training data to plot!")
            return
            
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name.upper()} Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{self.model_name.upper()} Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax3.semilogy(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.set_title(f'{self.model_name.upper()} Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        
        # Combined loss/accuracy
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        line2 = ax4_twin.plot(self.epochs, self.val_accuracies, 'b-', label='Val Accuracy', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='r')
        ax4_twin.set_ylabel('Accuracy', color='b')
        ax4.set_title(f'{self.model_name.upper()} Validation Metrics')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.save_dir / f"{self.model_name}_training_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {plot_file}")
        
        # Also save individual plots for paper
        self._save_individual_plots()
        
        plt.close()
        
    def _save_individual_plots(self):
        """Save individual plots for paper inclusion."""
        # Loss plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)
        plt.title(f'{self.model_name.upper()} Training Progress', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_file = self.save_dir / f"{self.model_name}_loss_curve.png"
        plt.savefig(loss_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'{self.model_name.upper()} Accuracy Progress', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        acc_file = self.save_dir / f"{self.model_name}_accuracy_curve.png"
        plt.savefig(acc_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Individual plots saved: {loss_file}, {acc_file}")

def compute_accuracy(model, dataloader, device):
    """Compute accuracy on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return accuracy_score(all_targets, all_preds)

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

def train_temporal_with_tracking(args, model, train_loader, val_loader, device, n_cls):
    """Train temporal model with full metric tracking."""
    
    # Initialize tracker
    tracker = TrainingTracker(args.temporal_model, "training_curves")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1e-7
    )
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    save_dir = Path(args.save_dir)
    best_model_path = save_dir / f"{args.temporal_model}_gesture_model_best.pth"
    
    logger.info(f"Starting tracked training for {args.epochs} epochs...")
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
        
        # Compute training accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        
        # Validation phase
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_loss, val_acc = validate_temporal_model(model, val_loader, criterion, device)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update tracker
            tracker.update(epoch + 1, avg_train_loss, train_acc, val_loss, val_acc, current_lr)
            
            logger.info(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, '
                       f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                logger.info(f'Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}')
            
            # Save best model
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
            if new_lr < 1e-7:
                logger.info(f'Learning rate too small ({new_lr:.2e}), stopping training.')
                break
        else:
            # For non-validation epochs, still track with previous validation metrics
            current_lr = optimizer.param_groups[0]['lr']
            if tracker.val_losses:  # Use last validation metrics
                tracker.update(epoch + 1, avg_train_loss, train_acc, 
                             tracker.val_losses[-1], tracker.val_accuracies[-1], current_lr)
            
            logger.info(f'Epoch {epoch+1}/{args.epochs} completed. Train Loss: {avg_train_loss:.4f}, '
                       f'Train Acc: {train_acc:.4f}')
    
    # Save metrics and create plots
    tracker.save_metrics()
    tracker.create_training_curves()
    
    return best_model_path, best_val_loss, best_val_acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/custom",
                    help="Root folder that contains train_X.npy etc.")
    ap.add_argument("--model", choices=["small", "large", "both"], default="small",
                    help="Which baseline to train (or 'both' to loop over both).")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save-dir", default="checkpoints")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument('--seq-len', type=int, default=16)
    ap.add_argument('--temporal-model', choices=['gru','tcn','lstm','transformer'], default=None)
    ap.add_argument('--patience', type=int, default=100, 
                    help="Early stopping patience for temporal models")
    ap.add_argument('--val-every', type=int, default=5,
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
        logger.info(f"Training {args.temporal_model} temporal model with curve tracking...")
        train_loader, val_loader, test_loader = get_seq_dataloaders(
            args.data_dir, args.seq_len, batch_size=args.batch_size, augment=args.augment)
        
        # Auto-detect input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[-1]
        logger.info(f"Auto-detected input dimension: {input_dim}")
        
        # Create model
        if args.temporal_model == 'gru':
            model = GRUGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'tcn':
            model = TemporalConvGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'lstm':
            model = LSTMGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        elif args.temporal_model == 'transformer':
            model = TransformerGesture(input_dim=input_dim, num_classes=n_cls).to(device)
        
        # Train with full tracking
        best_model_path, best_val_loss, best_val_acc = train_temporal_with_tracking(
            args, model, train_loader, val_loader, device, n_cls
        )
        
        # Load best model for final save
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best model with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f}")
        
        # Save final model (for compatibility)
        final_model_path = save_dir / f"{args.temporal_model}_gesture_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"✓ Final temporal model saved to {final_model_path}")
        logger.info(f"✓ Training curves saved to training_curves/{args.temporal_model}_*")
        
    else:
        logger.error("This script is designed for temporal models only. Use regular train.py for MLPs.") 