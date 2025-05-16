from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb

from mp_drone_control.models.mobilenet import (
    LandmarkClassifier,
    LargeLandmarkClassifier,
)
from mp_drone_control.data.loaders import get_dataloader


def train(
    data_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
    project_name: str = "hand-gesture-recognition",
    model_name: str = "small",  # 'small' or 'large'
):
    # Setup device
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"📟 Using device: {device}")

    # Load dataset
    train_loader = get_dataloader(data_dir, split="train", batch_size=batch_size, normalize=False)

    # Initialize model
    if model_name == "large":
        model = LargeLandmarkClassifier(input_dim=63, num_classes=11).to(device)
        wandb_model_name = "LargeLandmarkClassifier"
    else:
        model = LandmarkClassifier(input_dim=63, num_classes=11).to(device)
        wandb_model_name = "LandmarkClassifier"
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb
    wandb.init(
        project=project_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": wandb_model_name,
            "input_dim": 63,
            "num_classes": 11,
        },
    )
    wandb.watch(model, log="all")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        acc = correct / total
        wandb.log({"epoch": epoch + 1, "loss": running_loss, "accuracy": acc})
        print(
            f"📈 Epoch {epoch+1}/{num_epochs} — Loss: {running_loss:.4f} | Acc: {acc:.4f}"
        )

    if save_path:
        torch.save(model.state_dict(), save_path)
        wandb.save(str(save_path))
        print(f"💾 Model saved to {save_path}")

    wandb.finish()


def train_and_save_all_models(
    data_dir: Path,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None,
    project_name: str = "hand-gesture-recognition",
):
    """Train and save both small and large models."""
    save_dir = save_dir or Path("models/")
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name in ["small", "large"]:
        save_path = save_dir / f"{model_name}_model.pth"
        print(f"\n=== Training {model_name} model ===")
        train(
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            save_path=save_path,
            project_name=project_name,
            model_name=model_name,
        )
        print(f"{model_name.capitalize()} model saved to {save_path}")


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    data_dir: Path,
    split: str = "test",
    device: Optional[str] = None,
):
    """Evaluate a saved model on a given dataset split."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "large":
        model = LargeLandmarkClassifier(input_dim=63, num_classes=11).to(device)
    else:
        model = LandmarkClassifier(input_dim=63, num_classes=11).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    loader = get_dataloader(data_dir, split=split, batch_size=64, shuffle=False, normalize=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=3))
    return acc, f1


def export_model_to_onnx(model_name: str, checkpoint_path: Path, export_path: Path):
    """Export a trained model to ONNX format for mobile deployment."""
    if model_name == "large":
        model = LargeLandmarkClassifier(input_dim=63, num_classes=11)
    else:
        model = LandmarkClassifier(input_dim=63, num_classes=11)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    dummy_input = torch.randn(1, 63)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=12,
    )
    print(f"Model exported to {export_path}")
