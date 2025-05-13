from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb

from mp_drone_control.models.mobilenet import LandmarkClassifier
from mp_drone_control.data.loaders import get_dataloader


def train(
    data_dir: Path,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
    project_name: str = "hand-gesture-recognition",
):
    # Setup device
    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"📟 Using device: {device}")

    # Load dataset
    train_loader = get_dataloader(data_dir, split="train", batch_size=batch_size)

    # Initialize model
    model = LandmarkClassifier(input_dim=63, num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Initialize wandb
    wandb.init(
        project=project_name,
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model": "LandmarkClassifier",
            "input_dim": 63,
            "num_classes": 10,
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
