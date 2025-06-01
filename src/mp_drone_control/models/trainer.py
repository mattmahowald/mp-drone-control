#!/usr/bin/env python3
"""trainer.py – training / evaluation helpers for landmark‑MLP baselines.

This *replaces* the previous `LandmarkClassifier` / `LargeLandmarkClassifier`
(18‑layer Sequential) with the compact two‑layer networks defined in
`mp_drone_control.models.landmark_mlp`:

* **LandmarkMLPSmall**  – 63 → 128 → 64 → N (≈35 k params)
* **LandmarkMLPLarge**  – 63 → 256 → 128 → N (≈130 k params)

All training / evaluation code now infers `num_classes` from the dataset, so
there’s no hard‑coded **11** in sight.  WandB logging is kept, but symlink
creation is disabled to avoid WinError 1314 on Windows.
"""

from pathlib import Path
from typing import Optional

import os
os.environ["WANDB_DISABLE_SYMLINKS"] = "true"   # ← add this once, before wandb.init


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb

from mp_drone_control.models.landmark_mlp import (
    LandmarkMLPSmall,
    LandmarkMLPLarge,
)
from mp_drone_control.data.loaders import get_dataloader

# ---------------------------------------------------------------------------
MODEL_MAP = {
    "small": LandmarkMLPSmall,
    "large": LandmarkMLPLarge,
}

# ---------------------------------------------------------------------------

def _init_device(device_flag: Optional[str] = None) -> str:
    if device_flag is not None:
        return device_flag
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"

# ---------------------------------------------------------------------------

def train(
    *,
    data_dir: Path,
    num_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_path: Optional[Path] = None,
    project_name: str = "hand-gesture-recognition",
    model_name: str = "small",  # "small" | "large"
):
    """Train one MLP variant and optionally save the best checkpoint."""

    device = _init_device(device)
    print(f"📟 Using device: {device}")

    # ── Dataloaders ───────────────────────────────────────────────────────
    train_loader = get_dataloader(
        data_dir, split="train", batch_size=batch_size, normalize=False
    )
    num_classes = len(train_loader.dataset.label2idx)

    # ── Model / optimiser / loss ─────────────────────────────────────────
    model_cls = MODEL_MAP[model_name]
    model: nn.Module = model_cls(num_classes).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── WandB – disable symlinks to avoid WinError‑1314 on Windows ───────
    wandb.init(
        project=project_name,
        config=dict(epochs=num_epochs, batch_size=batch_size, lr=lr, model=model_name),
    )
    wandb.watch(model, log="all")

    # ── Training loop ────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        wandb.log({"epoch": epoch, "loss": epoch_loss, "acc": epoch_acc})
        print(f"📈 Epoch {epoch:02d}/{num_epochs} — Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if save_path and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"💾 New best model saved to {save_path} (acc={best_acc:.4f})")

    wandb.finish()

# ---------------------------------------------------------------------------

def train_and_save_all_models(
    *,
    data_dir: Path,
    num_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[str] = None,
    save_dir: Path = Path("checkpoints"),
    project_name: str = "hand-gesture-recognition",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    for m in ["small", "large"]:
        ckpt = save_dir / f"landmark_mlp_{m}_best.pth"
        train(
            data_dir=data_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            save_path=ckpt,
            project_name=project_name,
            model_name=m,
        )

# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    ckpt_path: Path,
    data_dir: Path,
    *,
    split: str = "test",
    batch_size: int = 256,
    device: Optional[str] = None,
):
    """Return (accuracy, macro‑F1) on the requested split."""
    from sklearn.metrics import accuracy_score, f1_score

    device = _init_device(device)
    loader = get_dataloader(data_dir, split=split, batch_size=batch_size, normalize=False)
    num_classes = len(loader.dataset.label2idx)

    model_cls = MODEL_MAP[model_name]
    model = model_cls(num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(yb)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"✅ {model_name} — acc={acc:.4f}  macro‑F1={f1:.4f}")
    return acc, f1

# ---------------------------------------------------------------------------

def export_model_to_onnx(model_name: str, ckpt_path: Path, export_path: Path):
    model_cls = MODEL_MAP[model_name]
    dummy_input = torch.randn(1, 63)

    # Number of classes inferred from checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    num_classes = state["out.weight"].shape[0]
    model = model_cls(num_classes)
    model.load_state_dict(state)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    print(f"🛫 ONNX model exported to {export_path}")

