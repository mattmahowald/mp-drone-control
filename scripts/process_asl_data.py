#!/usr/bin/env python3
"""process_asl_data.py  ‑‑  Generic splitter for *landmark* gesture datasets

Re‑implemented so it works for **our recorded hand‑gesture clips** instead of
 expecting the ASL‑digits *image* dataset under a hard‑coded "Dataset" folder.

Input folder structure (``--raw-dir``):
    data/raw/<gesture_label>/*.npy
Each ``.npy`` file can be either
    • (T, 21, 3)   ‑‑ per‑frame MediaPipe 3‑D landmarks
    • (T, 63)      ‑‑ already flattened per frame

The script:
    1. Recursively loads every ``.npy`` clip for every gesture label.
    2. Converts each frame to a 63‑D vector (x₁…x₂₁, y₁…y₂₁, z₁…z₂₁).
    3. Produces **class‑balanced** train/val/test splits using
       ``sklearn.model_selection.train_test_split``.
    4. Saves
          processed_dir/train_X.npy, train_y.npy
          processed_dir/val_X.npy,   val_y.npy
          processed_dir/test_X.npy,  test_y.npy
       where *X* is ``float32`` (N, 63) and *y* is ``str`` (N,) for readability.

Example
-------
$ poetry run python scripts/process_asl_data.py \
        --raw-dir       data/raw \
        --processed-dir data/processed/custom \
        --val-size 0.15 --test-size 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split
import logging

# --------------------------------------------------------------------------------------
# Utility funcs
# --------------------------------------------------------------------------------------

def _flatten_clip(arr: np.ndarray) -> np.ndarray:
    """Return (frames, 63) regardless of input shape."""
    if arr.ndim == 3 and arr.shape[1:] == (21, 3):
        # (T, 21, 3) → (T, 63)
        return arr.reshape(arr.shape[0], 63)
    if arr.ndim == 2 and arr.shape[1] == 63:
        return arr
    raise ValueError(f"Unsupported landmark shape {arr.shape}, expected (T,21,3) or (T,63).")


def _load_dataset(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all .npy clips from the raw directory."""
    clips = []
    labels = []
    for label_dir in raw_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for clip_path in label_dir.glob("*.npy"):
            try:
                clip = np.load(clip_path, allow_pickle=True)
                if clip is None or clip.size == 0:
                    logging.warning(f"Skipping empty clip: {clip_path}")
                    continue
                clips.append(clip)
                labels.append(label)
            except Exception as e:
                logging.error(f"Error loading {clip_path}: {e}")
                continue

    if not clips:
        raise ValueError(f"No .npy clips found for any label in {raw_dir}")

    return np.array(clips), np.array(labels)


# --------------------------------------------------------------------------------------
# Main CLI entry
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert raw gesture .npy clips into train/val/test splits.")
    ap.add_argument("--raw-dir", type=Path, required=True, help="Folder with per‑gesture subfolders of .npy clips.")
    ap.add_argument("--processed-dir", type=Path, required=True, help="Output folder for the split .npy arrays.")
    ap.add_argument("--val-size", type=float, default=0.15, help="Fraction of data to use for validation.")
    ap.add_argument("--test-size", type=float, default=0.15, help="Fraction of data to use for testing.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    raw_root: Path = args.raw_dir
    out_root: Path = args.processed_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading raw clips from {raw_root} …")
    X, y = _load_dataset(raw_root)
    print(f"   → {X.shape[0]:,} frames, {len(set(y))} gesture classes")

    # First split off the test set, then split the remaining into train/val so that
    # val_size is exactly the user‑specified proportion of the *original* dataset.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=args.seed)

    # Save arrays
    np.save(out_root / "train_X.npy", X_train)
    np.save(out_root / "train_y.npy", y_train)
    np.save(out_root / "val_X.npy", X_val)
    np.save(out_root / "val_y.npy", y_val)
    np.save(out_root / "test_X.npy", X_test)
    np.save(out_root / "test_y.npy", y_test)

    # Metadata for humans / reproducibility
    meta = {
        "num_frames": int(X.shape[0]),
        "num_classes": len(set(y)),
        "train_frames": int(X_train.shape[0]),
        "val_frames": int(X_val.shape[0]),
        "test_frames": int(X_test.shape[0]),
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": args.seed,
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved splits to {out_root} (see meta.json for stats)")


if __name__ == "__main__":
    main()
