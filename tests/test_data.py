import numpy as np
from pathlib import Path
import shutil

from mp_drone_control.data.loaders import get_dataloader


def generate_fake_data(out_dir: Path, num_samples: int = 100):
    """
    Generate fake landmark data with (21, 3) landmarks per sample and integer labels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.random.rand(num_samples, 21, 3).astype(np.float32)
    y = np.random.randint(0, 5, size=(num_samples,)).astype(np.int64)

    np.save(out_dir / "train_X.npy", X)
    np.save(out_dir / "train_y.npy", y)
    print(f"✔️  Fake dataset generated at {out_dir}")


def cleanup_fake_data(data_dir: Path):
    """
    Delete the fake dataset.
    """
    try:
        shutil.rmtree(data_dir)
        print(f"✅ Fake dataset deleted at {data_dir}")
    except FileNotFoundError:
        print(f"✅ No fake dataset found at {data_dir}")


def test_dataloader():
    temp_data_dir = Path("tests/temp_data")
    generate_fake_data(temp_data_dir)

    dataloader = get_dataloader(temp_data_dir, split="train", batch_size=16)

    for i, (x_batch, y_batch) in enumerate(dataloader):
        print(f"Batch {i} — X: {x_batch.shape}, Y: {y_batch.shape}")
        if i == 1:  # Print 2 batches then stop
            break

    assert x_batch.shape[-1] == 63, "Expected flattened landmark vector of size 63"
    assert len(y_batch.shape) == 1, "Expected 1D label tensor"

    cleanup_fake_data(temp_data_dir)
