import numpy as np
import torch
from pathlib import Path
import tempfile
import pytest

from mp_drone_control.data.loaders import get_dataloader, HandLandmarkDataset


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create sample data
        X = np.random.randn(100, 21, 3).astype(
            np.float32
        )  # 100 samples, 21 landmarks, 3 coordinates
        y = np.random.randint(0, 5, size=(100,)).astype(np.int64)  # 5 classes

        # Save to temporary directory
        np.save(tmp_path / "train_X.npy", X)
        np.save(tmp_path / "train_y.npy", y)
        np.save(tmp_path / "val_X.npy", X[:20])  # 20 samples for validation
        np.save(tmp_path / "val_y.npy", y[:20])
        np.save(tmp_path / "test_X.npy", X[:10])  # 10 samples for testing
        np.save(tmp_path / "test_y.npy", y[:10])

        yield tmp_path


def test_dataset_creation(temp_data_dir):
    """Test creating a HandLandmarkDataset instance."""
    dataset = HandLandmarkDataset(
        data_path=temp_data_dir / "train_X.npy",
        label_path=temp_data_dir / "train_y.npy",
        normalize=True,
        augment=True,
    )

    assert len(dataset) == 100
    x, y = dataset[0]
    assert x.shape == (63,)  # Flattened landmarks
    assert isinstance(y, int)
    assert 0 <= y < 5


def test_dataset_normalization(temp_data_dir):
    """Test dataset normalization."""
    # Test with normalization
    dataset_norm = HandLandmarkDataset(
        data_path=temp_data_dir / "train_X.npy",
        label_path=temp_data_dir / "train_y.npy",
        normalize=True,
        augment=True,
    )

    # Test without normalization
    dataset_raw = HandLandmarkDataset(
        data_path=temp_data_dir / "train_X.npy",
        label_path=temp_data_dir / "train_y.npy",
        normalize=False,
        augment=True,
    )

    x_norm, _ = dataset_norm[0]
    x_raw, _ = dataset_raw[0]

    # Check that normalized data is different from raw data
    assert not torch.equal(x_norm, x_raw)

    # Check that normalized data has expected properties
    assert torch.all(torch.isfinite(x_norm))  # No NaN or inf values
    assert torch.max(torch.abs(x_norm)) <= 1.0  # Values should be normalized


def test_dataloader_basic(temp_data_dir):
    """Test basic dataloader functionality."""
    dataloader = get_dataloader(
        data_dir=temp_data_dir,
        split="train",
        batch_size=16,
        shuffle=True,
        normalize=True,
        num_workers=0,  # Use 0 for testing
    )

    # Test batch iteration
    for i, (x_batch, y_batch) in enumerate(dataloader):
        assert x_batch.shape == (16, 63)  # Batch size 16, flattened landmarks
        assert y_batch.shape == (16,)  # Batch size 16
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.int64
        if i == 1:  # Test first two batches
            break


def test_dataloader_splits(temp_data_dir):
    """Test dataloader with different splits."""
    for split in ["train", "val", "test"]:
        dataloader = get_dataloader(
            data_dir=temp_data_dir,
            split=split,
            batch_size=8,
            shuffle=False,
            normalize=True,
            num_workers=0,
        )

        # Get all batches
        all_x = []
        all_y = []
        for x_batch, y_batch in dataloader:
            all_x.append(x_batch)
            all_y.append(y_batch)

        # Check total samples
        if split == "train":
            expected_samples = 100
        elif split == "val":
            expected_samples = 20
        else:  # test
            expected_samples = 10

        total_samples = sum(x.shape[0] for x in all_x)
        assert total_samples == expected_samples


def test_dataloader_edge_cases(temp_data_dir):
    """Test dataloader with edge cases."""
    # Test with batch size larger than dataset
    dataloader = get_dataloader(
        data_dir=temp_data_dir,
        split="test",  # Only 10 samples
        batch_size=20,  # Larger than dataset
        shuffle=False,
        normalize=True,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(dataloader))
    assert x_batch.shape[0] == 10  # Should return all samples
    assert y_batch.shape[0] == 10

    # Test with batch size 1
    dataloader = get_dataloader(
        data_dir=temp_data_dir,
        split="train",
        batch_size=1,
        shuffle=False,
        normalize=True,
        num_workers=0,
    )

    x_batch, y_batch = next(iter(dataloader))
    assert x_batch.shape == (1, 63)
    assert y_batch.shape == (1,)


def test_dataloader_shuffle(temp_data_dir):
    """Test dataloader shuffling."""
    # Get two dataloaders with same seed
    dataloader1 = get_dataloader(
        data_dir=temp_data_dir,
        split="train",
        batch_size=16,
        shuffle=True,
        normalize=True,
        num_workers=0,
    )

    dataloader2 = get_dataloader(
        data_dir=temp_data_dir,
        split="train",
        batch_size=16,
        shuffle=True,
        normalize=True,
        num_workers=0,
    )

    # Get first batch from each
    x1, y1 = next(iter(dataloader1))
    x2, y2 = next(iter(dataloader2))

    # Batches should be different due to shuffling
    assert not torch.equal(x1, x2)
    assert not torch.equal(y1, y2)
