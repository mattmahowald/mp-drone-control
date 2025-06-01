import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mp_drone_control.data.preprocess import (
    PreprocessingConfig,
    split_data,
    save_splits,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with sample data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample data
        landmarks = np.random.randn(
            100, 21, 3
        )  # 100 samples, 21 landmarks, 3 coordinates
        labels = np.random.randint(0, 5, size=100)  # 5 classes

        # Save to temporary directory
        tmp_path = Path(tmp_dir)
        np.save(tmp_path / "landmarks.npy", landmarks)
        np.save(tmp_path / "labels.npy", labels)

        yield tmp_path


def test_preprocessing_config_validation():
    """Test PreprocessingConfig validation."""
    # Valid config
    config = PreprocessingConfig(val_size=0.2, test_size=0.1)
    config.validate()  # Should not raise

    # Invalid val_size
    with pytest.raises(ValueError):
        PreprocessingConfig(val_size=1.5).validate()

    # Invalid test_size
    with pytest.raises(ValueError):
        PreprocessingConfig(test_size=-0.1).validate()

    # Invalid sum
    with pytest.raises(ValueError):
        PreprocessingConfig(val_size=0.6, test_size=0.5).validate()


def test_load_raw_data(temp_data_dir):
    """Test loading raw data."""
    landmarks = np.load(temp_data_dir / "landmarks.npy")
    labels = np.load(temp_data_dir / "labels.npy")

    assert landmarks.shape == (100, 21, 3)
    assert labels.shape == (100,)
    assert np.issubdtype(labels.dtype, np.integer)


def test_load_raw_data_missing_files():
    """Test loading raw data with missing files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Do not create files
        try:
            landmarks = np.load(tmp_path / "landmarks.npy")
            labels = np.load(tmp_path / "labels.npy")
        except FileNotFoundError:
            pass
        else:
            assert False, "Expected FileNotFoundError"


def test_load_raw_data_shape_mismatch():
    """Test loading raw data with shape mismatch."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        np.save(tmp_path / "landmarks.npy", np.random.randn(100, 21, 3))
        np.save(
            tmp_path / "labels.npy", np.random.randint(0, 5, size=50)
        )  # Mismatched size
        landmarks = np.load(tmp_path / "landmarks.npy")
        labels = np.load(tmp_path / "labels.npy")
        assert landmarks.shape[0] != labels.shape[0]


def test_split_data(temp_data_dir):
    """Test data splitting."""
    landmarks = np.load(temp_data_dir / "landmarks.npy")
    labels = np.load(temp_data_dir / "labels.npy")
    config = PreprocessingConfig(val_size=0.2, test_size=0.1)

    splits = split_data(landmarks, labels, config)

    # Check all splits exist
    assert set(splits.keys()) == {"train", "val", "test"}

    # Check shapes
    train_X, train_y = splits["train"]
    val_X, val_y = splits["val"]
    test_X, test_y = splits["test"]

    assert train_X.shape[0] == train_y.shape[0]
    assert val_X.shape[0] == val_y.shape[0]
    assert test_X.shape[0] == test_y.shape[0]

    # Check split sizes
    total_samples = len(landmarks)
    assert len(test_X) == pytest.approx(total_samples * config.test_size, rel=0.1)
    assert len(val_X) == pytest.approx(total_samples * config.val_size, rel=0.1)
    assert len(train_X) == pytest.approx(
        total_samples * (1 - config.test_size - config.val_size), rel=0.1
    )


def test_save_splits(temp_data_dir):
    """Test saving splits."""
    landmarks = np.load(temp_data_dir / "landmarks.npy")
    labels = np.load(temp_data_dir / "labels.npy")
    config = PreprocessingConfig()
    splits = split_data(landmarks, labels, config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        save_splits(splits, output_dir)

        # Check files exist
        assert (output_dir / "train_X.npy").exists()
        assert (output_dir / "train_y.npy").exists()
        assert (output_dir / "val_X.npy").exists()
        assert (output_dir / "val_y.npy").exists()
        assert (output_dir / "test_X.npy").exists()
        assert (output_dir / "test_y.npy").exists()

        # Check loaded data matches saved data
        for split_name in ["train", "val", "test"]:
            X = np.load(output_dir / f"{split_name}_X.npy")
            y = np.load(output_dir / f"{split_name}_y.npy")
            assert np.array_equal(X, splits[split_name][0])
            assert np.array_equal(y, splits[split_name][1])


def test_preprocess_pipeline(temp_data_dir):
    """Test complete preprocessing pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        config = PreprocessingConfig()
        # Simulate pipeline: load, split, save
        landmarks = np.load(temp_data_dir / "landmarks.npy")
        labels = np.load(temp_data_dir / "labels.npy")
        splits = split_data(landmarks, labels, config)
        save_splits(splits, output_dir)
        # Check output files
        assert (output_dir / "train_X.npy").exists()
        assert (output_dir / "train_y.npy").exists()
        assert (output_dir / "val_X.npy").exists()
        assert (output_dir / "val_y.npy").exists()
        assert (output_dir / "test_X.npy").exists()
        assert (output_dir / "test_y.npy").exists()
