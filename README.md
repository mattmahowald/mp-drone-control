# MP Drone Control

A machine learning project for drone control using MediaPipe and PyTorch.

## Prerequisites

- Python 3.12.3
- Poetry (Python package manager)

## Installing Poetry

### macOS / Linux / WSL

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Windows (PowerShell)

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

After installation, you may need to add Poetry to your PATH. The installer will tell you the exact command to run.

To verify the installation:

```bash
poetry --version
```

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/mp-drone-control.git
   cd mp-drone-control
   ```

2. **Install dependencies using Poetry**

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

## Using Make Commands

This project includes a Makefile with common development commands. Here are some useful commands:

```bash
make help        # Show all available commands
make setup       # Install Poetry and project dependencies
make install     # Install project dependencies
make test        # Run tests
make format      # Format code
make lint        # Run type checking
make clean       # Clean up cache files
make train       # Run training script
make inference   # Run inference script
```

## Project Structure

- `src/mp_drone_control/`: Main source code
  - `inference/`: Code for model inference
  - `models/`: Model architectures and training code
  - `data/`: Data processing utilities
    - `preprocess.py`: Reusable data preprocessing module (stratified splitting, config, saving splits)
  - `evaluation/`: Model evaluation scripts
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `tests/`: Unit tests
- `data/`: Dataset storage
- `wandb/`: Weights & Biases logging directory

## Data Preprocessing & Splitting

The project provides a reusable data preprocessing module at `src/mp_drone_control/data/preprocess.py`:

- `PreprocessingConfig`: Configuration for split sizes and reproducibility.
- `split_data`: Stratified train/val/test split using scikit-learn.
- `save_splits`: Save splits to disk as .npy files.

**Example usage:**

```python
from mp_drone_control.data.preprocess import PreprocessingConfig, split_data, save_splits

config = PreprocessingConfig(val_size=0.15, test_size=0.15, random_state=42)
splits = split_data(landmarks, labels, config)
save_splits(splits, output_dir)
```

The main data processing script (`scripts/process_asl_data.py`) uses this module for all splitting and saving operations.

## Usage

### Training and Evaluating Small & Large Models

This project supports automated training and evaluation of both a small and a large MLP model for hand gesture recognition. Results are summarized for easy comparison.

**Train both models:**

```bash
poetry run python scripts/train.py
```

**Evaluate both models and generate a comparison file:**

```bash
poetry run python scripts/evaluate.py
```

- The evaluation script writes a summary of test accuracy and F1 for both models to `models/model_comparison.txt`.
- Model checkpoints are saved in the `models/` directory as `small_model.pth` and `large_model.pth`.

**Next steps:**

- Export both models to ONNX for mobile deployment.
- Benchmark inference speed and resource usage on a mobile device or emulator.

### Training

1. **Prepare your dataset**

   - Place your training data in the `data/` directory
   - Follow the data format specified in the documentation

2. **Start training**
   ```bash
   python -m src.mp_drone_control.models.train
   ```

### Inference

To run inference with a trained model:

```bash
python -m src.mp_drone_control.inference.run_inference
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Type checking: `mypy .`

## Dependencies

Key dependencies include:

- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Transformers
- Weights & Biases

For a complete list of dependencies, see `pyproject.toml`.

## License

This project is licensed under the terms of the included LICENSE file.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
