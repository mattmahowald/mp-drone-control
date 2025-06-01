# Project Context: Data Preprocessing & Splitting

- The data preprocessing and stratified splitting logic is implemented in `src/mp_drone_control/data/preprocess.py`.
- This module provides:
  - `PreprocessingConfig`: Configuration for split sizes and reproducibility.
  - `split_data`: Stratified train/val/test split using scikit-learn.
  - `save_splits`: Save splits to disk as .npy files.
- The main data processing script (`scripts/process_asl_data.py`) uses this module for all splitting and saving operations.
- All logic is tested and reusable for future pipelines or experiments.

# CONTEXT.md

- Both small (LandmarkClassifier) and large (LargeLandmarkClassifier) MLP models are implemented in src/mp_drone_control/models/mobilenet.py.
- Training for both models is automated via scripts/train.py, which saves checkpoints to models/.
- Evaluation for both models is automated via scripts/evaluate.py, which writes a comparison summary to models/model_comparison.txt.
- Model selection is handled via a model_name argument ("small" or "large").
- Next steps: export both models to ONNX for mobile deployment, and benchmark on a mobile device or emulator.
