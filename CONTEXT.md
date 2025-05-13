# Project Context: Data Preprocessing & Splitting

- The data preprocessing and stratified splitting logic is implemented in `src/mp_drone_control/data/preprocess.py`.
- This module provides:
  - `PreprocessingConfig`: Configuration for split sizes and reproducibility.
  - `split_data`: Stratified train/val/test split using scikit-learn.
  - `save_splits`: Save splits to disk as .npy files.
- The main data processing script (`scripts/process_asl_data.py`) uses this module for all splitting and saving operations.
- All logic is tested and reusable for future pipelines or experiments.
