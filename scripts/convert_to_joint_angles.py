#!/usr/bin/env python3
"""
Convert existing landmark dataset to joint angles representation.
"""

import argparse
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from mp_drone_control.data.joint_angles import process_dataset_to_joint_angles
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Convert landmark dataset to joint angles")
    parser.add_argument("--input-dir", type=Path, default="data/processed/custom",
                        help="Input directory with landmark data")
    parser.add_argument("--output-dir", type=Path, default="data/processed/joint_angles",
                        help="Output directory for joint angle data")
    
    args = parser.parse_args()
    
    logger.info(f"Converting landmark data from {args.input_dir} to joint angles...")
    logger.info(f"Output will be saved to {args.output_dir}")
    
    process_dataset_to_joint_angles(args.input_dir, args.output_dir)
    
    logger.info("Conversion complete!")
    logger.info(f"You can now train models using: --data-dir {args.output_dir}")

if __name__ == "__main__":
    main() 