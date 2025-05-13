from pathlib import Path
from mp_drone_control.models.trainer import evaluate_model
from mp_drone_control.utils.logging_config import setup_logging
import sys

logger = setup_logging()

if __name__ == "__main__":
    data_dir = Path("data/processed/asl_digits")
    results = {}
    for model_name in ["small", "large"]:
        ckpt = Path(f"models/{model_name}_model.pth")
        logger.info(f"Evaluating {model_name} model...")
        acc, f1 = evaluate_model(model_name, ckpt, data_dir, split="test")
        results[model_name] = {"accuracy": acc, "f1": f1}

    # Write comparison to file
    comparison_path = Path("models/model_comparison.txt")
    with open(comparison_path, "w") as f:
        f.write("Model Comparison (Test Set)\n")
        f.write("============================\n")
        for model_name in ["small", "large"]:
            f.write(f"{model_name.capitalize()} Model:\n")
            f.write(f"  Accuracy: {results[model_name]['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {results[model_name]['f1']:.4f}\n\n")
    logger.info(f"Comparison written to {comparison_path}")
