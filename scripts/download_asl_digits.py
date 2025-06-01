import zipfile
import requests
import shutil
from pathlib import Path
from mp_drone_control.utils.logging_config import setup_logging

logger = setup_logging()


def download_and_extract(url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "asl_digits.zip"
    temp_dir = output_dir / "temp_extract"

    if not zip_path.exists():
        logger.info(f"Downloading dataset to {zip_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        logger.info("Zip already exists. Skipping download.")

    logger.info("Unzipping...")
    # Extract to a temporary directory first
    temp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Move the Dataset directory to the correct location
    extracted_dir = next(temp_dir.glob("Sign-Language-Digits-Dataset-*"))
    dataset_dir = extracted_dir / "Dataset"
    if dataset_dir.exists():
        target_dir = output_dir / "Dataset"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(dataset_dir), str(target_dir))
        logger.info(f"Dataset organized in {target_dir}")
    else:
        logger.error("Could not find Dataset directory in extracted files")

    # Clean up temporary directory and zip file
    shutil.rmtree(temp_dir)
    zip_path.unlink()


if __name__ == "__main__":
    DATA_URL = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
    download_and_extract(DATA_URL, Path("data/raw/asl_digits"))
