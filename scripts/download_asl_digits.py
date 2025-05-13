import zipfile
import requests
from pathlib import Path


def download_and_extract(url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "asl_digits.zip"

    if not zip_path.exists():
        print(f"⬇️  Downloading dataset to {zip_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("✅ Zip already exists. Skipping download.")

    print("📦 Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"✅ Dataset extracted to {output_dir}")


if __name__ == "__main__":
    DATA_URL = "https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip"
    download_and_extract(DATA_URL, Path("data/raw/asl_digits"))
