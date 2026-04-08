import os
import zipfile
import subprocess
from pathlib import Path

def setup_kaggle_and_download():
    """
    Downloads and extracts the FER-2013 dataset using the Kaggle API.
    Assumes kaggle.json is either in ~/.kaggle/ or the KAGGLE_CONFIG_DIR environment variable is set.
    """
    dataset_name = "msambare/fer2013"
    download_dir = Path("data")
    zip_file = "fer2013.zip"

    # 1. Create data directory
    if not download_dir.exists():
        download_dir.mkdir(parents=True)
        print(f"Created directory: {download_dir}")

    # 2. Check if data is already extracted
    if (download_dir / "train").exists() and (download_dir / "test").exists():
        print("Dataset already exists in 'data/'. Skipping download.")
        return

    # 3. Download dataset using Kaggle API
    print(f"Downloading {dataset_name} from Kaggle...")
    try:
        # Note: If kaggle.json is missing, this will fail. 
        # In Colab, the user must upload kaggle.json first.
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure kaggle.json is properly configured.")
        return
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please run 'pip install kaggle'.")
        return

    # 4. Extract dataset
    print(f"Extracting {zip_file} to {download_dir}...")
    if not os.path.exists(zip_file):
        # Check if it was downloaded with a different name or to current dir
        if os.path.exists("fer2013.zip"):
            zip_file = "fer2013.zip"
        else:
            print(f"Error: {zip_file} not found after download.")
            return

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    # 5. Clean up zip file
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print(f"Removed {zip_file}")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    setup_kaggle_and_download()
