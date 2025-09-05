# Download FSDNoisy18k dataset
import os
import subprocess

from scripts.constants import FSDNOISY18K_DIR

def download_fsdnoisy18k(output_dir: str = FSDNOISY18K_DIR):
    """Download FSDNoisy18k dataset"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FSDNoisy18k dataset from Zenodo API
    url = "https://zenodo.org/api/records/2529934/files-archive"
    archive_name = "fsdnoisy18k_files.zip"
    
    print("Downloading FSDNoisy18k dataset...")
    subprocess.run(["wget", url, "-O", os.path.join(output_dir, archive_name)])
    
    print("Extracting FSDNoisy18k dataset...")
    subprocess.run(["unzip", os.path.join(output_dir, archive_name), "-d", output_dir])
    
    # Remove archive after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("FSDNoisy18k dataset download complete.")

if __name__ == "__main__":
    download_fsdnoisy18k()