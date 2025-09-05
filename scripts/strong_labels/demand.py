# Download DEMAND (Diverse Environments Multi-channel Acoustic Noise Database)
import os
import subprocess

from scripts.constants import DEMAND_DIR

def download_demand(output_dir: str = DEMAND_DIR):
    """Download DEMAND noise corpus"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # DEMAND corpus from Zenodo API
    url = "https://zenodo.org/api/records/1227121/files-archive"
    archive_name = "demand_files.zip"
    
    print("Downloading DEMAND corpus...")
    subprocess.run(["wget", url, "-O", os.path.join(output_dir, archive_name)])
    
    print("Extracting DEMAND corpus...")
    subprocess.run(["unzip", os.path.join(output_dir, archive_name), "-d", output_dir])
    
    # Remove archive after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("DEMAND corpus download complete.")

if __name__ == "__main__":
    download_demand()