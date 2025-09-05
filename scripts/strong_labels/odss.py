# Download ODSS (Open Dataset of Synthetic Speech) 
import os
import subprocess

from scripts.constants import ODSS_DIR

def download_odss(output_dir: str = ODSS_DIR):
    """Download ODSS dataset for synthetic speech detection"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ODSS dataset from Zenodo
    url = "https://zenodo.org/records/8370669/files/odss.zip"
    archive_name = "odss.zip"
    
    print("Downloading ODSS dataset...")
    subprocess.run(["wget", url, "-P", output_dir])
    
    print("Extracting ODSS dataset...")
    subprocess.run(["unzip", os.path.join(output_dir, archive_name), "-d", output_dir])
    
    # Remove zip file after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("ODSS dataset download complete.")

if __name__ == "__main__":
    download_odss()