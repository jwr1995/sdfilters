# Download OpenMIC-2018 music dataset
import os
import subprocess

from scripts.constants import OPENMIC_2018_DIR

def download_openmic2018(output_dir: str = OPENMIC_2018_DIR):
    """Download OpenMIC-2018 dataset"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # OpenMIC-2018 dataset from Zenodo
    url = "https://zenodo.org/records/1432913/files/openmic-2018-v1.0.0.tgz"
    archive_name = "openmic-2018-v1.0.0.tgz"
    
    print("Downloading OpenMIC-2018 dataset...")
    subprocess.run(["wget", url, "-P", output_dir])
    
    print("Extracting OpenMIC-2018 dataset...")
    subprocess.run(["tar", "-xzf", os.path.join(output_dir, archive_name), "-C", output_dir])
    
    # Remove archive after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("OpenMIC-2018 dataset download complete.")

if __name__ == "__main__":
    download_openmic2018()