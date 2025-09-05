# Download FoR (Fake or Real) antispoofing dataset
import os
import subprocess

from scripts.constants import FOR_DIR

def download_for(output_dir: str = FOR_DIR):
    """Download FoR dataset for synthetic speech detection"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FoR dataset archive
    url = "https://bil.eecs.yorku.ca/share/for-original.tar.gz"
    archive_name = "for-original.tar.gz"
    
    print("Downloading FoR dataset...")
    subprocess.run(["wget", url, "-P", output_dir])
    
    print("Extracting FoR dataset...")
    subprocess.run(["tar", "-xzf", os.path.join(output_dir, archive_name), "-C", output_dir])
    
    # Remove archive after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("FoR dataset download complete.")

if __name__ == "__main__":
    download_for()