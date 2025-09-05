# Download MUSAN (Music, Speech, and Noise) corpus
import os
import subprocess

from scripts.constants import MUSAN_DIR

def download_musan(output_dir: str = MUSAN_DIR):
    """Download MUSAN corpus"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # MUSAN corpus from OpenSLR
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    archive_name = "musan.tar.gz"
    
    print("Downloading MUSAN corpus...")
    subprocess.run(["wget", url, "-P", output_dir])
    
    print("Extracting MUSAN corpus...")
    subprocess.run(["tar", "-xzf", os.path.join(output_dir, archive_name), "-C", output_dir])
    
    # Remove archive after extraction
    os.remove(os.path.join(output_dir, archive_name))
    
    print("MUSAN corpus download complete.")

if __name__ == "__main__":
    download_musan()