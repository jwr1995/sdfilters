# Download librispeech clean sets
# output data table as csv with headers: filename, 
import os
import subprocess

from scripts.constants import LIBRISPEECH_DIR

def download_librispeech_clean(output_dir: str = None, splits: list = None):
    if output_dir is None:
        output_dir = LIBRISPEECH_DIR
    if splits is None:
        splits = ['dev-clean']  # Default to smallest split
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    available_splits = {
        'train-clean-100': 'train-clean-100.tar.gz',
        'train-clean-360': 'train-clean-360.tar.gz', 
        'train-other-500': 'train-other-500.tar.gz',
        'dev-clean': 'dev-clean.tar.gz',
        'dev-other': 'dev-other.tar.gz',
        'test-clean': 'test-clean.tar.gz',
        'test-other': 'test-other.tar.gz'
    }
    
    for split in splits:
        if split not in available_splits:
            print(f"Split {split} not available. Skipping.")
            continue
            
        filename = available_splits[split]
        url = f"https://www.openslr.org/resources/12/{filename}"
        
        print(f"Downloading {split}...")
        subprocess.run(["wget", url, "-P", output_dir])
        
        print(f"Extracting {filename}...")
        subprocess.run(["tar", "-xzf", os.path.join(output_dir, filename), "-C", output_dir])
        
        # Remove archive after extraction
        os.remove(os.path.join(output_dir, filename))
    
    print("Librispeech clean sets downloaded and extracted.")

if __name__ == "__main__":
    download_librispeech_clean()