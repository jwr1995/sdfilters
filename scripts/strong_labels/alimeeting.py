# Download AliMeeting corpus
import os
import subprocess
from pathlib import Path
from tqdm.auto import tqdm

from scripts.constants import ALIMEETING_DIR

def resumable_download(url, filename, force_download=False):
    """Download file using wget with resume capability"""
    if os.path.exists(filename) and not force_download:
        print(f"File {filename} already exists, skipping download")
        return True
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        result = subprocess.run(["wget", "-c", url, "-O", str(filename)], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to download {url}: {result.stderr}")
            return False
        else:
            print(f"Successfully downloaded {filename}")
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def extract_archive(archive_path, extract_dir):
    """Extract tar.gz archive"""
    try:
        result = subprocess.run(["tar", "-xzf", str(archive_path), "-C", str(extract_dir)], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to extract {archive_path}: {result.stderr}")
            return False
        else:
            print(f"Successfully extracted {archive_path}")
            return True
    except Exception as e:
        print(f"Failed to extract {archive_path}: {e}")
        return False

def download_alimeeting(output_dir: str = ALIMEETING_DIR, force_download=False):
    """Download AliMeeting corpus
    
    Downloads from OpenSLR with direct URLs found at https://www.openslr.org/119/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AliMeeting corpus URLs
    base_url = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr"
    
    files_to_download = [
        ("Train_Ali_far.tar.gz", "Train set - 8-channel microphone array speech"),
        ("Train_Ali_near.tar.gz", "Train set - headset microphone speech"),
        ("Eval_Ali.tar.gz", "Evaluation set (8-channel and headset mics)"),
        ("Test_Ali.tar.gz", "Test set (8-channel and headset mics)")
    ]
    
    print(f"Downloading AliMeeting corpus to {output_dir}")
    print("Source: https://www.openslr.org/119/")
    print()
    
    for filename, description in tqdm(files_to_download, desc="Downloading AliMeeting"):
        url = f"{base_url}/{filename}"
        archive_path = output_dir / filename
        
        print(f"Downloading {description}...")
        success = resumable_download(url, archive_path, force_download)
        
        if success and archive_path.exists():
            print(f"Extracting {filename}...")
            extract_success = extract_archive(archive_path, output_dir)
            
            if extract_success:
                # Optionally remove archive after extraction
                # archive_path.unlink()
                print(f"Extraction of {filename} complete")
            else:
                print(f"Extraction of {filename} failed")
        else:
            print(f"Download of {filename} failed")
    
    print(f"\nAliMeeting download process completed.")
    print(f"Files saved to: {output_dir}")
    
    # List what was downloaded
    if output_dir.exists():
        extracted_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if extracted_dirs:
            print(f"Extracted directories: {[d.name for d in extracted_dirs]}")
    
    return output_dir

if __name__ == "__main__":
    download_alimeeting()