# Download Multilingual LibriSpeech (MLS) corpus
import os
import subprocess

from scripts.constants import MLS_DIR

def download_mls(output_dir: str = MLS_DIR, languages: list = None):
    """Download MLS corpus for specified languages"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default to Italian, Portuguese, Polish (smaller datasets)
    if languages is None:
        languages = ['italian', 'portuguese', 'polish']
    
    base_url = "https://dl.fbaipublicfiles.com/mls"
    
    available_languages = {
        'english': 'english',
        'german': 'german', 
        'dutch': 'dutch',
        'french': 'french',
        'spanish': 'spanish',
        'italian': 'italian',
        'portuguese': 'portuguese',
        'polish': 'polish'
    }
    
    for lang in languages:
        if lang not in available_languages:
            print(f"Language {lang} not available. Skipping.")
            continue
            
        lang_name = available_languages[lang]
        lang_dir = os.path.join(output_dir, lang_name)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Download complete language package
        archive_name = f"mls_{lang_name}.tar.gz"
        url = f"{base_url}/{archive_name}"
        
        print(f"Downloading MLS {lang} dataset...")
        result = subprocess.run(["wget", url, "-P", lang_dir], capture_output=True)
        
        archive_path = os.path.join(lang_dir, archive_name)
        if result.returncode == 0 and os.path.exists(archive_path):
            print(f"Extracting {archive_name}...")
            subprocess.run(["tar", "-xzf", archive_path, "-C", lang_dir])
            
            # Remove archive after extraction
            os.remove(archive_path)
            print(f"MLS {lang} download complete.")
        else:
            print(f"Download failed for MLS {lang}, skipping extraction")
    
    print("MLS download complete.")

if __name__ == "__main__":
    download_mls()