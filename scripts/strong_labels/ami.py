# Download AMI corpus with audio files
import os
import itertools
import subprocess
from pathlib import Path
from tqdm.auto import tqdm

from scripts.constants import AMI_DIR

def resumable_download(url, filename, force_download=False):
    """Download file using wget with resume capability"""
    if os.path.exists(filename) and not force_download:
        print(f"File {filename} already exists, skipping download")
        return
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        result = subprocess.run(["wget", "-c", url, "-O", str(filename)], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to download {url}: {result.stderr}")
        else:
            print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Complete AMI meetings list from dump.py
MEETINGS = {
    'EN2001': ['EN2001a', 'EN2001b', 'EN2001d', 'EN2001e'],
    'EN2002': ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d'],
    'EN2003': ['EN2003a'],
    'EN2004': ['EN2004a'],
    'EN2005': ['EN2005a'],
    'EN2006': ['EN2006a','EN2006b'],
    'EN2009': ['EN2009b','EN2009c','EN2009d'],
    'ES2002': ['ES2002a','ES2002b','ES2002c','ES2002d'],
    'ES2003': ['ES2003a','ES2003b','ES2003c','ES2003d'],
    'ES2004': ['ES2004a','ES2004b','ES2004c','ES2004d'],
    'ES2005': ['ES2005a','ES2005b','ES2005c','ES2005d'],
    'ES2006': ['ES2006a','ES2006b','ES2006c','ES2006d'],
    'ES2007': ['ES2007a','ES2007b','ES2007c','ES2007d'],
    'ES2008': ['ES2008a','ES2008b','ES2008c','ES2008d'],
    'ES2009': ['ES2009a','ES2009b','ES2009c','ES2009d'],
    'ES2010': ['ES2010a','ES2010b','ES2010c','ES2010d'],
    'ES2011': ['ES2011a','ES2011b','ES2011c','ES2011d'],
    'ES2012': ['ES2012a','ES2012b','ES2012c','ES2012d'],
    'ES2013': ['ES2013a','ES2013b','ES2013c','ES2013d'],
    'ES2014': ['ES2014a','ES2014b','ES2014c','ES2014d'],
    'ES2015': ['ES2015a','ES2015b','ES2015c','ES2015d'],
    'ES2016': ['ES2016a','ES2016b','ES2016c','ES2016d'],
    'IB4001': ['IB4001'],
    'IB4002': ['IB4002'],
    'IB4003': ['IB4003'],
    'IB4004': ['IB4004'],
    'IB4005': ['IB4005'],
    'IB4010': ['IB4010'],
    'IB4011': ['IB4011'],
    'IN1001': ['IN1001'],
    'IN1002': ['IN1002'],
    'IN1005': ['IN1005'],
    'IN1007': ['IN1007'],
    'IN1008': ['IN1008'],
    'IN1009': ['IN1009'],
    'IN1012': ['IN1012'],
    'IN1013': ['IN1013'],
    'IN1014': ['IN1014'],
    'IN1016': ['IN1016'],
    'IS1000': ['IS1000a','IS1000b','IS1000c','IS1000d'],
    'IS1001': ['IS1001a','IS1001b','IS1001c','IS1001d'],
    'IS1002': ['IS1002b','IS1002c','IS1002d'],
    'IS1003': ['IS1003a','IS1003b','IS1003c','IS1003d'],
    'IS1004': ['IS1004a','IS1004b','IS1004c','IS1004d'],
    'IS1005': ['IS1005a','IS1005b','IS1005c'],
    'IS1006': ['IS1006a','IS1006b','IS1006c','IS1006d'],
    'IS1007': ['IS1007a','IS1007b','IS1007c','IS1007d'],
    'IS1008': ['IS1008a','IS1008b','IS1008c','IS1008d'],
    'IS1009': ['IS1009a','IS1009b','IS1009c','IS1009d'],
    'TS3003': ['TS3003a','TS3003b','TS3003c','TS3003d'],
}

def download_audio(target_dir, force_download=False, url="http://groups.inf.ed.ac.uk/ami", mic="ihm"):
    """Download AMI audio files for all meetings"""
    target_dir = Path(target_dir)
    
    # Download audio for all meetings
    for item in tqdm(
        itertools.chain.from_iterable(MEETINGS.values()),
        desc="Downloading AMI meetings",
    ):
        if mic == "ihm":
            # Individual headset microphones (4 or 5 channels)
            headset_num = 5 if item in ("EN2001a", "EN2001d", "EN2001e") else 4
            for m in range(headset_num):
                wav_name = f"{item}.Headset-{m}.wav"
                wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
                wav_dir = target_dir / "wav_db" / item / "audio"
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / wav_name
                resumable_download(
                    wav_url,
                    filename=wav_path,
                    force_download=force_download,
                )
        elif mic == "ihm-mix":
            wav_name = f"{item}.Mix-Headset.wav"
            wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
            wav_dir = target_dir / "wav_db" / item / "audio"
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / wav_name
            resumable_download(
                wav_url, filename=wav_path, force_download=force_download
            )
        elif mic == "sdm":
            # Single distant microphone
            wav_name = f"{item}.Array1-01.wav"
            wav_url = f"{url}/AMICorpusMirror/amicorpus/{item}/audio/{wav_name}"
            wav_dir = target_dir / "wav_db" / item / "audio"
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / wav_name
            resumable_download(
                wav_url, filename=wav_path, force_download=force_download
            )

def download_ami(target_dir=AMI_DIR, annotations=None, force_download=False, url="http://groups.inf.ed.ac.uk/ami", mic="ihm"):
    """Download AMI Meeting Corpus with annotations and audio"""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    annotations = (
        target_dir / "ami_public_manual_1.6.2.zip" if not annotations else annotations
    )

    # Download audio files
    print(f"Downloading AMI audio files (mic: {mic})...")
    download_audio(target_dir, force_download, url, mic)
    print("AMI audio download complete")

    # Download annotations
    print("Downloading AMI annotations...")
    if annotations.exists() and not force_download:
        print(f"Skip downloading annotations as they exist in: {annotations}")
        return target_dir
    
    annotations_url = f"{url}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    resumable_download(annotations_url, annotations, force_download=force_download)
    
    print("AMI corpus setup complete.")
    return target_dir

if __name__ == "__main__":
    download_ami()