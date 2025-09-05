#!/usr/bin/env python3
"""
Extract DEMAND dataset and create 30-second chunks

This script extracts the DEMAND noise dataset from zip archives and creates
30-second segments suitable for training. The DEMAND dataset contains various
environmental noise types recorded in 17 different acoustic environments.

Usage:
    python scripts/extract_and_chunk_demand.py
    
Output:
    - Extracts 16kHz WAV files from DEMAND zip archives
    - Creates annotations for 30-second chunks 
    - Generates 2,720 total chunks (22.7 hours) across 17 environments
"""
import os
import glob
import zipfile
import subprocess
from pathlib import Path
from scripts.constants import DEMAND_DIR, DEMAND_CHUNKED_DIR, OUTPUT_DIR, ANNOTATIONS_DIR
from scripts.generate_annotations import generate_demand_chunked_annotations

def extract_demand_archives():
    """Extract all DEMAND zip files to a temporary directory"""
    demand_path = Path(DEMAND_DIR)
    extract_path = demand_path / "extracted"
    extract_path.mkdir(exist_ok=True)
    
    # Find all 16k zip files (better for training than 48k)
    zip_files = list(demand_path.glob("*_16k.zip"))
    
    print(f"Found {len(zip_files)} DEMAND 16kHz archives to extract")
    
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            print(f"Failed to extract {zip_file}: {e}")
    
    print(f"Extraction complete to {extract_path}")
    return extract_path

def create_demand_chunked():
    """Extract DEMAND files and create chunked annotations"""
    # Extract archives
    extracted_path = extract_demand_archives()
    
    # Generate chunked annotations using the extracted files
    output_file = os.path.join(ANNOTATIONS_DIR, 'demand-chunked_annotations.csv')
    generate_demand_chunked_annotations(str(extracted_path), output_file, chunk_length=30.0)
    
    print(f"DEMAND-chunked dataset created successfully")
    print(f"Annotations saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    create_demand_chunked()