#!/usr/bin/env python3
"""
AITW Dataset Preparation Script

This script downloads and prepares the Audio in the Wild (AITW) dataset for use with
the Whilter model. The dataset contains filemaps that reference YODAS and EMILIA files
which need to be downloaded separately.

Dataset source: https://zenodo.org/api/records/15534662/files-archive

Usage:
    python scripts/aitw_dataset.py --output_dir /path/to/aitw_dataset
    python scripts/aitw_dataset.py --output_dir /path/to/aitw_dataset --download_audio
"""

import os
import sys
import json
import argparse
import requests
import zipfile
import shutil
import tarfile
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import logging
import csv
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset configuration
ZENODO_RECORD_ID = "15534662"
ZENODO_API_BASE = "https://zenodo.org/api/records"
DATASET_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files-archive"


class AITWDatasetDownloader:
    """Handles downloading and preparation of the AITW dataset."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "aitw_dataset"
        self.filemaps_dir = self.dataset_dir / "filemaps"
        self.audio_dir = self.dataset_dir / "audio"
        self.temp_dir = self.dataset_dir / "temp"
        
    def create_directories(self):
        """Create necessary directories."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.filemaps_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories in {self.output_dir}")
        
    def download_dataset(self) -> bool:
        """Download the AITW dataset from Zenodo."""
        logger.info(f"Downloading AITW dataset from {DATASET_URL}")
        
        try:
            # Download the dataset
            response = requests.get(DATASET_URL, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            zip_path = self.output_dir / "aitw_dataset.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded dataset to {zip_path}")
            
            # Extract the dataset
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
            
            # Clean up zip file
            zip_path.unlink()
            logger.info("Dataset extraction completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def parse_yodas_filemap(self, csv_path: Path) -> Dict[str, List[Dict]]:
        """Parse YODAS filemap to group by archive file."""
        archive_groups = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    archive_url = row['Original File Path']
                    yodas_filename = row['YODAS File Name']
                    aitw_path = row['AITW File Path']
                    start_time = float(row['Timestamp Start'])
                    end_time = float(row['Timestamp End'])
                    duration = float(row['Duration'])
                    
                    if archive_url not in archive_groups:
                        archive_groups[archive_url] = []
                    
                    archive_groups[archive_url].append({
                        'yodas_filename': yodas_filename,
                        'aitw_path': aitw_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
                            
        except Exception as e:
            logger.error(f"Failed to parse YODAS filemap {csv_path}: {e}")
            
        return archive_groups
    
    def parse_emilia_filemap(self, csv_path: Path) -> Dict[str, List[Dict]]:
        """Parse EMILIA filemap to group by YouTube URL."""
        youtube_groups = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    youtube_url = row['Original File Path']
                    aitw_path = row['AITW File Path']
                    
                    # Handle empty timestamp values
                    try:
                        start_time = float(row['Timestamp Start']) if row['Timestamp Start'].strip() else 0.0
                        end_time = float(row['Timestamp End']) if row['Timestamp End'].strip() else 0.0
                        duration = float(row['Duration']) if row['Duration'].strip() else 0.0
                    except (ValueError, KeyError):
                        logger.warning(f"Skipping row with invalid timestamps: {aitw_path}")
                        continue
                    
                    # Skip rows with invalid timestamps
                    if start_time == 0.0 and end_time == 0.0:
                        logger.warning(f"Skipping row with zero timestamps: {aitw_path}")
                        continue
                    
                    if youtube_url not in youtube_groups:
                        youtube_groups[youtube_url] = []
                    
                    youtube_groups[youtube_url].append({
                        'aitw_path': aitw_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
                            
        except Exception as e:
            logger.error(f"Failed to parse EMILIA filemap {csv_path}: {e}")
            
        return youtube_groups
    
    def download_and_extract_yodas_archive(self, archive_url: str, segments: List[Dict]) -> int:
        """Download YODAS archive, extract specific files, segment them, and cleanup."""
        downloaded_count = 0
        
        try:
            # Create temporary directory for this archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download archive
                logger.info(f"Downloading YODAS archive: {archive_url}")
                archive_path = temp_path / "archive.tar.gz"
                
                response = requests.get(archive_url, stream=True)
                response.raise_for_status()
                
                with open(archive_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extract archive
                logger.info(f"Extracting archive...")
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                # Process each segment
                for segment in segments:
                    yodas_filename = segment['yodas_filename']
                    aitw_path = segment['aitw_path']
                    start_time = segment['start_time']
                    end_time = segment['end_time']
                    
                    # Find the audio file in extracted archive
                    audio_file = None
                    for extracted_file in temp_path.rglob(yodas_filename):
                        audio_file = extracted_file
                        break
                    
                    if audio_file and audio_file.exists():
                        # Create output directory
                        output_path = self.audio_dir / aitw_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extract segment using ffmpeg
                        if self.extract_audio_segment(audio_file, output_path, start_time, end_time):
                            downloaded_count += 1
                            logger.info(f"  ✓ Extracted: {aitw_path}")
                        else:
                            logger.error(f"  ✗ Failed to extract: {aitw_path}")
                    else:
                        logger.warning(f"  ⚠ File not found in archive: {yodas_filename}")
                
        except Exception as e:
            logger.error(f"Failed to process YODAS archive {archive_url}: {e}")
        
        return downloaded_count
    
    def download_and_extract_youtube_video(self, youtube_url: str, segments: List[Dict]) -> int:
        """Download YouTube video, extract audio, segment it, and cleanup."""
        downloaded_count = 0
        
        try:
            # Create temporary directory for this video
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download video using yt-dlp
                logger.info(f"Downloading YouTube video: {youtube_url}")
                video_path = temp_path / "video.mp4"
                
                cmd = [
                    'yt-dlp',
                    '--extract-audio',
                    '--audio-format', 'wav',
                    '--audio-quality', '0',
                    '--output', str(video_path),
                    youtube_url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
                if result.returncode != 0:
                    logger.error(f"Failed to download YouTube video: {result.stderr}")
                    return 0
                
                # Find the downloaded audio file
                audio_file = None
                for file in temp_path.glob("*.wav"):
                    audio_file = file
                    break
                
                if audio_file and audio_file.exists():
                    # Process each segment
                    for segment in segments:
                        aitw_path = segment['aitw_path']
                        start_time = segment['start_time']
                        end_time = segment['end_time']
                        
                        # Create output directory
                        output_path = self.audio_dir / aitw_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Extract segment using ffmpeg
                        if self.extract_audio_segment(audio_file, output_path, start_time, end_time):
                            downloaded_count += 1
                            logger.info(f"  ✓ Extracted: {aitw_path}")
                        else:
                            logger.error(f"  ✗ Failed to extract: {aitw_path}")
                else:
                    logger.error(f"Failed to find downloaded audio file for {youtube_url}")
                
        except Exception as e:
            logger.error(f"Failed to process YouTube video {youtube_url}: {e}")
        
        return downloaded_count
    
    def extract_audio_segment(self, input_file: Path, output_file: Path, start_time: float, end_time: float) -> bool:
        """Extract audio segment using ffmpeg."""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:a', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',  # Overwrite output file
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to extract audio segment: {e}")
            return False
    
    def process_filemaps(self, download_audio: bool = False, max_sources: Optional[int] = None) -> Dict[str, int]:
        """Process all filemaps and optionally download audio files."""
        stats = {
            'yodas_archives': 0,
            'youtube_videos': 0,
            'files_downloaded': 0,
            'files_failed': 0
        }
        
        # Look for CSV filemaps in the main dataset directory
        yodas_csv = self.dataset_dir / "yodas_file_map.csv"
        emilia_csv = self.dataset_dir / "emilia_file_map.csv"
        
        if not yodas_csv.exists() or not emilia_csv.exists():
            logger.error("Required CSV filemaps not found")
            return stats
        
        if not download_audio:
            # Just count sources
            yodas_groups = self.parse_yodas_filemap(yodas_csv)
            emilia_groups = self.parse_emilia_filemap(emilia_csv)
            
            stats['yodas_archives'] = len(yodas_groups)
            stats['youtube_videos'] = len(emilia_groups)
            logger.info(f"Found {len(yodas_groups)} YODAS archives and {len(emilia_groups)} YouTube videos")
            return stats
        
        # Process YODAS archives
        logger.info("Processing YODAS archives...")
        yodas_groups = self.parse_yodas_filemap(yodas_csv)
        
        yodas_sources = list(yodas_groups.items())
        if max_sources:
            yodas_sources = yodas_sources[:max_sources]
        
        for i, (archive_url, segments) in enumerate(yodas_sources):
            logger.info(f"Processing YODAS archive {i+1}/{len(yodas_sources)}: {archive_url}")
            downloaded = self.download_and_extract_yodas_archive(archive_url, segments)
            stats['files_downloaded'] += downloaded
            stats['yodas_archives'] += 1
            
            # Clean up temp files
            self.cleanup_temp_files()
        
        # Process EMILIA YouTube videos
        logger.info("Processing EMILIA YouTube videos...")
        emilia_groups = self.parse_emilia_filemap(emilia_csv)
        
        emilia_sources = list(emilia_groups.items())
        if max_sources:
            emilia_sources = emilia_sources[:max_sources]
        
        for i, (youtube_url, segments) in enumerate(emilia_sources):
            logger.info(f"Processing YouTube video {i+1}/{len(emilia_sources)}: {youtube_url}")
            downloaded = self.download_and_extract_youtube_video(youtube_url, segments)
            stats['files_downloaded'] += downloaded
            stats['youtube_videos'] += 1
            
            # Clean up temp files
            self.cleanup_temp_files()
        
        return stats
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
    
    def validate_dataset(self) -> bool:
        """Validate that the dataset was downloaded and extracted correctly."""
        required_files = [
            self.dataset_dir / "yodas_file_map.csv",
            self.dataset_dir / "emilia_file_map.csv"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        logger.info("Dataset validation passed. Found YODAS and EMILIA filemaps.")
        return True
    
    def get_dataset_info(self) -> Dict:
        """Get information about the downloaded dataset."""
        info = {
            'dataset_path': str(self.dataset_dir),
            'yodas_archives': 0,
            'youtube_videos': 0,
            'audio_files_count': 0,
            'total_audio_size_gb': 0
        }
        
        # Count sources
        yodas_csv = self.dataset_dir / "yodas_file_map.csv"
        emilia_csv = self.dataset_dir / "emilia_file_map.csv"
        
        if yodas_csv.exists():
            yodas_groups = self.parse_yodas_filemap(yodas_csv)
            info['yodas_archives'] = len(yodas_groups)
        
        if emilia_csv.exists():
            emilia_groups = self.parse_emilia_filemap(emilia_csv)
            info['youtube_videos'] = len(emilia_groups)
        
        # Count audio files and calculate size
        if self.audio_dir.exists():
            audio_files = list(self.audio_dir.rglob("*.wav"))
            info['audio_files_count'] = len(audio_files)
            
            total_size = sum(f.stat().st_size for f in audio_files)
            info['total_audio_size_gb'] = round(total_size / (1024**3), 2)
        
        return info


def main():
    """Main function to run the AITW dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the AITW dataset for Whilter model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/aitw_dataset.py --output_dir ./datasets/aitw
  python scripts/aitw_dataset.py --output_dir ./datasets/aitw --download_audio
  python scripts/aitw_dataset.py --output_dir ./datasets/aitw --download_audio --max_sources 10
  python scripts/aitw_dataset.py --output_dir ./datasets/aitw --info
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for the AITW dataset'
    )
    
    parser.add_argument(
        '--download_audio',
        action='store_true',
        help='Download and process audio files (requires ffmpeg and yt-dlp)'
    )
    
    parser.add_argument(
        '--max_sources',
        type=int,
        help='Maximum number of sources to process (for testing)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information without downloading'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset already exists'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = AITWDatasetDownloader(args.output_dir)
    
    # Show info only
    if args.info:
        if downloader.dataset_dir.exists():
            info = downloader.get_dataset_info()
            print("\nAITW Dataset Information:")
            print(f"  Dataset path: {info['dataset_path']}")
            print(f"  YODAS archives: {info['yodas_archives']}")
            print(f"  YouTube videos: {info['youtube_videos']}")
            print(f"  Audio files: {info['audio_files_count']}")
            print(f"  Total audio size: {info['total_audio_size_gb']} GB")
        else:
            print("Dataset not found. Run without --info to download.")
        return
    
    # Create directories
    downloader.create_directories()
    
    # Check if dataset already exists
    if downloader.dataset_dir.exists() and not args.force:
        logger.info("Dataset already exists. Use --force to re-download.")
        if downloader.validate_dataset():
            info = downloader.get_dataset_info()
            logger.info(f"Dataset ready: {info['yodas_archives']} YODAS archives, {info['youtube_videos']} YouTube videos")
            
            # If audio files are requested, process filemaps even if dataset exists
            if args.download_audio:
                logger.info("Processing filemaps for audio download...")
                stats = downloader.process_filemaps(download_audio=True, max_sources=args.max_sources)
                
                # Print summary
                print("\nAITW Dataset Preparation Summary:")
                print(f"  YODAS archives processed: {stats['yodas_archives']}")
                print(f"  YouTube videos processed: {stats['youtube_videos']}")
                print(f"  Files downloaded: {stats['files_downloaded']}")
                print(f"  Files failed: {stats['files_failed']}")
                
                # Show final dataset info
                info = downloader.get_dataset_info()
                print(f"\nDataset location: {info['dataset_path']}")
                print(f"Total audio size: {info['total_audio_size_gb']} GB")
                print("\nDataset preparation completed successfully!")
        return
    
    # Download dataset
    if not downloader.download_dataset():
        logger.error("Failed to download dataset")
        sys.exit(1)
    
    # Validate dataset
    if not downloader.validate_dataset():
        logger.error("Dataset validation failed")
        sys.exit(1)
    
    # Process filemaps and optionally download audio
    stats = downloader.process_filemaps(download_audio=args.download_audio, max_sources=args.max_sources)
    
    # Print summary
    print("\nAITW Dataset Preparation Summary:")
    print(f"  YODAS archives: {stats['yodas_archives']}")
    print(f"  YouTube videos: {stats['youtube_videos']}")
    
    if args.download_audio:
        print(f"  Files downloaded: {stats['files_downloaded']}")
        print(f"  Files failed: {stats['files_failed']}")
    else:
        print("  Audio files not downloaded (use --download_audio to download)")
    
    # Show final dataset info
    info = downloader.get_dataset_info()
    print(f"\nDataset location: {info['dataset_path']}")
    print(f"Total audio size: {info['total_audio_size_gb']} GB")
    
    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()



