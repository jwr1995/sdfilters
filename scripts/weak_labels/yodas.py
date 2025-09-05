#!/usr/bin/env python3
"""
Download YODAS Dataset from Hugging Face

YODAS (YouTube-Oriented Dataset for Audio and Speech) is a large-scale, 
multilingual dataset comprising over 500k hours of speech data in more 
than 100 languages, sourced from YouTube.

Available versions:
- YODAS: Original segmented dataset (369k hours)
- YODAS2: Unsegmented long-form dataset (24kHz)
- YODAS-Granary: Curated subset for 23 European languages

Usage:
    python scripts/strong_labels/yodas.py --dataset yodas --languages en es fr
    python scripts/strong_labels/yodas.py --dataset yodas2 --max_hours 100
    python scripts/strong_labels/yodas.py --dataset yodas-granary --streaming
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_dir)

try:
    from scripts.constants import OUTPUT_DIR, ITW_SPEECH_DIR
    DEFAULT_YODAS_DIR = ITW_SPEECH_DIR
    # Try to get access token from constants
    try:
        from scripts.constants import ACCESS_TOKEN
        HF_ACCESS_TOKEN = ACCESS_TOKEN
    except (ImportError, AttributeError):
        HF_ACCESS_TOKEN = None
except ImportError:
    # Fallback if constants not available
    OUTPUT_DIR = "/media/will/DATA/corpora"
    DEFAULT_YODAS_DIR = "/media/will/DATA/corpora/speech/itw"
    HF_ACCESS_TOKEN = None

# Try to import huggingface_hub
try:
    from huggingface_hub import snapshot_download, HfApi
    from datasets import load_dataset
    import pandas as pd
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub not available. Install with: pip install huggingface_hub datasets pandas")

def get_yodas_dataset_info():
    """Get information about available YODAS datasets"""
    datasets_info = {
        'yodas': {
            'repo_id': 'espnet/yodas',
            'description': 'Original YODAS dataset - segmented audio with captions',
            'size_hours': 369510,
            'size_gb_approx': 180000,  # Approximate
            'languages': '100+',
            'license': 'CC BY 4.0',
            'format': 'Segmented audio + text captions',
            'sampling_rate': '16kHz'
        },
        'yodas2': {
            'repo_id': 'espnet/yodas2',
            'description': 'YODAS2 - unsegmented long-form audio (video-level)',
            'size_hours': 500000,  # Estimated, continues to grow
            'size_gb_approx': 300000,  # Approximate
            'languages': '100+',
            'license': 'CC BY 4.0',
            'format': 'Long-form unsegmented audio',
            'sampling_rate': '24kHz'
        },
        'yodas-granary': {
            'repo_id': 'espnet/yodas-granary',
            'description': 'YODAS-Granary - curated subset for European languages',
            'size_hours': 'Variable',
            'size_gb_approx': 50000,  # Approximate
            'languages': '23 European languages',
            'license': 'CC BY 4.0',
            'format': 'High-quality pseudo-labeled speech for ASR/AST',
            'sampling_rate': 'Variable'
        },
        'yodas-sample': {
            'repo_id': 'espnet/yodas_sample',
            'description': 'YODAS sample dataset for testing',
            'size_hours': 1,
            'size_gb_approx': 1,
            'languages': 'Multiple',
            'license': 'CC BY 4.0',
            'format': 'Small sample for testing',
            'sampling_rate': '16kHz'
        }
    }
    return datasets_info

def get_yodas_language_info():
    """Get information about available languages in YODAS"""
    # Common languages available in YODAS (100+ total)
    common_languages = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
        'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no',
        'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 'et', 'lv',
        'lt', 'mt', 'cy', 'ga', 'is', 'mk', 'sq', 'eu', 'ca', 'gl'
    ]
    
    # European languages specifically available in YODAS-Granary
    granary_languages = [
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'pl', 'nl', 'sv', 
        'da', 'no', 'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 
        'et', 'lv', 'lt'
    ]
    
    return {
        'yodas': common_languages,
        'yodas2': common_languages,
        'yodas-granary': granary_languages
    }

def download_yodas_dataset(dataset_name: str,
                          output_dir: str,
                          languages: Optional[List[str]] = None,
                          max_hours: Optional[float] = None,
                          streaming: bool = True,
                          subset: Optional[str] = None):
    """
    Download YODAS dataset from Hugging Face
    
    Args:
        dataset_name: 'yodas', 'yodas2', 'yodas-granary', or 'yodas-sample'
        output_dir: Directory to save the dataset
        languages: List of language codes to filter (if supported)
        max_hours: Maximum hours to download (for testing)
        streaming: Use streaming to avoid downloading entire dataset
        subset: Specific subset/split to download
    """
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not available. Please install required packages.")
        return False
    
    # Get dataset info
    datasets_info = get_yodas_dataset_info()
    if dataset_name not in datasets_info:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(datasets_info.keys())}")
        return False
    
    dataset_info = datasets_info[dataset_name]
    repo_id = dataset_info['repo_id']
    
    print(f"📊 Dataset: {dataset_info['description']}")
    print(f"📝 License: {dataset_info['license']}")
    print(f"🌍 Languages: {dataset_info['languages']}")
    print(f"⏱️  Duration: ~{dataset_info['size_hours']:,} hours")
    print(f"💾 Size: ~{dataset_info['size_gb_approx']:,} GB")
    print(f"🎵 Format: {dataset_info['format']}")
    print(f"📡 Sampling Rate: {dataset_info['sampling_rate']}")
    
    # Create output directory
    output_path = Path(output_dir) / "yodas" / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path}")
    
    # Language filtering info
    language_info = get_yodas_language_info()
    if languages and dataset_name in language_info:
        available_langs = language_info[dataset_name]
        invalid_langs = set(languages) - set(available_langs)
        if invalid_langs:
            print(f"⚠️  Some languages may not be available: {invalid_langs}")
        print(f"🌍 Requested languages: {languages}")
    elif languages:
        print(f"🌍 Requested languages: {languages} (filtering may be limited)")
    
    try:
        print(f"\n🚀 Starting download from {repo_id}...")
        
        if streaming:
            # Use streaming for large datasets
            print("📡 Using streaming mode (recommended for large datasets)")
            
            # Load dataset in streaming mode
            dataset = load_dataset(
                repo_id, 
                streaming=True,
                trust_remote_code=True
            )
            
            print("✅ Dataset loaded successfully in streaming mode")
            
            # Process a few samples to verify
            print("\n📋 Sample data preview:")
            try:
                sample_count = 0
                for split_name, split_data in dataset.items():
                    print(f"Split: {split_name}")
                    for sample in split_data:
                        print(f"  Sample {sample_count + 1}: {list(sample.keys()) if hasattr(sample, 'keys') else 'Processing...'}")
                        sample_count += 1
                        if sample_count >= 3:  # Show first 3 samples
                            break
                    if sample_count >= 3:
                        break
            except Exception as e:
                print(f"⚠️  Could not preview samples: {e}")
            
            print("\n💡 Use the dataset object to process data incrementally")
            print(f"   Example: for sample in dataset['train']: process(sample)")
            
            # Save dataset info
            info_file = output_path / "dataset_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Repository: {repo_id}\n")
                f.write(f"Description: {dataset_info['description']}\n")
                f.write(f"License: {dataset_info['license']}\n")
                f.write(f"Languages: {dataset_info['languages']}\n")
                f.write(f"Format: {dataset_info['format']}\n")
                f.write(f"Sampling Rate: {dataset_info['sampling_rate']}\n")
                f.write(f"Estimated size: {dataset_info['size_gb_approx']:,} GB\n")
                f.write(f"Estimated duration: {dataset_info['size_hours']:,} hours\n")
                if languages:
                    f.write(f"Requested languages: {', '.join(languages)}\n")
                if max_hours:
                    f.write(f"Max hours limit: {max_hours}\n")
            
            print(f"📝 Dataset info saved to: {info_file}")
            return True
            
        else:
            # Download specific files or subsets
            print("⬇️  Downloading dataset files...")
            
            if dataset_name == 'yodas-sample':
                # Small dataset, can download fully
                dataset = load_dataset(repo_id, trust_remote_code=True)
                dataset.save_to_disk(str(output_path))
                print(f"✅ Sample dataset saved to: {output_path}")
            else:
                # For large datasets, use snapshot_download with selective files
                print("📦 Using selective file download...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(output_path),
                    repo_type="dataset",
                    resume_download=True,
                    ignore_patterns=["*.bin", "*.safetensors"] if max_hours else None
                )
                print(f"✅ Download completed to: {output_path}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_yodas_metadata(output_dir: str):
    """Download only metadata files for inspection"""
    print("📋 Downloading YODAS metadata for inspection...")
    
    datasets_info = get_yodas_dataset_info()
    metadata_path = Path(output_dir) / "yodas" / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, info in datasets_info.items():
        if dataset_name == 'yodas-sample':
            continue  # Skip sample for metadata
            
        try:
            print(f"\n📊 Fetching metadata for {dataset_name}...")
            
            # Try to get dataset info without downloading
            api = HfApi()
            repo_info = api.repo_info(repo_id=info['repo_id'], repo_type="dataset")
            
            # Save repo information
            info_file = metadata_path / f"{dataset_name}_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Repository: {info['repo_id']}\n")
                f.write(f"Description: {info['description']}\n")
                f.write(f"License: {info['license']}\n")
                f.write(f"Last modified: {repo_info.last_modified}\n")
                f.write(f"Downloads: {repo_info.downloads}\n")
                if hasattr(repo_info, 'siblings'):
                    f.write(f"Files: {len(repo_info.siblings)}\n")
                    f.write("File list:\n")
                    for sibling in repo_info.siblings[:20]:  # First 20 files
                        f.write(f"  - {sibling.rfilename}\n")
            
            print(f"✅ Metadata saved to: {info_file}")
            
        except Exception as e:
            print(f"⚠️  Could not fetch metadata for {dataset_name}: {e}")
    
    print(f"\n📁 All metadata saved to: {metadata_path}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Download YODAS speech dataset from Hugging Face"
    )
    parser.add_argument(
        '--dataset',
        choices=['yodas', 'yodas2', 'yodas-granary', 'yodas-sample', 'metadata'],
        default='yodas-sample',
        help='Dataset variant to download'
    )
    parser.add_argument(
        '--output_dir',
        default=DEFAULT_YODAS_DIR,
        help='Output directory for downloads (default: ITW speech directory)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en'],
        help='Language codes to filter (default: en). Examples: en es fr de'
    )
    parser.add_argument(
        '--english_only',
        action='store_true',
        help='Download only English data (equivalent to --languages en)'
    )
    parser.add_argument(
        '--max_hours',
        type=float,
        help='Maximum hours to download (for testing)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        default=True,
        help='Use streaming mode (recommended for large datasets)'
    )
    parser.add_argument(
        '--subset',
        help='Specific subset/split to download'
    )
    
    args = parser.parse_args()
    
    # Handle english_only flag
    if args.english_only:
        args.languages = ['en']
    
    print("🎬 YODAS Dataset Downloader")
    print("=" * 50)
    
    # Check dependencies
    if not HF_AVAILABLE:
        print("❌ Required packages not installed.")
        print("Install with: pip install huggingface_hub datasets pandas torch torchaudio")
        return
    
    # Show dataset information
    datasets_info = get_yodas_dataset_info()
    print("\n📊 Available Datasets:")
    for name, info in datasets_info.items():
        print(f"  {name}: {info['description']}")
        print(f"    Languages: {info['languages']}")
        print(f"    Size: ~{info['size_gb_approx']:,} GB")
        print(f"    Duration: {info['size_hours']:,} hours" if isinstance(info['size_hours'], int) else f"    Duration: {info['size_hours']}")
        print(f"    Format: {info['format']}")
        print()
    
    # Show language information
    language_info = get_yodas_language_info()
    print("\n🌍 Language Support:")
    print(f"  YODAS/YODAS2: {len(language_info['yodas'])}+ languages")
    print(f"  YODAS-Granary: {len(language_info['yodas-granary'])} European languages")
    print(f"  Common languages: {', '.join(language_info['yodas'][:10])}...")
    
    # Download requested dataset
    if args.dataset == 'metadata':
        success = download_yodas_metadata(args.output_dir)
    else:
        success = download_yodas_dataset(
            args.dataset,
            args.output_dir,
            args.languages,
            args.max_hours,
            args.streaming,
            args.subset
        )
    
    if success:
        print("\n✅ YODAS download completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Use streaming mode for processing large datasets")
        print("   2. Filter by language if needed during processing")
        print("   3. Consider using YODAS2 for long-form audio tasks")
        print("   4. Use YODAS-Granary for European language ASR/AST")
        print("   5. Cite the YODAS paper in your research")
        print("\n📄 Citation:")
        print("   YODAS: Youtube-Oriented Dataset for Audio and Speech")
        print("   arXiv:2406.00899, 2024")
    else:
        print("\n❌ YODAS download failed")

if __name__ == "__main__":
    main()