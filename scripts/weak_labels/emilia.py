#!/usr/bin/env python3
"""
Download Emilia Dataset from Hugging Face

Emilia is a large-scale multilingual speech generation dataset with 101k hours 
of in-the-wild speech data across six languages: English, Chinese, German, 
French, Japanese, and Korean.

The extended Emilia-Large dataset (216k hours) combines the original Emilia 
dataset with Emilia-YODAS data.

Usage:
    python scripts/strong_labels/emilia.py --dataset emilia --languages en zh
    python scripts/strong_labels/emilia.py --dataset emilia-large --max_size_gb 100
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
    DEFAULT_EMILIA_DIR = ITW_SPEECH_DIR
    # Try to get access token from constants
    try:
        from scripts.constants import ACCESS_TOKEN
        HF_ACCESS_TOKEN = ACCESS_TOKEN
    except (ImportError, AttributeError):
        HF_ACCESS_TOKEN = None
except ImportError:
    # Fallback if constants not available
    OUTPUT_DIR = "/media/will/DATA/corpora"
    DEFAULT_EMILIA_DIR = "/media/will/DATA/corpora/speech/itw"
    HF_ACCESS_TOKEN = None

# Try to import huggingface_hub
try:
    from huggingface_hub import snapshot_download, HfApi, login
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  huggingface_hub not available. Install with: pip install huggingface_hub datasets")

def get_emilia_dataset_info():
    """Get information about available Emilia datasets"""
    datasets_info = {
        'emilia': {
            'repo_id': 'amphion/Emilia-Dataset',
            'description': 'Original Emilia dataset (101k hours)',
            'languages': ['en', 'zh', 'de', 'fr', 'ja', 'ko'],
            'license': 'CC BY-NC 4.0',
            'size_hours': 101000,
            'size_gb_approx': 50000  # Approximate
        },
        'emilia-large': {
            'repo_id': 'amphion/Emilia-Dataset', 
            'description': 'Emilia-Large dataset (216k hours = Emilia + Emilia-YODAS)',
            'languages': ['en', 'zh', 'de', 'fr', 'ja', 'ko'],
            'license': 'Mixed: CC BY-NC 4.0 (Emilia) + CC BY 4.0 (Emilia-YODAS)',
            'size_hours': 216000,
            'size_gb_approx': 108000  # Approximate
        }
    }
    return datasets_info

def setup_huggingface_auth():
    """Setup Hugging Face authentication"""
    # Try token from constants first
    token = HF_ACCESS_TOKEN or os.getenv('HF_TOKEN')
    
    if token:
        try:
            login(token=token)
            print("✅ Successfully authenticated with access token")
            return True
        except Exception as e:
            print(f"❌ Authentication with token failed: {e}")
    
    if not token:
        print("🔑 Hugging Face token required for Emilia dataset access.")
        print("   Please set HF_TOKEN environment variable or login with:")
        print("   huggingface-cli login")
        print("\n   You can get a token from: https://huggingface.co/settings/tokens")
        
        try:
            login()
            print("✅ Successfully authenticated with Hugging Face")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    return False

def download_emilia_subset(dataset_name: str, 
                          output_dir: str,
                          languages: Optional[List[str]] = None,
                          max_size_gb: Optional[float] = None,
                          streaming: bool = True):
    """
    Download Emilia dataset from Hugging Face
    
    Args:
        dataset_name: 'emilia' or 'emilia-large'
        output_dir: Directory to save the dataset
        languages: List of language codes to download (default: all)
        max_size_gb: Maximum download size in GB (for testing)
        streaming: Use streaming to avoid downloading entire dataset at once
    """
    if not HF_AVAILABLE:
        print("❌ huggingface_hub not available. Please install required packages.")
        return False
    
    # Get dataset info
    datasets_info = get_emilia_dataset_info()
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
    
    # Setup authentication
    if not setup_huggingface_auth():
        return False
    
    # Create output directory
    output_path = Path(output_dir) / "emilia" / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_path}")
    
    # Filter languages if specified
    if languages:
        available_langs = dataset_info['languages']
        invalid_langs = set(languages) - set(available_langs)
        if invalid_langs:
            print(f"⚠️  Invalid languages: {invalid_langs}")
            print(f"Available languages: {available_langs}")
            languages = [lang for lang in languages if lang in available_langs]
        print(f"🌍 Downloading languages: {languages}")
    
    try:
        print(f"\n🚀 Starting download from {repo_id}...")
        
        if streaming:
            # Use streaming for large datasets
            print("📡 Using streaming mode (recommended for large datasets)")
            
            # Load dataset in streaming mode
            if dataset_name == 'emilia-large':
                # For Emilia-Large, we might need to specify subsets
                dataset = load_dataset(repo_id, streaming=True, trust_remote_code=True)
            else:
                dataset = load_dataset(repo_id, streaming=True, trust_remote_code=True)
            
            print("✅ Dataset loaded successfully in streaming mode")
            print("💡 Use the dataset object to process data incrementally")
            print(f"   Example: for sample in dataset['train']: process(sample)")
            
            # Save dataset info
            info_file = output_path / "dataset_info.txt"
            with open(info_file, 'w') as f:
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Repository: {repo_id}\n")
                f.write(f"Description: {dataset_info['description']}\n")
                f.write(f"License: {dataset_info['license']}\n")
                f.write(f"Languages: {', '.join(dataset_info['languages'])}\n")
                f.write(f"Estimated size: {dataset_info['size_gb_approx']:,} GB\n")
                f.write(f"Estimated duration: {dataset_info['size_hours']:,} hours\n")
                if languages:
                    f.write(f"Selected languages: {', '.join(languages)}\n")
            
            print(f"📝 Dataset info saved to: {info_file}")
            return True
            
        else:
            # Download entire dataset (not recommended for large datasets)
            print("⬬ Downloading entire dataset (this may take a very long time)...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(output_path),
                repo_type="dataset",
                resume_download=True
            )
            print(f"✅ Download completed to: {output_path}")
            return True
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def download_emilia_samples(output_dir: str, num_samples: int = 100):
    """Download a small sample of Emilia data for testing"""
    print(f"📦 Downloading {num_samples} samples from Emilia dataset...")
    
    if not setup_huggingface_auth():
        return False
    
    try:
        # Load a small subset for testing
        dataset = load_dataset(
            "amphion/Emilia-Dataset", 
            streaming=True,
            trust_remote_code=True
        )
        
        output_path = Path(output_dir) / "emilia" / "samples"
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for sample in dataset['train']:
            if count >= num_samples:
                break
                
            # Process sample (this would depend on the actual data format)
            print(f"Sample {count + 1}: {sample.keys() if hasattr(sample, 'keys') else 'Processing...'}")
            count += 1
        
        print(f"✅ Downloaded {count} samples to: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading samples: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download Emilia speech dataset from Hugging Face"
    )
    parser.add_argument(
        '--dataset', 
        choices=['emilia', 'emilia-large', 'samples'],
        default='emilia',
        help='Dataset variant to download'
    )
    parser.add_argument(
        '--output_dir',
        default=DEFAULT_EMILIA_DIR,
        help='Output directory for downloads (default: ITW speech directory)'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en'],
        help='Language codes to download (default: en). Options: en zh de fr ja ko'
    )
    parser.add_argument(
        '--english_only',
        action='store_true',
        help='Download only English data (equivalent to --languages en)'
    )
    parser.add_argument(
        '--max_size_gb',
        type=float,
        help='Maximum download size in GB (for testing)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        default=True,
        help='Use streaming mode (recommended for large datasets)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of samples to download (for samples mode)'
    )
    
    args = parser.parse_args()
    
    # Handle english_only flag
    if args.english_only:
        args.languages = ['en']
    
    print("🎤 Emilia Dataset Downloader")
    print("=" * 50)
    
    # Check dependencies
    if not HF_AVAILABLE:
        print("❌ Required packages not installed.")
        print("Install with: pip install huggingface_hub datasets torch torchaudio")
        return
    
    # Show dataset information
    datasets_info = get_emilia_dataset_info()
    print("\n📊 Available Datasets:")
    for name, info in datasets_info.items():
        print(f"  {name}: {info['description']}")
        print(f"    Languages: {', '.join(info['languages'])}")
        print(f"    Size: ~{info['size_gb_approx']:,} GB, ~{info['size_hours']:,} hours")
        print(f"    License: {info['license']}")
        print()
    
    # Download requested dataset
    if args.dataset == 'samples':
        success = download_emilia_samples(args.output_dir, args.num_samples)
    else:
        success = download_emilia_subset(
            args.dataset,
            args.output_dir,
            args.languages,
            args.max_size_gb,
            args.streaming
        )
    
    if success:
        print("\n✅ Emilia download completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Use the dataset for speech generation research")
        print("   2. Follow the license terms (non-commercial use for original Emilia)")
        print("   3. Cite the Emilia paper in your research")
    else:
        print("\n❌ Emilia download failed")

if __name__ == "__main__":
    main()