#!/usr/bin/env python3
"""
Multi-speaker Utterance Sampling Script

This script samples multi-speaker utterances from AliMeeting and AMI datasets
for listening tests and quality evaluation. It extracts audio segments and
provides metadata for manual inspection of multi-speaker detection accuracy.

Usage:
    python scripts/sample_multispeaker.py --dataset ami --num_samples 20
    python scripts/sample_multispeaker.py --dataset alimeeting --num_samples 10
    python scripts/sample_multispeaker.py --dataset both --num_samples 15
"""
import os
import sys
import csv
import random
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.constants import AMI_DIR, ALIMEETING_DIR, OUTPUT_DIR

def read_annotations(csv_file):
    """Read annotations from CSV file"""
    annotations = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append(row)
    return annotations

def sample_ami_multispeaker(num_samples=10, splits=['train', 'dev', 'test']):
    """Sample multi-speaker segments from AMI dataset"""
    samples = []
    
    for split in splits:
        ami_multi_file = os.path.join(ANNOTATIONS_DIR, f'ami_{split}_separated_multi.csv')
        
        if not os.path.exists(ami_multi_file):
            print(f"Warning: {ami_multi_file} not found, skipping {split} split")
            continue
            
        annotations = read_annotations(ami_multi_file)
        
        # Sample from this split
        split_samples = min(num_samples // len(splits), len(annotations))
        if split_samples > 0:
            sampled = random.sample(annotations, split_samples)
            
            for annotation in sampled:
                samples.append({
                    'dataset': 'ami',
                    'split': split,
                    'filename': annotation['filename'],
                    'start_time': float(annotation['start_time']),
                    'end_time': float(annotation['end_time']),
                    'duration': float(annotation['duration']),
                    'num_speakers': int(annotation['num_speakers']),
                    'audio_path': os.path.join(AMI_DIR, annotation['filename']),
                    'sample_id': f"ami_{split}_{len(samples):03d}"
                })
    
    return samples

def sample_alimeeting_multispeaker(num_samples=10, splits=['train', 'dev', 'test']):
    """Sample multi-speaker segments from AliMeeting dataset"""
    samples = []
    
    # Check which split files exist
    available_splits = []
    for split in splits:
        ali_multi_file = os.path.join(OUTPUT_DIR, 'annotations', f'alimeeting_annotations_{split}_multi.csv')
        if os.path.exists(ali_multi_file):
            available_splits.append(split)
    
    # If no split files exist, check for train test file
    if not available_splits:
        test_file = os.path.join(OUTPUT_DIR, 'annotations', 'alimeeting_train_test_train_multi.csv')
        if os.path.exists(test_file):
            print("Using AliMeeting train test file")
            annotations = read_annotations(test_file)
            sampled = random.sample(annotations, min(num_samples, len(annotations)))
            
            for annotation in sampled:
                samples.append({
                    'dataset': 'alimeeting',
                    'split': 'train_test',
                    'filename': annotation['filename'],
                    'start_time': float(annotation['start_time']),
                    'end_time': float(annotation['end_time']),
                    'duration': float(annotation['duration']),
                    'num_speakers': int(annotation['num_speakers']),
                    'audio_path': os.path.join(ALIMEETING_DIR, annotation['filename']),
                    'sample_id': f"alimeeting_train_{len(samples):03d}"
                })
            return samples
    
    # Process available splits
    for split in available_splits:
        ali_multi_file = os.path.join(OUTPUT_DIR, 'annotations', f'alimeeting_annotations_{split}_multi.csv')
        annotations = read_annotations(ali_multi_file)
        
        # Sample from this split
        split_samples = min(num_samples // len(available_splits), len(annotations))
        if split_samples > 0:
            sampled = random.sample(annotations, split_samples)
            
            for annotation in sampled:
                samples.append({
                    'dataset': 'alimeeting',
                    'split': split,
                    'filename': annotation['filename'],
                    'start_time': float(annotation['start_time']),
                    'end_time': float(annotation['end_time']),
                    'duration': float(annotation['duration']),
                    'num_speakers': int(annotation['num_speakers']),
                    'audio_path': os.path.join(ALIMEETING_DIR, annotation['filename']),
                    'sample_id': f"alimeeting_{split}_{len(samples):03d}"
                })
    
    return samples

def find_audio_file(base_path, filename):
    """Find the actual audio file path, handling different naming conventions"""
    # For AMI: annotations have paths like "wav_db/EN2001a/audio/EN2001a.Array1-01.wav"
    # For AliMeeting: annotations have paths like "Train_Ali_far/audio_dir/R0003_M0046.wav"
    #                 but actual files are "Train_Ali_far/audio_dir/R0003_M0046_MS002.wav"
    
    full_path = os.path.join(base_path, filename)
    
    # If exact path exists, return it
    if os.path.exists(full_path):
        return full_path
    
    # For AliMeeting, try with different microphone suffixes
    if 'alimeeting' in base_path.lower():
        base_name = os.path.splitext(full_path)[0]  # Remove .wav
        audio_dir = os.path.dirname(full_path)
        
        # Try common AliMeeting suffixes
        for suffix in ['_MS002', '_MS004', '_MS006', '_MS008']:
            candidate = f"{base_name}{suffix}.wav"
            if os.path.exists(candidate):
                return candidate
    
    return None

def extract_audio_segment(input_path, output_path, start_time, duration):
    """Extract audio segment using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',
            '-y',  # overwrite output files
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error extracting {input_path}: {e}")
        return False

def create_listening_test_samples(samples, output_dir):
    """Extract audio samples and create metadata file"""
    samples_dir = Path(output_dir) / "multispeaker_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    successful_extractions = 0
    
    print(f"\nExtracting {len(samples)} audio samples...")
    
    for i, sample in enumerate(samples):
        # Create output filename
        output_filename = f"{sample['sample_id']}_dur{sample['duration']:.1f}s_spk{sample['num_speakers']}.wav"
        output_path = samples_dir / output_filename
        
        print(f"[{i+1}/{len(samples)}] Extracting {sample['sample_id']}...")
        print(f"  Audio: {sample['audio_path']}")
        print(f"  Time: {sample['start_time']:.2f}-{sample['end_time']:.2f}s ({sample['duration']:.2f}s)")
        print(f"  Speakers: {sample['num_speakers']}")
        
        # Find the actual audio file
        if sample['dataset'] == 'ami':
            actual_audio_path = sample['audio_path']  # AMI paths should be correct
        else:
            # For AliMeeting, find the actual file with microphone suffix
            actual_audio_path = find_audio_file(
                os.path.dirname(sample['audio_path']), 
                os.path.basename(sample['filename'])
            )
            if actual_audio_path is None:
                actual_audio_path = find_audio_file(ALIMEETING_DIR, sample['filename'])
        
        # Check if source audio exists
        if not actual_audio_path or not os.path.exists(actual_audio_path):
            print(f"  ❌ Source audio not found: {sample['audio_path']}")
            continue
        
        print(f"  📁 Using: {os.path.basename(actual_audio_path)}")
            
        # Extract audio segment
        success = extract_audio_segment(
            actual_audio_path,
            str(output_path),
            sample['start_time'],
            sample['duration']
        )
        
        if success and output_path.exists():
            print(f"  ✅ Extracted to: {output_filename}")
            successful_extractions += 1
            
            # Add to metadata
            metadata.append({
                'sample_id': sample['sample_id'],
                'filename': output_filename,
                'dataset': sample['dataset'],
                'split': sample['split'],
                'original_file': sample['filename'],
                'start_time': sample['start_time'],
                'end_time': sample['end_time'],
                'duration': sample['duration'],
                'num_speakers': sample['num_speakers'],
                'notes': ''  # For manual annotation during listening test
            })
        else:
            print(f"  ❌ Failed to extract")
    
    # Write metadata file
    metadata_file = samples_dir / "samples_metadata.csv"
    with open(metadata_file, 'w', newline='') as f:
        fieldnames = ['sample_id', 'filename', 'dataset', 'split', 'original_file', 
                     'start_time', 'end_time', 'duration', 'num_speakers', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n✅ Successfully extracted {successful_extractions}/{len(samples)} samples")
    print(f"📁 Samples saved to: {samples_dir}")
    print(f"📊 Metadata saved to: {metadata_file}")
    
    return samples_dir, metadata_file

def print_sample_statistics(samples):
    """Print statistics about the samples"""
    print("\n📊 Sample Statistics:")
    
    # By dataset
    by_dataset = {}
    by_speakers = {}
    durations = []
    
    for sample in samples:
        dataset = sample['dataset']
        speakers = sample['num_speakers']
        duration = sample['duration']
        
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        by_speakers[speakers] = by_speakers.get(speakers, 0) + 1
        durations.append(duration)
    
    print(f"Total samples: {len(samples)}")
    print(f"By dataset: {dict(by_dataset)}")
    print(f"By speaker count: {dict(sorted(by_speakers.items()))}")
    
    if durations:
        print(f"Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"Average duration: {sum(durations)/len(durations):.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Sample multi-speaker utterances for listening tests")
    parser.add_argument('--dataset', choices=['ami', 'alimeeting', 'both'], default='both',
                       help='Dataset to sample from (default: both)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples per dataset (default: 20)')
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                       help='Dataset splits to sample from (default: train dev test)')
    parser.add_argument('--output_dir', default=OUTPUT_DIR,
                       help='Output directory for samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--no_extract', action='store_true',
                       help='Only show sample info, do not extract audio')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("🎧 Multi-speaker Utterance Sampling for Listening Tests")
    print("=" * 60)
    
    samples = []
    
    # Sample from AMI
    if args.dataset in ['ami', 'both']:
        print(f"\n📻 Sampling from AMI dataset...")
        ami_samples = sample_ami_multispeaker(args.num_samples, args.splits)
        samples.extend(ami_samples)
        print(f"   Found {len(ami_samples)} AMI samples")
    
    # Sample from AliMeeting  
    if args.dataset in ['alimeeting', 'both']:
        print(f"\n🇨🇳 Sampling from AliMeeting dataset...")
        ali_samples = sample_alimeeting_multispeaker(args.num_samples, args.splits)
        samples.extend(ali_samples)
        print(f"   Found {len(ali_samples)} AliMeeting samples")
    
    if not samples:
        print("❌ No samples found. Check that annotation files exist.")
        return
    
    # Shuffle all samples together
    random.shuffle(samples)
    
    # Print statistics
    print_sample_statistics(samples)
    
    if args.no_extract:
        print("\n📋 Sample preview (--no_extract mode):")
        for i, sample in enumerate(samples[:5]):
            print(f"  {i+1}. {sample['sample_id']} ({sample['dataset']}) - "
                  f"{sample['duration']:.1f}s, {sample['num_speakers']} speakers")
        if len(samples) > 5:
            print(f"  ... and {len(samples)-5} more samples")
        return
    
    # Extract audio samples
    samples_dir, metadata_file = create_listening_test_samples(samples, args.output_dir)
    
    print("\n🎧 Listening Test Ready!")
    print(f"   Open {samples_dir} and play the audio files")
    print(f"   Use {metadata_file} to record your observations")
    print("\n💡 Listening Test Tips:")
    print("   - Listen for clear speaker boundaries and overlaps")
    print("   - Note any false positives (single speaker detected as multi)")
    print("   - Check if all speakers are audible in multi-speaker segments")
    print("   - Record quality issues or annotation accuracy in 'notes' column")

if __name__ == "__main__":
    main()