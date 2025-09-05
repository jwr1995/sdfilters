#!/usr/bin/env python3
"""
Generate CSV annotations for strong-label datasets in AITW format
"""
import os
import csv
import glob
import librosa
from pathlib import Path

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        duration = librosa.get_duration(path=audio_path)
        return round(duration, 6)
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return 0.0

def generate_librispeech_annotations(librispeech_dir, output_file):
    """Generate annotations for LibriSpeech (clean single-speaker English)"""
    annotations = []
    
    # Find all .flac files recursively
    audio_files = glob.glob(os.path.join(librispeech_dir, "**/*.flac"), recursive=True)
    
    for audio_path in audio_files:
        # Get relative path from librispeech_dir
        rel_path = os.path.relpath(audio_path, librispeech_dir)
        duration = get_audio_duration(audio_path)
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 1,
            'noise': 0,
            'background_music': 0,
            'foreign_language': 0,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} LibriSpeech annotations to {output_file}")

def generate_mls_annotations(mls_dir, output_file):
    """Generate annotations for MLS (multilingual single-speaker)"""
    annotations = []
    
    # Find all .flac files recursively
    audio_files = glob.glob(os.path.join(mls_dir, "**/*.flac"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, mls_dir)
        duration = get_audio_duration(audio_path)
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 1,
            'noise': 0,
            'background_music': 0,
            'foreign_language': 1,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} MLS annotations to {output_file}")

def generate_for_annotations(for_dir, output_file):
    """Generate annotations for FoR antispoofing dataset"""
    annotations = []
    
    # Find all audio files
    audio_files = []
    for ext in ['*.wav', '*.flac', '*.mp3']:
        audio_files.extend(glob.glob(os.path.join(for_dir, "**", ext), recursive=True))
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, for_dir)
        duration = get_audio_duration(audio_path)
        
        # Determine if synthetic based on filename/path
        is_synthetic = 1 if 'fake' in rel_path.lower() or 'synthetic' in rel_path.lower() else 0
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 1,
            'noise': 0,
            'background_music': 0,
            'foreign_language': 0,
            'synthetic': is_synthetic
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} FoR annotations to {output_file}")

def generate_odss_annotations(odss_dir, output_file):
    """Generate annotations for ODSS synthetic speech dataset
    
    ODSS contains English, German, and Spanish speech samples.
    Both natural and synthetic versions are included.
    """
    annotations = []
    
    audio_files = glob.glob(os.path.join(odss_dir, "**/*.wav"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, odss_dir)
        duration = get_audio_duration(audio_path)
        
        # Determine if synthetic based on directory structure
        is_synthetic = 1 if 'synthetic' in rel_path.lower() or 'generated' in rel_path.lower() else 0
        
        # Determine language based on path structure
        # ODSS contains VCTK (English), HUI-ACG (German), SLR-ES (Spanish)
        is_foreign = 0  # Default to English
        
        # Check for German indicators
        if any(indicator in rel_path.lower() for indicator in ['german', 'hui-acg', 'de_']):
            is_foreign = 1
        # Check for Spanish indicators  
        elif any(indicator in rel_path.lower() for indicator in ['spanish', 'slr-es', 'es_']):
            is_foreign = 1
        # If it contains VCTK or HiFi-TTS indicators, it's English
        elif any(indicator in rel_path.lower() for indicator in ['vctk', 'hifitts']):
            is_foreign = 0
            
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 1,
            'noise': 0,
            'background_music': 0,
            'foreign_language': is_foreign,
            'synthetic': is_synthetic
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} ODSS annotations to {output_file}")

def generate_musan_annotations(musan_dir, output_file):
    """Generate annotations for MUSAN (music, speech, noise)"""
    annotations = []
    
    audio_files = glob.glob(os.path.join(musan_dir, "**/*.wav"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, musan_dir)
        duration = get_audio_duration(audio_path)
        
        # Determine category from path
        is_music = 1 if 'music' in rel_path.lower() else 0
        is_noise = 1 if 'noise' in rel_path.lower() else 0
        is_speech = 1 if 'speech' in rel_path.lower() else 0
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 1 if is_speech else 0,
            'noise': is_noise,
            'background_music': is_music,
            'foreign_language': 0,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} MUSAN annotations to {output_file}")

def generate_demand_annotations(demand_dir, output_file):
    """Generate annotations for DEMAND noise dataset"""
    annotations = []
    
    audio_files = glob.glob(os.path.join(demand_dir, "**/*.wav"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, demand_dir)
        duration = get_audio_duration(audio_path)
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 0,
            'noise': 1,
            'background_music': 0,
            'foreign_language': 0,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} DEMAND annotations to {output_file}")

def generate_demand_chunked_annotations(demand_dir, output_file, chunk_length=30.0):
    """Generate chunked annotations for DEMAND noise dataset
    
    Splits long DEMAND noise files into fixed-length segments for training.
    
    Args:
        demand_dir: Path to original DEMAND dataset
        output_file: Output CSV file path
        chunk_length: Chunk length in seconds (default: 30.0)
    """
    annotations = []
    chunk_count = 0
    
    audio_files = glob.glob(os.path.join(demand_dir, "**/*.wav"), recursive=True)
    
    print(f"Chunking {len(audio_files)} DEMAND files into {chunk_length}s segments...")
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, demand_dir)
        duration = get_audio_duration(audio_path)
        
        if duration <= 0:
            continue
            
        # Create chunks from the audio file
        num_chunks = int(duration // chunk_length)
        
        for chunk_idx in range(num_chunks):
            start_time = chunk_idx * chunk_length
            end_time = min((chunk_idx + 1) * chunk_length, duration)
            chunk_duration = end_time - start_time
            
            # Only include chunks that are at least 90% of target length
            if chunk_duration >= chunk_length * 0.9:
                annotations.append({
                    'filename': rel_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': chunk_duration,
                    'num_speakers': 0,
                    'noise': 1,
                    'background_music': 0,
                    'foreign_language': 0,
                    'synthetic': 0
                })
                chunk_count += 1
        
        # Handle remainder if it's substantial (>50% of chunk length)
        remainder_start = num_chunks * chunk_length
        if remainder_start < duration:
            remainder_duration = duration - remainder_start
            if remainder_duration >= chunk_length * 0.5:
                annotations.append({
                    'filename': rel_path,
                    'start_time': remainder_start,
                    'end_time': duration,
                    'duration': remainder_duration,
                    'num_speakers': 0,
                    'noise': 1,
                    'background_music': 0,
                    'foreign_language': 0,
                    'synthetic': 0
                })
                chunk_count += 1
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} DEMAND-chunked annotations ({chunk_count} total chunks, avg {chunk_length}s each) to {output_file}")

def generate_fsdnoisy18k_annotations(fsd_dir, output_file):
    """Generate annotations for FSDNoisy18k dataset"""
    annotations = []
    
    audio_files = glob.glob(os.path.join(fsd_dir, "**/*.wav"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, fsd_dir)
        duration = get_audio_duration(audio_path)
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 0,
            'noise': 1,
            'background_music': 0,
            'foreign_language': 0,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} FSDNoisy18k annotations to {output_file}")

def generate_openmic_annotations(openmic_dir, output_file):
    """Generate annotations for OpenMIC-2018 music dataset"""
    annotations = []
    
    audio_files = glob.glob(os.path.join(openmic_dir, "**/*.ogg"), recursive=True)
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, openmic_dir)
        duration = get_audio_duration(audio_path)
        
        annotations.append({
            'filename': rel_path,
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'num_speakers': 0,
            'noise': 0,
            'background_music': 1,
            'foreign_language': 0,
            'synthetic': 0
        })
    
    write_csv(annotations, output_file)
    print(f"Generated {len(annotations)} OpenMIC-2018 annotations to {output_file}")

def get_ami_splits():
    """Get AMI train/dev/test splits for Full-corpus-ASR partition
    
    Returns:
        dict: {'train': [...], 'dev': [...], 'test': [...]}
    """
    # Full-corpus-ASR partition from AMI manual
    splits = {
        'train': [
            # SA (TRAINING PART OF SEEN DATA)
            'ES2002', 'ES2003', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 
            'ES2012', 'ES2013', 'ES2014', 'ES2015', 'ES2016',
            'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007',
            'TS3005', 'TS3006', 'TS3007', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012',
            'EN2001', 'EN2003', 'EN2004', 'EN2005', 'EN2006', 'EN2009',
            'IN1001', 'IN1002', 'IN1005', 'IN1007', 'IN1008', 'IN1009', 'IN1012', 'IN1013', 'IN1014', 'IN1016'
        ],
        'dev': [
            # SB (DEV PART OF SEEN DATA) 
            'ES2011', 'IS1008', 'TS3004',
            'IB4001', 'IB4002', 'IB4003', 'IB4004', 'IB4010', 'IB4011'
        ],
        'test': [
            # SC (UNSEEN DATA FOR EVALUATION)
            'ES2004', 'IS1009', 'TS3003', 'EN2002'
        ]
    }
    return splits

def generate_ami_annotations(ami_dir, output_file, split=None, separate_multispeaker=True):
    """Generate annotations for AMI conversational speech dataset
    
    Creates both single-speaker and multi-speaker segments using the AMI parser.
    
    Args:
        ami_dir: Path to AMI corpus
        output_file: Output CSV file path (for single-speaker, or combined if separate_multispeaker=False)
        split: Optional split filter ('train', 'dev', 'test') or None for all
        separate_multispeaker: If True, create separate files for single vs multi-speaker
    """
    from scripts.ami_parser import AMIParser
    
    single_speaker_annotations = []
    multi_speaker_annotations = []
    single_speaker_count = 0
    multi_speaker_count = 0
    parser = AMIParser(ami_dir)
    
    meeting_ids = parser.get_available_meetings()
    if not meeting_ids:
        print(f"No AMI meetings found in {ami_dir}")
        return
    
    # Filter meetings by split if specified
    if split:
        ami_splits = get_ami_splits()
        if split not in ami_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {list(ami_splits.keys())}")
        
        # Filter meeting IDs to match the split
        split_meetings = ami_splits[split]
        # Extract base meeting names (without session suffix) to match against split list
        filtered_meeting_ids = []
        for meeting_id in meeting_ids:
            base_meeting = meeting_id.rstrip('abcde')  # Remove session suffix (a, b, c, d, e)
            if base_meeting in split_meetings:
                filtered_meeting_ids.append(meeting_id)
        
        meeting_ids = filtered_meeting_ids
        print(f"Found {len(meeting_ids)} AMI meetings for {split} split")
    else:
        print(f"Found {len(meeting_ids)} AMI meetings (all splits)")
    
    for meeting_id in meeting_ids:
        print(f"Processing meeting {meeting_id}...")
        
        # Get single-speaker segments
        segments = parser.parse_meeting_segments(meeting_id)
        
        # Filter segments (remove very short/long segments)
        segments = parser.filter_segments(segments, min_duration=0.5, max_duration=15.0)
        
        for segment in segments:
            audio_file_path = parser.get_audio_file_path(meeting_id, mic_type='sdm')
            if audio_file_path:
                annotation = {
                    'filename': audio_file_path,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration,
                    'num_speakers': 1,
                    'noise': 0,  # Assume clean conversational speech
                    'background_music': 0,
                    'foreign_language': 0,  # AMI is English
                    'synthetic': 0
                }
                single_speaker_annotations.append(annotation)
                single_speaker_count += 1
        
        # Get multi-speaker overlapping segments
        overlaps = parser.find_overlapping_speech(meeting_id, min_overlap_duration=0.5)
        
        # Also get multi-speaker windows (non-overlapping but in proximity)
        windows = parser.find_multispeaker_windows(meeting_id, window_duration=8.0, 
                                                  min_speakers=2, max_gap=1.5)
        
        # Combine overlaps and windows
        all_multispeaker = overlaps + windows
        
        for overlap in all_multispeaker:
            audio_file_path = parser.get_audio_file_path(meeting_id, mic_type='sdm')
            if audio_file_path:
                annotation = {
                    'filename': audio_file_path,
                    'start_time': overlap.start_time,
                    'end_time': overlap.end_time,
                    'duration': overlap.duration,
                    'num_speakers': len(overlap.speakers),
                    'noise': 0,
                    'background_music': 0,
                    'foreign_language': 0,
                    'synthetic': 0
                }
                multi_speaker_annotations.append(annotation)
                multi_speaker_count += 1
    
    if separate_multispeaker:
        # Create separate files for single-speaker and multi-speaker
        base_path = os.path.splitext(output_file)[0]  # Remove .csv extension
        single_file = f"{base_path}_single.csv"
        multi_file = f"{base_path}_multi.csv"
        
        write_csv(single_speaker_annotations, single_file)
        write_csv(multi_speaker_annotations, multi_file)
        
        print(f"Generated {single_speaker_count} single-speaker AMI annotations to {single_file}")
        print(f"Generated {multi_speaker_count} multi-speaker AMI annotations to {multi_file}")
        print(f"Total: {single_speaker_count + multi_speaker_count} AMI annotations")
        
        return single_file, multi_file
    else:
        # Combined file (original behavior)
        all_annotations = single_speaker_annotations + multi_speaker_annotations
        write_csv(all_annotations, output_file)
        print(f"Generated {len(all_annotations)} AMI annotations ({single_speaker_count} single-speaker, {multi_speaker_count} multi-speaker) to {output_file}")
        return output_file

def generate_alimeeting_annotations(alimeeting_dir, output_file, split=None, separate_multispeaker=True):
    """Generate annotations for AliMeeting Chinese conversational dataset"""
    from scripts.alimeeting_parser import (
        parse_alimeeting_textgrid, find_overlapping_speech, 
        filter_segments, get_alimeeting_splits
    )
    
    # Get train/dev/test splits if requested
    if split:
        splits = get_alimeeting_splits()
        if split not in splits:
            raise ValueError(f"Invalid split: {split}. Valid splits: {list(splits.keys())}")
        meeting_ids_to_process = splits[split]
        print(f"Processing AliMeeting {split} split with {len(meeting_ids_to_process)} meeting IDs")
    else:
        meeting_ids_to_process = None
        print("Processing all AliMeeting meetings")
    
    single_speaker_annotations = []
    multi_speaker_annotations = []
    single_speaker_count = 0
    multi_speaker_count = 0
    
    # Process all splits (Train_Ali_far, Train_Ali_near, Eval_Ali, Test_Ali)
    for split_dir in ['Train_Ali_far', 'Train_Ali_near', 'Eval_Ali', 'Test_Ali']:
        textgrid_dir = os.path.join(alimeeting_dir, split_dir, 'textgrid_dir')
        audio_dir = os.path.join(alimeeting_dir, split_dir, 'audio_dir')
        
        if not os.path.exists(textgrid_dir):
            continue
            
        print(f"Processing {split_dir}...")
        
        # Find all TextGrid files
        textgrid_files = glob.glob(os.path.join(textgrid_dir, "*.TextGrid"))
        
        for textgrid_file in textgrid_files:
            meeting_id = os.path.basename(textgrid_file).replace('.TextGrid', '')
            
            # Skip if we're filtering by split and this meeting isn't in the split
            if meeting_ids_to_process:
                # Extract room ID (e.g., R0003 from R0003_M0046)
                room_id = meeting_id.split('_')[0]
                if room_id not in meeting_ids_to_process:
                    continue
            
            # Parse TextGrid file
            try:
                segments = parse_alimeeting_textgrid(textgrid_file)
                
                # Filter segments by duration
                segments = filter_segments(segments, min_duration=0.5, max_duration=15.0)
                
                # Generate single-speaker annotations
                for segment in segments:
                    # Audio file path relative to alimeeting_dir  
                    audio_filename = f"{split_dir}/audio_dir/{segment.filename}.wav"
                    
                    annotation = {
                        'filename': audio_filename,
                        'start_time': segment.start_time,
                        'end_time': segment.end_time,
                        'duration': segment.duration,
                        'num_speakers': 1,
                        'noise': 0,  # Clean conversational speech
                        'background_music': 0,
                        'foreign_language': 1,  # Chinese
                        'synthetic': 0
                    }
                    single_speaker_annotations.append(annotation)
                    single_speaker_count += 1
                
                # Find overlapping speech
                overlaps = find_overlapping_speech(segments, min_overlap=0.5)
                
                # Also find multi-speaker windows (non-overlapping but in proximity)
                from scripts.alimeeting_parser import find_multispeaker_windows
                windows = find_multispeaker_windows(segments, window_duration=8.0,
                                                   min_speakers=2, max_gap=1.5)
                
                # Combine overlaps and windows
                all_multispeaker = overlaps + windows
                
                for overlap in all_multispeaker:
                    audio_filename = f"{split_dir}/audio_dir/{overlap.filename}.wav"
                    
                    annotation = {
                        'filename': audio_filename,
                        'start_time': overlap.start_time,
                        'end_time': overlap.end_time,
                        'duration': overlap.duration,
                        'num_speakers': len(overlap.speakers),
                        'noise': 0,
                        'background_music': 0,
                        'foreign_language': 1,  # Chinese
                        'synthetic': 0
                    }
                    multi_speaker_annotations.append(annotation)
                    multi_speaker_count += 1
                    
            except Exception as e:
                print(f"Error processing {textgrid_file}: {e}")
                continue
    
    if separate_multispeaker:
        # Create separate files for single-speaker and multi-speaker
        base_path = os.path.splitext(output_file)[0]  # Remove .csv extension
        if split:
            single_file = f"{base_path}_{split}_single.csv"
            multi_file = f"{base_path}_{split}_multi.csv"
        else:
            single_file = f"{base_path}_single.csv"
            multi_file = f"{base_path}_multi.csv"
        
        write_csv(single_speaker_annotations, single_file)
        write_csv(multi_speaker_annotations, multi_file)
        
        print(f"Generated {single_speaker_count} single-speaker AliMeeting annotations to {single_file}")
        print(f"Generated {multi_speaker_count} multi-speaker AliMeeting annotations to {multi_file}")
        print(f"Total: {single_speaker_count + multi_speaker_count} AliMeeting annotations")
        
        return single_file, multi_file
    else:
        # Combined file
        all_annotations = single_speaker_annotations + multi_speaker_annotations
        write_csv(all_annotations, output_file)
        print(f"Generated {len(all_annotations)} AliMeeting annotations ({single_speaker_count} single-speaker, {multi_speaker_count} multi-speaker) to {output_file}")
        return output_file

def write_csv(annotations, output_file):
    """Write annotations to CSV file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    fieldnames = ['filename', 'start_time', 'end_time', 'duration', 'num_speakers', 'noise', 'background_music', 'foreign_language', 'synthetic']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotations)

if __name__ == "__main__":
    from scripts.constants import *
    
    # Generate annotations for each dataset
    datasets = [
        (LIBRISPEECH_DIR, 'librispeech', generate_librispeech_annotations),
        (MLS_DIR, 'mls', generate_mls_annotations),
        (FOR_DIR, 'for', generate_for_annotations),
        (ODSS_DIR, 'odss', generate_odss_annotations),
        (MUSAN_DIR, 'musan', generate_musan_annotations),
        (DEMAND_DIR, 'demand', generate_demand_annotations),
        (DEMAND_DIR, 'demand-chunked', generate_demand_chunked_annotations),
        (FSDNOISY18K_DIR, 'fsdnoisy18k', generate_fsdnoisy18k_annotations),
        (OPENMIC_2018_DIR, 'openmic2018', generate_openmic_annotations),
        (AMI_DIR, 'ami', generate_ami_annotations),
        (ALIMEETING_DIR, 'alimeeting', generate_alimeeting_annotations)
    ]
    
    for dataset_dir, dataset_name, generate_func in datasets:
        if os.path.exists(dataset_dir):
            output_file = os.path.join(ANNOTATIONS_DIR, f'{dataset_name}_annotations.csv')
            generate_func(dataset_dir, output_file)
        else:
            print(f"Dataset directory not found: {dataset_dir}")