#!/usr/bin/env python3
"""
AliMeeting TextGrid Parser Module

This module provides functionality to parse AliMeeting TextGrid files and extract
speech segments with temporal information. AliMeeting is a Chinese conversational
speech corpus with multi-channel recordings and detailed speaker annotations.

The TextGrid format contains multiple IntervalTier sections, each representing
a different speaker with time-aligned speech segments.

Usage:
    from scripts.alimeeting_parser import parse_alimeeting_textgrid, find_overlapping_speech
    
    segments = parse_alimeeting_textgrid('/path/to/meeting.TextGrid')
    overlaps = find_overlapping_speech(segments)
"""
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from pathlib import Path

@dataclass
class AliMeetingSegment:
    """Represents a single speech segment from AliMeeting TextGrid"""
    filename: str
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds"""
        return self.end_time - self.start_time

@dataclass  
class AliMeetingOverlap:
    """Represents overlapping speech between multiple speakers"""
    filename: str
    start_time: float
    end_time: float
    speakers: List[str]
    
    @property
    def duration(self) -> float:
        """Duration of the overlap in seconds"""
        return self.end_time - self.start_time

def parse_textgrid_content(content: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Parse TextGrid content and extract speaker intervals
    
    Args:
        content: Raw TextGrid file content
        
    Returns:
        Dictionary mapping speaker_id -> list of (start, end, text) intervals
    """
    speakers = {}
    
    # Split into items (speakers)
    items = re.findall(r'item \[\d+\]:(.*?)(?=item \[\d+\]:|$)', content, re.DOTALL)
    
    for item in items:
        # Extract speaker name
        name_match = re.search(r'name = "([^"]+)"', item)
        if not name_match:
            continue
        speaker_name = name_match.group(1)
        
        # Extract intervals
        intervals = re.findall(r'intervals \[\d+\]:\s*xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = "([^"]*)"', item)
        
        speaker_intervals = []
        for start_str, end_str, text in intervals:
            start_time = float(start_str)
            end_time = float(end_str) 
            
            # Skip empty intervals
            if text.strip():
                speaker_intervals.append((start_time, end_time, text.strip()))
        
        if speaker_intervals:
            speakers[speaker_name] = speaker_intervals
    
    return speakers

def parse_alimeeting_textgrid(textgrid_path: str) -> List[AliMeetingSegment]:
    """
    Parse an AliMeeting TextGrid file and return speech segments
    
    Args:
        textgrid_path: Path to the TextGrid file
        
    Returns:
        List of AliMeetingSegment objects
    """
    textgrid_path = Path(textgrid_path)
    filename = textgrid_path.stem
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    speakers = parse_textgrid_content(content)
    segments = []
    
    for speaker_id, intervals in speakers.items():
        for start_time, end_time, text in intervals:
            segments.append(AliMeetingSegment(
                filename=filename,
                speaker_id=speaker_id,
                start_time=start_time,
                end_time=end_time,
                text=text
            ))
    
    # Sort by start time
    segments.sort(key=lambda x: x.start_time)
    return segments

def find_overlapping_speech(segments: List[AliMeetingSegment], 
                          min_overlap: float = 0.1) -> List[AliMeetingOverlap]:
    """
    Find overlapping speech segments between different speakers
    
    Args:
        segments: List of speech segments
        min_overlap: Minimum overlap duration in seconds
        
    Returns:
        List of AliMeetingOverlap objects representing multi-speaker overlaps
    """
    overlaps = []
    
    # Group segments by filename
    by_filename = {}
    for segment in segments:
        if segment.filename not in by_filename:
            by_filename[segment.filename] = []
        by_filename[segment.filename].append(segment)
    
    # Find overlaps within each file
    for filename, file_segments in by_filename.items():
        file_segments.sort(key=lambda x: x.start_time)
        
        for i, seg1 in enumerate(file_segments):
            for seg2 in file_segments[i+1:]:
                # Stop if seg2 starts after seg1 ends
                if seg2.start_time >= seg1.end_time:
                    break
                
                # Skip if same speaker
                if seg1.speaker_id == seg2.speaker_id:
                    continue
                
                # Calculate overlap
                overlap_start = max(seg1.start_time, seg2.start_time)
                overlap_end = min(seg1.end_time, seg2.end_time)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration >= min_overlap:
                    # Check if this overlap already exists
                    existing = False
                    for existing_overlap in overlaps:
                        if (existing_overlap.filename == filename and
                            abs(existing_overlap.start_time - overlap_start) < 0.01 and
                            abs(existing_overlap.end_time - overlap_end) < 0.01):
                            # Add speaker if not already present
                            if seg1.speaker_id not in existing_overlap.speakers:
                                existing_overlap.speakers.append(seg1.speaker_id)
                            if seg2.speaker_id not in existing_overlap.speakers:
                                existing_overlap.speakers.append(seg2.speaker_id)
                            existing = True
                            break
                    
                    if not existing:
                        overlaps.append(AliMeetingOverlap(
                            filename=filename,
                            start_time=overlap_start,
                            end_time=overlap_end,
                            speakers=[seg1.speaker_id, seg2.speaker_id]
                        ))
    
    return overlaps

def find_multispeaker_windows(segments: List[AliMeetingSegment],
                             window_duration: float = 10.0,
                             min_speakers: int = 2,
                             max_gap: float = 2.0) -> List[AliMeetingOverlap]:
    """
    Find multi-speaker windows containing multiple speakers in proximity
    (not necessarily overlapping)
    
    Args:
        segments: List of speech segments
        window_duration: Duration of each window in seconds
        min_speakers: Minimum number of speakers required in window
        max_gap: Maximum gap between speakers to consider them "in proximity"
        
    Returns:
        List of AliMeetingOverlap objects representing multi-speaker windows
    """
    if not segments:
        return []
    
    # Group segments by filename
    by_filename = {}
    for segment in segments:
        if segment.filename not in by_filename:
            by_filename[segment.filename] = []
        by_filename[segment.filename].append(segment)
    
    all_windows = []
    
    # Process each file
    for filename, file_segments in by_filename.items():
        file_segments.sort(key=lambda x: x.start_time)
        
        # Find the time range of the file
        start_time = min(s.start_time for s in file_segments)
        end_time = max(s.end_time for s in file_segments)
        
        current_time = start_time
        
        while current_time + window_duration <= end_time:
            window_start = current_time
            window_end = current_time + window_duration
            
            # Find all speakers active in this window
            speakers_in_window = set()
            segments_in_window = []
            
            for segment in file_segments:
                # Check if segment overlaps with window
                if (segment.start_time < window_end and segment.end_time > window_start):
                    speakers_in_window.add(segment.speaker_id)
                    segments_in_window.append(segment)
            
            # Check if we have enough speakers and they're in reasonable proximity
            if len(speakers_in_window) >= min_speakers:
                # Group segments by speaker and check proximity
                speaker_segments = {}
                for segment in segments_in_window:
                    if segment.speaker_id not in speaker_segments:
                        speaker_segments[segment.speaker_id] = []
                    speaker_segments[segment.speaker_id].append(segment)
                
                # Check if speakers are in proximity (gaps between speakers <= max_gap)
                all_speaker_times = []
                for speaker, segs in speaker_segments.items():
                    for seg in segs:
                        all_speaker_times.append((seg.start_time, seg.end_time, speaker))
                
                all_speaker_times.sort()
                
                # Check if there are reasonable gaps between different speakers
                has_proximity = False
                for i in range(len(all_speaker_times) - 1):
                    curr_end, next_start = all_speaker_times[i][1], all_speaker_times[i + 1][0]
                    curr_speaker, next_speaker = all_speaker_times[i][2], all_speaker_times[i + 1][2]
                    
                    # Different speakers within max_gap
                    if (curr_speaker != next_speaker and 
                        next_start - curr_end <= max_gap):
                        has_proximity = True
                        break
                
                if has_proximity:
                    # Create multi-speaker window
                    window = AliMeetingOverlap(
                        filename=filename,
                        start_time=window_start,
                        end_time=window_end,
                        speakers=list(speakers_in_window)
                    )
                    all_windows.append(window)
            
            # Move window forward (with some overlap for better coverage)
            current_time += window_duration * 0.5  # 50% overlap
    
    return all_windows

def filter_segments(segments: List[AliMeetingSegment], 
                   min_duration: float = 0.5,
                   max_duration: float = 15.0) -> List[AliMeetingSegment]:
    """
    Filter segments by duration constraints
    
    Args:
        segments: List of speech segments
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
        
    Returns:
        Filtered list of segments
    """
    return [
        segment for segment in segments
        if min_duration <= segment.duration <= max_duration
    ]

def get_alimeeting_splits() -> Dict[str, List[str]]:
    """
    Get AliMeeting train/dev/test splits based on meeting IDs
    
    Returns:
        Dictionary with 'train', 'dev', 'test' keys and meeting ID lists
    """
    # Based on AliMeeting corpus documentation
    # Train: majority of meetings, Dev/Test: smaller evaluation sets
    return {
        'train': ['R0003', 'R0004', 'R0005', 'R0006', 'R0007', 'R0008', 'R0009', 
                 'R0010', 'R0011', 'R0012', 'R0013', 'R0014', 'R0015', 'R0016',
                 'R0017', 'R0018', 'R0019', 'R0020', 'R0021', 'R0022', 'R0023',
                 'R0024', 'R0025', 'R0026', 'R0027', 'R0028', 'R0029', 'R0030',
                 'R0031', 'R0032', 'R0033', 'R0034', 'R0035', 'R0036', 'R0037',
                 'R0038', 'R0039', 'R0040', 'R0041', 'R0042', 'R0043', 'R0044',
                 'R0045', 'R0046', 'R0047', 'R0048', 'R0049', 'R0050', 'R0051',
                 'R0052', 'R0053', 'R0054', 'R0055', 'R0056', 'R0057', 'R0058',
                 'R0059', 'R0060', 'R0061', 'R0062', 'R0063', 'R0064', 'R0065',
                 'R0066', 'R0067', 'R0068', 'R0069', 'R0070', 'R0071', 'R0072',
                 'R0073', 'R0074', 'R0075', 'R0076', 'R0077', 'R0078', 'R0079',
                 'R0080', 'R0081', 'R0082', 'R0083', 'R0084', 'R0085', 'R0086',
                 'R0087', 'R0088', 'R0089', 'R0090', 'R0091', 'R0092', 'R0093',
                 'R0094', 'R0095', 'R0096', 'R0097', 'R0098', 'R0099', 'R0100',
                 'R0101', 'R0102', 'R0103', 'R0104', 'R0105', 'R0106', 'R0107',
                 'R0108', 'R0109', 'R0110', 'R0111', 'R0112', 'R0113', 'R0114',
                 'R0115', 'R0116', 'R0117', 'R0118'],
        'dev': ['R0119', 'R0120', 'R0121', 'R0122', 'R0123', 'R0124', 'R0125',
                'R0126', 'R0127', 'R0128', 'R0129', 'R0130', 'R0131', 'R0132'],
        'test': ['R0133', 'R0134', 'R0135', 'R0136', 'R0137', 'R0138', 'R0139',
                 'R0140', 'R0141', 'R0142', 'R0143', 'R0144', 'R0145', 'R0146']
    }

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        textgrid_file = sys.argv[1]
        segments = parse_alimeeting_textgrid(textgrid_file)
        overlaps = find_overlapping_speech(segments)
        
        print(f"Parsed {len(segments)} segments")
        print(f"Found {len(overlaps)} overlapping speech regions")
        
        if segments:
            print(f"Duration range: {min(s.duration for s in segments):.2f}s - {max(s.duration for s in segments):.2f}s")