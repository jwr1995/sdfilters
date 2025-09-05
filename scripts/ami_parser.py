"""
AMI Corpus XML Parser

This module provides utilities for parsing AMI corpus annotation files,
including segment timing, speaker identification, and overlap detection.
Can be reused across different projects requiring AMI data processing.
"""
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class AMISegment:
    """Represents a single speech segment from AMI corpus"""
    meeting_id: str
    speaker_id: str
    start_time: float
    end_time: float
    duration: float
    segment_id: str
    channel: int = 0

@dataclass
class AMIOverlap:
    """Represents overlapping speech between multiple speakers"""
    meeting_id: str
    speakers: List[str]
    start_time: float
    end_time: float
    duration: float
    source_segments: List[str]

class AMIParser:
    """Parser for AMI corpus XML annotation files"""
    
    def __init__(self, ami_base_dir: str):
        """Initialize parser with AMI corpus base directory
        
        Args:
            ami_base_dir: Path to AMI corpus root directory
        """
        self.ami_base_dir = ami_base_dir
        self.segments_dir = os.path.join(ami_base_dir, 'segments')
        self.wav_dir = os.path.join(ami_base_dir, 'wav_db')
        
    def get_available_meetings(self) -> List[str]:
        """Get list of available meeting IDs
        
        Returns:
            List of meeting IDs found in wav_db directory
        """
        meeting_ids = []
        if os.path.exists(self.wav_dir):
            for item in os.listdir(self.wav_dir):
                item_path = os.path.join(self.wav_dir, item)
                if os.path.isdir(item_path):
                    meeting_ids.append(item)
        return sorted(meeting_ids)
    
    def get_meeting_speakers(self, meeting_id: str) -> List[str]:
        """Get list of speakers for a specific meeting
        
        Args:
            meeting_id: Meeting identifier (e.g., 'EN2001a')
            
        Returns:
            List of speaker IDs (e.g., ['A', 'B', 'C', 'D'])
        """
        speakers = []
        if os.path.exists(self.segments_dir):
            segment_files = [f for f in os.listdir(self.segments_dir) 
                           if f.startswith(f"{meeting_id}.") and f.endswith('.segments.xml')]
            
            for segment_file in segment_files:
                speaker_id = segment_file.split('.')[1]  # Extract speaker ID
                speakers.append(speaker_id)
        
        return sorted(speakers)
    
    def parse_meeting_segments(self, meeting_id: str) -> List[AMISegment]:
        """Parse all segments for a specific meeting
        
        Args:
            meeting_id: Meeting identifier (e.g., 'EN2001a')
            
        Returns:
            List of AMISegment objects for all speakers in the meeting
        """
        segments = []
        speakers = self.get_meeting_speakers(meeting_id)
        
        for speaker_id in speakers:
            speaker_segments = self.parse_speaker_segments(meeting_id, speaker_id)
            segments.extend(speaker_segments)
        
        return sorted(segments, key=lambda x: x.start_time)
    
    def parse_speaker_segments(self, meeting_id: str, speaker_id: str) -> List[AMISegment]:
        """Parse segments for a specific speaker in a meeting
        
        Args:
            meeting_id: Meeting identifier (e.g., 'EN2001a')
            speaker_id: Speaker identifier (e.g., 'A')
            
        Returns:
            List of AMISegment objects for the specified speaker
        """
        segments = []
        segment_file = f"{meeting_id}.{speaker_id}.segments.xml"
        segment_path = os.path.join(self.segments_dir, segment_file)
        
        if not os.path.exists(segment_path):
            return segments
        
        try:
            tree = ET.parse(segment_path)
            root = tree.getroot()
            
            for segment in root.findall('.//segment'):
                start_time = float(segment.get('transcriber_start', 0))
                end_time = float(segment.get('transcriber_end', 0))
                segment_id = segment.get('nite:id', '')
                channel = int(segment.get('channel', 0))
                
                if end_time > start_time:  # Valid segment
                    ami_segment = AMISegment(
                        meeting_id=meeting_id,
                        speaker_id=speaker_id,
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        segment_id=segment_id,
                        channel=channel
                    )
                    segments.append(ami_segment)
        
        except ET.ParseError as e:
            print(f"Error parsing {segment_file}: {e}")
        
        return segments
    
    def find_overlapping_speech(self, meeting_id: str, 
                               min_overlap_duration: float = 0.5,
                               max_speakers: int = 2) -> List[AMIOverlap]:
        """Find overlapping speech segments in a meeting
        
        Args:
            meeting_id: Meeting identifier
            min_overlap_duration: Minimum overlap duration in seconds
            max_speakers: Maximum number of speakers to consider in overlap
            
        Returns:
            List of AMIOverlap objects representing multi-speaker segments
        """
        segments = self.parse_meeting_segments(meeting_id)
        overlaps = []
        
        # Group segments by speaker
        speaker_segments = defaultdict(list)
        for segment in segments:
            speaker_segments[segment.speaker_id].append(segment)
        
        # Find pairwise overlaps
        speakers = list(speaker_segments.keys())
        for i, speaker1 in enumerate(speakers):
            for speaker2 in speakers[i+1:]:
                pairwise_overlaps = self._find_pairwise_overlaps(
                    speaker_segments[speaker1],
                    speaker_segments[speaker2],
                    meeting_id,
                    min_overlap_duration
                )
                overlaps.extend(pairwise_overlaps)
        
        return overlaps
    
    def _find_pairwise_overlaps(self, segments1: List[AMISegment], 
                               segments2: List[AMISegment],
                               meeting_id: str,
                               min_overlap_duration: float) -> List[AMIOverlap]:
        """Find overlaps between two speakers' segments"""
        overlaps = []
        
        for seg1 in segments1:
            for seg2 in segments2:
                # Check for temporal overlap
                overlap_start = max(seg1.start_time, seg2.start_time)
                overlap_end = min(seg1.end_time, seg2.end_time)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration >= min_overlap_duration:
                        overlap = AMIOverlap(
                            meeting_id=meeting_id,
                            speakers=[seg1.speaker_id, seg2.speaker_id],
                            start_time=overlap_start,
                            end_time=overlap_end,
                            duration=overlap_duration,
                            source_segments=[seg1.segment_id, seg2.segment_id]
                        )
                        overlaps.append(overlap)
        
        return overlaps
    
    def find_multispeaker_windows(self, meeting_id: str, 
                                 window_duration: float = 10.0,
                                 min_speakers: int = 2,
                                 max_gap: float = 2.0) -> List[AMIOverlap]:
        """
        Find multi-speaker windows containing multiple speakers in proximity
        (not necessarily overlapping)
        
        Args:
            meeting_id: Meeting identifier
            window_duration: Duration of each window in seconds
            min_speakers: Minimum number of speakers required in window
            max_gap: Maximum gap between speakers to consider them "in proximity"
            
        Returns:
            List of AMIOverlap objects representing multi-speaker windows
        """
        segments = self.parse_meeting_segments(meeting_id)
        if not segments:
            return []
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        
        # Find the time range of the meeting
        start_time = min(s.start_time for s in segments)
        end_time = max(s.end_time for s in segments)
        
        windows = []
        current_time = start_time
        
        while current_time + window_duration <= end_time:
            window_start = current_time
            window_end = current_time + window_duration
            
            # Find all speakers active in this window
            speakers_in_window = set()
            segments_in_window = []
            
            for segment in segments:
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
                    window = AMIOverlap(
                        meeting_id=meeting_id,
                        speakers=list(speakers_in_window),
                        start_time=window_start,
                        end_time=window_end,
                        duration=window_duration,
                        source_segments=[s.segment_id for s in segments_in_window]
                    )
                    windows.append(window)
            
            # Move window forward (with some overlap for better coverage)
            current_time += window_duration * 0.5  # 50% overlap
        
        return windows
    
    def get_audio_file_path(self, meeting_id: str, mic_type: str = 'sdm') -> Optional[str]:
        """Get the audio file path for a meeting
        
        Args:
            meeting_id: Meeting identifier
            mic_type: Microphone type ('sdm', 'ihm', 'ihm-mix')
            
        Returns:
            Relative path to audio file or None if not found
        """
        if mic_type == 'sdm':
            audio_file = f"{meeting_id}.Array1-01.wav"
        elif mic_type == 'ihm-mix':
            audio_file = f"{meeting_id}.Mix-Headset.wav"
        else:
            # ihm individual headset mics - return first one
            audio_file = f"{meeting_id}.Headset-0.wav"
        
        audio_path = os.path.join(self.wav_dir, meeting_id, 'audio', audio_file)
        if os.path.exists(audio_path):
            return f"wav_db/{meeting_id}/audio/{audio_file}"
        
        return None
    
    def filter_segments(self, segments: List[AMISegment],
                       min_duration: float = 0.1,
                       max_duration: float = 30.0) -> List[AMISegment]:
        """Filter segments by duration criteria
        
        Args:
            segments: List of segments to filter
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            
        Returns:
            Filtered list of segments
        """
        return [seg for seg in segments 
                if min_duration <= seg.duration <= max_duration]

# Convenience functions
def parse_ami_meeting(ami_base_dir: str, meeting_id: str) -> Tuple[List[AMISegment], List[AMIOverlap]]:
    """Parse a single AMI meeting and return segments and overlaps
    
    Args:
        ami_base_dir: Path to AMI corpus root directory
        meeting_id: Meeting identifier
        
    Returns:
        Tuple of (single_speaker_segments, multi_speaker_overlaps)
    """
    parser = AMIParser(ami_base_dir)
    segments = parser.parse_meeting_segments(meeting_id)
    overlaps = parser.find_overlapping_speech(meeting_id)
    return segments, overlaps

def get_all_ami_data(ami_base_dir: str) -> Dict[str, Tuple[List[AMISegment], List[AMIOverlap]]]:
    """Parse all available AMI meetings
    
    Args:
        ami_base_dir: Path to AMI corpus root directory
        
    Returns:
        Dictionary mapping meeting_id to (segments, overlaps)
    """
    parser = AMIParser(ami_base_dir)
    all_data = {}
    
    for meeting_id in parser.get_available_meetings():
        segments, overlaps = parse_ami_meeting(ami_base_dir, meeting_id)
        all_data[meeting_id] = (segments, overlaps)
        
    return all_data