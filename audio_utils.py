import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


def load_audio(file_path):
    """ Load WAV file and returns a tuple containing audio data and sample rate. """
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        return audio, 16000
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")


def save_audio(audio, file_path, sample_rate):
    """
    Save audio data to WAV file.
    
    Args:
        audio: Audio data as numpy array
        file_path: Path where to save the audio file
        sample_rate: Sample rate of the audio
    """
    try:
        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(file_path, audio, sample_rate)
    except Exception as e:
        raise ValueError(f"Error saving audio file {file_path}: {str(e)}")


def extract_segment(audio, start_time, end_time, sample_rate):
    """
    Extract audio segment between timestamps.
    
    Args:
        audio: Full audio data as numpy array
        start_time: Start time in seconds
        end_time: End time in seconds
        sample_rate: Sample rate of the audio
        
    Returns:
        numpy array: Extracted audio segment
    """
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Ensure indices are within bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample]


def combine_segments(segments, sample_rate):
    """
    Combine multiple audio segments into a single audio array.
    
    Args:
        segments: List of (start_time, end_time, audio_segment) tuples
        sample_rate: Sample rate of the audio
        
    Returns:
        numpy array: Combined audio
    """
    if not segments:
        return np.array([])
    
    # Calculate total duration
    total_duration = max(end_time for _, end_time, _ in segments)
    total_samples = int(total_duration * sample_rate)
    
    # Initialize output array
    combined = np.zeros(total_samples)
    
    # Add each segment at its corresponding time position
    for start_time, end_time, segment in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = start_sample + len(segment)
        
        if end_sample > len(combined):
            # Extend array if needed
            combined = np.pad(combined, (0, end_sample - len(combined)))
        
        combined[start_sample:end_sample] = segment
    
    return combined


def detect_overlapping_segments(segments):
    """
    Detect overlapping segments where multiple speakers are active simultaneously.
    
    Args:
        segments: List of (start_time, end_time, speaker) tuples from diarization
        
    Returns:
        tuple: (overlapping_intervals, non_overlapping_segments)
            - overlapping_intervals: List of (start, end, [speaker1, speaker2, ...]) tuples
            - non_overlapping_segments: List of (start, end, speaker) tuples
    """
    if not segments:
        return [], []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    overlapping_intervals = []
    non_overlapping_segments = []
    
    # Track active speakers at each point
    events = []
    for start, end, speaker in sorted_segments:
        events.append((start, 'start', speaker))
        events.append((end, 'end', speaker))
    
    # Sort events by time
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'start' else 1))
    
    active_speakers = set()
    overlap_start = None
    
    for time, event_type, speaker in events:
        if event_type == 'start':
            active_speakers.add(speaker)
            # If we now have 2+ speakers, start tracking overlap
            if len(active_speakers) >= 2 and overlap_start is None:
                overlap_start = time
        else:  # event_type == 'end'
            # Before removing, check if we're in an overlap
            if len(active_speakers) >= 2 and overlap_start is not None:
                # End the overlap interval
                overlapping_intervals.append((
                    overlap_start,
                    time,
                    list(active_speakers)
                ))
                overlap_start = None
            
            active_speakers.discard(speaker)
            
            # If we still have 2+ speakers after removal, continue overlap
            if len(active_speakers) >= 2:
                overlap_start = time
    
    # Build non-overlapping segments by checking each segment against overlaps
    # Only exclude segments that are FULLY contained within overlap intervals
    # Segments that partially overlap will be split to preserve non-overlapping portions
    for start, end, speaker in sorted_segments:
        # Check if this segment is fully contained within any overlap interval
        is_fully_overlapping = False
        for ovl_start, ovl_end, speakers in overlapping_intervals:
            # Segment is fully contained if it's completely within the overlap
            if ovl_start <= start and end <= ovl_end:
                is_fully_overlapping = True
                break
        
        if is_fully_overlapping:
            # Entire segment is within an overlap - will be handled by source separation
            continue
        
        # Check if segment partially overlaps or doesn't overlap at all
        has_partial_overlap = False
        overlapping_parts = []
        for ovl_start, ovl_end, speakers in overlapping_intervals:
            if not (end <= ovl_start or start >= ovl_end):
                # There's some overlap
                overlapping_parts.append((max(start, ovl_start), min(end, ovl_end)))
                has_partial_overlap = True
        
        if not has_partial_overlap:
            # No overlap at all - entire segment is non-overlapping
            non_overlapping_segments.append((start, end, speaker))
        else:
            # Segment partially overlaps - split to preserve non-overlapping portions
            # Merge overlapping parts first
            if overlapping_parts:
                overlapping_parts.sort(key=lambda x: x[0])
                merged_overlaps = [overlapping_parts[0]]
                for ovl_start, ovl_end in overlapping_parts[1:]:
                    if ovl_start <= merged_overlaps[-1][1]:
                        merged_overlaps[-1] = (merged_overlaps[-1][0], max(merged_overlaps[-1][1], ovl_end))
                    else:
                        merged_overlaps.append((ovl_start, ovl_end))
                
                # Extract non-overlapping portions
                current_start = start
                for ovl_start, ovl_end in merged_overlaps:
                    # Add segment before this overlap (if any)
                    if current_start < ovl_start:
                        non_overlapping_segments.append((current_start, ovl_start, speaker))
                    # Move past this overlap
                    current_start = max(current_start, ovl_end)
                
                # Add remaining segment after last overlap (if any)
                if current_start < end:
                    non_overlapping_segments.append((current_start, end, speaker))
    
    return overlapping_intervals, non_overlapping_segments


def merge_segments_with_overlaps(non_overlapping_segments, overlapping_segments, sample_rate):
    """
    combine non-overlapping segments and separated overlapping segments.
    
    Args:
        non_overlapping_segments: Dict mapping speaker to list of (start, end, audio) tuples
        overlapping_segments: Dict mapping speaker to list of (start, end, audio) tuples from separation
        sample_rate: Sample rate of the audio
        
    Returns:
        Dict mapping speaker to combined audio array
    """
    all_speakers = set(non_overlapping_segments.keys()) | set(overlapping_segments.keys())
    merged_audio = {}
    
    for speaker in all_speakers:
        # Combine all segments for this speaker
        all_segments = []
        
        # Add non-overlapping segments
        if speaker in non_overlapping_segments:
            all_segments.extend(non_overlapping_segments[speaker])
        
        # Add overlapping segments
        if speaker in overlapping_segments:
            all_segments.extend(overlapping_segments[speaker])
        
        # Sort by start time
        all_segments.sort(key=lambda x: x[0])
        
        # Use combine_segments to merge
        merged_audio[speaker] = combine_segments(all_segments, sample_rate)
    
    return merged_audio
