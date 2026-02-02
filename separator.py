import os
import torch
import torch.serialization
from pyannote.audio import Pipeline
from audio_utils import load_audio, save_audio, extract_segment, combine_segments, detect_overlapping_segments, merge_segments_with_overlaps
from pathlib import Path
import numpy as np
import warnings

import torchaudio

if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg']

from pyannote.audio.core.task import Problem, Resolution, Specifications
torch.serialization.add_safe_globals([Problem, 
                                    Resolution, 
                                    Specifications, 
                                    torch.torch_version.TorchVersion, 
                                    np.core.multiarray.scalar,
                                    np.dtype])

# class that uses pyannote.audio for diarization and separates audio into individual speaker files.
class SpeakerSeparator:
    
    def __init__(self, model_name="pyannote/speaker-diarization-3.1", use_source_separation=True):
        """
        Initialize the speaker separator with a diarization pipeline.
        
        Args:
            model_name: Name of the diarization model to use
            use_source_separation: Whether to enable source separation for overlapping speech
        """
        # Check for Hugging Face token
        hf_token = os.environ.get('HF_TOKEN')
        
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable not set. "
                "Please set it with: export HF_TOKEN='your_token_here'\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )
        
        # # Fix for PyTorch 2.6+ security: allow required classes in model files
        # try:
        #     from pyannote.audio.core.task import Specifications
        #     torch.serialization.add_safe_globals([
        #         torch.torch_version.TorchVersion,
        #         Specifications
        #     ])
        # except (AttributeError, ImportError):
        #     # Older PyTorch versions or import issues - try without Specifications
        #     try:
        #         torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
        #     except AttributeError:
        #         # Older PyTorch versions don't have this, which is fine
        #         pass
        
        try:
            # Load the diarization pipeline
            print("Loading speaker diarization model...")
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                token=hf_token
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                print("Using GPU for processing")
            else:
                print("Using CPU for processing")
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization model: {str(e)}\n"
                "Make sure you have:\n"
                "1. Accepted the model license at https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Set your HF_TOKEN environment variable correctly"
            )
        
        # Initialize source separator if enabled
        self.use_source_separation = use_source_separation
        self.source_separator = None
        if use_source_separation:
            try:
                from source_separator import SourceSeparator
                self.source_separator = SourceSeparator()
            except Exception as e:
                warnings.warn(
                    f"Could not initialize source separator: {e}. "
                    "Overlapping speech will be handled with basic method."
                )
                self.use_source_separation = False
    
    def _map_separated_sources_to_speakers(self, separated_sources, overlap_start, overlap_end, 
                                           speaker_segments, sample_rate, consistency_mapping=None):
        """
        Map separated sources from SpeechBrain SepFormer to diarization speaker labels.
        
        Uses temporal alignment: matches separated sources to speakers based on
        which source aligns better with adjacent non-overlapping segments.
        
        Args:
            separated_sources: Tuple of (source1, source2) numpy arrays
            overlap_start: Start time of overlap interval
            overlap_end: End time of overlap interval
            speaker_segments: Dict mapping speaker labels to lists of (start, end, audio) segments
            sample_rate: Sample rate of the audio
            consistency_mapping: Optional tuple (source1_speaker, source2_speaker) for consistency
            
        Returns:
            Tuple of (mapping_dict, source_mapping_info) where:
            - mapping_dict: Dict mapping speaker labels to separated audio arrays
            - source_mapping_info: Tuple (source1_speaker, source2_speaker) indicating which source went to which speaker
        """
        source1, source2 = separated_sources
        speaker_labels = sorted(speaker_segments.keys())
        
        if len(speaker_labels) < 2:
            # Only one speaker, return first source
            return ({speaker_labels[0]: source1}, (speaker_labels[0], None))
        
        # Get non-overlapping segments adjacent to the overlap for each speaker
        speaker1 = speaker_labels[0]
        speaker2 = speaker_labels[1]
        
        # Strategy 0: Use consistency mapping if available (maintains consistency across overlaps)
        if consistency_mapping is not None:
            source1_speaker, source2_speaker = consistency_mapping
            return ({source1_speaker: source1, source2_speaker: source2}, (source1_speaker, source2_speaker))
        
        # Find segments just before or after the overlap for each speaker
        speaker1_segments = speaker_segments[speaker1]
        speaker2_segments = speaker_segments[speaker2]
        
        # Simple approach: use correlation with adjacent segments
        # Find the segment closest to overlap for each speaker
        speaker1_adjacent = None
        speaker2_adjacent = None
        
        for start, end, audio in speaker1_segments:
            # Check if segment is close to overlap (within 1 second)
            if abs(end - overlap_start) < 1.0 or abs(start - overlap_end) < 1.0:
                if speaker1_adjacent is None or abs(end - overlap_start) < abs(speaker1_adjacent[1] - overlap_start):
                    speaker1_adjacent = (start, end, audio)
        
        for start, end, audio in speaker2_segments:
            if abs(end - overlap_start) < 1.0 or abs(start - overlap_end) < 1.0:
                if speaker2_adjacent is None or abs(end - overlap_start) < abs(speaker2_adjacent[1] - overlap_start):
                    speaker2_adjacent = (start, end, audio)
        
        # If we have adjacent segments, use correlation to match
        if speaker1_adjacent is not None and speaker2_adjacent is not None:
            import numpy as np
            # Extract a small window from the end of adjacent segments
            window_size = min(len(speaker1_adjacent[2]), len(speaker2_adjacent[2]), 
                            int(0.5 * sample_rate))  # 0.5 seconds
            
            speaker1_window = speaker1_adjacent[2][-window_size:]
            speaker2_window = speaker2_adjacent[2][-window_size:]
            
            # Extract corresponding windows from separated sources
            source1_window = source1[:window_size]
            source2_window = source2[:window_size]
            
            # Normalize for correlation
            def normalize(x):
                x = x - np.mean(x)
                std = np.std(x)
                return x / (std + 1e-10) if std > 1e-10 else x
            
            speaker1_norm = normalize(speaker1_window)
            speaker2_norm = normalize(speaker2_window)
            source1_norm = normalize(source1_window)
            source2_norm = normalize(source2_window)
            
            # Compute correlations
            corr1_with_speaker1 = np.corrcoef(speaker1_norm, source1_norm)[0, 1]
            corr1_with_speaker2 = np.corrcoef(speaker2_norm, source1_norm)[0, 1]
            corr2_with_speaker1 = np.corrcoef(speaker1_norm, source2_norm)[0, 1]
            corr2_with_speaker2 = np.corrcoef(speaker2_norm, source2_norm)[0, 1]
            
            # Determine best mapping
            # Option 1: source1 -> speaker1, source2 -> speaker2
            score1 = corr1_with_speaker1 + corr2_with_speaker2
            # Option 2: source1 -> speaker2, source2 -> speaker1
            score2 = corr1_with_speaker2 + corr2_with_speaker1
            
            # Only use correlation if correlations are strong enough (threshold: 0.15)
            # If both correlations are weak/negative, fall back to energy matching
            if (abs(corr1_with_speaker1) > 0.15 or abs(corr1_with_speaker2) > 0.15 or 
                abs(corr2_with_speaker1) > 0.15 or abs(corr2_with_speaker2) > 0.15):
                if score1 >= score2:
                    return ({speaker1: source1, speaker2: source2}, (speaker1, speaker2))
                else:
                    return ({speaker1: source2, speaker2: source1}, (speaker2, speaker1))
        
        # Strategy 2: Use energy matching with adjacent segments
        # Match based on which source has energy profile closer to adjacent segments
        if speaker1_adjacent is not None or speaker2_adjacent is not None:
            source1_energy = np.sum(np.abs(source1))
            source2_energy = np.sum(np.abs(source2))
            
            if speaker1_adjacent is not None and speaker2_adjacent is not None:
                speaker1_energy = np.sum(np.abs(speaker1_adjacent[2]))
                speaker2_energy = np.sum(np.abs(speaker2_adjacent[2]))
                
                # Match based on energy similarity
                energy_diff1 = abs(source1_energy - speaker1_energy) + abs(source2_energy - speaker2_energy)
                energy_diff2 = abs(source1_energy - speaker2_energy) + abs(source2_energy - speaker1_energy)
                
                if energy_diff1 <= energy_diff2:
                    return ({speaker1: source1, speaker2: source2}, (speaker1, speaker2))
                else:
                    return ({speaker1: source2, speaker2: source1}, (speaker2, speaker1))
            elif speaker1_adjacent is not None:
                # Only speaker1 has adjacent segment - match source with higher energy to speaker1
                speaker1_energy = np.sum(np.abs(speaker1_adjacent[2]))
                if abs(source1_energy - speaker1_energy) <= abs(source2_energy - speaker1_energy):
                    return ({speaker1: source1, speaker2: source2}, (speaker1, speaker2))
                else:
                    return ({speaker1: source2, speaker2: source1}, (speaker2, speaker1))
            else:  # speaker2_adjacent is not None
                speaker2_energy = np.sum(np.abs(speaker2_adjacent[2]))
                if abs(source2_energy - speaker2_energy) <= abs(source1_energy - speaker2_energy):
                    return ({speaker1: source1, speaker2: source2}, (speaker1, speaker2))
                else:
                    return ({speaker1: source2, speaker2: source1}, (speaker2, speaker1))
        
        # Strategy 3: Default mapping (fallback - should be rare)
        return ({speaker1: source1, speaker2: source2}, (speaker1, speaker2))
    
    def separate(self, input_file, output_dir="./output"):
        """ Separate two speakers from an audio file into separate WAV files. """
        # Validate input file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not input_file.lower().endswith('.wav'):
            print(f"Input file is not a .wav file: {input_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        print(f"Loading audio file: {input_file}")
        audio, sample_rate = load_audio(input_file)
        print(f"Audio loaded: {len(audio)/sample_rate:.2f} seconds, {sample_rate} Hz sample rate")
        
        # Run diarization
        print("Running speaker diarization...")
        try:
            # Try to set num_speakers in pipeline parameters if supported
            # Some pyannote versions need it set differently
            try:
                # Try as call parameter first
                diarization = self.pipeline(input_file, num_speakers=2)
            except (TypeError, ValueError) as e:
                # If that doesn't work, try setting it in pipeline parameters
                try:
                    # Try to modify pipeline parameters
                    if hasattr(self.pipeline, 'parameters'):
                        self.pipeline.parameters['num_speakers'] = 2
                    diarization = self.pipeline(input_file)
                except:
                    # Last resort: just run without num_speakers
                    diarization = self.pipeline(input_file)
        except Exception as e:
            raise RuntimeError(f"Error during diarization: {str(e)}")
        
        # Extract speaker segments from diarization
        print("Extracting speaker segments...")
        all_segments = []
        
        # # New pyannote.audio 4.0+ API: DiarizeOutput.serialize() returns dict
        # diarization_dict = diarization.serialize()
        # for segment_info in diarization_dict.get('diarization', []):
        #     start = segment_info['start']
        #     end = segment_info['end']
        #     speaker = segment_info['speaker']
        #     all_segments.append((start, end, speaker))
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            all_segments.append((turn.start, turn.end, speaker))
        
        # Debug: Print diarization results
        unique_speakers = set(seg[2] for seg in all_segments)
        print(f"Diarization found {len(unique_speakers)} unique speaker(s): {sorted(unique_speakers)}")
        print(f"Total segments detected: {len(all_segments)}")
        if len(all_segments) > 0:
            print(f"First few segments: {all_segments[:3]}")
        
        
        # Detect overlapping segments
        overlapping_intervals, non_overlapping_segments_list = detect_overlapping_segments(all_segments)
        
        print(f"Detected {len(overlapping_intervals)} overlapping intervals and {len(non_overlapping_segments_list)} non-overlapping segments")
        
        # Organize non-overlapping segments by speaker
        non_overlapping_by_speaker = {}
        for start, end, speaker in non_overlapping_segments_list:
            if speaker not in non_overlapping_by_speaker:
                non_overlapping_by_speaker[speaker] = []
            segment_audio = extract_segment(audio, start, end, sample_rate)
            non_overlapping_by_speaker[speaker].append((start, end, segment_audio))
        
        # Handle overlapping segments with source separation
        overlapping_by_speaker = {}
        # Track source-to-speaker mapping for consistency across overlaps
        # Maps (speaker1, speaker2) tuple to (source1_speaker, source2_speaker) tuple
        source_speaker_mapping = {}
        
        if overlapping_intervals and self.use_source_separation and self.source_separator:
            print(f"Processing {len(overlapping_intervals)} overlapping intervals with source separation...")
            for ovl_start, ovl_end, speakers in overlapping_intervals:
                # Extract the overlapping audio segment
                overlap_audio = extract_segment(audio, ovl_start, ovl_end, sample_rate)
                
                # Skip empty or too short segments
                if len(overlap_audio) < 100:  # Minimum samples needed
                    # Assign to first speaker as fallback
                    if speakers:
                        speaker = speakers[0]
                        if speaker not in overlapping_by_speaker:
                            overlapping_by_speaker[speaker] = []
                        overlapping_by_speaker[speaker].append((ovl_start, ovl_end, overlap_audio))
                    continue
                
                # Apply source separation
                try:
                    separated_sources = self.source_separator.separate(overlap_audio, sample_rate)
                    source1, source2 = separated_sources
                    
                    # Map separated sources to speaker labels
                    # First, we need speaker_segments dict for mapping function
                    temp_speaker_segments = {}
                    for speaker in speakers:
                        temp_speaker_segments[speaker] = non_overlapping_by_speaker.get(speaker, [])
                    
                    # Get speaker pair key for consistency tracking
                    speaker_pair = tuple(sorted(speakers))
                    
                    # Get existing mapping for consistency, or None if first time
                    existing_mapping = source_speaker_mapping.get(speaker_pair) if speaker_pair in source_speaker_mapping else None
                    
                    mapped_sources, source_mapping_info = self._map_separated_sources_to_speakers(
                        separated_sources, ovl_start, ovl_end,
                        temp_speaker_segments, sample_rate,
                        existing_mapping
                    )
                    
                    # Update consistency mapping: track which source (0 or 1) maps to which speaker
                    if speaker_pair not in source_speaker_mapping:
                        # First time seeing this speaker pair - store the mapping info returned by the function
                        source1_speaker, source2_speaker = source_mapping_info
                        if source1_speaker and source2_speaker:
                            source_speaker_mapping[speaker_pair] = (source1_speaker, source2_speaker)
                    
                    # Add separated segments to overlapping_by_speaker
                    for speaker, separated_audio in mapped_sources.items():
                        if speaker not in overlapping_by_speaker:
                            overlapping_by_speaker[speaker] = []
                        overlapping_by_speaker[speaker].append((ovl_start, ovl_end, separated_audio))
                        
                except Exception as e:
                    warnings.warn(f"Error separating overlap [{ovl_start:.2f}-{ovl_end:.2f}]: {e}. Using fallback.")
                    # Fallback: assign overlap to first speaker
                    if speakers:
                        speaker = speakers[0]
                        if speaker not in overlapping_by_speaker:
                            overlapping_by_speaker[speaker] = []
                        overlapping_by_speaker[speaker].append((ovl_start, ovl_end, overlap_audio))
        elif overlapping_intervals:
            # No source separation available, assign overlaps to speakers (fallback)
            print(f"Source separation not available. Assigning {len(overlapping_intervals)} overlaps to speakers...")
            for ovl_start, ovl_end, speakers in overlapping_intervals:
                overlap_audio = extract_segment(audio, ovl_start, ovl_end, sample_rate)
                # Assign to first speaker as fallback
                if speakers:
                    speaker = speakers[0]
                    if speaker not in overlapping_by_speaker:
                        overlapping_by_speaker[speaker] = []
                    overlapping_by_speaker[speaker].append((ovl_start, ovl_end, overlap_audio))
        # Merge non-overlapping and overlapping segments
        merged_audio = {}
        speaker_labels = sorted(unique_speakers)
        
        for speaker in speaker_labels:
            # Create a silent canvas the exact length of the original audio
            speaker_final_audio = np.zeros_like(audio)
            
            placed_segments = []
            
            # 1. Add Clean (Non-overlapping) Segments
            if speaker in non_overlapping_by_speaker:
                for start, end, segment_audio in non_overlapping_by_speaker[speaker]:
                    start_sample = int(start * sample_rate)
                    # Ensure start_sample is within bounds
                    if start_sample >= len(speaker_final_audio):
                        continue  # Skip segments that are beyond the audio length
                    length = min(len(segment_audio), len(speaker_final_audio) - start_sample)
                    if length <= 0:
                        continue  # Skip zero or negative length segments
                    speaker_final_audio[start_sample : start_sample + length] = segment_audio[:length]
                    placed_segments.append((start, end, "non_overlapping"))
            
            # 2. Add Separated (Cleaned) Overlapping Segments
            if speaker in overlapping_by_speaker:
                for start, end, separated_audio in overlapping_by_speaker[speaker]:
                    start_sample = int(start * sample_rate)
                    # Ensure start_sample is within bounds
                    if start_sample >= len(speaker_final_audio):
                        continue  # Skip segments that are beyond the audio length
                    length = min(len(separated_audio), len(speaker_final_audio) - start_sample)
                    if length <= 0:
                        continue  # Skip zero or negative length segments
                    # This replaces the 'mixed' audio with the 'separated' speaker voice
                    speaker_final_audio[start_sample : start_sample + length] = separated_audio[:length]
                    placed_segments.append((start, end, "overlapping"))
            
            merged_audio[speaker] = speaker_final_audio

        # SPECIAL CASE: If only 1 speaker found, but we expect 2, 
        # run source separation on the whole thing as a last resort.
        # Otherwise, use hybrid approach: diarization for non-overlapping, source separation for overlaps
        if len(unique_speakers) == 1 and self.use_source_separation and self.source_separator:
            print("Only 1 speaker detected by diarization. Running full-file source separation to extract both speakers...")
            try:
                # Separates the entire audio into two tracks using source separation
                source1, source2 = self.source_separator.separate(audio, sample_rate)
                
                # Replace merged_audio with the separated sources
                merged_audio = {
                    "SPEAKER_00": source1,
                    "SPEAKER_01": source2
                }
                # Update unique_speakers so the rest of the script saves both
                unique_speakers = {"SPEAKER_00", "SPEAKER_01"}
                speaker_labels = sorted(unique_speakers)
            except Exception as e:
                print(f"Full separation fallback failed: {e}")
                warnings.warn("Source separation failed. Output may only contain 1 speaker.")

        
        # Currently hardcoded to support only 2 speakers.
        num_speakers = len(merged_audio)
        if num_speakers == 0:
            raise ValueError("No speakers detected in the audio file")
        elif num_speakers == 1:
            print(f"Warning: Only 1 speaker detected. Expected 2 speakers.")
        elif num_speakers > 2:
            print(f"Warning: {num_speakers} speakers detected. Expected 2. Using first 2 speakers.")
            # Take only the first 2 speakers (by total audio length)
            speaker_lengths = {
                speaker: len(audio_data)
                for speaker, audio_data in merged_audio.items()
            }
            top_speakers = sorted(speaker_lengths.items(), key=lambda x: x[1], reverse=True)[:2]
            merged_audio = {
                speaker: merged_audio[speaker]
                for speaker, _ in top_speakers
            }
        
        # Get speaker labels (sorted for consistent output)
        speaker_labels = sorted(merged_audio.keys())
        
        # Check final output quality (correlation between outputs)
        if len(speaker_labels) >= 2:
            audio1 = merged_audio[speaker_labels[0]]
            audio2 = merged_audio[speaker_labels[1]]
            if len(audio1) == len(audio2):
                correlation = float(np.corrcoef(audio1, audio2)[0, 1])
                # Print separation quality info
                if abs(correlation) < 0.2:
                    print(f"✓ Excellent separation quality (correlation: {correlation:.3f} - close to 0 is ideal)")
                elif abs(correlation) < 0.5:
                    print(f"✓ Good separation quality (correlation: {correlation:.3f})")
                else:
                    print(f"⚠ Moderate separation quality (correlation: {correlation:.3f} - lower is better)")
        
        # Save output files
        print("Saving output files...")
        for i, speaker in enumerate(speaker_labels[:2], 1):  # Only process first 2 speakers
            combined_audio = merged_audio[speaker]
            
            output_file = output_path / f"speaker_{i}.wav"
            save_audio(combined_audio, str(output_file), sample_rate)
            
            total_duration = len(combined_audio) / sample_rate
            print(f"Speaker {i} ({speaker}): {total_duration:.2f} seconds -> {output_file}")
        
        print(f"\nSeparation complete! Output files saved to: {output_dir}")
        return speaker_labels[:2]
