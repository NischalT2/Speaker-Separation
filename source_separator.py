import os
import torch
import numpy as np
from typing import Tuple
import warnings


class SourceSeparator:
    """
    Source separator using SpeechBrain SepFormer model for separating overlapping speakers.
    """
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the source separator with a SpeechBrain SepFormer model.
        
        Args:
            model_name: Hugging Face model name or None to use default
            device: torch device ('cuda' or 'cpu'), None for auto-detection
        """
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        # Use sepformer-wsj02mix model
        self.model_name = model_name or "speechbrain/sepformer-wsj02mix"
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the source separation model."""
        try:
            print("Loading source separation model...")
            
            try:
                from speechbrain.inference.separation import SepformerSeparation
                from huggingface_hub import hf_hub_download
                
                # Patch hf_hub_download to accept use_auth_token for compatibility
                original_hf_hub_download = hf_hub_download
                def patched_hf_hub_download(*args, **kwargs):
                    if 'use_auth_token' in kwargs:
                        # Convert use_auth_token to token for newer huggingface_hub
                        token = kwargs.pop('use_auth_token')
                        if token:
                            kwargs['token'] = token
                    return original_hf_hub_download(*args, **kwargs)
                
                # Temporarily patch the function
                import huggingface_hub
                huggingface_hub.hf_hub_download = patched_hf_hub_download
                
                try:
                    # Load SpeechBrain SepFormer model (wsj02mix)
                    model_to_load = self.model_name or "speechbrain/sepformer-wsj02mix"
                    savedir = f"pretrained_models/{model_to_load.replace('/', '_')}"
                    
                    # First, create a dummy custom.py file if it doesn't exist
                    # This prevents SpeechBrain from trying to download it
                    import os
                    os.makedirs(savedir, exist_ok=True)
                    custom_py_path = os.path.join(savedir, "custom.py")
                    if not os.path.exists(custom_py_path):
                        # Create an empty custom.py file
                        with open(custom_py_path, 'w') as f:
                            f.write("# Empty custom.py - not required for this model\n")
                    
                    # Load model
                    self.model = SepformerSeparation.from_hparams(
                        source=model_to_load,
                        savedir=savedir,
                        run_opts={"device": str(self.device)},
                        use_auth_token=False  # Model is public, no token needed
                    )
                    self.model_name = model_to_load  # Update model name for correct sample rate
                    print(f"Loaded model {model_to_load} from SpeechBrain")
                finally:
                    # Restore original function
                    huggingface_hub.hf_hub_download = original_hf_hub_download
                    
            except ImportError:
                raise RuntimeError(
                    "SpeechBrain library not found. Install with: pip install speechbrain"
                )
                
        except Exception as e:
            warnings.warn(
                f"Could not load SpeechBrain SepFormer model: {e}. "
                "Source separation will use fallback method."
            )
            self.model = None


    def separate(self, mixed_audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate mixed audio into two speaker sources.
        
        Args:
            mixed_audio: Mixed audio signal as numpy array (1D)
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (source1, source2) as numpy arrays
        """
        if self.model is None:
            return self._fallback_separation(mixed_audio)

        try:
            import librosa
            
            # 1. CRITICAL: Resample to 8kHz (wsj02mix model requirement)
            target_rate = 8000
            if sample_rate != target_rate:
                mixed_audio_resampled = librosa.resample(mixed_audio, orig_sr=sample_rate, target_sr=target_rate)
            else:
                mixed_audio_resampled = mixed_audio

            # 2. Convert to Tensor (Batch, Time)
            mixed_tensor = torch.from_numpy(mixed_audio_resampled).float().unsqueeze(0).to(self.device)
            
            # 3. Separate
            with torch.no_grad():
                # SpeechBrain returns [Batch, Time, Sources]
                est_sources = self.model.separate_batch(mixed_tensor)
                
                source1_8k = est_sources[0, :, 0].cpu().numpy()
                source2_8k = est_sources[0, :, 1].cpu().numpy()
            
            # 4. Resample BACK to your original sample rate (e.g., 16kHz)
            if sample_rate != target_rate:
                source1 = librosa.resample(source1_8k, orig_sr=target_rate, target_sr=sample_rate)
                source2 = librosa.resample(source2_8k, orig_sr=target_rate, target_sr=sample_rate)
            else:
                source1, source2 = source1_8k, source2_8k
            
            # 5. Balance volumes and normalize outputs
            # First, check energy levels to balance volumes
            source1_energy = np.sum(np.abs(source1))
            source2_energy = np.sum(np.abs(source2))
            
            # Balance volumes: boost quieter source to be closer to louder source
            # But don't over-amplify (max 3x boost to avoid noise amplification)
            if source2_energy > 0 and source1_energy > 0:
                energy_ratio = source1_energy / source2_energy
                if energy_ratio > 2.0:
                    # Source1 is much louder, boost source2
                    boost_factor = min(energy_ratio / 2.0, 3.0)  # Cap at 3x
                    source2 = source2 * boost_factor
                elif energy_ratio < 0.5:
                    # Source2 is much louder, boost source1
                    boost_factor = min((1.0 / energy_ratio) / 2.0, 3.0)
                    source1 = source1 * boost_factor
            
            # Normalize both to same peak level to prevent clipping
            def normalize_audio(audio, target_max=0.95):
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    return audio * (target_max / max_val)
                return audio
            
            source1 = normalize_audio(source1)
            source2 = normalize_audio(source2)
            
            return source1, source2
                
        except Exception as e:
            print(f"SpeechBrain separation failed: {e}")
            return self._fallback_separation(mixed_audio)
    
    def _fallback_separation(self, mixed_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Basic fallback separation method (not ideal, but provides output).
        Splits audio into two channels using simple energy-based approach.
        """
        # Very basic approach: split by energy or use simple masking
        # This is a placeholder - real separation needs proper models
        half_len = len(mixed_audio) // 2
        
        # Simple split (this is not real separation, just a fallback)
        source1 = np.concatenate([mixed_audio[:half_len], np.zeros(len(mixed_audio) - half_len)])
        source2 = np.concatenate([np.zeros(half_len), mixed_audio[half_len:]])
        
        return source1, source2
    
    # def separate(self, mixed_audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Separate mixed audio into two speaker sources.
        
    #     Args:
    #         mixed_audio: Mixed audio signal as numpy array (1D)
    #         sample_rate: Sample rate of the audio
            
    #     Returns:
    #         Tuple of (source1, source2) as numpy arrays
    #     """
    #     # Validate input - skip empty or too short segments
    #     if len(mixed_audio) == 0:
    #         return np.zeros(1), np.zeros(1)
        
    #     # SpeechBrain models typically need at least a few hundred samples
    #     min_samples = 100
    #     if len(mixed_audio) < min_samples:
    #         return self._fallback_separation(mixed_audio)
        
    #     if self.model is None:
    #         # Fallback: simple energy-based separation (very basic)
    #         # This is not ideal but provides a fallback
    #         warnings.warn(
    #             "Source separation model not available. Using basic fallback method. "
    #             "Results may be poor. Consider installing a proper model."
    #         )
    #         return self._fallback_separation(mixed_audio)
        
    #     try:
    #         # SpeechBrain SepFormer expects audio as torch tensor
    #         # Convert numpy array to torch tensor: (samples,)
    #         if len(mixed_audio.shape) == 1:
    #             mixed_tensor = torch.from_numpy(mixed_audio).float()
    #         else:
    #             # If multi-channel, take first channel
    #             mixed_tensor = torch.from_numpy(mixed_audio[:, 0] if mixed_audio.shape[1] > 1 else mixed_audio).float()
            
    #         # SpeechBrain SepFormer expects 16kHz audio
    #         if sample_rate != 16000:
    #             import librosa
    #             mixed_audio_resampled = librosa.resample(mixed_audio, orig_sr=sample_rate, target_sr=16000)
    #             mixed_tensor = torch.from_numpy(mixed_audio_resampled).float()
            
    #         # Move to device
    #         mixed_tensor = mixed_tensor.to(self.device)
            
    #         # SpeechBrain SepFormer separates the audio
    #         # The model returns separated sources as a tensor
    #         with torch.no_grad():
    #             # Separate sources - SpeechBrain returns (sources, samples) or (batch, sources, samples)
    #             separated = self.model.separate_batch(mixed_tensor.unsqueeze(0))
                
    #             # Handle different output shapes
    #             if len(separated.shape) == 2:
    #                 # Shape: (sources, samples)
    #                 if separated.shape[0] >= 2:
    #                     source1 = separated[0].cpu().numpy()
    #                     source2 = separated[1].cpu().numpy()
    #                 else:
    #                     source1 = separated[0].cpu().numpy()
    #                     source2 = np.zeros_like(source1)
    #             elif len(separated.shape) == 3:
    #                 # Shape: (batch, sources, samples) or (batch, samples, sources)
    #                 # Check if it's (batch, samples, sources) or (batch, sources, samples)
    #                 if separated.shape[2] >= 2:  # Likely (batch, samples, sources)
    #                     source1 = separated[0, :, 0].cpu().numpy()
    #                     source2 = separated[0, :, 1].cpu().numpy()
    #                 elif separated.shape[1] >= 2:  # Likely (batch, sources, samples)
    #                     source1 = separated[0, 0].cpu().numpy()
    #                     source2 = separated[0, 1].cpu().numpy()
    #                 else:
    #                     source1 = separated[0, 0].cpu().numpy()
    #                     source2 = np.zeros_like(source1)
    #             else:
    #                 # Fallback if shape is unexpected
    #                 warnings.warn(f"Unexpected output shape from SepFormer: {separated.shape}. Using fallback.")
    #                 return self._fallback_separation(mixed_audio)
            
    #         # Resample back to original sample rate if needed
    #         if sample_rate != 16000:
    #             import librosa
    #             source1 = librosa.resample(source1, orig_sr=16000, target_sr=sample_rate)
    #             source2 = librosa.resample(source2, orig_sr=16000, target_sr=sample_rate)
            
    #         return source1, source2
            
    #     except Exception as e:
    #         warnings.warn(f"Error during source separation: {e}. Using fallback method.")
    #         return self._fallback_separation(mixed_audio)
    
    # def _fallback_separation(self, mixed_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Basic fallback separation method (not ideal, but provides output).
    #     Splits audio into two channels using simple energy-based approach.
    #     """
    #     # Very basic approach: split by energy or use simple masking
    #     # This is a placeholder - real separation needs proper models
    #     half_len = len(mixed_audio) // 2
        
    #     # Simple split (this is not real separation, just a fallback)
    #     source1 = np.concatenate([mixed_audio[:half_len], np.zeros(len(mixed_audio) - half_len)])
    #     source2 = np.concatenate([np.zeros(half_len), mixed_audio[half_len:]])
        
        # return source1, source2
