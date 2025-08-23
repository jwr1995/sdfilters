"""Whisper model wrappers for feature extraction and audio processing.

This module provides wrapper classes around Hugging Face Whisper models
for extracting audio features and mel spectrograms for speech quality assessment.
"""

from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperForConditionalGeneration


# Audio processing constants
SAMPLE_RATE = 16000  # Required sample rate for Whisper
N_FFT = 400  # FFT window size
N_MELS = 80  # Number of mel frequency bins
HOP_LENGTH = 160  # STFT hop length
CHUNK_LENGTH = 30  # Maximum audio chunk length in seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # Token stride for convolutions


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute log-Mel spectrogram from audio waveform.
    
    This function converts audio waveform to log-Mel spectrogram representation
    compatible with Whisper models. The implementation follows the Whisper
    preprocessing pipeline.

    Args:
        audio: Audio waveform tensor, numpy array, or file path (16 kHz)
        n_mels: Number of Mel frequency filters (only 80 supported)
        padding: Number of zero samples to pad on the right
        device: Device to perform computation on

    Returns:
        Log-Mel spectrogram tensor of shape (n_mels, n_frames)
        
    Raises:
        AssertionError: If n_mels is not 80
    """
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

@lru_cache(maxsize=None)
def mel_filters(device: torch.device, n_mels: int = N_MELS) -> torch.Tensor:
    """Load Mel filterbank matrix for STFT to Mel spectrogram conversion.
    
    This function loads pre-computed Mel filterbank coefficients to avoid
    librosa dependency. The filters were generated using:
    
    ```python
    import librosa
    import numpy as np
    mel_80 = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    np.savez_compressed("mel_filters.npz", mel_80=mel_80)
    ```
    
    Args:
        device: Device to load the filterbank on
        n_mels: Number of Mel filters (only 80 supported)
        
    Returns:
        Mel filterbank matrix of shape (n_mels, n_fft//2 + 1)
        
    Raises:
        AssertionError: If n_mels is not 80
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("models/mel_filters.npz",allow_pickle=True) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)



def pad_or_trim(
    array: Union[torch.Tensor, np.ndarray], 
    length: int = N_SAMPLES, 
    *, 
    axis: int = -1
) -> Union[torch.Tensor, np.ndarray]:
    """Pad or trim audio array to specified length.
    
    This function ensures the audio array has exactly the specified length
    by either padding with zeros or trimming excess samples.
    
    Args:
        array: Input audio array (torch.Tensor or numpy.ndarray)
        length: Target length in samples (default: 30 seconds at 16kHz)
        axis: Axis along which to pad/trim (default: -1)
        
    Returns:
        Array with exactly `length` samples along the specified axis
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array







class WhisperWrapper_full(nn.Module):
    """Full Whisper model wrapper for decoder feature extraction.
    
    This wrapper uses the complete Whisper model (encoder + decoder) to extract
    hidden states from the decoder layers. Useful for tasks that require
    full sequence-to-sequence processing.
    
    Args:
        layer: Which layer to extract (-1 for all layers, None for last layer)
        use_feat_extractor: Whether to use internal feature extraction
        pretrained_model: Pretrained model name (default: "openai/whisper-small")
        num_layers: Number of decoder layers to process (default: 12)
    """
    
    def __init__(
        self, 
        layer: Optional[int] = None, 
        use_feat_extractor: bool = False, 
        pretrained_model: Optional[str] = None, 
        num_layers: int = 12,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.num_layers = num_layers
        self.use_feat_extractor = use_feat_extractor
        self.layer = 12 if layer is None else layer
        
        # Load pretrained model
        model_name = pretrained_model or "openai/whisper-small"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through full Whisper model.
        
        Args:
            data: Input audio tensor or mel spectrogram
            
        Returns:
            Decoder hidden states tensor
        """
        if self.use_feat_extractor:
            # Convert audio to mel spectrogram
            data = log_mel_spectrogram(data, padding=N_SAMPLES)
            data = pad_or_trim(data, length=3000).to(self.device)

        # Generate with hidden state output
        outputs = self.model.generate(
            input_features=data,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
        # Extract hidden states based on layer configuration
        if self.layer == -1:
            # Extract all layers
            decoder_hidden = []
            for layer in range(self.num_layers):
                hidden = torch.stack([
                    outputs.decoder_hidden_states[word][layer][:][:] 
                    for word in range(len(outputs.decoder_hidden_states))
                ])
                # Reshape: ('word', batch, layer, feat_dim) -> (batch, 'word', feat_dim, layer)
                hidden = hidden.permute(1, 0, 3, 2)
                decoder_hidden.append(hidden)
            decoder_hidden = torch.stack(decoder_hidden, dim=-1).squeeze(3)
            
        elif self.layer is None:
            # Extract final layer
            decoder_hidden = torch.stack([
                outputs.decoder_hidden_states[word][self.num_layers-1][0][0] 
                for word in range(len(outputs.decoder_hidden_states))
            ])
            decoder_hidden = decoder_hidden.unsqueeze(0)
            
        else:
            # Extract specific layer
            decoder_hidden = torch.stack([
                outputs.decoder_hidden_states[word][self.layer][0][0] 
                for word in range(len(outputs.decoder_hidden_states))
            ])
            decoder_hidden = decoder_hidden.unsqueeze(0)
            
        return decoder_hidden


class WhisperWrapper_encoder(nn.Module):
    """Whisper encoder wrapper for feature extraction.
    
    This wrapper uses only the Whisper encoder to extract audio features,
    which is more efficient for tasks that don't require sequence generation.
    Supports extracting features from specific layers or all layers.
    
    Args:
        layer: Which encoder layer to extract (-1 for all, None for final)
        use_feat_extractor: Whether to use internal mel-spectrogram extraction
        pretrained_model: Pretrained model name (default: "openai/whisper-small")
    """
    
    def __init__(
        self, 
        layer: Optional[int] = None, 
        use_feat_extractor: bool = False, 
        pretrained_model: Optional[str] = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.use_feat_extractor = use_feat_extractor
        self.layer = layer
        
        # Initialize feature extractor if needed
        if not use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-small"
            )
            
        # Load encoder model
        model_name = pretrained_model or "openai/whisper-small"
        model = WhisperModel.from_pretrained(model_name)
        self.model = model.encoder
        
        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through Whisper encoder.
        
        Args:
            data: Input audio tensor
            
        Returns:
            Encoder hidden states tensor
        """
        if self.use_feat_extractor:
            # Internal mel spectrogram extraction
            data_padded = pad_or_trim(data, length=N_SAMPLES).to(self.device)
            data_feats = log_mel_spectrogram(data_padded)
        else:
            # Use HuggingFace feature extractor
            d_list = [d.to('cpu').tolist() for d in data]
            processed_data = self.feature_extractor(
                d_list, 
                sampling_rate=16000, 
                return_tensors='pt'
            )
            data_feats = processed_data.input_features.to(self.device)
            
        # Extract features based on layer configuration
        if self.layer is None:
            # Final layer output
            outputs = self.model(
                input_features=data_feats, 
                return_dict=True
            )
            data = outputs[0]
            
        elif self.layer == -1:
            # All layers
            outputs = self.model(
                input_features=data_feats, 
                return_dict=True,
                output_hidden_states=True
            )
            layers = [layer_output for layer_output in outputs.hidden_states]
            data = torch.stack(layers, dim=-1)
            
        else:
            # Specific layer
            outputs = self.model(
                input_features=data_feats, 
                return_dict=True,
                output_hidden_states=True
            )
            data = outputs.hidden_states[self.layer]

        return data
    

