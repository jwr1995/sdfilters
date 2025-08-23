"""Configuration classes and utilities for transformer-based models.

This module provides configuration management for various transformer architectures
used in speech quality assessment, including model dimensions, layer counts,
and preprocessing parameters.
"""

from enum import Enum
from typing import Optional

import torch
from torch.nn.functional import pad

class Input(Enum):
    """Enumeration of supported input feature types."""
    MFCC = 0  # Mel-frequency cepstral coefficients
    XLSR = 1  # Cross-lingual Speech Representations (e.g., from Whisper)

class CenterCrop(torch.nn.Module):
    """Center crop or pad sequences to fixed length.
    
    This module ensures that input sequences have exactly the specified length
    by center cropping long sequences or zero-padding short sequences.
    
    Args:
        seq_len: Target sequence length
    """
    
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply center cropping or padding to input tensor.
        
        Args:
            x: Input tensor of shape (N, L, C) or (L, C)
            
        Returns:
            Tensor with sequence length exactly equal to self.seq_len
        """
        # Handle 2D input by temporarily adding batch dimension
        unsqueezed = False
        if x.dim() == 2:
            unsqueezed = True
            x = x.unsqueeze(0)
            
        assert x.dim() == 3, f"Expected 3D tensor (N, L, C), got {x.dim()}D"

        current_len = x.size(1)
        
        if current_len > self.seq_len:
            # Center crop
            center_start = (current_len - self.seq_len) // 2
            x = x[:, center_start:center_start + self.seq_len, :]
            
        elif current_len < self.seq_len:
            # Zero pad at the end
            to_pad = self.seq_len - current_len
            x = pad(x, (0, 0, 0, to_pad, 0, 0), mode="constant", value=0.0)

        # Remove temporary batch dimension if it was added
        if unsqueezed:
            x = x.squeeze(0)

        return x
    
# Model architecture specifications
MODEL_SPECS = {
    # XLS-R models (from XLS-R paper Table 2)
    "wav2vec2-xls-r-300m": {"layers": 24, "hidden_size": 1024},
    "wav2vec2-xls-r-1b": {"layers": 48, "hidden_size": 1280},
    "wav2vec2-xls-r-2b": {"layers": 48, "hidden_size": 1920},
    
    # HuBERT models
    "hubert_encoder": {"layers": -1, "hidden_size": 512},
    "hubert_encoder_t": {"layers": -1, "hidden_size": 384},
    "hubert_full": {"layers": -1, "hidden_size": 768},
    "hubert_full_t": {"layers": -1, "hidden_size": 384},
    
    # Whisper models
    "whisper_encoder": {"layers": -1, "hidden_size": 768},
    "whisper_encoder_ref": {"layers": -1, "hidden_size": 768 * 2},
    "whisper_encoder_t": {"layers": -1, "hidden_size": 1500},
    "whisper_full": {"layers": -1, "hidden_size": 768},
    "whisper_full_t": {"layers": -1, "hidden_size": 384},
}


class Config:
    """Configuration class for transformer-based models.
    
    This class manages all hyperparameters and architectural choices
    for transformer models used in speech quality assessment.
    
    Args:
        name: Configuration name/identifier
        input: Input feature type (MFCC or XLSR)
        feat_seq_len: Length of input feature sequence
        dim_transformer: Transformer hidden dimension
        xlsr_name: Name of pre-trained model (for XLSR input)
        nhead_transformer: Number of transformer attention heads
        nlayers_transformer: Number of transformer layers
    """

    def __init__(
        self,
        name: str,
        input: Input,
        feat_seq_len: int,
        dim_transformer: Optional[int] = None,
        xlsr_name: Optional[str] = None,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):
        # Validate inputs
        if feat_seq_len <= 0:
            raise ValueError("feat_seq_len must be positive")
            
        if input == Input.MFCC:
            xlsr_name = None

        # Store basic parameters
        self.name = name
        self.input = input
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer
        
        # Configure input dimensions based on model type
        self._configure_input_dimensions()
        
        # Set output configuration
        self.dim_head_in = self.dim_transformer
        self.dim_head_out = 1
        self.dropout = 0.1
        
    def _configure_input_dimensions(self):
        """Configure input dimensions based on model specification."""
        if self.xlsr_name is not None:
            # Use pre-trained model specifications
            if self.xlsr_name not in MODEL_SPECS:
                raise ValueError(f"Unknown model: {self.xlsr_name}")
                
            spec = MODEL_SPECS[self.xlsr_name]
            self.xlsr_layers = spec["layers"] + 1 if spec["layers"] > 0 else None
            self.dim_input = spec["hidden_size"]
            
        else:
            # MFCC input configuration
            self.xlsr_layers = None
            if self.feat_seq_len == 80:  # Handle transposed MFCC
                self.dim_input = 3000
            else:
                self.dim_input = 80
                
    def __repr__(self) -> str:
        return (f"Config(name='{self.name}', input={self.input}, "
                f"feat_seq_len={self.feat_seq_len}, "
                f"dim_transformer={self.dim_transformer}, "
                f"xlsr_name='{self.xlsr_name}')")

# Default feature sequence length
DEFAULT_FEAT_SEQ_LEN = 256


####################### TRANSFORMER_32DEEP_CONFIG ####################
MFCC_TRANSFORMER_32DEEP_CONFIG = Config(
    "MFCC_TRANSFORMER_32DEEP_CONFIG",
    Input.MFCC,
    feat_seq_len=DEFAULT_FEAT_SEQ_LEN,
    dim_transformer=256,
    xlsr_name=None,
    nhead_transformer=4,
    nlayers_transformer=4,
)

HUBERT_ENCODER_CONFIG= Config(
    "HUBERT_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=256,
    dim_transformer=256,
    xlsr_name="hubert_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



HUBERT_ENCODER_CONFIG_T = Config(
    "HUBERT_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=512,
    dim_transformer=256,
    xlsr_name="hubert_encoder_t",
    nhead_transformer=4,
    nlayers_transformer=4,
)





HUBERT_FULL_CONFIG = Config(
    "HUBERT_FULL_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="hubert_full",
    nhead_transformer=4,
    nlayers_transformer=4,
)


WHISPER_ENCODER_CONFIG = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=768,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG = Config(
    "WHISPER_ENCODER_CONFIG_REF",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=768,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)


WHISPER_ENCODER_CONFIG_MEDIUM = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=512,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



WHISPER_ENCODER_CONFIG_SMALL = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)
WHISPER_ENCODER_CONFIG_SMALL_T = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG_MEL = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.MFCC,
    feat_seq_len=3000,
    dim_transformer=256,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)



WHISPER_ENCODER_CONFIG_SMALLER = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=1500,
    dim_transformer=128,
    xlsr_name="whisper_encoder",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_ENCODER_CONFIG_SMALLER_T = Config(
    "WHISPER_ENCODER_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=128,
    xlsr_name="whisper_encoder_t",
    nhead_transformer=4,
    nlayers_transformer=4,
)

WHISPER_FULL_CONFIG_SMALL= Config(
    "WHISPER_FULL_CONFIG",
    Input.XLSR,
    feat_seq_len=768,
    dim_transformer=256,
    xlsr_name="whisper_full",
    nhead_transformer=4,
    nlayers_transformer=4,
)


XLSR_300M_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=DEFAULT_FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-300m",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_1B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=DEFAULT_FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-1b",
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_2B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
    feat_seq_len=DEFAULT_FEAT_SEQ_LEN,
    dim_transformer=32,
    xlsr_name="wav2vec2-xls-r-2b",
    nhead_transformer=4,
    nlayers_transformer=4,
)