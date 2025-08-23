"""Transformer wrapper with positional encoding for speech quality assessment.

This module provides a transformer encoder wrapper with configurable positional
encoding strategies for processing audio features.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

try:
    from config import Config
except ImportError:
    from models.config import Config

class TransformerWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        # Normalization.
        self.norm = nn.BatchNorm1d(config.dim_input)

        # Position encoding.
        # if config.xlsr_name == "hubert_encoder" or config.xlsr_name == "hubert_full" or config.xlsr_name == "whisper_full":
        #     self.position_encoding = PositionalEncodingVariable(config)
        # else:
        #     self.position_encoding = PositionalEncoding(config)
        self.position_encoding = PositionalEncoding(config)

        # Down-projection to transformer dim.
        self.linear_proj = nn.Linear(
            in_features=config.dim_input,
            out_features=config.dim_transformer,
        )
        self.linear_proj_drop = nn.Dropout(config.dropout)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_transformer,
            dim_feedforward=config.dim_transformer*2,
            nhead=config.nhead_transformer,
            batch_first=True,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.nlayers_transformer,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:

        # Normalization.
        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm(x.transpose(-1,-2)).transpose(-1,-2)

        # Linear projection down to transformer dim.
        x = self.linear_proj(x)
        x = self.linear_proj_drop(x)

        # Position encoding + transformer.
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, mask)

        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        d_model: int = config.dim_transformer
        seq_len: int = config.feat_seq_len
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*seq_len) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe.expand(x.shape)
        return x

class PositionalEncodingVariable(nn.Module):
    """Variable-length positional encoding for dynamic sequences.
    
    This encoding computes positional embeddings dynamically based on
    the actual sequence length of the input, rather than using a fixed
    maximum length. Useful for sequences with varying lengths.
    
    Args:
        config: Configuration containing transformer parameters
    """

    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.dim_transformer

    def forward(self, x: Tensor) -> Tensor:
        """Apply dynamic positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Create position indices
        position_ids = torch.arange(seq_len, device=device).unsqueeze(1).float()
        
        # Compute division term
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        # Compute positional encoding
        pe = torch.zeros(1, seq_len, self.d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position_ids * div_term)
        pe[0, :, 1::2] = torch.cos(position_ids * div_term)
        
        return x + pe



