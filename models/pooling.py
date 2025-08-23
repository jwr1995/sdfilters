"""Pooling layers for sequence aggregation and final prediction.

This module contains pooling mechanisms that aggregate sequential features
into final quality predictions for speech quality assessment.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PoolAttFF(nn.Module):
    """Attention-based pooling module with feed-forward network.
    
    This module applies attention-based pooling to aggregate sequence features
    into a single representation, followed by a feed-forward network for
    final prediction.
    
    Args:
        dim_head_in: Input dimension size
        dropout_rate: Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim_head_in: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Attention mechanism layers
        self.attention_proj = nn.Linear(dim_head_in, 2 * dim_head_in)
        self.attention_score = nn.Linear(2 * dim_head_in, 1)
        
        # Output projection
        self.output_proj = nn.Linear(dim_head_in, 1)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply attention pooling to input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Pooled tensor of shape (batch_size, 1)
        """
        # Compute attention weights
        attention_features = F.relu(self.attention_proj(x))
        attention_features = self.dropout(attention_features)
        attention_scores = self.attention_score(attention_features)  # (B, L, 1)
        
        # Apply attention weights
        attention_weights = F.softmax(attention_scores.transpose(2, 1), dim=2)  # (B, 1, L)
        pooled_features = torch.bmm(attention_weights, x)  # (B, 1, features)
        pooled_features = pooled_features.squeeze(1)  # (B, features)
        
        # Final projection
        output = self.output_proj(pooled_features)  # (B, 1)
        
        return output


class GlobalAveragePooling(nn.Module):
    """Simple global average pooling for sequence aggregation.
    
    Alternative to attention pooling that simply averages all sequence positions.
    """
    
    def __init__(self, dim_head_in: int):
        super().__init__()
        self.output_proj = nn.Linear(dim_head_in, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply global average pooling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Pooled tensor of shape (batch_size, 1)
        """
        # Global average pooling across sequence dimension
        pooled = x.mean(dim=1)  # (B, features)
        output = self.output_proj(pooled)  # (B, 1)
        return output


class MaxPooling(nn.Module):
    """Max pooling across sequence for feature aggregation.
    
    Alternative pooling strategy that takes the maximum activation
    across the sequence dimension.
    """
    
    def __init__(self, dim_head_in: int):
        super().__init__()
        self.output_proj = nn.Linear(dim_head_in, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply max pooling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Pooled tensor of shape (batch_size, 1)
        """
        # Max pooling across sequence dimension
        pooled, _ = x.max(dim=1)  # (B, features)
        output = self.output_proj(pooled)  # (B, 1)
        return output