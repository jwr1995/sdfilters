"""Pooling layers for sequence aggregation and classification.

This module contains pooling mechanisms that aggregate sequential features
into final predictions for multi-task audio classification tasks.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PoolAttFF(nn.Module):
    """Attention-based pooling module with feed-forward network.
    
    This module applies attention-based pooling to aggregate sequence features
    into a single representation, followed by a feed-forward network for
    final binary classification prediction.
    
    Args:
        dim_head_in: Input dimension size
        dropout_rate: Dropout probability (default: 0.1)
    """
    
    def __init__(self, dim_head_in: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Attention mechanism layers
        self.attention_proj = nn.Linear(dim_head_in, 2 * dim_head_in)
        self.attention_score = nn.Linear(2 * dim_head_in, 1)
        
        # Output projection for binary classification
        self.output_proj = nn.Linear(dim_head_in, 1)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: Tensor) -> Tensor:
        """Apply attention pooling to input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Raw logit tensor of shape (batch_size, 1) for binary classification
        """
        # Compute attention weights
        attention_features = F.relu(self.attention_proj(x))
        attention_features = self.dropout(attention_features)
        attention_scores = self.attention_score(attention_features)  # (B, L, 1)
        
        # Apply attention weights
        attention_weights = F.softmax(attention_scores.transpose(2, 1), dim=2)  # (B, 1, L)
        pooled_features = torch.bmm(attention_weights, x)  # (B, 1, features)
        pooled_features = pooled_features.squeeze(1)  # (B, features)
        
        # Final projection (raw logits, no activation)
        logits = self.output_proj(pooled_features)  # (B, 1)
        
        return logits


class MultiHeadClassificationPool(nn.Module):
    """Multi-head classification pooling with named outputs.
    
    This module creates separate attention pooling heads for each classification
    task, allowing for flexible multi-task learning with named outputs.
    
    Args:
        dim_input: Input feature dimension
        task_labels: Dictionary mapping task names to task descriptions
                    e.g., {'multispeaker': 'Multiple speakers present', 
                          'music': 'Background music detected', ...}
        dropout_rate: Dropout probability for attention pooling
    """
    
    def __init__(
        self, 
        dim_input: int, 
        task_labels: dict[str, str],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.task_labels = task_labels
        self.task_names = list(task_labels.keys())
        self.n_tasks = len(self.task_names)
        
        # Create named classification heads
        self.classification_heads = nn.ModuleDict({
            task_name: PoolAttFF(dim_input, dropout_rate)
            for task_name in self.task_names
        })
        
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Apply multi-head classification pooling.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Dictionary mapping task names to logit tensors of shape (batch_size, 1)
        """
        outputs = {}
        for task_name, head in self.classification_heads.items():
            outputs[task_name] = head(x)
            
        return outputs
    
    def get_task_info(self) -> dict[str, str]:
        """Get task information dictionary.
        
        Returns:
            Dictionary mapping task names to descriptions
        """
        return self.task_labels.copy()


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
            Classification logits of shape (batch_size, 1)
        """
        # Global average pooling across sequence dimension
        pooled = x.mean(dim=1)  # (B, features)
        logits = self.output_proj(pooled)  # (B, 1)
        return logits


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
            Classification logits of shape (batch_size, 1)
        """
        # Max pooling across sequence dimension
        pooled, _ = x.max(dim=1)  # (B, features)
        logits = self.output_proj(pooled)  # (B, 1)
        return logits