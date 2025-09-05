"""Whilter: Multi-task audio classification model.

This module implements the Whilter model for multi-task binary classification
of audio samples, designed to filter "in-the-wild" speech datasets.

Based on the paper: "Whilter: A Whisper-based Data Filter for 'In-the-Wild' 
Speech Corpora Using Utterance-level Multi-Task Classification"
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from feature_extractor import (
        WhisperWrapper_encoder, 
        pad_or_trim, 
        log_mel_spectrogram
    )
    from transformer import TransformerWrapper
    from config import Config, Input
    from pooling import MultiHeadClassificationPool, PoolAttFF
except ImportError:
    from models.feature_extractor import (
        WhisperWrapper_encoder, 
        pad_or_trim, 
        log_mel_spectrogram
    )
    from models.transformer import TransformerWrapper
    from models.config import Config, Input
    from models.pooling import MultiHeadClassificationPool, PoolAttFF


# Default Whilter task configuration from the paper
WHILTER_TASKS = {
    "multispeaker": "Multiple speakers present in audio",
    "music": "Background music detected", 
    "foreign_language": "Non-English speech detected",
    "noise": "Noisy or reverberant speech",
    "synthetic": "Synthetic or artificially generated speech"
}

# Alternative task configurations
SPEECH_QUALITY_TASKS = {
    "mos": "Mean Opinion Score quality",
    "noisiness": "Level of background noise",
    "coloration": "Spectral coloration artifacts", 
    "discontinuity": "Temporal discontinuities",
    "loudness": "Perceived loudness level"
}

GENERAL_AUDIO_TASKS = {
    "speech": "Human speech present",
    "music": "Musical content present",
    "noise": "Environmental noise present",
    "silence": "Silence or very quiet audio"
}


class WhilterModel(nn.Module):
    """Whilter: Multi-task audio classification model.
    
    This model uses a frozen Whisper encoder with learnable layer weights,
    followed by a transformer network and separate attention pooling heads
    for each classification task.
    
    Architecture:
    - Frozen Whisper encoder (12 layers) with learnable layer combination weights
    - 4-layer transformer network (768 -> 256 dimensions)  
    - Separate attention pooling head for each classification task
    - Binary cross-entropy loss for each task
    
    Args:
        task_config: Dictionary mapping task names to descriptions
        sequence_length: Expected sequence length from feature extractor (default: 1500)
        transformer_dim: Transformer hidden dimension (default: 256)
        n_heads: Number of transformer attention heads (default: 4)
        n_layers: Number of transformer layers (default: 4)
        dropout_rate: Dropout rate for attention pooling (default: 0.1)
    """
    
    def __init__(
        self,
        task_config: Dict[str, str] = WHILTER_TASKS,
        sequence_length: int = 1500,
        transformer_dim: int = 256, 
        n_heads: int = 4,
        n_layers: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.task_config = task_config
        self.task_names = list(task_config.keys())
        self.n_tasks = len(self.task_names)
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(768)  # Whisper encoder output dim
        
        # Frozen Whisper encoder with learnable layer weights
        self.feature_extractor = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feature_extractor.requires_grad_(False)
        
        # Learnable weights for combining encoder layers
        self.layer_weights = nn.Parameter(torch.ones(13))  # 12 layers + input
        self.layer_softmax = nn.Softmax(dim=0)
        
        # Transformer configuration
        self.config = Config(
            "WHILTER_CONFIG",
            Input.XLSR,
            feat_seq_len=sequence_length,
            dim_transformer=transformer_dim,
            xlsr_name="whisper_encoder",
            nhead_transformer=n_heads,
            nlayers_transformer=n_layers,
        )
        
        # Transformer network
        self.transformer = TransformerWrapper(self.config)
        
        # Multi-head classification pooling
        self.classification_pool = MultiHeadClassificationPool(
            dim_input=transformer_dim,
            task_labels=task_config,
            dropout_rate=dropout_rate
        )
        
    def forward(self, x: Tensor, return_dict: bool = True) -> Union[Dict[str, Tensor], Tensor]:
        """Forward pass for multi-task classification.
        
        Args:
            x: Input audio tensor of shape (batch_size, audio_length)  
            return_dict: If True, return dict of task outputs. If False, return stacked tensor.
            
        Returns:
            If return_dict=True: Dictionary mapping task names to logit tensors (batch_size, 1)
            If return_dict=False: Stacked tensor of shape (batch_size, n_tasks)
        """
        # Extract features from all encoder layers
        layer_features = self.feature_extractor(x)  # Shape: (B, 1500, 768, 13)
        
        # Weighted combination of encoder layers
        layer_weights_norm = self.layer_softmax(self.layer_weights)
        combined_features = layer_features @ layer_weights_norm  # (B, 1500, 768)
        
        # Normalize input features
        features_norm = self.input_norm(
            combined_features.permute(0, 2, 1)
        ).permute(0, 2, 1)  # Back to (B, 1500, 768)
        
        # Process through transformer
        transformer_out = self.transformer(features_norm)  # (B, 1500, 256)
        
        # Apply multi-head classification pooling
        task_logits = self.classification_pool(transformer_out)
        
        if return_dict:
            return task_logits  # Dict[str, Tensor] 
        else:
            # Stack outputs in consistent order
            stacked = torch.stack([task_logits[name] for name in self.task_names], dim=1)
            return stacked.squeeze(-1)  # (B, n_tasks)
    
    def predict(self, x: Tensor, threshold: float = 0.5) -> Dict[str, Tensor]:
        """Generate binary predictions for all tasks.
        
        Args:
            x: Input audio tensor
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Dictionary mapping task names to binary predictions (0/1)
        """
        with torch.no_grad():
            logits = self.forward(x, return_dict=True)
            predictions = {}
            for task_name, task_logits in logits.items():
                probs = torch.sigmoid(task_logits)
                predictions[task_name] = (probs > threshold).float()
            return predictions
    
    def get_task_info(self) -> Dict[str, str]:
        """Get information about classification tasks.
        
        Returns:
            Dictionary mapping task names to descriptions
        """
        return self.task_config.copy()
    
    def get_layer_weights(self) -> torch.Tensor:
        """Get the learned layer combination weights.
        
        Returns:
            Normalized layer weights of shape (13,)
        """
        return self.layer_softmax(self.layer_weights).detach()


class FlexibleWhilterModel(WhilterModel):
    """Flexible version of Whilter with configurable architecture.
    
    This version allows more customization of the underlying components
    while maintaining the same interface as the base WhilterModel.
    """
    
    def __init__(
        self,
        task_config: Dict[str, str] = WHILTER_TASKS,
        sequence_length: int = 1500,
        transformer_dim: int = 256,
        whisper_model_size: str = "small",  # "small", "base", "large" 
        freeze_whisper: bool = True,
        use_layer_weights: bool = True,
        **kwargs
    ):
        super().__init__(task_config, sequence_length, transformer_dim, **kwargs)
        
        self.whisper_model_size = whisper_model_size
        self.freeze_whisper = freeze_whisper
        self.use_layer_weights = use_layer_weights
        
        # Re-initialize feature extractor with options
        if not freeze_whisper:
            self.feature_extractor.requires_grad_(True)
            
        if not use_layer_weights:
            # Use only the final layer
            self.feature_extractor = WhisperWrapper_encoder(
                use_feat_extractor=True, 
                layer=None  # Final layer only
            )
            self.layer_weights = None
            self.layer_softmax = None
    
    def forward(self, x: Tensor, return_dict: bool = True) -> Union[Dict[str, Tensor], Tensor]:
        """Forward pass with flexible architecture."""
        if self.use_layer_weights:
            return super().forward(x, return_dict)
        else:
            # Simplified forward pass without layer weighting
            features = self.feature_extractor(x)  # (B, 1500, 768)
            
            features_norm = self.input_norm(
                features.permute(0, 2, 1)
            ).permute(0, 2, 1)
            
            transformer_out = self.transformer(features_norm)
            task_logits = self.classification_pool(transformer_out)
            
            if return_dict:
                return task_logits
            else:
                stacked = torch.stack([task_logits[name] for name in self.task_names], dim=1)
                return stacked.squeeze(-1)


def create_whilter_model(model_type: str = "default") -> WhilterModel:
    """Factory function to create different Whilter model configurations.
    
    Args:
        model_type: Type of model configuration
            - "default": Standard Whilter from paper
            - "speech_quality": For speech quality assessment tasks
            - "general_audio": For general audio classification 
            - "lightweight": Smaller model for faster inference
            
    Returns:
        Configured WhilterModel instance
    """
    if model_type == "default":
        return WhilterModel(task_config=WHILTER_TASKS)
        
    elif model_type == "speech_quality":
        return WhilterModel(task_config=SPEECH_QUALITY_TASKS)
        
    elif model_type == "general_audio":
        return WhilterModel(task_config=GENERAL_AUDIO_TASKS)
        
    elif model_type == "lightweight":
        return WhilterModel(
            task_config=WHILTER_TASKS,
            transformer_dim=128,
            n_heads=2,
            n_layers=2
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")