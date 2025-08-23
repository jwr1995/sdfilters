"""Whisper-based speech quality prediction models.

This module contains various neural network architectures for speech quality
assessment based on Whisper encoder features. Models support both single-dimensional
MOS (Mean Opinion Score) prediction and multi-dimensional quality assessment.
"""

from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from feature_extractor import (
        WhisperWrapper_full,
        WhisperWrapper_encoder, 
        pad_or_trim, 
        log_mel_spectrogram
    )
    from transformer import TransformerWrapper
    from config import CenterCrop, Config, Input
    from pooling import PoolAttFF
except ImportError:
    from models.feature_extractor import (
        WhisperWrapper_full,
        WhisperWrapper_encoder, 
        pad_or_trim, 
        log_mel_spectrogram
    )
    from models.transformer import TransformerWrapper
    from models.config import CenterCrop, Config, Input
    from models.pooling import PoolAttFF



    
class whisperMetricPredictorEncoderTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorEncoderTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder returns (B, 1500, 512)
        out_feats = out_feats.permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)

        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out
    
class SingleMOSPredictor(nn.Module):
    """Single MOS (Mean Opinion Score) predictor using Whisper encoder layers.
    
    This model extracts features from all Whisper encoder layers, learns
    optimal weights to combine them, and predicts a single quality score
    using transformer processing and attention pooling.
    
    Based on: https://github.com/lcn-kul/xls-r-analysis-sqa/
    
    Args:
        sequence_length: Expected sequence length from feature extractor (default: 1500)
    """
    def __init__(self, sequence_length: int = 1500):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(768)
        
        # Feature extraction (frozen)
        self.feature_extractor = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feature_extractor.requires_grad_(False)
        
        # Layer weighting
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_softmax = nn.Softmax(dim=0)
        
        # Transformer configuration
        self.config = Config(
            "WHISPER_ENCODER_CONFIG",
            Input.XLSR,
            feat_seq_len=sequence_length,
            dim_transformer=256,
            xlsr_name="whisper_encoder",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        
        self.transformer = TransformerWrapper(self.config)
        self.attention_pool = PoolAttFF(self.config.dim_transformer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single MOS prediction.
        
        Args:
            x: Input audio tensor
            
        Returns:
            MOS score tensor in range [0, 1]
        """
        # Extract features from all encoder layers
        layer_features = self.feature_extractor(x)  # Shape: (B, 1500, 768, 13)
        
        # Weighted combination of layers
        layer_weights_norm = self.layer_softmax(self.layer_weights)
        combined_features = layer_features @ layer_weights_norm  # (B, 1500, 768)
        
        # Normalize input
        features_norm = self.input_norm(
            combined_features.permute(0, 2, 1)
        ).permute(0, 2, 1)  # Back to (B, 1500, 768)
        
        # Process through transformer
        transformer_out = self.transformer(features_norm)  # (B, 1500, 256)
        
        # Pool and predict
        pooled = self.attention_pool(transformer_out)  # (B, 1)
        mos_score = self.sigmoid(pooled)  # (B, 1)
        
        return mos_score
class MultiDimPredictor(nn.Module):
    """Multi-dimensional speech quality predictor.
    
    Predicts 5 quality dimensions: MOS, Noisiness, Coloration, 
    Discontinuity, and Loudness using separate attention pooling heads.
    
    Based on: https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    
    QUALITY_DIMENSIONS = ["MOS", "Noisiness", "Coloration", "Discontinuity", "Loudness"]
    
    def __init__(self, sequence_length: int = 1500, n_dimensions: int = 5):
        super().__init__()
        
        self.n_dimensions = n_dimensions
        
        self.input_norm = nn.BatchNorm1d(768)
        
        # Feature extraction (frozen)
        self.feature_extractor = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feature_extractor.requires_grad_(False)
        
        # Layer weighting
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_softmax = nn.Softmax(dim=0)
        
        # Transformer configuration
        self.config = Config(
            "WHISPER_ENCODER_CONFIG",
            Input.XLSR,
            feat_seq_len=sequence_length,
            dim_transformer=256,
            xlsr_name="whisper_encoder",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        
        self.transformer = TransformerWrapper(self.config)
        
        # Separate attention pooling heads for each quality dimension
        self.attention_pools = nn.ModuleList([
            PoolAttFF(self.config.dim_transformer) 
            for _ in range(n_dimensions)
        ])
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-dimensional quality prediction.
        
        Args:
            x: Input audio tensor
            
        Returns:
            Quality scores tensor of shape (batch_size, n_dimensions)
            Each dimension: [MOS, Noisiness, Coloration, Discontinuity, Loudness]
        """
        # Extract and combine encoder layer features
        layer_features = self.feature_extractor(x)
        layer_weights_norm = self.layer_softmax(self.layer_weights)
        combined_features = layer_features @ layer_weights_norm
        
        # Normalize and transform
        features_norm = self.input_norm(
            combined_features.permute(0, 2, 1)
        ).permute(0, 2, 1)
        
        transformer_out = self.transformer(features_norm)
        
        # Apply separate pooling heads for each quality dimension
        dimension_scores = []
        for pool in self.attention_pools:
            score = self.sigmoid(pool(transformer_out))
            dimension_scores.append(score)
        
        # Stack all dimensions: [MOS, Noisiness, Coloration, Discontinuity, Loudness]
        multi_dim_scores = torch.stack(dimension_scores, dim=1)  # (B, 5, 1)
        
        return multi_dim_scores.squeeze(-1)  # (B, 5)
class whisperMetricPredictorEncoderLayersTransformerSmallRef(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.norm_input_ref = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_weights_ref = nn.Parameter(torch.ones(13))
        self.softmax_ref = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG_REF",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder_ref",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool1 = PoolAttFF(self.config.dim_transformer)
      
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        
        out_feats = out_feats @ self.softmax(self.layer_weights)  # weighted sum of the 13 tensors
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        out_feats_ref = self.feat_extract(y) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats_ref = out_feats_ref @ self.softmax_ref(self.layer_weights_ref)  # weighted sum of the 13 tensors
        out_feats_ref = self.norm_input_ref(out_feats_ref.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        
        #concatenate the two inputs
        # Concatenate degraded and reference features
        out_feats = torch.cat([out_feats, out_feats_ref], dim=2)


        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)


        out1 = self.attenPool1(out) #attenPool returns (B, 1)
        out1 = self.sigmoid(out1) #sigmoid returns (B, 1)
        
        
        return out1#,out_feats
class whisperMetricPredictorEncoderLayersTransformerSmallDimRef(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.norm_input_ref = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_weights_ref = nn.Parameter(torch.ones(13))
        self.softmax_ref = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG_REF",
        Input.XLSR,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name="whisper_encoder_ref",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool1 = PoolAttFF(self.config.dim_transformer)
        self.attenPool2 = PoolAttFF(self.config.dim_transformer)
        self.attenPool3 = PoolAttFF(self.config.dim_transformer)
        self.attenPool4 = PoolAttFF(self.config.dim_transformer)
        self.attenPool5 = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        
        out_feats = out_feats @ self.softmax(self.layer_weights)  # weighted sum of the 13 tensors
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        out_feats_ref = self.feat_extract(y) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats_ref = out_feats_ref @ self.softmax_ref(self.layer_weights_ref)  # weighted sum of the 13 tensors
        out_feats_ref = self.norm_input_ref(out_feats_ref.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        
        
        #concatenate the two inputs
        # Concatenate degraded and reference features
        out_feats = torch.cat([out_feats, out_feats_ref], dim=2)


        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)


        out1 = self.attenPool1(out) #attenPool returns (B, 1)
        out1 = self.sigmoid(out1) #sigmoid returns (B, 1)
        out2 = self.attenPool2(out) #attenPool returns (B, 1)
        out2 = self.sigmoid(out2)

        out3 = self.attenPool3(out) #attenPool returns (B, 1)
        out3 = self.sigmoid(out3)

        out4 = self.attenPool4(out) #attenPool returns (B, 1)
        out4 = self.sigmoid(out4)

        out5 = self.attenPool5(out) #attenPool returns (B, 1)
        out5 = self.sigmoid(out5)

        # Return all 5 outputs on a new dimension
        out = torch.stack([out1, out2, out3, out4, out5], dim=1)
        
        return out#,out_feats



class whisperMetricPredictorEncoderLayersTransformerSmallT(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(
        self, feat_seq=1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)

        self.config  = Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=768,
        dim_transformer=256,
        xlsr_name="whisper_encoder_t",
        nhead_transformer=4,
        nlayers_transformer=4,
        )
        self.transformer = TransformerWrapper(self.config)

        
        
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
    
        out_feats = self.feat_extract(x) #whisper encoder a list of 13 tensors of shape (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights)  # weighted sum of the 13 tensors

        out_feats = out_feats.permute(0,2,1) #swap axes to (B, 512, 1500)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)
        out = self.transformer(out_feats) # transformer returns (B, 1500, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorMelTransformerSmall(nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=feat_seq,
        dim_transformer=256,
        xlsr_name=None,
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(80)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded).swapaxes(1,2) #returns (B, 3000, 80)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 3000, 80)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out


class whisperMetricPredictorMelTransformerSmallT (nn.Module):
    """Transformer based varient on metric estimator

    based on https://github.com/lcn-kul/xls-r-analysis-sqa/
    """
    def __init__(self, feat_seq=3000):
        super().__init__()


        self.config = Config(
        "MFCC_TRANSFORMER_32DEEP_CONFIG",
        Input.MFCC,
        feat_seq_len=80,
        dim_transformer=256,
        xlsr_name="mel_T",
        nhead_transformer=4,
        nlayers_transformer=4,
    )
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)

        self.attenPool = PoolAttFF(self.config.dim_transformer)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N_SAMPLES = 16000*30
        data_padded = pad_or_trim(x, length=N_SAMPLES) #pad or trim to 30 seconds, returns (B, 480000)
        data_feats = log_mel_spectrogram(data_padded) #returns (B, 80, 3000)
    
        data_feats = self.norm_input(data_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 80, 3000)
        out_trans = self.transformer(data_feats) # transformer returns (B, 3000, 256)
        out = self.attenPool(out_trans) #attenPool returns (B, 1)
        out = self.sigmoid(out)

        return out



class whisperMetricPredictorFullTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, 1500, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out

class whisperMetricPredictorFullTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)


        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x)[:,:,:,-1] #whisper encoder returns (B, W, 768)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out



class whisperMetricPredictorFullLayersTransformerSmall(nn.Module):
    def __init__(self, feat_seq=768//2):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=feat_seq,
            dim_transformer=256,
            xlsr_name="whisper_full",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(768)
        self.transformer = TransformerWrapper(self.config)
        self.norm_input = nn.BatchNorm1d(768)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 384, 768)
        out = self.transformer(out_feats) # transformer returns (B, 384, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out


class whisperMetricPredictorFullLayersTransformerSmallT(nn.Module):
    def __init__(self, feat_seq=384):
        super().__init__()



        self.feat_extract = WhisperWrapper_full(layer=-1,use_feat_extractor=True)
        self.feat_extract.requires_grad_(False)
        self.config = Config(
            "WHISPER_FULL_CONFIG",
            Input.XLSR,
            feat_seq_len=768,
            dim_transformer=256,
            xlsr_name="whisper_full_t",
            nhead_transformer=4,
            nlayers_transformer=4,
        )
        self.cc = CenterCrop(feat_seq)
        self.norm_input = nn.BatchNorm1d(feat_seq)

        self.transformer = TransformerWrapper(self.config)
        self.attenPool = PoolAttFF(self.config.dim_transformer)
        self.layer_weights = nn.Parameter(torch.ones(12))
        self.softmax = nn.Softmax(dim=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):        
        out_feats = self.feat_extract(x) #whisper encoder returns list (B, 1500, 768,12)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 12 tensors (B, 1500, 768) 
        print(self.layer_weights)
        out_feats = self.cc (out_feats) #center crop to 384
        
        out_feats= out_feats.permute(0,2,1) #swap axes to (B, 768, 384)

        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 768, 384)
        
        out = self.transformer(out_feats) # transformer returns (B, 768, 256)
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out

