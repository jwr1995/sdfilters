"""Speech Quality Assessment using Whisper-based models.

This module provides functionality to evaluate audio quality using
trained Whisper-based neural network models.
"""

import sys
import argparse
from pathlib import Path
from typing import Union, Tuple

import torch
import torchaudio

from models.models import (
    SingleMOSPredictor,
    MultiDimPredictor
)

def _get_device() -> torch.device:
    """Determine the best available device for model inference.
    
    Returns:
        torch.device: The optimal device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # for M1 Macs
    else:
        return torch.device("cpu")  # May be slow!


def _load_model(model_type: str, device: torch.device) -> torch.nn.Module:
    """Load the appropriate model based on type.
    
    Args:
        model_type: Type of model ('single' for MOS only, 'multi' for multidimensional)
        device: Device to load the model on
        
    Returns:
        Loaded PyTorch model ready for inference
        
    Raises:
        ValueError: If model_type is not supported
        FileNotFoundError: If checkpoint file is not found
    """
    checkpoint_dir = Path("checkpoints")
    
    if model_type == "single":
        model = SingleMOSPredictor()
        checkpoint_path = checkpoint_dir / "single_head_model.pt"
    elif model_type == "multi":
        model = MultiDimPredictor()
        checkpoint_path = checkpoint_dir / "multi_head_model.pt"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'single' or 'multi'.")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model


def _validate_audio(waveform: torch.Tensor, sample_rate: int) -> None:
    """Validate audio format requirements.
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate of the audio
        
    Raises:
        ValueError: If audio doesn't meet requirements
    """
    if waveform.shape[0] != 1:
        raise ValueError(f"Audio must be mono (1 channel), got {waveform.shape[0]} channels")
    
    if sample_rate != 16000:
        raise ValueError(f"Sample rate must be 16000 Hz, got {sample_rate} Hz")


def get_score(audio_file: Union[str, Path], model_type: str = "single") -> torch.Tensor:
    """Evaluate audio quality using Whisper-based models.
    
    This function loads an audio file and evaluates its quality using either
    a single-dimensional MOS (Mean Opinion Score) model or a multi-dimensional
    model that predicts MOS, Noisiness, Coloration, Discontinuity, and Loudness.

    Args:
        audio_file: Path to the audio file (must be mono, 16kHz sample rate)
        model_type: Model type - 'single' for MOS only, 'multi' for multidimensional
                   scoring [MOS, Noisiness, Coloration, Discontinuity, Loudness]

    Returns:
        Quality scores as a tensor:
        - For 'single': Shape (1,) with MOS score (0-1 range)
        - For 'multi': Shape (5,) with [MOS, Noisiness, Coloration, 
                      Discontinuity, Loudness] (0-1 range each)
                      
    Raises:
        ValueError: If audio format is incorrect or model type is unsupported
        FileNotFoundError: If audio file or model checkpoint is not found
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    device = _get_device()
    model = _load_model(model_type, device)
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    
    _validate_audio(waveform, sample_rate)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        score = model(waveform)
        
    if model_type == "multi":
        score = score.squeeze(0)
        
    return score 



def _format_scores(scores: torch.Tensor, model_type: str) -> Tuple[str, ...]:
    """Format scores for display.
    
    Args:
        scores: Raw model output scores (0-1 range)
        model_type: Type of model used
        
    Returns:
        Formatted score strings
    """
    if model_type == "single":
        mos = scores.item() * 5  # Convert to 1-5 MOS scale
        return (f"MOS: {mos:.3f}",)
    else:
        # Multi-dimensional scores
        dimension_names = ["MOS", "Noisiness", "Coloration", "Discontinuity", "Loudness"]
        scaled_scores = scores.cpu() * 5  # Convert to 1-5 scale
        return tuple(f"{name}: {score:.3f}" for name, score in zip(dimension_names, scaled_scores))


def main() -> None:
    """Main entry point for the speech quality assessment tool."""
    parser = argparse.ArgumentParser(
        description="Evaluate audio quality using Whisper-based models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python get_score.py audio.wav                    # Single MOS score
  python get_score.py audio.wav --model_type multi # Multi-dimensional scores
        """
    )
    
    parser.add_argument(
        "audio_file", 
        type=str, 
        help="Path to the audio file (must be mono, 16kHz)"
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["single", "multi"],
        default="single",
        help="Model type: 'single' for MOS only, 'multi' for multidimensional scores"
    )
    
    args = parser.parse_args()
    
    try:
        scores = get_score(args.audio_file, args.model_type)
        formatted_scores = _format_scores(scores, args.model_type)
        
        print(f"\nAudio Quality Assessment Results:")
        print(f"File: {args.audio_file}")
        print(f"Model: {args.model_type}")
        print("-" * 40)
        
        for score_str in formatted_scores:
            print(score_str)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
