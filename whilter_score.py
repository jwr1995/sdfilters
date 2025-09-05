"""Whilter: Multi-task audio classification for "in-the-wild" speech filtering.

This module provides functionality to classify audio samples using the Whilter model
for detecting multiple speakers, background music, foreign languages, noise, and 
synthetic speech in audio recordings.

Based on the paper: "Whilter: A Whisper-based Data Filter for 'In-the-Wild' 
Speech Corpora Using Utterance-level Multi-Task Classification"
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Union, Tuple

import torch
import torchaudio

from models.whilter import WhilterModel, create_whilter_model


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
        return torch.device("cpu")


def _load_whilter_model(model_type: str, device: torch.device) -> WhilterModel:
    """Load the Whilter model based on type.
    
    Args:
        model_type: Type of Whilter model configuration
        device: Device to load the model on
        
    Returns:
        Loaded WhilterModel ready for inference
        
    Raises:
        ValueError: If model_type is not supported
        FileNotFoundError: If checkpoint file is not found
    """
    checkpoint_dir = Path("checkpoints")
    
    # Create model based on type
    if model_type == "default":
        model = create_whilter_model("default")
        checkpoint_path = checkpoint_dir / "whilter_model.pt"
    elif model_type == "speech_quality":
        model = create_whilter_model("speech_quality")
        checkpoint_path = checkpoint_dir / "whilter_speech_quality.pt"
    elif model_type == "general_audio":
        model = create_whilter_model("general_audio")
        checkpoint_path = checkpoint_dir / "whilter_general_audio.pt"
    elif model_type == "lightweight":
        model = create_whilter_model("lightweight")
        checkpoint_path = checkpoint_dir / "whilter_lightweight.pt"
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'default', 'speech_quality', 'general_audio', or 'lightweight'.")
    
    # Load checkpoint if available
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}. Using untrained model.")
    
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


def classify_audio(audio_file: Union[str, Path], model_type: str = "default", 
                  threshold: float = 0.5) -> Dict[str, Union[float, bool]]:
    """Classify audio using the Whilter model.
    
    This function loads an audio file and classifies it using the Whilter model
    to detect multiple speakers, background music, foreign languages, noise, 
    and synthetic speech.

    Args:
        audio_file: Path to the audio file (must be mono, 16kHz sample rate)
        model_type: Whilter model type - 'default', 'speech_quality', 'general_audio', or 'lightweight'
        threshold: Classification threshold for binary predictions (default: 0.5)

    Returns:
        Dictionary containing:
        - 'probabilities': Dict mapping task names to probability scores (0-1)
        - 'predictions': Dict mapping task names to binary predictions (True/False)
        - 'task_info': Dict mapping task names to descriptions
                      
    Raises:
        ValueError: If audio format is incorrect or model type is unsupported
        FileNotFoundError: If audio file is not found
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    device = _get_device()
    model = _load_whilter_model(model_type, device)
    
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    
    _validate_audio(waveform, sample_rate)
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # Get raw logits from model
        logits = model(waveform, return_dict=True)
        
        # Convert to probabilities
        probabilities = {}
        predictions = {}
        for task_name, task_logits in logits.items():
            prob = torch.sigmoid(task_logits).item()
            probabilities[task_name] = prob
            predictions[task_name] = prob > threshold
        
        # Get task information
        task_info = model.get_task_info()
        
    return {
        'probabilities': probabilities,
        'predictions': predictions,
        'task_info': task_info
    }


def _format_results(results: Dict[str, Union[float, bool, Dict]], 
                   show_probabilities: bool = True) -> Tuple[str, ...]:
    """Format classification results for display.
    
    Args:
        results: Results dictionary from classify_audio()
        show_probabilities: Whether to show probability scores
        
    Returns:
        Formatted result strings
    """
    formatted_lines = []
    
    # Header
    formatted_lines.append("Whilter Audio Classification Results:")
    formatted_lines.append("=" * 50)
    
    # Task results
    probabilities = results['probabilities']
    predictions = results['predictions']
    task_info = results['task_info']
    
    for task_name in task_info.keys():
        prob = probabilities[task_name]
        pred = predictions[task_name]
        description = task_info[task_name]
        
        # Format prediction status
        status = "✓ DETECTED" if pred else "✗ NOT DETECTED"
        
        if show_probabilities:
            line = f"{task_name.upper():<15} {status:<15} ({prob:.3f})"
        else:
            line = f"{task_name.upper():<15} {status}"
            
        formatted_lines.append(line)
        
        # Add description on next line
        formatted_lines.append(f"{'':<15} {description}")
        formatted_lines.append("")  # Empty line for spacing
    
    return tuple(formatted_lines)


def main() -> None:
    """Main entry point for the Whilter audio classification tool."""
    parser = argparse.ArgumentParser(
        description="Classify audio using Whilter multi-task model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python whilter_score.py audio.wav                           # Default classification
  python whilter_score.py audio.wav --model_type speech_quality  # Speech quality tasks
  python whilter_score.py audio.wav --threshold 0.7          # Higher threshold
  python whilter_score.py audio.wav --no-probabilities       # Hide probability scores
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
        choices=["default", "speech_quality", "general_audio", "lightweight"],
        default="default",
        help="Whilter model type for different classification tasks"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Classification threshold for binary predictions (default: 0.5)"
    )
    
    parser.add_argument(
        "--no-probabilities", 
        action="store_true",
        help="Hide probability scores in output"
    )
    
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    try:
        results = classify_audio(args.audio_file, args.model_type, args.threshold)
        
        if args.json:
            import json
            # Convert numpy types to native Python types for JSON serialization
            json_results = {
                'file': args.audio_file,
                'model_type': args.model_type,
                'threshold': args.threshold,
                'probabilities': {k: float(v) for k, v in results['probabilities'].items()},
                'predictions': {k: bool(v) for k, v in results['predictions'].items()},
                'task_info': results['task_info']
            }
            print(json.dumps(json_results, indent=2))
        else:
            formatted_results = _format_results(results, not args.no_probabilities)
            
            print(f"\nFile: {args.audio_file}")
            print(f"Model: {args.model_type}")
            print(f"Threshold: {args.threshold}")
            print()
            
            for line in formatted_results:
                print(line)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
