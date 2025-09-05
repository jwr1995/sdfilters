"""Test script for Whilter model functionality.

This script tests the basic functionality of the Whilter model to ensure
it can be loaded and run inference without errors.
"""

import torch
import torchaudio
from pathlib import Path

from models.whilter import WhilterModel, create_whilter_model


def create_test_audio(duration_seconds: float = 5.0, sample_rate: int = 16000) -> torch.Tensor:
    """Create a simple test audio signal.
    
    Args:
        duration_seconds: Duration of the test audio
        sample_rate: Sample rate of the audio
        
    Returns:
        Test audio tensor of shape (1, samples) - mono audio
    """
    # Create a simple sine wave
    t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    signal = torch.sin(2 * torch.pi * 440 * t)  # 440 Hz sine wave
    
    # Add some noise
    noise = torch.randn_like(signal) * 0.1
    signal = signal + noise
    
    # Normalize
    signal = signal / torch.max(torch.abs(signal))
    
    # Add batch dimension only (mono audio)
    signal = signal.unsqueeze(0)  # (1, samples)
    
    return signal


def test_whilter_model():
    """Test the Whilter model functionality."""
    print("Testing Whilter model...")
    
    # Create test audio
    print("Creating test audio...")
    test_audio = create_test_audio(duration_seconds=5.0)
    print(f"Test audio shape: {test_audio.shape}")
    
    # Test different model configurations
    model_configs = [
        ("default", "Standard Whilter for 'in-the-wild' speech filtering"),
        ("speech_quality", "Speech quality assessment tasks"),
        ("general_audio", "General audio classification"),
        ("lightweight", "Lightweight model for faster inference")
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for model_type, description in model_configs:
        print(f"\nTesting {model_type} model: {description}")
        
        try:
            # Create model
            model = create_whilter_model(model_type)
            model.to(device)
            model.eval()
            
            # Get model info
            task_info = model.get_task_info()
            print(f"  Tasks: {list(task_info.keys())}")
            print(f"  Number of tasks: {len(task_info)}")
            
            # Test forward pass
            with torch.no_grad():
                # Move audio to device
                audio_device = test_audio.to(device)
                
                # Forward pass
                logits = model(audio_device, return_dict=True)
                
                # Convert to probabilities
                probabilities = {}
                for task_name, task_logits in logits.items():
                    prob = torch.sigmoid(task_logits).item()
                    probabilities[task_name] = prob
                
                print(f"  Probabilities: {probabilities}")
                
                # Test stacked output
                stacked_output = model(audio_device, return_dict=False)
                print(f"  Stacked output shape: {stacked_output.shape}")
                
                # Test predictions
                predictions = model.predict(audio_device, threshold=0.5)
                print(f"  Predictions: {predictions}")
                
                # Test layer weights
                layer_weights = model.get_layer_weights()
                print(f"  Layer weights shape: {layer_weights.shape}")
                
            print(f"  ✓ {model_type} model test passed!")
            
        except Exception as e:
            print(f"  ✗ {model_type} model test failed: {e}")
            import traceback
            traceback.print_exc()


def test_audio_loading():
    """Test audio loading functionality."""
    print("\nTesting audio loading...")
    
    # Create a temporary test audio file
    test_audio = create_test_audio(duration_seconds=3.0)
    test_file = Path("test_audio.wav")
    
    try:
        # Save test audio (torchaudio expects (channels, samples) format)
        audio_for_save = test_audio.squeeze(0).unsqueeze(0)  # (1, samples) -> (1, samples) for mono
        torchaudio.save(test_file, audio_for_save, sample_rate=16000)
        print(f"  Saved test audio to: {test_file}")
        
        # Load test audio
        loaded_audio, sample_rate = torchaudio.load(test_file)
        print(f"  Loaded audio shape: {loaded_audio.shape}")
        print(f"  Sample rate: {sample_rate}")
        
        # Verify format
        if loaded_audio.shape[0] != 1:
            raise ValueError(f"Audio must be mono, got {loaded_audio.shape[0]} channels")
        
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000 Hz, got {sample_rate} Hz")
        
        print("  ✓ Audio loading test passed!")
        
    except Exception as e:
        print(f"  ✗ Audio loading test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
            print(f"  Cleaned up: {test_file}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Whilter Model Test Suite")
    print("=" * 60)
    
    # Test model functionality
    test_whilter_model()
    
    # Test audio loading
    test_audio_loading()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
