# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Testing
```bash
# Run the Whilter model test suite
python test_whilter.py

# Test single audio file classification
python whilter_score.py audio.wav

# Test legacy speech quality assessment
python get_score.py audio.wav
```

### Main Usage
```bash
# Whilter multi-task audio classification (primary tool)
python whilter_score.py audio.wav                           # Default classification
python whilter_score.py audio.wav --model_type speech_quality  # Speech quality tasks
python whilter_score.py audio.wav --threshold 0.7          # Adjust threshold
python whilter_score.py audio.wav --json                   # JSON output

# Legacy speech quality assessment
python get_score.py audio.wav                              # Single MOS score
python get_score.py audio.wav --model_type multi          # Multi-dimensional scores
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

## Architecture Overview

This is **Whilter**, a Whisper-based multi-task audio classification system for filtering "in-the-wild" speech datasets, published at Interspeech 2025. The codebase has two main components:

### Primary System: Whilter Model (Interspeech 2025)
- **Core Implementation**: `models/whilter.py` - Main WhilterModel class with multi-task classification
- **Entry Point**: `whilter_score.py` - Command-line interface for audio classification
- **Research Paper**: "Whilter: A Whisper-based Data Filter for 'In-the-Wild' Speech Corpora Using Utterance-level Multi-Task Classification"

**Architecture Pipeline**: 
`Audio Input (30s, 16kHz)` → `Whisper Encoder (frozen, 12 layers)` → `Learnable Layer Weights` → `4-layer Transformer (768→256d)` → `Multi-head Attention Pooling` → `5 Task-specific Binary Classifiers`

**Default Tasks** (for "in-the-wild" speech filtering):
- **Multispeaker**: Multiple speakers present in audio
- **Music**: Background music detected  
- **Foreign Language**: Non-English speech detected
- **Noise**: Noisy or reverberant speech (including convolutive noise)
- **Synthetic**: Synthetic or artificially generated speech

**Performance**: Achieves F1 scores above 85% and equal error rates of 6.5-7.8% for three of five subtasks, outperforming BEATs classifier on speech-specific tasks.

### Legacy System: Speech Quality Assessment
- **Core Implementation**: `models/models.py` - SingleMOSPredictor and MultiDimPredictor
- **Entry Point**: `get_score.py` - Command-line interface for quality scoring
- **Purpose**: Mean Opinion Score (MOS) and multi-dimensional quality assessment

### Key Architecture Components

**Feature Extraction** (`models/feature_extractor.py`):
- `WhisperWrapper_encoder`: Frozen Whisper encoder (openai/whisper-small) with configurable layer output
- `pad_or_trim`: Audio preprocessing to 30-second chunks
- `log_mel_spectrogram`: Mel spectrogram computation

**Transformer Processing** (`models/transformer.py`):
- `TransformerWrapper`: 4-layer transformer network (768 → 256 dimensions)
- Configured via `models/config.py` Config class

**Attention Pooling** (`models/pooling.py`):
- `MultiHeadClassificationPool`: Separate attention pooling heads for each task
- `PoolAttFF`: Attention-based feature pooling with feedforward layers

### Model Variants
- `default`: Standard Whilter (5 tasks for speech filtering)
- `speech_quality`: Speech quality assessment tasks
- `general_audio`: General audio classification (speech/music/noise/silence)
- `lightweight`: Optimized for faster inference (fewer layers/dimensions)

## Audio Requirements

- **Format**: Mono (1 channel), 16kHz sample rate
- **Duration**: Up to 30 seconds (automatically padded/trimmed)
- **Supported Formats**: WAV, MP3, FLAC (via torchaudio)

## Model Checkpoints

Models expect checkpoints in `checkpoints/` directory:
- `whilter_model.pt` - Default Whilter model
- `whilter_speech_quality.pt` - Speech quality variant
- `whilter_general_audio.pt` - General audio variant  
- `whilter_lightweight.pt` - Lightweight variant
- `single_head_model.pt` - Legacy single MOS model
- `multi_head_model.pt` - Legacy multi-dimensional model

## Dataset Scripts

The `scripts/` directory contains dataset preparation utilities:

### AITW Dataset (Annotated In-The-Wild)
- **Main Script**: `weak_labels/aitw_dataset.py` - Downloads and prepares the AITW dataset
- **Dataset Details**: 21,414 manually annotated samples (≈64 hours) from Emilia and YODAS datasets
- **Split**: 18,346 training, 1,353 validation, 1,716 test samples
- **Annotation Tool**: Uses Label Studio GUI for consistent labeling by experienced annotators

### Training Data Preparation
- **Two-stage Training**: 
  1. Simulated training on mixed non-ITW datasets with dynamic mixing
  2. Fine-tuning on real AITW annotated data
- **Dynamic Mixing**: Artificial combinations at SNRs/SSRs of -5 to 10 dB
- **Datasets Used**: AMI, AliMeeting, MLS, MUSAN, OpenMic-2018, FSDNoisy-18k, DEMAND, LibriSpeech, FOR, ODSS

### Strong Labels Scripts
- Various labeled dataset processors in `strong_labels/` (LibriSpeech, MUSAN, AMI, AliMeeting, etc.)
- Dataset scripts handle downloading, processing, and annotation generation

## Development Notes

- **GPU Support**: GPU acceleration recommended (~0.033s per 30s audio on A10), CPU inference supported
- **Dependencies**: PyTorch 2.1.0+, transformers 4.35.0, torchaudio, soundfile, librosa
- **Training Details**: 
  - ADAM optimizer with exponential learning rate decay
  - Binary cross-entropy (BCE) loss for each task
  - Weighted random sampling to address class imbalances
  - Data augmentation: frequency dropping, frame dropping, bit-resolution reduction, speed perturbation
- **Model Architecture**: 
  - Whisper encoder layers frozen during training/inference
  - 12 learnable layer combination weights (learned via softmax)
  - Multi-task binary classification with separate attention pooling per task
- **Performance Insights**:
  - Speech-specific foundation models (Whisper) outperform general audio models (BEATs) on speech tasks
  - Significant processing time improvements vs single-task model combinations
  - Best performance on multispeaker, foreign language, and synthetic speech detection