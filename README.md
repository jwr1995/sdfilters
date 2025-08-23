# Whisper-based Speech Quality Assessment (WhiSQA)

A neural network-based speech quality assessment system built on OpenAI's Whisper models. This system predicts speech quality metrics including MOS (Mean Opinion Score) and multi-dimensional quality assessments.

## Features

- **Single MOS Prediction**: Predicts overall speech quality as a Mean Opinion Score (1-5 scale)
- **Multi-dimensional Assessment**: Predicts 5 quality dimensions:
  - MOS (Mean Opinion Score)
  - Noisiness
  - Coloration
  - Discontinuity
  - Loudness
- **Whisper-based**: Uses pre-trained Whisper encoder features for robust audio understanding
- **GPU Accelerated**: Supports CUDA and MPS (Apple Silicon) acceleration

## Requirements

- Python 3.8+
- PyTorch 2.1.0+
- transformers 4.35.0+
- torchaudio 2.1.0+
- soundfile
- git-lfs (for model checkpoints)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sdfilters
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure git-lfs is installed and initialize:
```bash
git lfs pull
```

## Usage

### Basic Usage

```bash
# Single MOS score prediction
python get_score.py /path/to/audio.wav

# Multi-dimensional quality assessment
python get_score.py /path/to/audio.wav --model_type multi
```

### Audio Requirements

- **Sample Rate**: 16 kHz
- **Channels**: Mono (single channel)
- **Format**: WAV (other formats supported via torchaudio)
- **Duration**: Any length (automatically padded/trimmed to 30 seconds)

### Example Output

```bash
# Single MOS
python get_score.py audio.wav
Audio Quality Assessment Results:
File: audio.wav
Model: single
----------------------------------------
MOS: 3.742

# Multi-dimensional
python get_score.py audio.wav --model_type multi
Audio Quality Assessment Results:
File: audio.wav
Model: multi
----------------------------------------
MOS: 3.742
Noisiness: 2.156
Coloration: 1.893
Discontinuity: 2.234
Loudness: 3.891
```

## Model Architecture

The system uses transformer-based architectures with:

- **Feature Extraction**: Frozen Whisper encoder (openai/whisper-small)
- **Processing**: Multi-layer transformer with attention pooling
- **Output**: Single or multi-dimensional quality predictions

### Model Variants

1. **whisperMetricPredictorEncoderLayersTransformerSmall**: Single MOS prediction
2. **whisperMetricPredictorEncoderLayersTransformerSmalldim**: Multi-dimensional prediction

## Results

![Results](results.png)

## File Structure

```
sdfilters/
├── get_score.py                    # Main evaluation script
├── models/                         # Model implementations
│   ├── whisper_ni_predictors.py    # Quality prediction models
│   ├── whisper_wrapper.py          # Whisper model wrappers
│   ├── transformer_wrapper.py     # Transformer utilities
│   ├── transformer_config.py      # Configuration classes
│   └── mel_filters.npz            # Pre-computed mel filterbank
├── checkpoints/                    # Trained model weights
│   ├── single_head_model.pt       # Single MOS model
│   └── multi_head_model.pt        # Multi-dimensional model
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Technical Details

### Performance
- **GPU Memory**: ~2GB VRAM for inference
- **CPU Fallback**: Supported but slower
- **Speed**: ~0.5 seconds per 30-second audio clip (GPU)

### Limitations
- Optimized for speech quality assessment
- Requires 16kHz mono audio input
- Maximum processing length: 30 seconds

## Citation

If you use this code in your research, please cite:

```bibtex
@article{whisqa2024,
  title={Whisper-based Speech Quality Assessment},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Add your license information here]
