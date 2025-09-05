# Whilter: Whisper-based Data Filter for "In-the-Wild" Speech Corpora

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A neural network-based multi-task classification system built on OpenAI's Whisper models for filtering "in-the-wild" speech datasets. Whilter can automatically detect and classify audio recordings based on multiple criteria, making it ideal for preprocessing large speech corpora.

**Based on the paper:** [Whilter: A Whisper-based Data Filter for 'In-the-Wild' Speech Corpora Using Utterance-level Multi-Task Classification](https://arxiv.org/abs/XXXX.XXXXX) (Interspeech 2025)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Classify an audio file
python whilter_score.py audio.wav

# Get JSON output for scripting
python whilter_score.py audio.wav --json
```

## ✨ Features

### Multi-Task Audio Classification
Whilter performs simultaneous classification across 5 key categories:

- **🎤 Multi-speaker Detection**: Identifies audio with multiple speakers
- **🎵 Background Music Detection**: Detects presence of musical content  
- **🌍 Foreign Language Detection**: Identifies non-English speech
- **🔊 Noise Detection**: Detects noisy or reverberant speech
- **🤖 Synthetic Speech Detection**: Identifies artificially generated speech

### Model Variants
- **`default`**: Standard Whilter for "in-the-wild" speech filtering
- **`speech_quality`**: Speech quality assessment tasks
- **`general_audio`**: General audio classification (speech/music/noise/silence)
- **`lightweight`**: Optimized for faster inference

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.1.0 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **Audio**: 16kHz mono WAV files (other formats supported via torchaudio)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sdfilters.git
   cd sdfilters
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model checkpoints (optional):**
   ```bash
   git lfs pull  # If using git-lfs for model weights
   ```

## 📖 Usage

### Basic Classification

```bash
# Default multi-task classification
python whilter_score.py /path/to/audio.wav

# Specify model type
python whilter_score.py audio.wav --model_type speech_quality

# Adjust classification threshold (default: 0.5)
python whilter_score.py audio.wav --threshold 0.7

# Output in JSON format for scripting
python whilter_score.py audio.wav --json
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio_file` | Path to audio file (required) | - |
| `--model_type` | Model variant: `default`, `speech_quality`, `general_audio`, `lightweight` | `default` |
| `--threshold` | Classification threshold (0.0-1.0) | `0.5` |
| `--no-probabilities` | Hide probability scores | `False` |
| `--json` | Output in JSON format | `False` |

### Example Output

```bash
$ python whilter_score.py sample.wav

File: sample.wav
Model: default
Threshold: 0.5

Whilter Audio Classification Results:
==================================================
MULTISPEAKER     ✓ DETECTED        (0.823)
                 Multiple speakers present in audio

MUSIC            ✗ NOT DETECTED    (0.234)
                 Background music detected

FOREIGN_LANGUAGE ✗ NOT DETECTED    (0.156)
                 Non-English speech detected

NOISE            ✓ DETECTED        (0.678)
                 Noisy or reverberant speech

SYNTHETIC        ✗ NOT DETECTED    (0.089)
                 Synthetic or artificially generated speech
```

### JSON Output

```bash
$ python whilter_score.py sample.wav --json
{
  "file": "sample.wav",
  "model_type": "default",
  "threshold": 0.5,
  "probabilities": {
    "multispeaker": 0.823,
    "music": 0.234,
    "foreign_language": 0.156,
    "noise": 0.678,
    "synthetic": 0.089
  },
  "predictions": {
    "multispeaker": true,
    "music": false,
    "foreign_language": false,
    "noise": true,
    "synthetic": false
  }
}
```

## 🏗️ Architecture

### Model Overview
Whilter uses a transformer-based architecture with:

- **Feature Extraction**: Frozen Whisper encoder (`openai/whisper-small`) with learnable layer weights
- **Processing**: 4-layer transformer with attention pooling
- **Output**: Multi-task binary classification with separate attention heads for each task

### Key Components

```
Audio Input (16kHz mono)
         ↓
Whisper Encoder (frozen)
         ↓
Learnable Layer Weights
         ↓
4-Layer Transformer
         ↓
Multi-Head Attention Pooling
         ↓
Task-Specific Classifiers
         ↓
Binary Predictions
```

## 📊 Performance

- **Speed**: ~0.5 seconds per 30-second audio clip (GPU)
- **Memory**: ~2GB VRAM for inference
- **CPU Support**: Available but slower
- **Audio Length**: Up to 30 seconds (automatically padded/trimmed)

## 📁 Project Structure

```
sdfilters/
├── whilter_score.py              # Main classification script
├── test_whilter.py               # Test suite
├── scripts/
│   └── aitw_dataset.py           # AITW dataset preparation script
├── models/
│   ├── whilter.py                # Whilter model implementation
│   ├── models.py                 # Legacy speech quality models
│   ├── whisper_wrapper.py        # Whisper model wrappers
│   ├── transformer_wrapper.py    # Transformer utilities
│   ├── transformer_config.py     # Configuration classes
│   ├── pooling.py                # Attention pooling layers
│   └── mel_filters.npz           # Pre-computed mel filterbank
├── checkpoints/                  # Trained model weights
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python test_whilter.py
```

This will test all model variants and verify audio loading functionality.

## 📊 Dataset Preparation

### AITW Dataset

The Audio in the Wild (AITW) dataset can be prepared using the provided script:

```bash
# Download dataset structure and filemaps (small)
python scripts/aitw_dataset.py --output_dir ./datasets/aitw

# Download dataset with audio files (large - several GB)
python scripts/aitw_dataset.py --output_dir ./datasets/aitw --download_audio

# Check dataset information
python scripts/aitw_dataset.py --output_dir ./datasets/aitw --info
```

The script downloads the dataset from Zenodo and processes filemaps to download the relevant YODAS and EMILIA audio files.

## 🔧 Advanced Usage

### Batch Processing

```bash
# Process multiple files
for file in *.wav; do
    python whilter_score.py "$file" --json > "${file%.wav}.json"
done
```

### Integration with Python

```python
from models.whilter import create_whilter_model
import torch

# Load model
model = create_whilter_model('default')
model.eval()

# Process audio tensor (shape: [batch_size, audio_length])
audio = torch.randn(1, 16000 * 30)  # 30 seconds at 16kHz
with torch.no_grad():
    results = model(audio)
```

## 📚 Legacy Features

This repository also includes legacy speech quality assessment models:

```bash
# Single MOS prediction
python get_score.py audio.wav

# Multi-dimensional quality assessment  
python get_score.py audio.wav --model_type multi
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Citation

If you use Whilter in your research, please cite:

```bibtex
@inproceedings{ravenscroft2025whilter,
  title={Whilter: A Whisper-based Data Filter for "In-the-Wild" Speech Corpora Using Utterance-level Multi-Task Classification},
  author={Ravenscroft, William and Close, George and Bower-Morris, Kit and Stacey, Jamie and Sityaev, Dmitry and Hong, Kris Y.},
  booktitle={Interspeech 2025},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on OpenAI's [Whisper](https://github.com/openai/whisper) models
- Inspired by the need for robust "in-the-wild" speech data filtering
- Thanks to the Interspeech 2025 reviewers for their valuable feedback

---

**Note**: This is a research implementation. For production use, consider additional validation and error handling.
