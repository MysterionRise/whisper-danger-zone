# speech-transcription-toolkit

[![CI](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/MysterionRise/speech-transcription-toolkit/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MysterionRise/speech-transcription-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/MysterionRise/speech-transcription-toolkit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Speech-to-text transcription with pluggable backends (Whisper, Voxtral) and optional speaker diarization (no cloud APIs required).

## Features

- 🎤 **Multi-Backend Support**: OpenAI Whisper (default) or Mistral Voxtral models
- 👥 **Speaker Diarization**: Optional speaker identification using pyannote.audio
- 🔄 **Audio Conversion**: Batch convert OGG/Opus files to WAV
- 📝 **Multiple Output Formats**: Plain text, speaker-labeled text, JSON
- 🚀 **Fully Offline**: No cloud APIs, runs completely on your machine
- 🔌 **Extensible Architecture**: Easy to add new transcription backends
- 🧪 **Enterprise-Grade Quality**: Full test suite, CI/CD, 80% coverage minimum

## Installation

### Prerequisites

- Python 3.10+
- ffmpeg (required for audio processing)

**Install ffmpeg:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

### Install Python Dependencies

```bash
# Main dependencies
pip install -r requirements.txt

# Development dependencies (for testing)
pip install -r requirements-dev.txt
```

## Usage

### Basic Transcription

```bash
# Transcribe audio file to stdout (uses Whisper by default)
python main.py audio.wav

# Save to file
python main.py audio.mp3 -o transcript.txt

# Use specific model
python main.py audio.mp3 --model large

# Quiet mode (no progress bars)
python main.py audio.mp3 --quiet
```

### Backend Selection

```bash
# List available backends
python main.py --list-backends

# List models for a backend
python main.py --list-models
python main.py --list-models --backend voxtral

# Use Whisper (default)
python main.py audio.mp3 --backend whisper --model turbo

# Use Voxtral Mini (requires extra dependencies)
python main.py audio.mp3 --backend voxtral --model voxtral-mini
```

**Available Backends:**
| Backend | Models | Description |
|---------|--------|-------------|
| `whisper` | tiny, base, small, medium, large, turbo | OpenAI Whisper - fast, accurate, multilingual |
| `voxtral` | voxtral-mini, voxtral-small | Mistral Voxtral - strong multilingual support |

### Speaker Diarization

```bash
# Transcribe with speaker labels
python main.py audio.mp3 --diarize --hf-token $HUGGINGFACE_TOKEN -o transcript.txt

# Set HF token as environment variable (recommended for security)
export HUGGINGFACE_TOKEN=your_token_here
python main.py audio.mp3 --diarize -o transcript.txt
```

### Advanced Options

```bash
# Translate to English
python main.py audio.mp3 --task translate

# Specify language
python main.py audio.mp3 --language en

# Save JSON output with timestamps
python main.py audio.mp3 --json result.json

# Use GPU acceleration
python main.py audio.mp3 --device cuda
```

### Audio Conversion

```bash
# Convert single file
python convert.py audio.ogg

# Convert entire directory
python convert.py ./recordings --outdir ./wav --rate 16000 --channels 1

# Overwrite existing files
python convert.py *.ogg --overwrite
```

## Available Whisper Models

| Model | Parameters | VRAM | Speed | Quality |
|-------|------------|------|-------|---------|
| tiny | 39M | ~1GB | Fastest | Basic |
| base | 74M | ~1GB | Very Fast | Good |
| small | 244M | ~2GB | Fast | Better |
| medium | 769M | ~5GB | Moderate | Great |
| large | 1550M | ~10GB | Slow | Best |
| turbo | 809M | ~6GB | Fast | Excellent |

**Default:** `turbo` (best speed/quality tradeoff)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py -v

# Run tests in parallel
pytest -n auto
```

### Code Quality

**Recommended: Use pre-commit hooks** to automatically run quality checks before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

**Manual commands:**

```bash
# Format code
black .

# Sort imports
isort --profile black .

# Lint code
flake8 .

# Type checking
mypy main.py convert.py --ignore-missing-imports

# Security scan
bandit -r .
```

### CI/CD

GitHub Actions automatically runs (all checks are **blocking** - merges require passing CI):
- Unit tests (Python 3.10, 3.11) with 80% coverage threshold
- Code quality checks (black, isort, flake8)
- Security scanning (bandit, safety)
- Type checking (mypy)
- Integration tests

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for details.

## Architecture

```
speech-transcription-toolkit/
├── main.py                  # Main CLI for transcription + diarization
├── convert.py               # Audio format conversion utility
├── backends/                # Pluggable transcription backends
│   ├── __init__.py          # Backend registry
│   ├── base.py              # TranscriptionBackend abstract class
│   ├── whisper_backend.py   # OpenAI Whisper implementation
│   └── voxtral_backend.py   # Mistral Voxtral implementation
├── tests/                   # Comprehensive test suite
│   ├── test_main.py         # Tests for main.py
│   ├── test_convert.py      # Tests for convert.py
│   └── test_backends.py     # Tests for backend system
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── pyproject.toml           # Project config (black, isort, mypy, coverage)
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── .github/workflows/       # CI/CD pipeline (blocking quality gates)
├── CONTRIBUTING.md          # Contribution guidelines
├── SECURITY.md              # Security policy
└── GAPS.md                  # Enterprise quality gap analysis
```

## Analysis & Roadmap

See [ANALYSIS.md](ANALYSIS.md) for:
- 📊 Detailed code analysis
- 🔒 Security vulnerability assessment
- ⚡ Performance optimization recommendations
- 🎯 Feature improvement roadmap
- 📈 Prioritized enhancement phases

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines, including:
- Development environment setup
- Code style and conventions
- Testing requirements (80% coverage minimum)
- Pull request process

Quick start:
```bash
pip install -r requirements-dev.txt
pre-commit install
pytest
```

## Security

See [SECURITY.md](SECURITY.md) for our security policy and vulnerability reporting process.

## License

MIT License - See LICENSE file for details

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'pyannote.audio'"**
- Install optional diarization dependencies: `pip install pyannote.audio`

**"ffmpeg not found"**
- Install ffmpeg (see Prerequisites above)

**"CUDA not available"**
- Remove `--device cuda` flag to use CPU
- Or install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/

**"Hugging Face token error"**
- Get token from https://huggingface.co/settings/tokens
- Set as environment variable: `export HUGGINGFACE_TOKEN=your_token`

**Out of memory errors**
- Use a smaller Whisper model: `--model small`
- Process shorter audio segments
- Use CPU instead of GPU: `--device cpu`