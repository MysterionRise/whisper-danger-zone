# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech-to-text transcription tool with pluggable backends (Whisper, Voxtral) and optional speaker diarization. Runs fully offline without cloud APIs.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks (recommended)
pre-commit install

# Run all tests
pytest

# Run specific test file or test
pytest tests/test_main.py -v
pytest tests/test_backends.py::TestWhisperBackend -v

# Format and lint (run before commits)
black .
isort --profile black .
flake8 .

# Type checking
mypy main.py convert.py backends/ --ignore-missing-imports

# Security scan
bandit -r . -x ./tests
```

## Code Style

- Line length: 120 characters
- Formatter: Black with isort (profile=black)
- Type hints: Used throughout, checked with mypy
- Python version: 3.10+
- Coverage minimum: 80%

## Architecture

### Pluggable Backend System

```
backends/
├── __init__.py           # Registry: list_backends(), get_backend()
├── base.py               # TranscriptionBackend ABC, TranscriptionResult
├── whisper_backend.py    # OpenAI Whisper implementation
└── voxtral_backend.py    # Mistral Voxtral Mini/Small implementation
```

**Adding a new backend:**
1. Create `backends/mybackend.py` with class inheriting `TranscriptionBackend`
2. Implement `available_models()`, `load_model()`, `transcribe()` methods
3. Register in `backends/__init__.py`

### CLI (main.py)

- `parse_args()` → CLI argument parsing with `--backend` flag
- `run_transcription()` → Load backend and transcribe (replaces old `run_whisper()`)
- `show_backends()` → Display available backends (`--list-backends`)
- `show_models()` → Display available models (`--list-models`)
- `load_diarization_pipeline()` → Lazy load pyannote (only when --diarize used)
- `diarize_audio()` → Run speaker diarization
- `merge_diarization()` → Merge segments with speaker labels via midpoint lookup
- `write_outputs()` → Write text/JSON output

### Audio Conversion (convert.py)

- `collect_ogg_files()` → Recursively find OGG files
- `convert_file()` → Convert to 16-bit PCM WAV using pydub/ffmpeg

## Key Design Patterns

- **Pluggable backends**: Abstract `TranscriptionBackend` base class with registry
- **Optional dependencies**: pyannote.audio and Voxtral deps gracefully fall back if not installed
- **Lazy loading**: Diarization pipeline and Voxtral deps loaded only when needed
- **Error handling**: Uses `sys.exit()` for CLI errors (tests mock this with `SystemExit`)

## Testing Notes

- Tests use pytest with extensive mocking (unittest.mock, pytest-mock)
- Backend tests in `tests/test_backends.py`, CLI tests in `tests/test_main.py`
- When testing `sys.exit()` calls, the mock must raise `SystemExit` to match real behavior
- CI runs on Python 3.10 and 3.11 with 80% coverage requirement

## External Requirements

- ffmpeg must be installed for audio processing
- Speaker diarization requires a Hugging Face token (via `--hf-token` or `HUGGINGFACE_TOKEN` env var)
- Voxtral backend requires: torch, transformers>=4.36.0, librosa
