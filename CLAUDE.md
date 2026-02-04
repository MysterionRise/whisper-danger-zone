# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech-to-text transcription tool with optional speaker diarization using OpenAI Whisper and pyannote.audio. Runs fully offline without cloud APIs.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run specific test file or test
pytest tests/test_main.py -v
pytest tests/test_main.py::TestParseArgs::test_basic_args -v

# Format and lint (run before commits)
black .
isort --profile black .
flake8 .

# Type checking
mypy main.py convert.py --ignore-missing-imports

# Security scan
bandit -r .
```

## Code Style

- Line length: 120 characters
- Formatter: Black with isort (profile=black)
- Type hints: Used throughout, checked with mypy
- Python version: 3.10+

## Architecture

Two CLI tools with functional design (minimal classes):

**main.py** - Transcription with optional diarization
- `parse_args()` → CLI argument parsing
- `run_whisper()` → Load Whisper model and transcribe
- `load_diarization_pipeline()` → Lazy load pyannote (only when --diarize used)
- `diarize_audio()` → Run speaker diarization
- `merge_diarization()` → Merge Whisper segments with speaker labels via midpoint lookup
- `write_outputs()` → Write text/JSON output

**convert.py** - Batch audio conversion (OGG/Opus → WAV)
- `collect_ogg_files()` → Recursively find OGG files
- `convert_file()` → Convert to 16-bit PCM WAV using pydub/ffmpeg

## Key Design Patterns

- **Optional dependencies**: pyannote.audio gracefully falls back if not installed
- **Lazy loading**: Diarization pipeline loaded only when needed
- **Error handling**: Uses `sys.exit()` for CLI errors (tests mock this with `SystemExit`)

## Testing Notes

- Tests use pytest with extensive mocking (unittest.mock, pytest-mock)
- When testing `sys.exit()` calls, the mock must raise `SystemExit` to match real behavior
- CI runs on Python 3.10 and 3.11

## External Requirements

- ffmpeg must be installed for audio processing
- Speaker diarization requires a Hugging Face token (via `--hf-token` or `HUGGINGFACE_TOKEN` env var)
