# Contributing to speech-transcription-toolkit

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up your development environment (see below)
4. Create a branch for your changes
5. Make your changes and test them
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10+
- ffmpeg (for audio processing)
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/speech-transcription-toolkit.git
cd speech-transcription-toolkit

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest

# Run code quality checks
pre-commit run --all-files
```

## Code Style

This project follows these conventions:

- **Formatter**: Black with 120 character line length
- **Import Sorting**: isort with Black profile
- **Linting**: flake8
- **Type Hints**: Used throughout, checked with mypy

### Quick Commands

```bash
# Format code
black .

# Sort imports
isort --profile black .

# Lint
flake8 .

# Type check
mypy main.py convert.py --ignore-missing-imports
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_main.py -v

# Run specific test
pytest tests/test_main.py::TestParseArgs::test_basic_args -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test classes `Test*`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Mock external dependencies (Whisper, pyannote, file I/O)

### Coverage Requirements

The project requires **80% minimum code coverage**. PRs that reduce coverage below this threshold will fail CI.

## Submitting Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-voxtral-backend`
- `fix/handle-empty-audio`
- `docs/update-readme`
- `refactor/extract-transcription-logic`

### Commit Messages

Write clear, descriptive commit messages:

```
feat: add Voxtral backend support

- Add VoxtralBackend class implementing TranscriptionBackend protocol
- Support both Mini (3B) and Small (24B) models
- Add --backend and --voxtral-model CLI arguments
```

Follow these prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Pull Request Process

1. **Ensure all checks pass**
   - All tests pass
   - Code coverage >= 80%
   - Pre-commit hooks pass
   - CI pipeline passes

2. **Update documentation**
   - Update README if needed
   - Add docstrings for new functions
   - Update CLAUDE.md if adding new commands

3. **Create the PR**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Reference any related issues

4. **Review process**
   - Address reviewer feedback
   - Keep the PR focused on a single concern
   - Rebase on main if needed

5. **After approval**
   - Squash commits if requested
   - Ensure final CI pass
   - A maintainer will merge

## Questions?

If you have questions about contributing, please open an issue with the "question" label.

Thank you for contributing!
