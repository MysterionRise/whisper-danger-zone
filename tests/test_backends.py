"""Tests for the pluggable backend system."""

import pathlib
from unittest.mock import MagicMock, patch

import pytest

from backends import (
    DEFAULT_BACKEND,
    TranscriptionBackend,
    TranscriptionResult,
    get_backend,
    get_backend_class,
    list_backends,
    register_backend,
)
from backends.base import TranscriptionBackend as BaseBackend
from backends.whisper_backend import WhisperBackend


class TestTranscriptionResult:
    """Tests for TranscriptionResult class."""

    def test_basic_creation(self):
        """Test creating a basic TranscriptionResult."""
        result = TranscriptionResult(
            text="Hello world",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        )
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language is None
        assert result.raw == {}

    def test_with_language_and_raw(self):
        """Test creating TranscriptionResult with all fields."""
        result = TranscriptionResult(
            text="Bonjour",
            segments=[{"start": 0.0, "end": 0.5, "text": "Bonjour"}],
            language="fr",
            raw={"custom_field": "value"},
        )
        assert result.text == "Bonjour"
        assert result.language == "fr"
        assert result.raw["custom_field"] == "value"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TranscriptionResult(
            text="Test",
            segments=[{"start": 0.0, "end": 1.0, "text": "Test"}],
            language="en",
            raw={"extra": "data"},
        )
        d = result.to_dict()
        assert d["text"] == "Test"
        assert d["segments"] == [{"start": 0.0, "end": 1.0, "text": "Test"}]
        assert d["language"] == "en"
        assert d["extra"] == "data"  # raw fields merged in

    def test_to_dict_empty_segments(self):
        """Test to_dict with empty segments."""
        result = TranscriptionResult(text="", segments=[])
        d = result.to_dict()
        assert d["text"] == ""
        assert d["segments"] == []
        assert d["language"] is None


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = list_backends()
        assert isinstance(backends, list)
        assert "whisper" in backends
        assert "voxtral" in backends

    def test_get_backend_whisper(self):
        """Test getting Whisper backend."""
        backend = get_backend("whisper")
        assert isinstance(backend, WhisperBackend)
        assert backend.name == "whisper"

    def test_get_backend_voxtral(self):
        """Test getting Voxtral backend."""
        backend = get_backend("voxtral")
        assert backend.name == "voxtral"

    def test_get_backend_invalid(self):
        """Test getting non-existent backend."""
        with pytest.raises(ValueError) as exc_info:
            get_backend("nonexistent")
        assert "Unknown backend" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_backend_class(self):
        """Test getting backend class."""
        cls = get_backend_class("whisper")
        assert cls is WhisperBackend

    def test_get_backend_class_invalid(self):
        """Test getting non-existent backend class."""
        with pytest.raises(ValueError):
            get_backend_class("nonexistent")

    def test_default_backend(self):
        """Test that default backend is whisper."""
        assert DEFAULT_BACKEND == "whisper"

    def test_register_custom_backend(self):
        """Test registering a custom backend."""

        class CustomBackend(TranscriptionBackend):
            name = "custom"
            description = "Custom test backend"

            @classmethod
            def available_models(cls):
                return ["model1"]

            def load_model(self, model_name, device=None):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
                return TranscriptionResult(text="custom", segments=[])

        register_backend("custom_test", CustomBackend)
        assert "custom_test" in list_backends()

        backend = get_backend("custom_test")
        assert isinstance(backend, CustomBackend)


class TestWhisperBackend:
    """Tests for WhisperBackend."""

    def test_available_models(self):
        """Test available models list."""
        models = WhisperBackend.available_models()
        assert "tiny" in models
        assert "base" in models
        assert "small" in models
        assert "medium" in models
        assert "large" in models
        assert "turbo" in models

    def test_default_model(self):
        """Test default model is turbo."""
        assert WhisperBackend.default_model() == "turbo"

    def test_backend_properties_before_load(self):
        """Test backend properties before model is loaded."""
        backend = WhisperBackend()
        assert backend.is_loaded is False
        assert backend.model_name is None
        assert backend.device is None

    @patch("backends.whisper_backend.whisper")
    def test_load_model(self, mock_whisper):
        """Test loading a model."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("tiny")

        mock_whisper.load_model.assert_called_once_with("tiny", device=None)
        assert backend.is_loaded is True
        assert backend.model_name == "tiny"

    @patch("backends.whisper_backend.whisper")
    def test_load_model_with_device(self, mock_whisper):
        """Test loading a model with specific device."""
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("base", device="cuda")

        mock_whisper.load_model.assert_called_once_with("base", device="cuda")

    def test_load_model_invalid(self):
        """Test loading invalid model raises error."""
        backend = WhisperBackend()
        with pytest.raises(ValueError) as exc_info:
            backend.load_model("invalid_model")
        assert "Unknown Whisper model" in str(exc_info.value)

    def test_transcribe_without_model(self):
        """Test transcribing without loading model raises error."""
        backend = WhisperBackend()
        with pytest.raises(RuntimeError) as exc_info:
            backend.transcribe(pathlib.Path("audio.mp3"))
        assert "No model loaded" in str(exc_info.value)

    @patch("backends.whisper_backend.whisper")
    def test_transcribe_missing_file(self, mock_whisper, tmp_path):
        """Test transcribing missing file raises error."""
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("tiny")

        with pytest.raises(FileNotFoundError):
            backend.transcribe(tmp_path / "nonexistent.mp3")

    @patch("backends.whisper_backend.whisper")
    def test_transcribe_success(self, mock_whisper, tmp_path):
        """Test successful transcription."""
        # Create mock audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        # Mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": " Hello world",
                    "tokens": [1, 2, 3],
                    "temperature": 0.0,
                    "avg_logprob": -0.5,
                    "compression_ratio": 1.2,
                    "no_speech_prob": 0.01,
                }
            ],
            "language": "en",
        }
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("tiny")
        result = backend.transcribe(audio_file)

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert len(result.segments) == 1
        assert result.language == "en"
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 1.0

    @patch("backends.whisper_backend.whisper")
    def test_transcribe_with_language(self, mock_whisper, tmp_path):
        """Test transcription with specified language."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.transcribe.return_value = {"text": "Bonjour", "segments": [], "language": "fr"}
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("tiny")
        backend.transcribe(audio_file, language="fr")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "fr"

    @patch("backends.whisper_backend.whisper")
    def test_transcribe_translate_task(self, mock_whisper, tmp_path):
        """Test transcription with translate task."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])
        mock_model.transcribe.return_value = {"text": "Hello", "segments": []}
        mock_whisper.load_model.return_value = mock_model

        backend = WhisperBackend()
        backend.load_model("tiny")
        backend.transcribe(audio_file, task="translate")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["task"] == "translate"


class TestBaseBackend:
    """Tests for base TranscriptionBackend class."""

    def test_cannot_instantiate_abstract(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBackend()

    def test_default_model_empty_list(self):
        """Test default_model with empty available_models."""

        class EmptyBackend(BaseBackend):
            name = "empty"
            description = "Empty test backend"

            @classmethod
            def available_models(cls):
                return []

            def load_model(self, model_name, device=None):
                pass

            def transcribe(self, audio_path, language=None, task="transcribe", verbose=True):
                return TranscriptionResult(text="", segments=[])

        assert EmptyBackend.default_model() == ""
