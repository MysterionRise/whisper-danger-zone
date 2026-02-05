"""OpenAI Whisper transcription backend.

This backend wraps the openai-whisper library for local speech-to-text transcription.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import whisper

from .base import TranscriptionBackend, TranscriptionResult


class WhisperBackend(TranscriptionBackend):
    """OpenAI Whisper transcription backend.

    Supports all Whisper model sizes from tiny to large, plus the optimized
    turbo model for faster inference.
    """

    name = "whisper"
    description = "OpenAI Whisper - offline speech-to-text with multiple model sizes"

    # Model sizes in order of speed (fastest first) and accuracy (smallest first)
    MODELS = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "turbo"]

    @classmethod
    def available_models(cls) -> List[str]:
        """Return list of available Whisper model names."""
        return cls.MODELS.copy()

    @classmethod
    def default_model(cls) -> str:
        """Return 'turbo' as the default - best balance of speed and accuracy."""
        return "turbo"

    def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a Whisper model.

        Args:
            model_name: One of the available model sizes (tiny, base, small, medium, large, turbo).
            device: 'cpu', 'cuda', or None for auto-detection.

        Raises:
            ValueError: If model_name is not recognized.
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown Whisper model: {model_name}. Available: {', '.join(self.MODELS)}")

        self._model = whisper.load_model(model_name, device=device)
        self._model_name = model_name
        self._device = device or str(next(self._model.parameters()).device)

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code or None for auto-detection.
            task: 'transcribe' or 'translate'.
            verbose: Show progress during transcription.

        Returns:
            TranscriptionResult with transcript text and segments.

        Raises:
            RuntimeError: If no model is loaded.
            FileNotFoundError: If audio file doesn't exist.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Build transcription kwargs
        kwargs: Dict[str, Any] = {
            "task": task,
            "verbose": verbose,
        }
        if language:
            kwargs["language"] = language

        # Run transcription
        raw_result: Dict[str, Any] = self._model.transcribe(str(audio_path), **kwargs)

        # Convert to standardized format
        segments = [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "id": seg.get("id"),
                "tokens": seg.get("tokens"),
                "temperature": seg.get("temperature"),
                "avg_logprob": seg.get("avg_logprob"),
                "compression_ratio": seg.get("compression_ratio"),
                "no_speech_prob": seg.get("no_speech_prob"),
            }
            for seg in raw_result.get("segments", [])
        ]

        return TranscriptionResult(
            text=raw_result.get("text", ""),
            segments=segments,
            language=raw_result.get("language"),
            raw=raw_result,
        )
