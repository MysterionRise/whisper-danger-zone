#!/usr/bin/env python3
"""Tests for main.py speech-to-text functionality."""
from __future__ import annotations

import json
import pathlib
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from main import (
    diarize_audio,
    load_diarization_pipeline,
    merge_diarization,
    parse_args,
    run_whisper,
    write_outputs,
)


class TestParseArgs:
    """Test command-line argument parsing for main.py."""

    def test_parse_args_minimal(self):
        """Test parsing with only required audio argument."""
        args = parse_args(["audio.mp3"])
        assert args.audio == pathlib.Path("audio.mp3")
        assert args.model == "turbo"
        assert args.task == "transcribe"
        assert args.diarize is False
        assert args.quiet is False

    def test_parse_args_custom_model(self):
        """Test parsing with custom Whisper model."""
        args = parse_args(["audio.mp3", "--model", "large"])
        assert args.model == "large"

    def test_parse_args_language(self):
        """Test parsing with language specification."""
        args = parse_args(["audio.mp3", "--language", "en"])
        assert args.language == "en"

    def test_parse_args_translate_task(self):
        """Test parsing with translate task."""
        args = parse_args(["audio.mp3", "--task", "translate"])
        assert args.task == "translate"

    def test_parse_args_device(self):
        """Test parsing with device specification."""
        args = parse_args(["audio.mp3", "--device", "cuda"])
        assert args.device == "cuda"

    def test_parse_args_quiet(self):
        """Test parsing with quiet flag."""
        args = parse_args(["audio.mp3", "--quiet"])
        assert args.quiet is True

    def test_parse_args_diarization(self):
        """Test parsing with diarization enabled."""
        args = parse_args(["audio.mp3", "--diarize", "--hf-token", "test_token"])
        assert args.diarize is True
        assert args.hf_token == "test_token"

    def test_parse_args_output_files(self):
        """Test parsing with output file specifications."""
        args = parse_args(["audio.mp3", "-o", "transcript.txt", "--json", "result.json"])
        assert args.output == pathlib.Path("transcript.txt")
        assert args.json == pathlib.Path("result.json")


class TestMergeDiarization:
    """Test merging Whisper segments with speaker diarization."""

    def test_merge_single_segment_single_speaker(self):
        """Test merging with one segment and one speaker."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world"}
            ]
        }
        spk_segments = [(0.0, 5.0, "SPEAKER_00")]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 1
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello world"

    def test_merge_multiple_segments_single_speaker(self):
        """Test merging multiple segments with one speaker."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.0, "end": 4.0, "text": "world"},
                {"start": 4.0, "end": 6.0, "text": "today"}
            ]
        }
        spk_segments = [(0.0, 6.0, "SPEAKER_00")]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 3
        assert all(seg["speaker"] == "SPEAKER_00" for seg in result)

    def test_merge_speaker_changes(self):
        """Test merging with speaker changes."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.5, "end": 4.5, "text": "Hi there"},
                {"start": 5.0, "end": 7.0, "text": "How are you?"}
            ]
        }
        spk_segments = [
            (0.0, 2.5, "SPEAKER_00"),
            (2.5, 5.0, "SPEAKER_01"),
            (5.0, 7.0, "SPEAKER_00")
        ]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 3
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_merge_segment_without_speaker(self):
        """Test segment that doesn't fall within any speaker interval."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Hello"},
                {"start": 10.0, "end": 11.0, "text": "World"}  # Gap in speaker timeline
            ]
        }
        spk_segments = [
            (0.0, 2.0, "SPEAKER_00")
        ]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "unknown"

    def test_merge_overlapping_speakers(self):
        """Test with overlapping speaker segments."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "First segment"}
            ]
        }
        # Midpoint is 1.5, which falls in SPEAKER_00's range
        spk_segments = [
            (0.0, 2.0, "SPEAKER_00"),
            (1.5, 4.0, "SPEAKER_01")
        ]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 1
        # Should match first speaker that contains the midpoint
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merge_preserves_whisper_metadata(self):
        """Test that original Whisper segment metadata is preserved."""
        whisper_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello",
                    "id": 1,
                    "seek": 0,
                    "tokens": [1, 2, 3],
                    "temperature": 0.0
                }
            ]
        }
        spk_segments = [(0.0, 2.0, "SPEAKER_00")]

        result = merge_diarization(whisper_result, spk_segments)

        assert result[0]["id"] == 1
        assert result[0]["seek"] == 0
        assert result[0]["tokens"] == [1, 2, 3]
        assert result[0]["temperature"] == 0.0
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_merge_empty_segments(self):
        """Test merging with empty segments."""
        whisper_result = {"segments": []}
        spk_segments = [(0.0, 5.0, "SPEAKER_00")]

        result = merge_diarization(whisper_result, spk_segments)

        assert len(result) == 0

    def test_merge_unordered_speaker_segments(self):
        """Test that speaker segments are sorted before merging."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 5.0, "end": 7.0, "text": "World"}
            ]
        }
        # Deliberately unordered
        spk_segments = [
            (5.0, 8.0, "SPEAKER_01"),
            (0.0, 3.0, "SPEAKER_00")
        ]

        result = merge_diarization(whisper_result, spk_segments)

        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"


class TestWriteOutputs:
    """Test output writing functionality."""

    def test_write_plain_transcript_to_stdout(self, capsys):
        """Test writing plain transcript to stdout."""
        result = {
            "text": "This is a test transcript.",
            "segments": []
        }
        args = MagicMock()
        args.diarize = False
        args.output = None
        args.json = None

        write_outputs(result, args)

        captured = capsys.readouterr()
        assert "This is a test transcript." in captured.out

    def test_write_transcript_to_file(self, tmp_path: pathlib.Path):
        """Test writing transcript to a file."""
        result = {
            "text": "File transcript test.",
            "segments": []
        }
        output_file = tmp_path / "transcript.txt"

        args = MagicMock()
        args.diarize = False
        args.output = output_file
        args.json = None

        write_outputs(result, args)

        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "File transcript test." in content

    def test_write_diarized_transcript(self, tmp_path: pathlib.Path):
        """Test writing transcript with speaker labels."""
        result = {
            "text": "Original text",
            "speaker_segments": [
                {"speaker": "SPEAKER_00", "text": "Hello there."},
                {"speaker": "SPEAKER_01", "text": "Hi!"},
                {"speaker": "SPEAKER_00", "text": "How are you?"}
            ]
        }
        output_file = tmp_path / "diarized.txt"

        args = MagicMock()
        args.diarize = True
        args.output = output_file
        args.json = None

        write_outputs(result, args)

        content = output_file.read_text(encoding="utf-8")
        assert "[SPEAKER_00]" in content
        assert "[SPEAKER_01]" in content
        assert "Hello there." in content
        assert "Hi!" in content

    def test_write_json_output(self, tmp_path: pathlib.Path):
        """Test writing JSON output."""
        result = {
            "text": "JSON test",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "JSON test"}
            ]
        }
        json_file = tmp_path / "result.json"

        args = MagicMock()
        args.diarize = False
        args.output = None
        args.json = json_file

        write_outputs(result, args)

        assert json_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["text"] == "JSON test"
        assert len(loaded["segments"]) == 1

    def test_write_both_transcript_and_json(self, tmp_path: pathlib.Path):
        """Test writing both transcript and JSON files."""
        result = {
            "text": "Both outputs test",
            "segments": []
        }
        txt_file = tmp_path / "transcript.txt"
        json_file = tmp_path / "result.json"

        args = MagicMock()
        args.diarize = False
        args.output = txt_file
        args.json = json_file

        write_outputs(result, args)

        assert txt_file.exists()
        assert json_file.exists()

    def test_write_speaker_changes_formatted(self, capsys):
        """Test speaker change formatting in output."""
        result = {
            "speaker_segments": [
                {"speaker": "SPEAKER_00", "text": "First sentence."},
                {"speaker": "SPEAKER_00", "text": "Second sentence."},
                {"speaker": "SPEAKER_01", "text": "Different speaker."}
            ]
        }
        args = MagicMock()
        args.diarize = True
        args.output = None
        args.json = None

        write_outputs(result, args)

        captured = capsys.readouterr()
        # Should only show speaker label when it changes
        assert captured.out.count("[SPEAKER_00]") == 1
        assert captured.out.count("[SPEAKER_01]") == 1


class TestRunWhisper:
    """Test Whisper transcription functionality."""

    @patch("main.whisper")
    def test_run_whisper_basic(self, mock_whisper, tmp_path: pathlib.Path):
        """Test basic Whisper transcription."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        args = MagicMock()
        args.audio = audio_file
        args.model = "turbo"
        args.device = None
        args.language = None
        args.task = "transcribe"
        args.quiet = False

        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "Test transcription"}

        result = run_whisper(args)

        mock_whisper.load_model.assert_called_once_with("turbo", device=None)
        assert result["text"] == "Test transcription"

    @patch("main.whisper")
    def test_run_whisper_with_language(self, mock_whisper, tmp_path: pathlib.Path):
        """Test Whisper with language specification."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        args = MagicMock()
        args.audio = audio_file
        args.model = "turbo"
        args.device = None
        args.language = "en"
        args.task = "transcribe"
        args.quiet = False

        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "English text"}

        result = run_whisper(args)

        # Verify language was passed to transcribe
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"

    @patch("main.whisper")
    def test_run_whisper_translate_task(self, mock_whisper, tmp_path: pathlib.Path):
        """Test Whisper with translate task."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        args = MagicMock()
        args.audio = audio_file
        args.model = "turbo"
        args.device = None
        args.language = None
        args.task = "translate"
        args.quiet = False

        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "Translated text"}

        result = run_whisper(args)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["task"] == "translate"

    @patch("main.whisper")
    def test_run_whisper_quiet_mode(self, mock_whisper, tmp_path: pathlib.Path):
        """Test Whisper in quiet mode."""
        audio_file = tmp_path / "test.mp3"
        audio_file.touch()

        args = MagicMock()
        args.audio = audio_file
        args.model = "turbo"
        args.device = None
        args.language = None
        args.task = "transcribe"
        args.quiet = True

        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        mock_model.transcribe.return_value = {"text": "Quiet transcription"}

        result = run_whisper(args)

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["verbose"] is False

    @patch("main.whisper")
    @patch("main.sys.exit")
    def test_run_whisper_missing_file(self, mock_exit, mock_whisper):
        """Test error handling for missing audio file."""
        args = MagicMock()
        args.audio = pathlib.Path("/nonexistent/audio.mp3")

        run_whisper(args)

        mock_exit.assert_called_once()
        assert "does not exist" in str(mock_exit.call_args[0][0])


class TestLoadDiarizationPipeline:
    """Test diarization pipeline loading."""

    @patch("main.Pipeline")
    @patch.dict("os.environ", {"HUGGINGFACE_TOKEN": "env_token"})
    def test_load_pipeline_with_env_token(self, mock_pipeline):
        """Test loading pipeline with environment variable token."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        pipeline = load_diarization_pipeline(None)

        mock_pipeline.from_pretrained.assert_called_once()
        call_kwargs = mock_pipeline.from_pretrained.call_args[1]
        assert call_kwargs["use_auth_token"] == "env_token"

    @patch("main.Pipeline")
    def test_load_pipeline_with_explicit_token(self, mock_pipeline):
        """Test loading pipeline with explicitly provided token."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        pipeline = load_diarization_pipeline("explicit_token")

        call_kwargs = mock_pipeline.from_pretrained.call_args[1]
        assert call_kwargs["use_auth_token"] == "explicit_token"

    @patch("main.sys.exit")
    def test_load_pipeline_pyannote_not_installed(self, mock_exit):
        """Test error when pyannote.audio is not installed."""
        # Import main and temporarily set Pipeline to None
        import main

        original_pipeline = main.Pipeline
        try:
            main.Pipeline = None
            load_diarization_pipeline("token")
        finally:
            main.Pipeline = original_pipeline

        mock_exit.assert_called_once()
        assert "not installed" in str(mock_exit.call_args[0][0])

    @patch("main.Pipeline")
    @patch.dict("os.environ", {}, clear=True)
    def test_load_pipeline_no_token_warning(self, mock_pipeline, capsys):
        """Test warning when no token is provided."""
        mock_pipeline.from_pretrained.return_value = MagicMock()

        load_diarization_pipeline(None)

        captured = capsys.readouterr()
        assert "No HF token provided" in captured.err


class TestDiarizeAudio:
    """Test audio diarization functionality."""

    @patch("main.Pipeline")
    def test_diarize_audio_basic(self, mock_pipeline, tmp_path: pathlib.Path):
        """Test basic audio diarization."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.touch()

        # Mock diarization annotation
        mock_turn_1 = MagicMock()
        mock_turn_1.start = 0.0
        mock_turn_1.end = 2.5

        mock_turn_2 = MagicMock()
        mock_turn_2.start = 2.5
        mock_turn_2.end = 5.0

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_turn_1, None, "SPEAKER_00"),
            (mock_turn_2, None, "SPEAKER_01")
        ]

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_annotation

        result = diarize_audio(audio_file, mock_pipeline_instance)

        assert len(result) == 2
        assert result[0] == (0.0, 2.5, "SPEAKER_00")
        assert result[1] == (2.5, 5.0, "SPEAKER_01")

    @patch("main.Pipeline")
    def test_diarize_audio_multiple_speakers(self, mock_pipeline, tmp_path: pathlib.Path):
        """Test diarization with multiple speakers."""
        audio_file = tmp_path / "conversation.mp3"
        audio_file.touch()

        # Mock multiple speaker turns
        turns = [
            (0.0, 1.5, "SPEAKER_00"),
            (1.5, 3.0, "SPEAKER_01"),
            (3.0, 4.5, "SPEAKER_00"),
            (4.5, 6.0, "SPEAKER_02")
        ]

        mock_turns = []
        for start, end, speaker in turns:
            mock_turn = MagicMock()
            mock_turn.start = start
            mock_turn.end = end
            mock_turns.append((mock_turn, None, speaker))

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = mock_turns

        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = mock_annotation

        result = diarize_audio(audio_file, mock_pipeline_instance)

        assert len(result) == 4
        assert result[0][2] == "SPEAKER_00"
        assert result[1][2] == "SPEAKER_01"
        assert result[2][2] == "SPEAKER_00"
        assert result[3][2] == "SPEAKER_02"
