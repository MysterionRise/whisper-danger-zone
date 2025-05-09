#!/usr/bin/env python3
"""Whisper CLI with optional speaker diarization.

This script wraps the OpenAI Whisper models and—optionally—the `pyannote.audio`
pre‑trained speaker‑diarization pipeline so you can label each segment with a
speaker ID.

Examples
--------
Transcribe only (no diarization):
    python main.py audio.mp3

Full diarization (requires `pyannote.audio` >=2.1):
    python main.py audio.mp3 --diarize --hf-token $HUGGINGFACE_TOKEN

Write combined transcript to a file while saving the raw JSON:
    python main.py audio.mp3 -o transcript.txt --json whisper.json --diarize

"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import whisper

# ``pyannote.audio`` is only required for diarization. Keep it optional.
try:
    from pyannote.audio import Pipeline  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Pipeline = None  # type: ignore

###############################################################################
# Argument parsing
###############################################################################

def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        prog="whisper-cli",
        description="Transcribe (and optionally diarize) audio with OpenAI Whisper.",
    )

    # Positional audio path
    parser.add_argument("audio", type=pathlib.Path, help="Path to the input audio/video file.")

    # Whisper‑related options
    parser.add_argument("-m", "--model", default="turbo", help="Whisper model size / name.")
    parser.add_argument("-l", "--language", default=None, help="Language code spoken in the audio.")
    parser.add_argument("-t", "--task", choices=("transcribe", "translate"), default="transcribe")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Force run device.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bars.")

    # Speaker‑diarization toggle
    parser.add_argument(
        "--diarize", action="store_true", help="Run speaker diarization with pyannote.audio.")
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="Hugging Face access token for downloading the diarization pipeline.\n"
        "If omitted, the HUGGINGFACE_TOKEN environment variable is used.",
    )

    # Output paths
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Write plain transcript here.")
    parser.add_argument("--json", type=pathlib.Path, help="Write raw Whisper result dict to JSON.")

    return parser.parse_args()

###############################################################################
# Whisper
###############################################################################

def run_whisper(args: argparse.Namespace) -> Dict[str, Any]:  # pragma: no cover
    if not args.audio.exists():
        sys.exit(f"Error: audio file '{args.audio}' does not exist.")

    model = whisper.load_model(args.model, device=args.device)

    kw: Dict[str, Any] = {}
    if args.language:
        kw["language"] = args.language
    kw["task"] = args.task

    # ``verbose`` is inverted relative to "quiet"
    kw["verbose"] = not args.quiet

    return model.transcribe(str(args.audio), **kw)

###############################################################################
# Diarization helpers
###############################################################################

def load_diarization_pipeline(token: Optional[str]) -> "Pipeline":  # type: ignore  # pragma: no cover
    """Load the PyAnnote speaker‑diarization pipeline (lazy)."""
    if Pipeline is None:
        sys.exit("pyannote.audio is not installed. Install with 'pip install pyannote.audio'.")

    auth_token = token or os.getenv("HUGGINGFACE_TOKEN")
    if auth_token is None:
        print("⚠️  No HF token provided. Proceeding without one (may fail for private models).", file=sys.stderr)

    # Using the official pretrained pipeline released under Apache‑2.0.
    return Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=auth_token)  # type: ignore


def diarize_audio(audio_path: pathlib.Path, pipeline: "Pipeline") -> List[Tuple[float, float, str]]:  # type: ignore  # pragma: no cover
    """Return list of (start, end, speaker_label)."""
    diarization = pipeline(str(audio_path))  # returns "pyannote.core.Annotation"

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments


def merge_diarization(
    whisper_result: Dict[str, Any],
    spk_segments: List[Tuple[float, float, str]],
) -> List[Dict[str, Any]]:
    """Attach speaker labels to each Whisper segment via midpoint lookup."""
    # Build timeline index
    indexed: List[Tuple[float, float, str]] = sorted(spk_segments, key=lambda x: x[0])

    output: List[Dict[str, Any]] = []
    for seg in whisper_result["segments"]:
        mid = (seg["start"] + seg["end"]) / 2.0
        speaker_id = "unknown"
        for s_start, s_end, label in indexed:
            if s_start <= mid <= s_end:
                speaker_id = label
                break
        output.append({**seg, "speaker": speaker_id})
    return output

###############################################################################
# Output helpers
###############################################################################

def write_outputs(result: Dict[str, Any], args: argparse.Namespace) -> None:  # pragma: no cover
    """Write transcript/plain text and optional JSON to disk or stdout."""
    if args.diarize and "speaker_segments" in result:
        # Pretty print with speaker labels
        lines: List[str] = []
        current_spk: Optional[str] = None
        for seg in result["speaker_segments"]:
            spk = seg["speaker"]
            if spk != current_spk:
                lines.append(f"\n[{spk}] ")
                current_spk = spk
            lines.append(seg["text"].strip())
        transcript = " ".join(lines).strip()
    else:
        transcript = result.get("text", "").strip()

    # Write / print transcript
    if args.output:
        args.output.write_text(transcript, encoding="utf-8")
    else:
        print(transcript)

    # Raw JSON output if requested
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

###############################################################################
# Main program flow
###############################################################################

def main() -> None:  # pragma: no cover
    args = parse_args()

    whisper_result = run_whisper(args)

    # Optionally run speaker diarization
    if args.diarize:
        pipeline = load_diarization_pipeline(args.hf_token)
        spk_segments = diarize_audio(args.audio, pipeline)
        speaker_segments = merge_diarization(whisper_result, spk_segments)
        whisper_result["speaker_segments"] = speaker_segments  # attach for JSON export

    write_outputs(whisper_result, args)


if __name__ == "__main__":  # pragma: no cover
    main()
