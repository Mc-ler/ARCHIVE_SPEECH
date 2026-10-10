#!/usr/bin/env python3
"""
Batch transcription for videos and audios using faster-whisper.

Supports common video/audio extensions.
Outputs: txt, srt, vtt, json.

Customize by editing defaults or passing CLI arguments.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

# Force unbuffered / line-buffered stdout so Git Bash shows output live.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
except Exception:
    pass

# os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from faster_whisper import WhisperModel
from tqdm import tqdm

from Src.progress import setup_logger, Phase, hhmmss


# ------------------------------------------------------------
# Supported media extensions for video and audio
# ------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


# ------------------------------------------------------------
# Timestamp helpers
# ------------------------------------------------------------
def fmt_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def fmt_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ------------------------------------------------------------
# Writers
# ------------------------------------------------------------
def write_txt(segments, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def write_srt(segments, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_timestamp_srt(seg.start)} --> {fmt_timestamp_srt(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")


def write_vtt(segments, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{fmt_timestamp_vtt(seg.start)} --> {fmt_timestamp_vtt(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")


def write_json(segments, info, path: Path, source_file: Path) -> None:
    data = {
        "file": str(source_file),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": [
            {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in (seg.words or [])
                ],
            }
            for i, seg in enumerate(segments, 1)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# File collection from source
# ------------------------------------------------------------
def collect_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Return list of media files from a file or directory."""
    if input_path.is_file():
        return [input_path]

    pattern = "**/*" if recursive else "*"
    files = []
    for p in input_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            files.append(p)
    return sorted(files)


# ------------------------------------------------------------
# Core transcription
# ------------------------------------------------------------
def transcribe_file(
    model: WhisperModel,
    media_path: Path,
    out_dir: Path,
    formats: List[str],
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    beam_size: int = 5,
    logger=None,
    live_text: bool = True,
    text_every_n: int = 5,
) -> None:
    """Transcribe one media file and write requested formats.

    logger is optional; if omitted, a plain stdout logger is created.
    live_text=True prints a line every text_every_n segments without
    breaking the tqdm bar.
    """
    if logger is None:
        logger = setup_logger()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = media_path.stem
    base = out_dir / stem

    # ---------- transcribe (with live progress) ----------
    with Phase(logger, f"Transcribe {media_path.name}"):
        t0 = time.time()
        segments, info = model.transcribe(
            str(media_path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )

        total = max(float(info.duration or 0.0), 0.001)
        logger.info(
            f"       language={info.language} "
            f"(p={info.language_probability:.2f}), "
            f"duration={hhmmss(info.duration)}"
        )

        collected = []
        last_end = 0.0
        n = 0

        pbar = tqdm(
            total=total,
            unit="s",
            desc=f"[{stem}]",
            bar_format=(
                "{l_bar}{bar}| {n:.1f}/{total:.1f}s "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
            dynamic_ncols=True,
            file=sys.stdout,
        )

        try:
            for seg in segments:
                collected.append(seg)
                n += 1

                step = max(seg.end - last_end, 0.0)
                last_end = seg.end
                pbar.update(step)

                if live_text and n % text_every_n == 0:
                    pbar.write(
                        f"  {hhmmss(seg.start)} -> {hhmmss(seg.end)}  "
                        f"{seg.text.strip()}"
                    )
        finally:
            # Snap the bar to the last known position even on error.
            pbar.n = min(pbar.total, max(pbar.n, last_end))
            pbar.refresh()
            pbar.close()

        dt = time.time() - t0
        speed = total / dt if dt > 0 else 0.0
        logger.info(
            f"       {n} segments, {dt:.1f}s wall time, ~{speed:.1f}x realtime"
        )

    # ---------- writes (each its own phase) ----------
    if "txt" in formats:
        with Phase(logger, f"Write {stem}.txt"):
            write_txt(collected, base.with_suffix(".txt"))
    if "srt" in formats:
        with Phase(logger, f"Write {stem}.srt"):
            write_srt(collected, base.with_suffix(".srt"))
    if "vtt" in formats:
        with Phase(logger, f"Write {stem}.vtt"):
            write_vtt(collected, base.with_suffix(".vtt"))
    if "json" in formats:
        with Phase(logger, f"Write {stem}.json"):
            write_json(collected, info, base.with_suffix(".json"), media_path)

    logger.info(f"OK     {media_path.name} -> {base}.*")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch transcribe videos/audios with faster-whisper."
    )
    parser.add_argument("--input", required=True, help="File or directory")
    parser.add_argument("--output", default="../output", help="Output directory")
    parser.add_argument(
        "--model",
        default="large-v3",
        help="tiny/base/small/medium/large-v3/distil-large-v3",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--compute_type",
        default="float16",
        help="float16/int8_float16/int8/default",
    )
    parser.add_argument("--language", default=None, help="zh/en/... or omit for auto")
    parser.add_argument(
        "--initial_prompt",
        default=None,
        help="Bias vocabulary, e.g. names and university terms",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["txt", "srt", "json"],
        choices=["txt", "srt", "vtt", "json"],
        help="Output formats",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD silence filter")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word timestamps")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--model-dir",
        default="../models",
        help="Where to cache downloaded models",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide live segment text")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output)

    log_path = out_dir / "transcribe.log"
    logger = setup_logger(log_path)
    logger.info(f"Python {sys.version.split()[0]}, cwd={Path.cwd()}")
    logger.info(f"Arguments: {vars(args)}")

    files = collect_files(input_path, recursive=args.recursive)
    if not files:
        raise SystemExit("No media files found.")
    logger.info(f"Found {len(files)} file(s) in {input_path}")

    # ---------- model load ----------
    with Phase(logger, f"Load model {args.model} on {args.device} ({args.compute_type})"):
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=args.model_dir,
        )

    # ---------- per-file loop ----------
    t_start = time.time()
    for i, f in enumerate(files, 1):
        logger.info(f"=== [{i}/{len(files)}] {f.name} ===")
        try:
            transcribe_file(
                model=model,
                media_path=f,
                out_dir=out_dir,
                formats=args.formats,
                language=args.language,
                initial_prompt=args.initial_prompt,
                vad_filter=not args.no_vad,
                word_timestamps=args.word_timestamps,
                beam_size=args.beam_size,
                logger=logger,
                live_text=not args.quiet,
            )
        except Exception as e:
            logger.exception(f"FAILED on {f.name}: {e}")
            continue

    logger.info(f"All done in {time.time() - t_start:.1f}s. Log: {log_path}")


if __name__ == "__main__":
    main()