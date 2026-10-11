# sync the data with .txt and .json from .srt format via script
"""
sync.py — reformat an edited SRT and regenerate .txt/.vtt/.json in sync.

Flow:
  1. transcribe.py  -> ../output/foo.srt, foo.txt, foo.json
  2. You intensively listen and edit foo.srt (fix words / timing).
  3. sync.py takes the edited foo.srt as the single source of truth,
     re-wraps and re-times it with tunable parameters, and writes a
     consistent set of .srt / .vtt / .txt / .json back out.

Examples:
  # Chinese subtitles, 20 chars per line, max 2 lines, merge tiny gaps
  python sync.py --input output/2018年研究生开学典礼.srt \
                   --output output --language zh \
                   --max-chars-per-line 20 --max-lines 2 \
                   --max-duration 8 --min-duration 0.8 --merge-gap 0.4

  # English subtitles, 42 chars per line
  python sync.py --input output/speech.srt --output output \
                   --language en --max-chars-per-line 42 --max-lines 2
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

# ------------------------------------------------------------
# Timestamps
# ------------------------------------------------------------
SRT_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)

def _to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0

def fmt_srt(t: float) -> str:
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t)
    if ms == 1000:
        s += 1
        ms = 0
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d},{ms:03d}"

def fmt_vtt(t: float) -> str:
    return fmt_srt(t).replace(",", ".")

# ------------------------------------------------------------
# Cue model
# ------------------------------------------------------------
@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

# ------------------------------------------------------------
# Language detection
# ------------------------------------------------------------
_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

def is_cjk_dominant(text: str) -> bool:
    if not text:
        return False
    cjk = len(_CJK.findall(text))
    return cjk / max(len(text), 1) > 0.3

# ------------------------------------------------------------
# Parse SRT
# ------------------------------------------------------------
def parse_srt(path: Path) -> List[Cue]:
    raw = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    cues: List[Cue] = []
    for blk in blocks:
        lines = [ln.rstrip() for ln in blk.splitlines()]
        if not lines:
            continue
        ts_line = next((ln for ln in lines if "-->" in ln), None)
        if ts_line is None:
            continue
        m = SRT_TS.search(ts_line)
        if not m:
            continue
        start = _to_sec(*m.group(1, 2, 3, 4))
        end = _to_sec(*m.group(5, 6, 7, 8))
        body_lines = lines[lines.index(ts_line) + 1:]
        text = " ".join(ln.strip() for ln in body_lines if ln.strip())
        if text:
            cues.append(Cue(start, end, text))
    return cues

# ------------------------------------------------------------
# Merge / split
# ------------------------------------------------------------
def merge_cues(cues, gap, max_dur, max_chars):
    out = []
    for c in cues:
        if out:
            p = out[-1]
            combined = (p.text + " " + c.text).strip()
            if ((c.start - p.end) <= gap
                    and (c.end - p.start) <= max_dur
                    and len(combined) <= max_chars):
                p.end = c.end
                p.text = combined
                continue
        out.append(Cue(c.start, c.end, c.text))
    return out

_PUNCT = set("。！？；!?;，,、 ")

def _split_text(text: str, n: int) -> List[str]:
    if n <= 1:
        return [text]
    target = len(text) / n
    pieces, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= target and ch in _PUNCT:
            pieces.append(buf.strip())
            buf = ""
    if buf.strip():
        pieces.append(buf.strip())

    while len(pieces) < n:
        i = max(range(len(pieces)), key=lambda k: len(pieces[k]))
        s = pieces[i]
        mid = len(s) // 2
        pieces[i:i+1] = [s[:mid].strip(), s[mid:].strip()]
    while len(pieces) > n:
        pieces[0] = pieces[0] + pieces[1]
        pieces.pop(1)
    return pieces

def split_long_cues(cues, max_dur, max_chars):
    out = []
    for c in cues:
        need_dur = int(c.duration // max_dur) + 1
        need_chars = int(len(c.text) // max_chars) + 1
        n = max(need_dur, need_chars, 1)
        if n == 1:
            out.append(c)
            continue
        pieces = _split_text(c.text, n)
        total = sum(len(p) for p in pieces) or 1
        t = c.start
        for p in pieces:
            share = (c.end - c.start) * (len(p) / total)
            out.append(Cue(t, t + share, p))
            t += share
        out[-1].end = c.end
    return out

# ------------------------------------------------------------
# Wrapping
# ------------------------------------------------------------
def wrap_text(text, max_chars, max_lines, cjk):
    text = re.sub(r"\s+", "", text) if cjk else re.sub(r"\s+", " ", text).strip()
    if cjk:
        lines = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    else:
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + ["".join(lines[max_lines - 1:])]
    return "\n".join(lines)

# ------------------------------------------------------------
# Writers
# ------------------------------------------------------------
def write_srt(cues, path, params):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_srt(c.start)} --> {fmt_srt(c.end)}\n")
            f.write(wrap_text(c.text, params["max_chars_per_line"], params["max_lines"], params["cjk"]))
            f.write("\n\n")

def write_vtt(cues, path, params):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for c in cues:
            f.write(f"{fmt_vtt(c.start)} --> {fmt_vtt(c.end)}\n")
            f.write(wrap_text(c.text, params["max_chars_per_line"], params["max_lines"], params["cjk"]))
            f.write("\n\n")

def write_txt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in cues:
            f.write(c.text.strip() + "\n")

def write_json(cues, path, source):
    data = {
        "source": str(source),
        "segments": [
            {"id": i, "start": c.start, "end": c.end,
             "duration": c.duration, "text": c.text.strip()}
            for i, c in enumerate(cues, 1)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reformat an edited SRT and sync other formats.")
    ap.add_argument("--input", required=True, help="Edited .srt (source of truth)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--formats", nargs="+", default=["srt", "txt", "vtt", "json"],
                    choices=["srt", "txt", "vtt", "json"])
    ap.add_argument("--max-chars-per-line", type=int, default=20,
                    help="Chinese ~18-22, English ~38-45")
    ap.add_argument("--max-lines", type=int, default=2)
    ap.add_argument("--max-duration", type=float, default=8.0, help="Split cues longer than this (seconds)")
    ap.add_argument("--min-duration", type=float, default=0.8, help="Nudge shorter cues")
    ap.add_argument("--merge-gap", type=float, default=0.4, help="Merge cues separated by less than this")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--language", default=None,
                    help="'zh' or 'en' to force wrapping style; omit for auto-detect")
    ap.add_argument("--suffix", default="", help="Append to output filename, e.g. '_v2'")
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem + args.suffix

    cues = parse_srt(src)
    print(f"[resync] parsed {len(cues)} cues from {src.name}")

    if not args.no_merge:
        cues = merge_cues(cues, args.merge_gap, args.max_duration,
                          args.max_chars_per_line * args.max_lines)
    if not args.no_split:
        cues = split_long_cues(cues, args.max_duration,
                               args.max_chars_per_line * args.max_lines)

    # Enforce min duration without overlapping the next cue
    for i, c in enumerate(cues):
        if c.duration < args.min_duration:
            desired = c.start + args.min_duration
            if i + 1 < len(cues):
                desired = min(desired, cues[i + 1].start)
            c.end = max(c.end, desired)

    cjk = (args.language == "zh") or (
        args.language is None and is_cjk_dominant(" ".join(c.text for c in cues[:20]))
    )
    params = {
        "cjk": cjk,
        "max_chars_per_line": args.max_chars_per_line,
        "max_lines": args.max_lines,
    }

    print(f"[resync] {len(cues)} cues after merge/split; cjk={cjk}")

    if "srt" in args.formats:
        write_srt(cues, out_dir / f"{stem}.srt", params)
    if "vtt" in args.formats:
        write_vtt(cues, out_dir / f"{stem}.vtt", params)
    if "txt" in args.formats:
        write_txt(cues, out_dir / f"{stem}.txt")
    if "json" in args.formats:
        write_json(cues, out_dir / f"{stem}.json", src)

    print(f"[resync] wrote {args.formats} -> {out_dir}")


if __name__ == "__main__":
    main()