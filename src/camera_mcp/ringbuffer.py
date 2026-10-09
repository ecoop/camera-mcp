# Copyright (c) 2026 Eric Cooper.
"""The ring buffer: the camera's main stream, copied (never re-encoded) into short files.

ffmpeg's segment muxer writes the stream as a run of MPEG-TS files, each starting at a
keyframe and at least `segment_seconds` long, named by the UTC second it started. Old
files are deleted as time passes, unless a motion window still needs them; when a window
closes, the files covering it (plus the pre and post time) are joined into one MP4 clip,
again without re-encoding. Copying packets keeps full camera quality and costs almost no
CPU, unlike holding decoded frames in memory.
"""

from __future__ import annotations

import calendar
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from camera_mcp.capture import FFMPEG
from camera_mcp.config import Camera

_NAME = "%Y%m%dT%H%M%SZ"  # segment file names: the UTC second the segment started


@dataclass(frozen=True)
class Segment:
    path: Path
    start: float  # epoch seconds
    end: float  # epoch seconds: when the file was last written


def recorder_args(camera: Camera, seg_dir: Path, segment_seconds: float) -> list[str]:
    """ffmpeg arguments that copy `camera`'s stream into rolling segment files."""
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if camera.is_stream:
        args += ["-rtsp_transport", "tcp", "-timeout", "10000000"]
    else:
        args += ["-re"]  # a file stands in for a camera: read it at its own speed
    return args + [
        "-i", camera.source,
        "-map", "0:v:0", "-c", "copy",  # video only for now; Tapo audio (G.711) doesn't fit MPEG-TS
        "-f", "segment", "-segment_time", str(segment_seconds), "-segment_format", "mpegts",
        "-reset_timestamps", "1", "-strftime", "1",
        str(seg_dir / f"{_NAME}.ts"),
    ]


def recorder_env() -> dict[str, str]:
    """ffmpeg names segments in local time; run it in UTC so names are unambiguous."""
    return {**os.environ, "TZ": "UTC"}


def list_segments(seg_dir: Path) -> list[Segment]:
    """Segments on disk, oldest first."""
    segments = []
    for path in seg_dir.glob("*.ts"):
        try:
            start = calendar.timegm(time.strptime(path.stem, _NAME))
            end = path.stat().st_mtime
        except (ValueError, FileNotFoundError):
            continue
        segments.append(Segment(path, float(start), max(end, float(start))))
    return sorted(segments, key=lambda s: s.start)


def covering(segments: list[Segment], start: float, end: float) -> list[Segment]:
    """The segments that overlap [start, end]."""
    return [s for s in segments if s.end >= start and s.start <= end]


def prune(seg_dir: Path, keep_from: float) -> int:
    """Delete segments that ended before `keep_from`, except the newest (being written)."""
    segments = list_segments(seg_dir)
    removed = 0
    for seg in segments[:-1]:
        if seg.end < keep_from:
            seg.path.unlink(missing_ok=True)
            removed += 1
    return removed


def assemble_clip(segments: list[Segment], out: Path, timeout: float = 60.0) -> Path:
    """Join `segments` into one MP4 at `out`, without re-encoding.

    The newest segment can be cut short when the recorder is stopped; if joining fails, it is
    retried without that last segment.
    """
    segments = [s for s in segments if s.path.exists() and s.path.stat().st_size >= 188]  # one TS packet
    if not segments:
        raise ValueError("no segments to join")
    try:
        return _concat(segments, out, timeout)
    except RuntimeError:
        if len(segments) < 2:
            raise
        return _concat(segments[:-1], out, timeout)


def _concat(segments: list[Segment], out: Path, timeout: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        for seg in segments:
            listing.write(f"file '{seg.path.resolve()}'\n")
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-f", "concat", "-safe", "0", "-i", listing.name,
             "-c", "copy", "-movflags", "+faststart", str(out)],
            capture_output=True, timeout=timeout,
        )
    finally:
        os.unlink(listing.name)
    if proc.returncode != 0 or not out.exists():
        detail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"joining segments failed: {detail[0]}")
    return out
