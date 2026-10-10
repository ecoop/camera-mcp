# Copyright (c) 2026 Eric Cooper.
"""Probe every configured camera: does it answer, is the login accepted, and what do its
streams deliver (codec, resolution, frame rate).

Read-only. Each stream is opened with `ffprobe`, its header is read, and it is closed;
nothing on the camera changes. Tapo cameras get both streams checked: /stream1 (main)
and /stream2 (low resolution). Needs $CAMERA_USER and $CAMERA_PASS for host cameras.

    python -m camera_mcp.probe [--config cameras.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction

from camera_mcp.config import Camera, ConfigError, default_config_path, load_cameras, redact

from camera_mcp.capture import find_tool

FFPROBE = find_tool("ffprobe", "CAMERA_MCP_FFPROBE")


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamInfo:
    codec: str
    width: int
    height: int
    fps: float | None
    seconds: float  # time to open the stream and read its header


def _fps(*rates: str | None) -> float | None:
    for rate in rates:
        try:
            value = float(Fraction(rate or ""))
        except (ValueError, ZeroDivisionError):
            continue
        if 0 < value < 1000:  # RTSP sources often report 0/0 or a 90 kHz timebase here
            return round(value, 2)
    return None


def probe_stream(camera: Camera, timeout: float = 10.0) -> StreamInfo:
    args = [FFPROBE, "-hide_banner", "-v", "error"]
    if camera.source.startswith(("rtsp://", "rtsps://")):
        args += ["-rtsp_transport", "tcp", "-timeout", str(int(timeout * 1_000_000))]
    args += ["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
             "-of", "json", "-i", camera.source]
    started = time.monotonic()
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout + 5)
    except FileNotFoundError:
        raise ProbeError("ffprobe not found; install ffmpeg or set CAMERA_MCP_FFPROBE") from None
    except subprocess.TimeoutExpired:
        raise ProbeError(f"no answer within {timeout:.0f}s") from None
    elapsed = time.monotonic() - started
    stderr = proc.stderr.decode(errors="replace")
    if "401" in stderr:
        raise ProbeError("login rejected: check CAMERA_USER and CAMERA_PASS (the Tapo Camera Account)")
    streams = json.loads(proc.stdout or b"{}").get("streams") or []
    if proc.returncode != 0 or not streams:
        detail = stderr.strip().splitlines()[-1:] or ["no video stream"]
        raise ProbeError(redact(detail[0]))
    s = streams[0]
    return StreamInfo(
        codec=s.get("codec_name", "?"),
        width=int(s.get("width") or 0),
        height=int(s.get("height") or 0),
        fps=_fps(s.get("avg_frame_rate"), s.get("r_frame_rate")),
        seconds=round(elapsed, 1),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m camera_mcp.probe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=default_config_path(),
                        help="camera YAML (default: $CAMERA_MCP_CONFIG, else cameras.yaml at the repo root)")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per stream (default 10)")
    args = parser.parse_args(argv)
    try:
        cameras = load_cameras(args.config)
    except ConfigError as e:
        print(f"probe: {e}", file=sys.stderr)
        raise SystemExit(2) from None

    failures = 0
    print(f"{'camera':14} {'stream':7} {'result':40} {'open':>5}")
    for camera in cameras.values():
        if not camera.enabled:
            print(f"{camera.name:14} {'-':7} disabled")
            continue
        variants = [("main", camera)] + ([("sub", sub)] if (sub := camera.substream()) else [])
        for which, cam in variants:
            try:
                info = probe_stream(cam, args.timeout)
            except ProbeError as e:
                failures += 1
                print(f"{camera.name:14} {which:7} FAIL {e}")
                continue
            fps = f"{info.fps:g} fps" if info.fps else "fps ?"
            result = f"{info.codec} {info.width}x{info.height} {fps}"
            print(f"{camera.name:14} {which:7} {result:40} {info.seconds:4.1f}s")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
