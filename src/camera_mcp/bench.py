# Copyright (c) 2026 Eric Cooper.
"""How fast can this machine identify animals? Times the detector on real frames.

Runs SpeciesNet the way the monitor does (every box classified), on the whole frame and
with wide-view tiling, and reports seconds per frame and what that means: how many busy
cameras this machine can keep up with at one analysed frame per second each.

    python -m camera_mcp.bench --clip some.mp4 [--clip ...] [--frames 8]
    python -m camera_mcp.bench --camera backyard --camera east_deck   # live frames

Read-only: clips are only read; cameras give one frame each, like `snapshot`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

from camera_mcp.capture import FFMPEG, CaptureError, grab_any
from camera_mcp.config import ConfigError, default_config_path, find_camera, load_cameras, load_site
from camera_mcp.detect import Detector, SpeciesNetDetector, available


@dataclass(frozen=True)
class Timing:
    mode: str  # "whole frame" or "tiled"
    seconds: list[float]
    boxes: list[int]

    @property
    def median(self) -> float:
        return statistics.median(self.seconds)


def clip_frames(path: str, count: int) -> list[np.ndarray]:
    """`count` frames spread across a video file, as RGB arrays."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height:format=duration", "-of", "json", path],
                           capture_output=True, text=True)
    info = json.loads(probe.stdout or "{}")
    stream = (info.get("streams") or [{}])[0]
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    if not (width and height and duration):
        raise ValueError(f"{path}: not a readable video")
    rate = count / duration
    out = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
                          "-vf", f"fps={rate}", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, timeout=300).stdout
    size = width * height * 3
    return [np.frombuffer(out[i * size:(i + 1) * size], np.uint8).reshape(height, width, 3)
            for i in range(len(out) // size)]


def time_detector(detector: Detector, frames: list[np.ndarray]) -> list[Timing]:
    """Time `detector` per frame, whole-frame and tiled. The first call (model loading) is
    a warm-up and not counted."""
    detector.analyze(frames[0])
    results = []
    for mode, tiles in (("whole frame", False), ("tiled", True)):
        seconds, boxes = [], []
        for frame in frames:
            start = time.perf_counter()
            boxes.append(len(detector.analyze(frame, tiles=tiles)))
            seconds.append(time.perf_counter() - start)
        results.append(Timing(mode, seconds, boxes))
    return results


def capacity(seconds_per_frame: float, frames_per_second: float = 1.0) -> float:
    """Busy cameras this machine can keep up with, analysing `frames_per_second` each."""
    return 1.0 / (seconds_per_frame * frames_per_second)


def device() -> str:
    try:
        import torch
    except ImportError:
        return "no PyTorch"
    kind = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return f"{kind} (PyTorch {torch.__version__})"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m camera_mcp.bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", default=[], help="a video file to take frames from (repeatable)")
    parser.add_argument("--camera", action="append", default=[], help="a camera to take one live frame from (repeatable)")
    parser.add_argument("--frames", type=int, default=6, help="frames per clip (default 6)")
    parser.add_argument("--config", default=default_config_path())
    args = parser.parse_args(argv)
    if not available():
        print("bench: SpeciesNet isn't installed: pip install 'camera-mcp[detect]'", file=sys.stderr)
        raise SystemExit(2)
    if not (args.clip or args.camera):
        parser.error("give at least one --clip or --camera")

    frames: list[np.ndarray] = []
    try:
        for path in args.clip:
            frames += clip_frames(path, args.frames)
        if args.camera:
            cameras = load_cameras(args.config)
            for name in args.camera:
                cam = find_camera(cameras, name)
                if cam is None:
                    raise ConfigError(f"unknown camera {name!r}")
                frame, note = grab_any(cam, timeout=10)
                frames.append(frame.image[:, :, ::-1].copy())  # BGR to RGB
                print(f"{cam.name}: one live frame, {frame.image.shape[1]}x{frame.image.shape[0]}{note}")
        site = load_site(args.config)
    except (ConfigError, CaptureError, ValueError) as e:
        print(f"bench: {e}", file=sys.stderr)
        raise SystemExit(2) from None

    detector = SpeciesNetDetector(site)
    print(f"timing SpeciesNet on {len(frames)} frames; the first run loads the model and isn't counted")
    timings = time_detector(detector, frames)
    print(f"\nmachine: {device()}\nmodel:   {detector.model}")
    for t in timings:
        print(f"{t.mode:12} median {t.median:5.2f} s/frame   (min {min(t.seconds):.2f}, max {max(t.seconds):.2f};"
              f" {statistics.mean(t.boxes):.1f} boxes per frame)")
    whole, tiled = (t.median for t in timings)
    print(f"\nAt one analysed frame per second per busy camera, this machine keeps up with about"
          f"\n  {capacity(whole):.2g} busy cameras without tiling, or {capacity(tiled):.2g} wide-view (tiled) cameras.")


if __name__ == "__main__":
    main()
