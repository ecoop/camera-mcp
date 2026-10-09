# Copyright (c) 2026 Eric Cooper.
"""One-shot frame capture and JPEG encoding.

Frames come from the `ffmpeg` binary, not from OpenCV's video backends: OpenCV's macOS
wheels do not reliably bundle FFmpeg (4.14 and 5.0 on Intel Macs have none, so no RTSP),
and the binary can use hardware decode (VideoToolbox) on Apple Silicon. OpenCV is used
only to decode and resize the still.

Each call opens the source, takes one frame, and closes it: no per-camera state is held
here. Continuous watching (detection, ring buffers, recording) belongs to the monitor,
not to the MCP server.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np

from camera_mcp.config import NO_LOGIN, Camera, redact

FFMPEG = os.environ.get("CAMERA_MCP_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR
    taken_at: float  # epoch seconds


def reachable(camera: Camera, timeout: float = 1.0) -> bool:
    """Cheap liveness check: a TCP connect for streams, a file check otherwise. No decode."""
    endpoint = camera.endpoint()
    if endpoint is None:
        return os.path.exists(camera.source)
    try:
        with socket.create_connection(endpoint, timeout=timeout):
            return True
    except OSError:
        return False


def _ffmpeg_args(camera: Camera, timeout: float) -> list[str]:
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if camera.source.startswith(("rtsp://", "rtsps://")):
        # TCP: UDP drops packets on busy Wi-Fi and yields smeared frames.
        args += ["-rtsp_transport", "tcp", "-timeout", str(int(timeout * 1_000_000))]
    return args + ["-i", camera.source, "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"]


def grab_frame(camera: Camera, timeout: float = 5.0) -> Frame:
    """Return one decoded frame from the camera (the first frame, for a file)."""
    if camera.needs_login:
        raise CaptureError(f"{camera.name}: {NO_LOGIN}")
    if not camera.is_stream and not os.path.exists(camera.source):
        raise CaptureError(f"{camera.name}: source file not found")
    try:
        proc = subprocess.run(_ffmpeg_args(camera, timeout), capture_output=True, timeout=timeout + 5)
    except FileNotFoundError:
        raise CaptureError("ffmpeg not found; install it or set CAMERA_MCP_FFMPEG") from None
    except subprocess.TimeoutExpired:
        raise CaptureError(f"{camera.name}: no frame within {timeout:.0f}s") from None
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
        raise CaptureError(redact(f"{camera.name}: {detail[0]}"))
    image = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise CaptureError(f"{camera.name}: frame did not decode")
    return Frame(image=image, taken_at=time.time())


def to_jpeg(image: np.ndarray, max_width: int = 1024, quality: int = 80) -> tuple[bytes, int, int]:
    """Downscale to `max_width` (never up) and JPEG-encode. Returns (bytes, width, height)."""
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CaptureError("JPEG encoding failed")
    return buf.tobytes(), w, h
