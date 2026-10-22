# Copyright (c) 2026 Eric Cooper.
"""The ffmpeg child processes a camera worker runs: the motion watcher's command line, stopping a
process for certain, collecting its last stderr lines, and stopping a recorder an earlier monitor
left running (found by its pid file)."""

from __future__ import annotations

import collections
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from camera_mcp.capture import FFMPEG
from camera_mcp.config import Camera
from camera_mcp.credentials import redact


def watcher_args(source: Camera, width: int, height: int, fps: float) -> list[str]:
    """ffmpeg arguments that decode `source` to raw YUV 4:2:0 frames on stdout: the Y plane
    (brightness) drives motion detection; the colour planes tell day from infrared night."""
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    args += ["-rtsp_transport", "tcp", "-timeout", "10000000"] if source.is_stream else ["-re"]
    return args + ["-i", source.source, "-an", "-vf", f"fps={fps},scale={width}:{height}",
                   "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]


def stop_process(proc: subprocess.Popen, grace: tuple[float, ...] = (2.0, 3.0)) -> None:
    """Stop an ffmpeg child for certain.

    ffmpeg 7 treats the first SIGTERM (or SIGINT) as "finish gracefully", and with the segment
    muxer that can hang indefinitely; a second signal makes it exit. SIGKILL is the backstop.
    """
    for wait in grace:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            pass
    proc.kill()
    proc.wait()


def _alive(pid: int) -> bool:
    """True if `pid` is running; a zombie (exited, not yet reaped) counts as gone."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    ps = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
    return ps.returncode == 0 and not ps.stdout.strip().startswith("Z")


def stop_stale_recorder(pid_file: Path) -> bool:
    """Stop a recorder left running by a monitor that died, if it's still writing into this
    camera's buffer. True if one was stopped. The command line must name the buffer directory,
    so an unrelated process that reused the pid is left alone."""
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    ps = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    stopped = False
    if ps.returncode == 0 and str(pid_file.parent) in ps.stdout:
        for sig in (signal.SIGTERM, signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            for _ in range(20):
                time.sleep(0.1)
                if not _alive(pid):
                    break
            else:
                continue
            break
        stopped = True
    pid_file.unlink(missing_ok=True)
    return stopped


def _tail(proc: subprocess.Popen) -> collections.deque[str]:
    """Collect the last few stderr lines of `proc` in the background (and keep the pipe drained)."""
    lines: collections.deque[str] = collections.deque(maxlen=5)

    def drain() -> None:
        for raw in proc.stderr:
            lines.append(redact(raw.decode(errors="replace").strip()))

    threading.Thread(target=drain, daemon=True).start()
    return lines
