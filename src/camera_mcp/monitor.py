# Copyright (c) 2026 Eric Cooper. All rights reserved.
"""The always-on monitor (phase 2a: motion and clips; no detection model yet).

For each enabled camera it runs:

- a **recorder**: ffmpeg copies the main stream into the ring buffer (see ringbuffer.py);
- a **watcher**: ffmpeg decodes the low-resolution stream to small grayscale frames at a
  few frames per second, and the motion detector opens and closes windows on them;
- **housekeeping**: once a window has closed and its post-roll is on disk, the covering
  segments are joined into a clip and the window is logged; old segments are deleted.

Streams that drop are reopened with backoff. A video file can stand in for a camera; it
is read at its own speed, once, and the monitor stops when every file has ended.

    camera-monitor [--config cameras.yaml] [--data-dir data] [--camera NAME ...] [--duration S]
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from camera_mcp import ringbuffer
from camera_mcp.capture import FFMPEG
from camera_mcp.config import Camera, ConfigError, default_config_path, find_camera, load_cameras, redact
from camera_mcp.motion import MotionDetector, MotionWindows, Window
from camera_mcp.probe import ProbeError, probe_stream
from camera_mcp.store import Store

log = logging.getLogger("camera_mcp.monitor")

BACKOFF = (2, 5, 10, 30, 60)  # seconds between reconnect attempts
STATUS_EVERY = 5.0  # seconds between status writes while watching


def default_data_dir() -> Path:
    """$CAMERA_MCP_DATA, else data/ at the repo root."""
    return Path(os.environ.get("CAMERA_MCP_DATA") or Path(__file__).resolve().parents[2] / "data")


def watcher_args(source: Camera, width: int, height: int, fps: float) -> list[str]:
    """ffmpeg arguments that decode `source` to raw grayscale frames on stdout."""
    args = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    args += ["-rtsp_transport", "tcp", "-timeout", "10000000"] if source.is_stream else ["-re"]
    return args + ["-i", source.source, "-an", "-vf", f"fps={fps},scale={width}:{height},format=gray",
                   "-f", "rawvideo", "-pix_fmt", "gray", "-"]


def _tail(proc: subprocess.Popen) -> collections.deque[str]:
    """Collect the last few stderr lines of `proc` in the background (and keep the pipe drained)."""
    lines: collections.deque[str] = collections.deque(maxlen=5)

    def drain() -> None:
        for raw in proc.stderr:
            lines.append(redact(raw.decode(errors="replace").strip()))

    threading.Thread(target=drain, daemon=True).start()
    return lines


class CameraWorker:
    def __init__(self, camera: Camera, data_dir: Path, store: Store, stop: threading.Event):
        self.camera = camera
        self.s = camera.monitor
        self.data_dir = data_dir
        self.seg_dir = data_dir / "buffer" / camera.name
        self.store = store
        self.stop = stop
        self.detector = MotionDetector(self.s)
        self.windows = MotionWindows(self.s.quiet_seconds)
        self.pending: list[Window] = []  # closed, waiting for their post-roll to reach disk
        self.lock = threading.Lock()
        self.procs: set[subprocess.Popen] = set()
        self.recorder_done = threading.Event()
        self.watcher_done = threading.Event()
        self.threads: list[threading.Thread] = []

    # ---- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.store.set_status(self.camera.name, "starting", now=time.time())
        for target in (self._record, self._watch, self._housekeep, self._reap):
            t = threading.Thread(target=target, name=f"{self.camera.name}:{target.__name__[1:]}", daemon=True)
            t.start()
            self.threads.append(t)

    def join(self, timeout: float | None = None) -> None:
        for t in self.threads:
            t.join(timeout)

    @property
    def done(self) -> bool:
        return self.recorder_done.is_set() and self.watcher_done.is_set() and not self.pending

    def _reap(self) -> None:
        """On stop, end this camera's ffmpeg processes so blocked reads return."""
        self.stop.wait()
        for proc in list(self.procs):
            proc.terminate()

    def _run(self, args: list[str], **kw) -> subprocess.Popen:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
        self.procs.add(proc)
        if self.stop.is_set():
            proc.terminate()
        return proc

    def _retry(self, attempt: int, error: str) -> bool:
        """Log a dropped stream and wait before reconnecting. False: give up (stop, or a file)."""
        if self.stop.is_set() or not self.camera.is_stream:
            return False
        delay = BACKOFF[min(attempt, len(BACKOFF) - 1)]
        log.warning("%s: %s; retrying in %ss", self.camera.name, error, delay)
        self.store.set_status(self.camera.name, "retrying", now=time.time(), error=error)
        return not self.stop.wait(delay)

    # ---- recorder -----------------------------------------------------------------

    def _record(self) -> None:
        attempt = 0
        try:
            while not self.stop.is_set():
                args = ringbuffer.recorder_args(self.camera, self.seg_dir, self.s.segment_seconds)
                proc = self._run(args, stdout=subprocess.DEVNULL, env=ringbuffer.recorder_env())
                errors = _tail(proc)
                started = time.monotonic()
                code = proc.wait()
                self.procs.discard(proc)
                if time.monotonic() - started > 60:
                    attempt = 0  # it ran for a while: start the backoff over
                if not self._retry(attempt, errors[-1] if errors else f"recorder exited ({code})"):
                    break
                attempt += 1
        finally:
            self.recorder_done.set()

    # ---- watcher ------------------------------------------------------------------

    def _watch(self) -> None:
        source = self.camera.substream() or self.camera
        attempt = 0
        try:
            while not self.stop.is_set():
                try:
                    info = probe_stream(source)
                except ProbeError as e:
                    if not self._retry(attempt, f"motion stream: {e}"):
                        break
                    attempt += 1
                    continue
                width = self.s.motion_width
                height = max(2, round(width * info.height / max(info.width, 1) / 2) * 2)
                proc = self._run(watcher_args(source, width, height, self.s.motion_fps), stdout=subprocess.PIPE)
                errors = _tail(proc)
                frames = self._read_frames(proc, width, height)
                proc.wait()
                self.procs.discard(proc)
                if frames:
                    attempt = 0
                if not self._retry(attempt, errors[-1] if errors else "motion stream ended"):
                    break
                attempt += 1
        finally:
            with self.lock:
                closed = self.windows.close()
            if closed:
                self._closed(closed)
            self.watcher_done.set()

    def _read_frames(self, proc: subprocess.Popen, width: int, height: int) -> int:
        size = width * height
        count = 0
        last_status = last_frame = 0.0
        while not self.stop.is_set():
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            now = last_frame = time.time()
            count += 1
            changed = self.detector.update(np.frombuffer(buf, np.uint8).reshape(height, width))
            moving = self.detector.is_motion(changed)
            with self.lock:
                was_open = self.windows.open is not None
                closed = self.windows.feed(now, changed, moving)
            if moving and not was_open:
                log.info("%s: motion (%.1f%% of pixels changed)", self.camera.name, changed * 100)
            if closed:
                self._closed(closed)
            if now - last_status >= STATUS_EVERY or moving and not was_open:
                self.store.set_status(self.camera.name, "watching", now=now, last_frame=now,
                                      last_motion=now if moving else None)
                last_status = now
        if count:  # the stream ended or we're stopping: record the last frame actually seen
            self.store.set_status(self.camera.name, "watching", now=time.time(), last_frame=last_frame)
        return count

    def _closed(self, window: Window) -> None:
        log.info("%s: motion window closed after %.1fs (%d motion frames)", self.camera.name,
                 window.last_motion - window.start, window.frames)
        with self.lock:
            self.pending.append(window)

    # ---- housekeeping -------------------------------------------------------------

    def _housekeep(self) -> None:
        while True:
            finished = self.recorder_done.is_set() and self.watcher_done.is_set()
            self._finish_pending(final=finished)
            self._prune()
            if finished and not self.pending:
                self.store.set_status(self.camera.name, "stopped", now=time.time())
                return
            time.sleep(0.5)

    def _finish_pending(self, final: bool) -> None:
        segments = ringbuffer.list_segments(self.seg_dir)
        with self.lock:
            pending = list(self.pending)
        for window in pending:
            clip_start = window.start - self.s.pre_seconds
            clip_end = window.last_motion + self.s.post_seconds
            # The post-roll is on disk once a later segment has started (or recording is over).
            if not (final or any(seg.start > clip_end for seg in segments)):
                continue
            usable = segments if self.recorder_done.is_set() else segments[:-1]
            clip = self._write_clip(ringbuffer.covering(usable, clip_start, clip_end), window)
            self.store.add_window(self.camera.name, window.start, window.last_motion, window.peak,
                                  window.frames, clip)
            with self.lock:
                self.pending.remove(window)

    def _write_clip(self, segments: list[ringbuffer.Segment], window: Window) -> str | None:
        if not segments:
            log.warning("%s: no recorded footage for the window at %s", self.camera.name,
                        datetime.fromtimestamp(window.start, timezone.utc).isoformat(timespec="seconds"))
            return None
        stamp = datetime.fromtimestamp(window.start, timezone.utc)
        out = self.data_dir / "clips" / self.camera.name / stamp.strftime("%Y-%m-%d") / stamp.strftime("%H%M%SZ.mp4")
        try:
            ringbuffer.assemble_clip(segments, out)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            log.warning("%s: %s", self.camera.name, e)
            return None
        log.info("%s: clip %s (%d segments)", self.camera.name, out.relative_to(self.data_dir), len(segments))
        return str(out.relative_to(self.data_dir))

    def _prune(self) -> None:
        keep = time.time() - self.s.pre_seconds - 2 * self.s.segment_seconds
        with self.lock:
            starts = [w.start for w in self.pending] + ([self.windows.open.start] if self.windows.open else [])
        if starts:
            keep = min(keep, min(starts) - self.s.pre_seconds - self.s.segment_seconds)
        ringbuffer.prune(self.seg_dir, keep)


class Monitor:
    def __init__(self, cameras: list[Camera], data_dir: Path):
        self.data_dir = data_dir
        self.store = Store(data_dir / "monitor.db")
        self.stop = threading.Event()
        self.workers = [CameraWorker(c, data_dir, self.store, self.stop) for c in cameras]

    def run(self, duration: float | None = None) -> None:
        """Watch until stopped, until `duration` seconds pass, or until every file source ends."""
        for w in self.workers:
            w.start()
        deadline = time.monotonic() + duration if duration else None
        while not all(w.done for w in self.workers):
            if self.stop.wait(0.5) or deadline and time.monotonic() >= deadline:
                break
        self.stop.set()
        for w in self.workers:
            w.join(timeout=30)
        self.store.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="camera-monitor", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=default_config_path(),
                        help="camera YAML (default: $CAMERA_MCP_CONFIG, else cameras.yaml at the repo root)")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="where the buffer, clips and monitor.db go (default: $CAMERA_MCP_DATA, else data/)")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME",
                        help="watch only this camera (repeatable; names match loosely)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cameras = load_cameras(args.config)
    except ConfigError as e:
        print(f"camera-monitor: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    chosen = [c for c in cameras.values() if c.enabled]
    if args.camera:
        found = {name: find_camera(cameras, name) for name in args.camera}
        if missing := [n for n, c in found.items() if c is None]:
            print(f"camera-monitor: unknown camera {', '.join(missing)}", file=sys.stderr)
            raise SystemExit(2)
        chosen = list({c.name: c for c in found.values()}.values())
    monitor = Monitor(chosen, args.data_dir)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: monitor.stop.set())
    log.info("watching %s; data in %s", ", ".join(c.name for c in chosen), args.data_dir)
    monitor.run(args.duration)


if __name__ == "__main__":
    main()
