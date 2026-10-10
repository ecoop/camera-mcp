# Copyright (c) 2026 Eric Cooper.
"""The always-on monitor.

For each enabled camera it runs:

- a **recorder**: ffmpeg copies the main stream into the ring buffer (see ringbuffer.py);
- a **watcher**: ffmpeg decodes the low-resolution stream to small frames at a few frames
  per second; the motion detector opens and closes windows on them, and every few minutes,
  while all is still, the view is checked against its reference (has the camera moved?);
- **housekeeping**: finished segments that overlap motion, or arrive while an animal is in
  view, go to the shared analyzer; closed visits get a thumbnail and a clip joined from the
  buffer; old segments are deleted.

The **analyzer** (one for all cameras) runs the detector (SpeciesNet) on frames decoded from
those segments and feeds each camera's visit tracker. Without a detector the monitor watches
for motion only and keeps a clip of every motion window.

Streams that drop are reopened with backoff. A video file can stand in for a camera; it is
read at its own speed, once, and the monitor stops when every file has ended.

    camera-monitor [--config cameras.yaml] [--data-dir data] [--camera NAME ...] [--duration S]
                   [--detector auto|speciesnet|none] [--reset-view NAME ...]
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, tzinfo
from pathlib import Path

import cv2
import numpy as np

from camera_mcp import ringbuffer
from camera_mcp.capture import FFMPEG
from camera_mcp.config import (
    Camera, ConfigError, default_config_path, default_data_dir, find_camera, load_cameras, load_site, redact,
)
from camera_mcp.detect import Detector, SpeciesNetDetector, available
from camera_mcp.motion import MotionDetector, MotionWindows, Window
from camera_mcp.probe import ProbeError, probe_stream
from camera_mcp.scenery import Scenery
from camera_mcp.store import Store
from camera_mcp.view import ViewCheck, is_colour
from camera_mcp.visits import Visit, VisitTracker

log = logging.getLogger("camera_mcp.monitor")


def watermark(frame: np.ndarray, text: str) -> np.ndarray:
    """A small camera label in the bottom-left corner over a dim strip, so a crop can be traced to
    its camera without occluding much. `frame` is RGB uint8; returns a labelled copy."""
    if not text:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.35, min(w, h) / 320 * 0.45)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 3
    band_h = min(h, th + base + 2 * pad)
    band_w = min(w, tw + 2 * pad)
    region = out[h - band_h:h, 0:band_w].astype(np.float32)
    out[h - band_h:h, 0:band_w] = (region * 0.35).astype(np.uint8)  # dim the strip behind the text
    cv2.putText(out, text, (pad, h - base - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out

BACKOFF = (2, 5, 10, 30, 60)  # seconds between reconnect attempts
STATUS_EVERY = 5.0  # seconds between status writes while watching
PAUSE_RECHECK = 30.0  # seconds between clock checks while a camera is outside its active window
SHUTDOWN_SECONDS = 120.0  # at most this long, after stopping, to save open visits and their media


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


class CameraWorker:
    def __init__(self, camera: Camera, data_dir: Path, store: Store, stop: threading.Event,
                 tz: tzinfo | None = None):
        self.camera = camera
        self.tz = tz or datetime.now().astimezone().tzinfo  # local time for the active window
        self.s = camera.monitor
        self.data_dir = data_dir
        self.seg_dir = data_dir / "buffer" / camera.name
        self.pid_file = self.seg_dir / "recorder.pid"
        self.store = store
        self.stop = stop
        self.motion = MotionDetector(self.s)
        self.windows = MotionWindows(self.s.quiet_seconds)
        self.view = ViewCheck(data_dir / "views", camera.name, every=self.s.view_check_seconds,
                              threshold=self.s.view_threshold)
        self.pending: list[Window] = []  # closed, waiting for their post-roll to reach disk
        # Detection (phase 2b), when the monitor has an analyzer:
        self.analyzer: Analyzer | None = None
        self.tracker = VisitTracker(camera.name, close_after=self.s.visit_close_seconds)
        self.scenery = Scenery(data_dir / "scenery" / f"{camera.name}.json")
        self.recent_windows: list[tuple[float, float]] = []  # (start, last motion) of closed windows
        self.considered_until = 0.0  # segments starting at or before this were already considered
        self.queued: dict[Path, float] = {}  # segments waiting for the analyzer -> their start
        self.pending_visits: list[Visit] = []  # closed visits waiting for their clip
        self.analysis_finished = threading.Event()
        self._dims: tuple[int, int] | None = None  # main-stream frame size, for decoding segments
        self.lock = threading.Lock()
        self.procs: set[subprocess.Popen] = set()
        self.recorder_done = threading.Event()
        self.watcher_done = threading.Event()
        self.threads: list[threading.Thread] = []

    # ---- lifecycle ----------------------------------------------------------------

    def start(self) -> None:
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        if stop_stale_recorder(self.pid_file):
            log.warning("%s: stopped a recorder left running by an earlier monitor", self.camera.name)
        self.store.set_status(self.camera.name, "starting", now=time.time())
        if stale := self.store.close_stale_visits(self.camera.name):
            log.warning("%s: closed %d visit(s) left open by an earlier monitor", self.camera.name, stale)
        for target in (self._record, self._watch, self._housekeep, self._reap):
            t = threading.Thread(target=target, name=f"{self.camera.name}:{target.__name__[1:]}", daemon=True)
            t.start()
            self.threads.append(t)

    def join(self, timeout: float | None = None) -> None:
        for t in self.threads:
            t.join(timeout)

    @property
    def done(self) -> bool:
        return (self.recorder_done.is_set() and self.watcher_done.is_set() and not self.pending
                and not self.pending_visits and (self.analyzer is None or self.analysis_finished.is_set()))

    def _active_now(self) -> bool:
        """Is the camera inside its watch window right now? (Always, with no window.)"""
        return self.camera.is_active(datetime.now(self.tz))

    def _await_active(self) -> bool:
        """Block while the camera is outside its active window. False means stop was set."""
        if self._active_now():
            return True
        self.store.set_status(self.camera.name, "paused", now=time.time())
        log.info("%s: outside its active window (%s); paused until it reopens",
                 self.camera.name, self.camera.active)
        while not self.stop.wait(PAUSE_RECHECK):
            if self._active_now():
                return True
        return False

    def _reap(self) -> None:
        """On stop, end this camera's ffmpeg processes so blocked reads return."""
        self.stop.wait()
        for proc in list(self.procs):
            stop_process(proc)

    def _run(self, args: list[str], **kw) -> subprocess.Popen:
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, **kw)
        self.procs.add(proc)
        if self.stop.is_set():
            stop_process(proc)
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
                if not self._await_active():
                    break
                args = ringbuffer.recorder_args(self.camera, self.seg_dir, self.s.segment_seconds)
                proc = self._run(args, stdout=subprocess.DEVNULL, env=ringbuffer.recorder_env())
                self.pid_file.write_text(str(proc.pid))
                errors = _tail(proc)
                started = time.monotonic()
                code = proc.wait()
                self.procs.discard(proc)
                self.pid_file.unlink(missing_ok=True)
                if time.monotonic() - started > 60:
                    attempt = 0  # it ran for a while: start the backoff over
                if not self._active_now():
                    continue  # window closed: re-gate and pause, not a stream error
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
                if not self._await_active():
                    break
                try:
                    info = probe_stream(source)
                except ProbeError as e:
                    if not self._retry(attempt, f"motion stream: {e}"):
                        break
                    attempt += 1
                    continue
                # Wide views decode motion frames at the low-res stream's full width, so a far-off
                # bird covers 4x the pixels.
                width = max(self.s.motion_width, 640) if "wide_view" in self.camera.tags else self.s.motion_width
                height = max(2, round(width * info.height / max(info.width, 1) / 2) * 2)
                proc = self._run(watcher_args(source, width, height, self.s.motion_fps), stdout=subprocess.PIPE)
                errors = _tail(proc)
                frames = self._read_frames(proc, width, height)
                proc.wait()
                self.procs.discard(proc)
                if frames:
                    attempt = 0
                if not self._active_now():
                    continue  # window closed: re-gate and pause, not a stream error
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
        luma = width * height
        size = luma * 3 // 2  # yuv420p: a full-size Y plane, then quarter-size U and V
        count = 0
        last_status = last_frame = 0.0
        span_start: float | None = None  # this stream session: the camera is being watched from here
        while not self.stop.is_set():
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            now = last_frame = time.time()
            span_start = span_start or now
            count += 1
            frame = np.frombuffer(buf, np.uint8)
            gray = frame[:luma].reshape(height, width)
            changed = self.motion.update(gray)
            moving = self.motion.is_motion(changed)
            with self.lock:
                was_open = self.windows.open is not None
                closed = self.windows.feed(now, changed, moving)
            if not moving and not was_open and self.view.due(now):
                self._check_view(now, gray, frame[luma:luma + luma // 4], frame[luma + luma // 4:])
            if moving and not was_open:
                log.info("%s: motion (%.1f%% of pixels changed)", self.camera.name, changed * 100)
            if closed:
                self._closed(closed)
            if now - last_status >= STATUS_EVERY or moving and not was_open:
                self.store.set_status(self.camera.name, "watching", now=now, last_frame=now,
                                      last_motion=now if moving else None)
                self.store.extend_uptime(self.camera.name, span_start, now)
                last_status = now
        if count:  # the stream ended or we're stopping: record the last frame actually seen
            self.store.set_status(self.camera.name, "watching", now=time.time(), last_frame=last_frame)
            self.store.extend_uptime(self.camera.name, span_start, last_frame)
        return count

    def _check_view(self, now: float, gray: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        result = self.view.check(now, gray, colour=is_colour(u, v))
        if result is None:
            return
        self.store.set_view(self.camera.name, result.score, result.moved, now=now)
        if result.changed and result.moved:
            log.warning("%s: the view no longer matches its reference (score %.2f): moved or "
                        "blocked? If it was re-aimed on purpose: camera-monitor --reset-view %s",
                        self.camera.name, result.score, self.camera.name)
        elif result.changed:
            log.info("%s: the view matches its reference again (score %.2f)", self.camera.name, result.score)

    def _closed(self, window: Window) -> None:
        log.info("%s: motion window closed after %.1fs (%d motion frames)", self.camera.name,
                 window.last_motion - window.start, window.frames)
        with self.lock:
            self.pending.append(window)
            self.recent_windows = [w for w in self.recent_windows if window.start - w[1] < 600]
            self.recent_windows.append((window.start, window.last_motion))

    # ---- housekeeping -------------------------------------------------------------

    def _housekeep(self) -> None:
        while True:
            if self.camera.active and not self._active_now():
                for proc in list(self.procs):  # window closed mid-stream: end it so we pause
                    stop_process(proc)
            finished = self.recorder_done.is_set() and self.watcher_done.is_set()
            for step in (lambda: self.analyzer and self._schedule(finished),
                         lambda: self._finish_pending(final=finished),
                         lambda: self._finish_visits(final=finished and self.analysis_finished.is_set()),
                         self._prune):
                try:
                    step()
                except Exception:  # one failure must not stop this camera's housekeeping
                    log.exception("%s: housekeeping step failed", self.camera.name)
            if self.done:
                self.store.set_status(self.camera.name, "stopped", now=time.time())
                return
            time.sleep(0.5)

    # ---- detection ----------------------------------------------------------------

    def _schedule(self, finished: bool) -> None:
        """Send finished segments to the analyzer if they overlap motion or an open visit."""
        segments = ringbuffer.list_segments(self.seg_dir)
        complete = segments if self.recorder_done.is_set() else segments[:-1]
        for seg in complete:
            if seg.start <= self.considered_until:
                continue
            self.considered_until = seg.start
            if self._needs_analysis(seg) and not self.stop.is_set():
                with self.lock:
                    self.queued[seg.path] = seg.start
                self.analyzer.submit(self, seg)
        if finished and not self.queued and not self.analysis_finished.is_set():
            with self.lock:
                closed = self.tracker.close_all()  # input is over: whatever is open has ended
            for visit in closed:
                self._visit_closed(visit)
            self.analysis_finished.set()

    def _needs_analysis(self, seg: ringbuffer.Segment, margin: float = 1.0) -> bool:
        with self.lock:
            if self.tracker.open:
                return True  # keep looking while an animal is here, moving or not
            spans = list(self.recent_windows) + ([(self.windows.open.start, float("inf"))] if self.windows.open else [])
        return any(seg.end >= start - margin and seg.start <= end + margin for start, end in spans)

    def frame_size(self, seg: ringbuffer.Segment) -> tuple[int, int]:
        if self._dims is None:
            info = probe_stream(Camera("segment", str(seg.path)))
            self._dims = (info.width, info.height)
        return self._dims

    def _visit_closed(self, visit: Visit) -> None:
        label = visit.label
        log.info("%s: %s visit ended: %s (%.0f%% of votes), %.0fs, up to %d at once", self.camera.name,
                 visit.key, label.name, label.score * 100, visit.last_seen - visit.start, visit.max_count)
        with self.lock:
            self.pending_visits.append(visit)

    def _finish_visits(self, final: bool) -> None:
        segments = ringbuffer.list_segments(self.seg_dir)
        with self.lock:
            pending = list(self.pending_visits)
        for visit in pending:
            clip_end = min(visit.last_seen + self.s.post_seconds, visit.start + self.s.max_clip_seconds)
            if not (final or any(seg.start > clip_end for seg in segments)):
                continue
            keep_images = visit.key != "person" or self.s.save_people_images
            thumbnail = clip = None
            if keep_images:
                stamp = datetime.fromtimestamp(visit.start, timezone.utc)
                name = f"{stamp:%H%M%SZ}-{re.sub(r'[^a-z0-9]+', '-', visit.key)}"
                try:
                    thumbnail = self._write_thumbnail(visit, f"{stamp:%Y-%m-%d}/{name}.jpg")
                    usable = segments if self.recorder_done.is_set() else segments[:-1]
                    clip = self._write_clip(ringbuffer.covering(usable, visit.start - self.s.pre_seconds, clip_end),
                                            visit.start, f"{name}.mp4")
                except Exception:  # keep the visit even if its media can't be written
                    log.exception("%s: couldn't save media for the %s visit", self.camera.name, visit.key)
            self.store.add_visit(visit, model=self.analyzer.model if self.analyzer else "", thumbnail=thumbnail, clip=clip)
            with self.lock:
                self.pending_visits.remove(visit)

    def _write_thumbnail(self, visit: Visit, rel: str) -> str | None:
        if visit.thumbnail is None or visit.thumbnail.size == 0:
            return None
        out = self.data_dir / "media" / self.camera.name / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        labelled = watermark(visit.thumbnail, self.camera.short)  # which camera this crop came from
        # imencode, not imwrite: SpeciesNet's yolov5 replaces cv2.imwrite with a 2-argument version.
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(labelled, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        out.write_bytes(jpeg.tobytes())
        return str(out.relative_to(self.data_dir))

    def _finish_pending(self, final: bool) -> None:
        segments = ringbuffer.list_segments(self.seg_dir)
        with self.lock:
            pending = list(self.pending)
        for window in pending:
            clip_start = window.start - self.s.pre_seconds
            clip_end = min(window.last_motion + self.s.post_seconds, window.start + self.s.max_clip_seconds)
            # The post-roll is on disk once a later segment has started (or recording is over).
            if not (final or (self.analyzer and not self.s.training_capture)
                    or any(seg.start > clip_end for seg in segments)):
                continue
            if self.analyzer and not self.s.training_capture:
                clip = None  # with detection on, clips are kept for visits, not for raw motion
            else:
                usable = segments if self.recorder_done.is_set() else segments[:-1]
                clip = self._write_clip(ringbuffer.covering(usable, clip_start, clip_end), window.start)
            self.store.add_window(self.camera.name, window.start, window.last_motion, window.peak,
                                  window.frames, clip)
            with self.lock:
                self.pending.remove(window)

    def _write_clip(self, segments: list[ringbuffer.Segment], start: float, name: str | None = None) -> str | None:
        stamp = datetime.fromtimestamp(start, timezone.utc)
        if not segments:
            log.warning("%s: no recorded footage for %s", self.camera.name, stamp.isoformat(timespec="seconds"))
            return None
        out = self.data_dir / "clips" / self.camera.name / f"{stamp:%Y-%m-%d}" / (name or f"{stamp:%H%M%SZ}.mp4")
        try:
            ringbuffer.assemble_clip(segments, out)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            log.warning("%s: %s", self.camera.name, e)
            return None
        log.info("%s: clip %s (%d segments)", self.camera.name, out.relative_to(self.data_dir), len(segments))
        return str(out.relative_to(self.data_dir))

    def _prune(self) -> None:
        keep = time.time() - self.s.pre_seconds - 2 * self.s.segment_seconds
        with self.lock:
            starts = [w.start for w in self.pending] + ([self.windows.open.start] if self.windows.open else [])
            starts += list(self.queued.values()) + [v.start for v in self.pending_visits]
            starts += [v.start for v in self.tracker.open.values()]
        if starts:
            keep = min(keep, min(starts) - self.s.pre_seconds - self.s.segment_seconds)
        ringbuffer.prune(self.seg_dir, keep)


class Analyzer:
    """One detection model shared by every camera, fed finished segments through a queue.

    Each segment is decoded at one frame per `detect_every_seconds`; when the queue backs up
    (a slow machine, many busy cameras), only one frame per segment is analysed until it
    catches up. On stop, queued segments are dropped and open visits are closed as they are.
    """

    BACKLOG = 8  # queued segments beyond which analysis thins out
    SKIP_REPORT_EVERY = 60.0  # seconds between "can't keep up" warnings

    def __init__(self, detector: Detector, stop: threading.Event):
        self.detector = detector
        self.stop = stop
        self.queue: queue.Queue[tuple[CameraWorker, ringbuffer.Segment]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="analyzer", daemon=True)
        self._behind = False
        self._skipped: dict[str, float] = defaultdict(float)  # seconds skipped per camera since the last report
        self._last_report = 0.0

    @property
    def model(self) -> str:
        return self.detector.model

    def submit(self, worker: CameraWorker, seg: ringbuffer.Segment) -> None:
        self.queue.put((worker, seg))

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                worker, seg = self.queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop.is_set():
                    return
                continue
            try:
                if self.stop.is_set():
                    pass
                elif time.time() - seg.end > worker.s.max_analysis_lag:
                    self._skip(worker, seg)
                else:
                    self._analyze(worker, seg)
            except Exception as e:  # a bad segment or a model error must not stop the monitor
                log.warning("%s: analysis failed for %s: %s", worker.camera.name, seg.path.name, e)
            finally:
                with worker.lock:
                    worker.queued.pop(seg.path, None)

    def _skip(self, worker: CameraWorker, seg: ringbuffer.Segment) -> None:
        """Too far behind to be useful: drop the segment, but record the blind spot."""
        worker.store.add_skipped(worker.camera.name, seg.start, seg.end)
        self._skipped[worker.camera.name] += seg.end - seg.start
        now = time.time()
        if now - self._last_report >= self.SKIP_REPORT_EVERY:
            log.warning("analysis can't keep up: skipped %s (motion more than %.0fs old); this machine is too "
                        "slow for this many busy cameras", ", ".join(f"{s:.0f}s on {c}" for c, s in self._skipped.items()),
                        worker.s.max_analysis_lag)
            self._skipped.clear()
            self._last_report = now

    def _analyze(self, worker: CameraWorker, seg: ringbuffer.Segment) -> None:
        s = worker.s
        behind = self.queue.qsize() > self.BACKLOG
        if behind != self._behind:
            self._behind = behind
            log.warning("analysis is %s", "falling behind: one frame per segment until it catches up"
                        if behind else "caught up")
        every = max(seg.end - seg.start, s.segment_seconds) if behind else s.detect_every_seconds
        width, height = worker.frame_size(seg)
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(seg.path),
             "-vf", f"fps=1/{every}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, timeout=60,
        )
        size = width * height * 3
        for i in range(len(proc.stdout) // size):
            if self.stop.is_set():
                return
            frame = np.frombuffer(proc.stdout[i * size:(i + 1) * size], np.uint8).reshape(height, width, 3)
            t = seg.start + i * every
            detections = self.detector.analyze(frame, tiles="wide_view" in worker.camera.tags,
                                               min_conf=s.min_box_conf, max_boxes=s.max_boxes,
                                               vehicles=s.detect_vehicles)
            detections, learned = worker.scenery.filter(t, detections, frame)
            for spot in learned:
                log.info("%s: learned scenery at x=%.2f y=%.2f (%.0f%% x %.0f%% of the frame): a \"%s\" that "
                         "keeps appearing in exactly the same place; ignoring it from now on",
                         worker.camera.name, spot.x, spot.y, spot.w * 100, spot.h * 100, spot.kind)
            with worker.lock:
                before = set(worker.tracker.open)
                closed = worker.tracker.feed(t, detections, frame)
                still_open = dict(worker.tracker.open)
            for key, visit in sorted(still_open.items()):
                if visit.db_id is None:  # arrived (or took over an unidentified visit's row)
                    if key not in before:
                        log.info("%s: %s arrived", worker.camera.name, key)
                    visit.db_id = worker.store.open_visit(visit, model=self.model)
                elif visit.last_seen == t:
                    worker.store.update_open_visit(visit)
            for visit in closed:
                worker._visit_closed(visit)


class Monitor:
    def __init__(self, cameras: list[Camera], data_dir: Path, detector: Detector | None = None,
                 tz: tzinfo | None = None):
        self.data_dir = data_dir
        self.store = Store(data_dir / "monitor.db")
        self.stop = threading.Event()
        self.workers = [CameraWorker(c, data_dir, self.store, self.stop, tz) for c in cameras]
        self.analyzer = Analyzer(detector, self.stop) if detector else None
        for w in self.workers:
            w.analyzer = self.analyzer

    def run(self, duration: float | None = None) -> None:
        """Watch until stopped, until `duration` seconds pass, or until every file source ends."""
        if self.analyzer:
            self.analyzer.start()
        for w in self.workers:
            w.start()
        deadline = time.monotonic() + duration if duration else None
        while not all(w.done for w in self.workers):
            if self.stop.wait(0.5) or deadline and time.monotonic() >= deadline:
                break
        self.stop.set()
        # Let every camera finish: close its open visits, write their thumbnails and clips, and
        # store them. Only then close the store; a visit still unsaved at the deadline is named.
        finish_by = time.monotonic() + SHUTDOWN_SECONDS
        while not all(w.done for w in self.workers) and time.monotonic() < finish_by:
            time.sleep(0.5)
        for w in self.workers:
            w.join(timeout=5)
            for visit in w.pending_visits:
                log.warning("%s: shutting down before the %s visit from %s was saved", w.camera.name, visit.key,
                            datetime.fromtimestamp(visit.start, timezone.utc).isoformat(timespec="seconds"))
        if self.analyzer:
            self.analyzer.thread.join(timeout=30)
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
    parser.add_argument("--detector", choices=("auto", "speciesnet", "none"), default="auto",
                        help="what identifies animals in motion: SpeciesNet if installed (auto, the default),"
                             " or none (motion and clips only)")
    parser.add_argument("--reset-view", action="append", default=[], metavar="NAME",
                        help="after re-aiming a camera on purpose: forget its reference view and learned"
                             " scenery, then exit (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cameras = load_cameras(args.config)
    except ConfigError as e:
        print(f"camera-monitor: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    if args.reset_view:
        for name in args.reset_view:
            cam = find_camera(cameras, name)
            if cam is None:
                print(f"camera-monitor: unknown camera {name}", file=sys.stderr)
                raise SystemExit(2)
            removed = ViewCheck(args.data_dir / "views", cam.name).reset()
            spots = Scenery(args.data_dir / "scenery" / f"{cam.name}.json").reset()
            print(f"{cam.name}: {'reference view forgotten' if removed else 'no reference view yet'};"
                  f" a new one is taken on the next check. Learned scenery forgotten ({spots} spot(s)).")
        return
    chosen = [c for c in cameras.values() if c.enabled]
    if args.camera:
        found = {name: find_camera(cameras, name) for name in args.camera}
        if missing := [n for n, c in found.items() if c is None]:
            print(f"camera-monitor: unknown camera {', '.join(missing)}", file=sys.stderr)
            raise SystemExit(2)
        chosen = list({c.name: c for c in found.values()}.values())
    detector = None
    if args.detector != "none":
        if available():
            detector = SpeciesNetDetector(load_site(args.config))
        elif args.detector == "speciesnet":
            print("camera-monitor: SpeciesNet isn't installed: pip install 'camera-mcp[detect]'", file=sys.stderr)
            raise SystemExit(2)
        else:
            log.info("SpeciesNet isn't installed: watching for motion only")
    monitor = Monitor(chosen, args.data_dir, detector, tz=load_site(args.config).tz())
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: monitor.stop.set())
    log.info("watching %s; data in %s", ", ".join(c.name for c in chosen), args.data_dir)
    monitor.run(args.duration)


if __name__ == "__main__":
    main()
