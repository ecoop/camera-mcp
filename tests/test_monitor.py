# Copyright (c) 2026 Eric Cooper. All rights reserved.
"""Phase 2a: config additions, motion detection, the ring buffer, and the monitor end to end
on a generated clip (still, then a moving box, then still). No network or hardware."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime

import numpy as np
import pytest

from camera_mcp.config import Camera, ConfigError, MonitorSettings, find_camera, load_cameras
from camera_mcp.monitor import Monitor
from camera_mcp.motion import MotionDetector, MotionWindows
from camera_mcp.probe import probe_stream
from camera_mcp.ringbuffer import covering, list_segments, prune

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")


def write(tmp_path, text):
    path = tmp_path / "cameras.yaml"
    path.write_text(text)
    return path


# ---- config ---------------------------------------------------------------------------

def test_monitor_settings_site_wide_and_per_camera(tmp_path):
    cams = load_cameras(write(tmp_path, """
monitor: {pre_seconds: 3, quiet_seconds: 4}
cameras:
  - {name: a, source: a.mp4}
  - {name: b, source: b.mp4, pre_seconds: 8, tags: [feeder, dog_yard]}
"""))
    assert cams["a"].monitor == MonitorSettings(pre_seconds=3, quiet_seconds=4)
    assert cams["b"].monitor.pre_seconds == 8 and cams["b"].monitor.quiet_seconds == 4
    assert cams["b"].tags == ("feeder", "dog_yard") and cams["a"].tags == ()


@pytest.mark.parametrize("text, message", [
    ("monitor: {pre_secs: 3}\ncameras: [{name: a, source: a.mp4}]", "unknown setting pre_secs"),
    ("cameras: [{name: a, source: a.mp4, post_seconds: soon}]", "must be a number"),
    ("cameras: [{name: a, source: a.mp4, segment_seconds: 1}]", "at least 2"),
    ("cameras: [{name: a, source: a.mp4, tags: [1, 2]}]", "list of strings"),
])
def test_bad_monitor_settings(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_cameras(write(tmp_path, text))


def test_names_match_loosely(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - {name: west_alley, source: a.mp4}
  - {name: backyard, source: b.mp4, aliases: [garden feeders]}
"""))
    assert find_camera(cams, "West Alley").name == "west_alley"
    assert find_camera(cams, "west-alley").name == "west_alley"
    assert find_camera(cams, "Garden Feeders").name == "backyard"
    assert find_camera(cams, "front door") is None


@pytest.mark.parametrize("cameras", [
    "[{name: west_alley, source: a.mp4}, {name: West Alley, source: b.mp4}]",
    "[{name: a, source: a.mp4}, {name: a, source: b.mp4}]",
    "[{name: a, source: a.mp4}, {name: b, source: b.mp4, aliases: [A]}]",
])
def test_names_that_clash_loosely_are_rejected(tmp_path, cameras):
    with pytest.raises(ConfigError, match="clashes"):
        load_cameras(write(tmp_path, f"cameras: {cameras}\n"))


# ---- motion ---------------------------------------------------------------------------

def test_motion_detector_ignores_a_still_scene_and_sees_a_change():
    detector = MotionDetector(MonitorSettings())
    still = np.full((180, 320), 100, np.uint8)
    assert detector.update(still) == 0.0  # first frame: becomes the background
    assert detector.update(still) == 0.0
    moved = still.copy()
    moved[60:100, 140:180] = 220  # a 40x40 patch: about 2.8% of the frame
    changed = detector.update(moved)
    assert 0.02 < changed < 0.04 and detector.is_motion(changed)


def test_motion_windows_open_and_close_after_quiet():
    windows = MotionWindows(quiet_seconds=2)
    assert windows.feed(0.0, 0.0, False) is None and windows.open is None
    windows.feed(1.0, 0.05, True)
    windows.feed(1.5, 0.10, True)
    assert windows.feed(3.0, 0.0, False) is None  # 1.5 s quiet: still open
    closed = windows.feed(3.6, 0.0, False)  # 2.1 s quiet: closes
    assert (closed.start, closed.last_motion, closed.peak, closed.frames) == (1.0, 1.5, 0.10, 2)
    assert windows.open is None


# ---- ring buffer ----------------------------------------------------------------------

def test_segments_are_listed_selected_and_pruned(tmp_path):
    base = 1_800_000_000  # 2027-01-15T08:00:00Z
    for i in range(5):  # five 2-second segments
        start = base + 2 * i
        path = tmp_path / time.strftime("%Y%m%dT%H%M%SZ.ts", time.gmtime(start))
        path.write_bytes(b"x")
        os.utime(path, (start + 2, start + 2))
    tmp_path.joinpath("notes.ts").write_bytes(b"x")  # not a segment name: ignored
    segments = list_segments(tmp_path)
    assert [s.start - base for s in segments] == [0, 2, 4, 6, 8]
    assert [s.start - base for s in covering(segments, base + 3, base + 5)] == [2, 4]
    assert prune(tmp_path, keep_from=base + 7) == 3
    assert [s.start - base for s in list_segments(tmp_path)] == [6, 8]
    assert prune(tmp_path, keep_from=base + 100) == 1  # the newest is never deleted
    assert len(list_segments(tmp_path)) == 1


# ---- the monitor, end to end ----------------------------------------------------------

@pytest.fixture
def still_move_still(tmp_path):
    """14 s at 10 fps, 640x360: gray, a box moving across from 5 s to 8 s, then gray again.
    A keyframe every second, like a camera's stream."""
    path = tmp_path / "still_move_still.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=10:d=14",
         "-f", "lavfi", "-i", "color=c=white:s=60x60:r=10:d=14",
         "-filter_complex", "[0][1]overlay=x='100+120*(t-5)':y=150:enable='between(t,5,8)'",
         "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_monitor_records_one_window_and_a_clip(tmp_path, still_move_still):
    settings = MonitorSettings(pre_seconds=1, post_seconds=1, quiet_seconds=2)
    camera = Camera("demo", str(still_move_still), monitor=settings)
    data = tmp_path / "data"
    started = time.time()
    Monitor([camera], data).run(duration=60)  # a file source: returns when the file has played

    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    windows = db.execute("SELECT * FROM motion_windows").fetchall()
    assert len(windows) == 1
    w = windows[0]
    start, end = (datetime.fromisoformat(w[k]).timestamp() - started for k in ("started_at", "ended_at"))
    assert 3.5 <= start <= 7.0, start  # the box appears 5 s in (reader start-up adds a little)
    assert 1.0 <= end - start <= 5.0, (start, end)  # ...and moves for 3 s
    clip = probe_stream(Camera("clip", str(data / w["clip"])))
    assert (clip.width, clip.height) == (640, 360)
    status = db.execute("SELECT state, error FROM camera_status WHERE camera = 'demo'").fetchone()
    assert tuple(status) == ("stopped", None)
