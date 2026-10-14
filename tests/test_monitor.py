# Copyright (c) 2026 Eric Cooper.
"""Phase 2a: config additions, motion detection, the ring buffer, and the monitor end to end
on a generated clip (still, then a moving box, then still). No network or hardware."""

from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from camera_mcp.config import (Camera, ConfigError, MonitorSettings, find_camera, load_cameras,
                               parse_active_window)
from camera_mcp.monitor import Monitor, stop_stale_recorder
from camera_mcp.motion import MotionDetector, MotionWindows
from camera_mcp.probe import probe_stream
from camera_mcp.ringbuffer import covering, list_segments, prune
from camera_mcp.view import ViewCheck, edges, is_colour, similarity

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


def _still_then_patch(detector):
    """Prime `detector` on a still frame, then feed a frame with a 40x40 bright patch; return changed."""
    still = np.full((180, 320), 100, np.uint8)
    detector.update(still)
    detector.update(still)
    moved = still.copy()
    moved[60:100, 140:180] = 220  # the same ~2.8% patch as the test above
    return detector.update(moved)


def test_motion_mask_ignores_change_in_a_dead_region():
    # An ignore rect covering the patch -> that change never counts.
    masked = MotionDetector(MonitorSettings(), ignore=((0.4, 0.3, 0.2, 0.3),))
    changed = _still_then_patch(masked)
    assert changed == 0.0 and not masked.is_motion(changed)


def test_motion_roi_counts_only_inside_it():
    # The patch is on the right; an ROI on the left half excludes it.
    outside = MotionDetector(MonitorSettings(), roi=(0.0, 0.0, 0.4, 1.0))
    assert not outside.is_motion(_still_then_patch(outside))
    # The same patch inside the ROI still counts.
    inside = MotionDetector(MonitorSettings(), roi=(0.3, 0.2, 0.5, 0.6))
    assert inside.is_motion(_still_then_patch(inside))


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

def _at(hhmm):
    from datetime import datetime
    h, m = map(int, hhmm.split(":"))
    return datetime(2026, 10, 10, h, m)


@pytest.mark.parametrize("spec, expected", [
    ("", None),
    ("07:00-19:00", (420, 1140)),
    (" 19:00 - 07:00 ", (1140, 420)),
])
def test_parse_active_window(spec, expected):
    assert parse_active_window(spec) == expected


@pytest.mark.parametrize("spec", ["19:00", "9-5", "24:00-07:00", "07:60-08:00", "08:00-08:00"])
def test_bad_active_window_is_rejected(spec):
    with pytest.raises(ConfigError):
        parse_active_window(spec)


def test_is_active_daytime_window():
    cam = Camera("deck", "d.mp4", active="07:00-19:00")  # watched by day only
    assert cam.is_active(_at("07:00")) and cam.is_active(_at("12:00"))
    assert not cam.is_active(_at("06:59")) and not cam.is_active(_at("19:00"))


def test_is_active_overnight_window_wraps_midnight():
    cam = Camera("deck", "d.mp4", active="19:00-07:00")  # the deck cameras: night only
    assert cam.is_active(_at("23:00")) and cam.is_active(_at("02:00")) and cam.is_active(_at("06:59"))
    assert not cam.is_active(_at("07:00")) and not cam.is_active(_at("12:00"))


def test_no_window_is_always_active():
    assert Camera("a", "a.mp4").is_active(_at("03:00")) is True


def test_active_window_loaded_and_validated(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - {name: backyard, source: b.mp4}
  - {name: east_deck, source: d.mp4, active: "19:00-07:00"}
"""))
    assert cams["backyard"].active == "" and cams["backyard"].is_active(_at("12:00"))
    assert cams["east_deck"].active == "19:00-07:00"
    assert not cams["east_deck"].is_active(_at("12:00"))
    with pytest.raises(ConfigError, match="active window"):
        load_cameras(write(tmp_path, 'cameras: [{name: a, source: a.mp4, active: "9-5"}]'))


def test_motion_regions_loaded_and_validated(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - {name: a, source: a.mp4, motion_mask: [[0.0, 0.0, 0.2, 0.3]], motion_roi: [0.1, 0.1, 0.8, 0.8]}
  - {name: b, source: b.mp4}
"""))
    assert cams["a"].motion_mask == ((0.0, 0.0, 0.2, 0.3),) and cams["a"].motion_roi == (0.1, 0.1, 0.8, 0.8)
    assert cams["b"].motion_mask == () and cams["b"].motion_roi is None


@pytest.mark.parametrize("snippet, msg", [
    ("motion_mask: [[0.5, 0.5, 0.6, 0.1]]", "out of bounds"),  # x+w > 1
    ("motion_mask: [[0.0, 0.0, 0.1]]", "fractions"),           # not four numbers
    ("motion_roi: [0.0, 0.0, 0.0, 0.5]", "out of bounds"),     # zero width
])
def test_bad_motion_regions_rejected(tmp_path, snippet, msg):
    with pytest.raises(ConfigError, match=msg):
        load_cameras(write(tmp_path, f"cameras: [{{name: a, source: a.mp4, {snippet}}}]"))


def test_schedule_overrides_by_time_of_day(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - name: backyard
    source: b.mp4
    tags: [wide_view]
    tile_rows: 2
    tile_cols: 2
    when:
      - { hours: "22:00-06:00", tile_rows: 1, tile_cols: 1, detect_every_seconds: 2 }
  - name: fence
    source: f.mp4
"""))
    b = cams["backyard"]
    day = b.settings_at(_at("12:00"))
    assert (day.tile_rows, day.tile_cols, day.detect_every_seconds) == (2, 2, 1.0)
    for t in ("23:00", "03:00", "05:59"):  # the window wraps midnight
        night = b.settings_at(_at(t))
        assert (night.tile_rows, night.tile_cols, night.detect_every_seconds) == (1, 1, 2.0)
    assert cams["fence"].schedule == () and cams["fence"].settings_at(_at("03:00")) == cams["fence"].monitor


def test_priority_parses_and_schedules(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - name: alley
    source: a.mp4
    priority: 5
    when:
      - { hours: "22:00-06:00", priority: 9 }
  - name: yard
    source: y.mp4
"""))
    alley = cams["alley"]
    assert alley.settings_at(_at("12:00")).priority == 5      # base
    assert alley.settings_at(_at("23:30")).priority == 9      # night boost via when:
    assert cams["yard"].settings_at(_at("12:00")).priority == 0  # default


@pytest.mark.parametrize("when, msg", [
    ("[{tile_rows: 1}]", "needs an 'hours'"),
    ('[{hours: "9-5", tile_rows: 1}]', "HH:MM"),
    ('[{hours: "22:00-06:00", max_boxes: 2}]', "cannot override"),
    ('[{hours: "22:00-06:00"}]', "no overrides"),
])
def test_bad_schedule_rejected(tmp_path, when, msg):
    with pytest.raises(ConfigError, match=msg):
        load_cameras(write(tmp_path, f"cameras: [{{name: a, source: a.mp4, when: {when}}}]"))


def test_tile_windows_count():
    from camera_mcp.detect import tile_windows
    assert len(tile_windows(1, 1)) == 1 and len(tile_windows(2, 2)) == 4 and len(tile_windows(1, 2)) == 2


# ---- analyzer work scheduler (Phase B: priority order under backlog) ------------------

class _StubWorker:
    """Minimal stand-in for CameraWorker in _Scheduler tests: a name and a live priority."""
    def __init__(self, name, priority=0):
        self.camera = SimpleNamespace(name=name)
        self._priority = priority

    def priority_now(self):
        return self._priority


def test_scheduler_preserves_per_camera_order():
    from camera_mcp.monitor import _Scheduler
    s = _Scheduler()
    w = _StubWorker("a")
    for i in range(3):
        s.put(w, i)
    assert [s.get(0.1)[1] for _ in range(3)] == [0, 1, 2]  # FIFO within a camera
    assert s.get(0.01) is None  # drained


def test_scheduler_drains_higher_priority_first():
    from camera_mcp.monitor import _Scheduler
    s = _Scheduler()
    lo, hi = _StubWorker("lo", 0), _StubWorker("hi", 5)
    for w, tag in [(lo, "lo0"), (hi, "hi0"), (lo, "lo1"), (hi, "hi1")]:
        s.put(w, tag)
    # the high-priority camera's whole backlog first, each camera still in its own order
    assert [s.get(0.1)[1] for _ in range(4)] == ["hi0", "hi1", "lo0", "lo1"]


def test_scheduler_round_robins_equal_priority():
    from camera_mcp.monitor import _Scheduler
    s = _Scheduler()
    a, b = _StubWorker("a", 1), _StubWorker("b", 1)
    for w, tag in [(a, "a0"), (a, "a1"), (b, "b0"), (b, "b1")]:
        s.put(w, tag)
    # equal priority: least-recently-served next, so the two cameras alternate
    assert [s.get(0.1)[1] for _ in range(4)] == ["a0", "b0", "a1", "b1"]


def test_scheduler_get_wakes_on_put():
    from camera_mcp.monitor import _Scheduler
    s = _Scheduler()
    w = _StubWorker("a")
    threading.Timer(0.05, lambda: s.put(w, "x")).start()
    assert s.get(1.0)[1] == "x"  # blocked, then woke when the segment arrived


def test_roi_parses_and_rejects_bad(tmp_path):
    cams = load_cameras(write(tmp_path, "cameras: [{name: a, source: a.mp4, roi: [0.1, 0.1, 0.8, 0.8]}]"))
    assert cams["a"].roi == (0.1, 0.1, 0.8, 0.8)
    assert load_cameras(write(tmp_path, "cameras: [{name: b, source: b.mp4}]"))["b"].roi is None
    with pytest.raises(ConfigError, match="out of bounds"):
        load_cameras(write(tmp_path, "cameras: [{name: a, source: a.mp4, roi: [0.5, 0.0, 0.8, 0.5]}]"))


def test_zone_overlay_merges_and_overrides(tmp_path):
    import json
    from camera_mcp.config import load_zone_overlay
    cfg = write(tmp_path, "cameras:\n"
                "  - {name: a, source: a.mp4, roi: [0.1, 0.1, 0.5, 0.5]}\n"
                "  - {name: b, source: b.mp4}\n")
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps({"cameras": {
        "a": {"roi": None, "motion_mask": [[0.0, 0.0, 0.2, 0.2]]},   # clear the yaml roi, add a mask
        "b": {"roi": [0.3, 0.3, 0.4, 0.4], "motion_roi": [0.1, 0.1, 0.8, 0.8]},
        "ghost": {"roi": [0, 0, 1, 1]},                               # unknown camera: ignored
    }}))
    # Without the overlay, the yaml values stand.
    assert load_cameras(cfg)["a"].roi == (0.1, 0.1, 0.5, 0.5)
    # With it, overlay fields win; absent fields keep the yaml/default.
    cams = load_cameras(cfg, zones=zones)
    assert cams["a"].roi is None and cams["a"].motion_mask == ((0.0, 0.0, 0.2, 0.2),)
    assert cams["b"].roi == (0.3, 0.3, 0.4, 0.4) and cams["b"].motion_roi == (0.1, 0.1, 0.8, 0.8)
    assert "ghost" not in cams
    # A missing overlay file is simply no overrides.
    assert load_zone_overlay(tmp_path / "nope.json") == {}


def test_bad_zone_overlay_is_rejected(tmp_path):
    cfg = write(tmp_path, "cameras: [{name: a, source: a.mp4}]")
    zones = tmp_path / "zones.json"
    zones.write_text('{"cameras": {"a": {"roi": [0.5, 0.0, 0.8, 0.5]}}}')  # x+w > 1
    with pytest.raises(ConfigError, match="out of bounds"):
        load_cameras(cfg, zones=zones)
    zones.write_text("{not json")
    with pytest.raises(ConfigError, match="unreadable"):
        load_cameras(cfg, zones=zones)


def test_analysis_width_parses_and_schedules(tmp_path):
    cams = load_cameras(write(tmp_path, """
cameras:
  - name: a
    source: a.mp4
    analysis_width: 1280
    when:
      - { hours: "22:00-06:00", analysis_width: 960 }
"""))
    a = cams["a"]
    assert a.monitor.analysis_width == 1280
    assert a.settings_at(_at("12:00")).analysis_width == 1280
    assert a.settings_at(_at("23:00")).analysis_width == 960


def test_scaled_decode_frames_reshape(tmp_path):
    # _analyze decodes with an explicit even scale=W:H, so every frame is exactly W*H*3 bytes.
    from camera_mcp.capture import FFMPEG
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc=size=640x360:rate=5:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    out = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(clip),
                          "-vf", "fps=1/1,scale=160:90", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, timeout=30).stdout
    size = 160 * 90 * 3
    assert len(out) > 0 and len(out) % size == 0


def test_analyzer_persists_stats(tmp_path):
    import threading
    from camera_mcp.monitor import Analyzer
    from camera_mcp.store import Store
    store = Store(tmp_path / "monitor.db")
    a = Analyzer(detector=None, stop=threading.Event(), store=store)
    a._last_stats = time.time() - 61        # pretend an interval has elapsed
    a._analyzed.update(east_fence=10, backyard=4)
    a._dropped["east_fence"] = 2
    a._infer_seconds.update(east_fence=1.5, backyard=0.5)
    a._max_lag.update(east_fence=42.0, backyard=5.0)
    a._flush_stats(force=True)
    db = sqlite3.connect(tmp_path / "monitor.db")
    agg = db.execute("SELECT queue_depth, max_lag, analyzed, dropped, infer_seconds FROM analysis_stats").fetchone()
    assert agg == (0, 42.0, 14, 2, 2.0)  # the aggregate is the sum (max for lag) across cameras
    percam = {r[0]: (r[1], r[2], r[3]) for r in
              db.execute("SELECT camera, analyzed, dropped, max_lag FROM analysis_camera_stats")}
    assert percam == {"east_fence": (10, 2, 42.0), "backyard": (4, 0, 5.0)}
    assert not (a._analyzed or a._dropped or a._max_lag or a._infer_seconds)  # all reset after flush


def test_alerter_fires_once_per_cooldown_for_alert_species(monkeypatch):
    from zoneinfo import ZoneInfo
    import camera_mcp.mailer as mailer
    from camera_mcp.monitor import Alerter
    sent = []
    monkeypatch.setattr(mailer, "send_email", lambda subj, body, settings: sent.append((body, settings["SMTP_TO"])))
    a = Alerter({"coyote"}, {"SMTP_HOST": "h", "SMTP_FROM": "f", "SMTP_TO": "brief@x"},
                "15551234567@txt.att.net", ZoneInfo("America/Los_Angeles"))
    assert a.active
    a.maybe_alert("backyard", "cat", 1_700_000_000.0)      # not an alert species -> nothing
    a.maybe_alert("backyard", "coyote", 1_700_000_000.0)   # fires
    a.maybe_alert("backyard", "coyote", 1_700_000_100.0)   # within cooldown -> suppressed
    assert len(sent) == 1
    assert sent[0][1] == "15551234567@txt.att.net" and sent[0][0].startswith("coyote at backyard")
    assert not Alerter({"coyote"}, {"SMTP_TO": "x"}, None, ZoneInfo("UTC")).active  # no recipient -> inactive
    assert not Alerter(set(), {"SMTP_TO": "x"}, "r", ZoneInfo("UTC")).active        # no species -> inactive


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


@pytest.fixture
def endless(tmp_path):
    """A 120 s moving test pattern: for the length of a test, it behaves like a live camera."""
    path = tmp_path / "endless.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:d=120",
         "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _started(proc: subprocess.Popen, marker: str, timeout: float = 5.0) -> None:
    """Wait until `proc` shows `marker` in its command line (right after fork it still shows
    the parent's)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ps = subprocess.run(["ps", "-ww", "-p", str(proc.pid), "-o", "command="], capture_output=True, text=True)
        if marker in ps.stdout:
            return
        time.sleep(0.05)
    raise AssertionError(f"process {proc.pid} never showed {marker!r}")


def _children_reading(path) -> list[str]:
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [line for line in ps.splitlines() if str(path) in line and "ffmpeg" in line]


@needs_ffmpeg
def test_stopping_leaves_no_ffmpeg_running_and_saves_the_open_window(tmp_path, endless):
    """Regression: ffmpeg 7 can ignore a single SIGTERM, which left recorders running (and
    filling the disk) after the monitor exited, and lost the window open at shutdown."""
    camera = Camera("live", str(endless), monitor=MonitorSettings(pre_seconds=1, post_seconds=1))
    data = tmp_path / "data"
    started = time.monotonic()
    Monitor([camera], data).run(duration=4)  # stops while motion is ongoing
    assert time.monotonic() - started < 25
    assert _children_reading(endless) == []
    assert not (data / "buffer" / "live" / "recorder.pid").exists()
    db = sqlite3.connect(data / "monitor.db")
    assert db.execute("SELECT state FROM camera_status WHERE camera = 'live'").fetchone() == ("stopped",)
    (clip,) = db.execute("SELECT clip FROM motion_windows").fetchone()
    assert clip and probe_stream(Camera("clip", str(data / clip))).width == 640


@needs_ffmpeg
def test_a_recorder_left_by_a_dead_monitor_is_stopped_at_startup(tmp_path):
    seg_dir = tmp_path / "data" / "buffer" / "cam"
    seg_dir.mkdir(parents=True)
    # Stands in for an orphaned recorder: its command line names the buffer directory.
    stale = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(seg_dir)])
    (seg_dir / "recorder.pid").write_text(str(stale.pid))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _started(stale, str(seg_dir))
    _started(unrelated, "time.sleep(60)")
    try:
        assert stop_stale_recorder(seg_dir / "recorder.pid") is True
        assert stale.wait(5) is not None and not (seg_dir / "recorder.pid").exists()
        (seg_dir / "recorder.pid").write_text(str(unrelated.pid))  # a reused pid: not ours
        assert stop_stale_recorder(seg_dir / "recorder.pid") is False
        assert unrelated.poll() is None
    finally:
        for p in (stale, unrelated):
            p.kill()
            p.wait()


# ---- moved-camera check ---------------------------------------------------------------

def _scene(shift: int = 0, brightness: float = 1.0, seed: int = 1) -> np.ndarray:
    """A 320x180 stand-in for a camera view: fence boards and posts over a noisy ground."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(110, 12, (180, 320)), 0, 255).astype(np.uint8)
    for x in range(10, 320, 34):
        cv2.rectangle(img, (x, 20), (x + 26, 120), 175, -1)
        cv2.rectangle(img, (x, 20), (x + 26, 120), 60, 2)
    cv2.line(img, (0, 125), (319, 140), 40, 3)
    cv2.rectangle(img, (230, 60), (300, 170), 200, -1)
    if shift:
        img = np.roll(img, (shift, shift // 2), axis=(1, 0))
    return np.clip(img * brightness, 0, 255).astype(np.uint8)


def test_view_similarity_tells_a_moved_camera_from_changing_light():
    ref = edges(_scene())
    assert similarity(edges(_scene(seed=2)), ref) > 0.85  # same view, different noise
    assert similarity(edges(_scene(brightness=0.6)), ref) > 0.6  # 40% less light
    assert similarity(edges(_scene(shift=20)), ref) < 0.5  # tilted: everything 20 px over
    assert similarity(edges(np.full((180, 320), 30, np.uint8)), ref) is None  # too dark to judge


def test_view_check_flags_after_strikes_and_clears(tmp_path):
    check = ViewCheck(tmp_path, "cam", every=60, threshold=0.5, strikes=3)
    first = check.check(0, _scene(), colour=True)
    assert first.score is None and (tmp_path / "cam-day.png").exists()
    assert check.check(60, _scene(seed=3), colour=True).score > 0.85
    results = [check.check(t, _scene(shift=20), colour=True) for t in (120, 180, 240)]
    assert [r.moved for r in results] == [False, False, True] and results[-1].changed
    back = check.check(300, _scene(seed=4), colour=True)  # put back
    assert not back.moved and back.changed
    assert check.check(360, _scene(shift=20), colour=False).score is None  # night: its own reference
    assert check.reset() == 2 and not (tmp_path / "cam-day.png").exists()


def test_is_colour_tells_day_from_infrared():
    neutral = np.full((90, 160), 128, np.uint8)
    assert not is_colour(neutral, neutral)
    assert is_colour(neutral - 20, neutral + 25)


@needs_ffmpeg
def test_monitor_flags_a_camera_that_moved(tmp_path):
    """A still view for 6 s, then the same view shifted (a raccoon tilted the camera). The jump
    reads as motion for ~6 s while the motion background adapts; checks wait for stillness."""
    a, b, clip = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "tilted.mp4"
    cv2.imwrite(str(a), cv2.resize(_scene(), (640, 360)))
    cv2.imwrite(str(b), cv2.resize(_scene(shift=20), (640, 360)))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-loop", "1", "-t", "6", "-r", "10", "-i", str(a),
         "-loop", "1", "-t", "14", "-r", "10", "-i", str(b), "-filter_complex", "[0][1]concat=n=2,format=yuv420p",
         "-c:v", "libx264", "-g", "10", str(clip)],
        check=True,
    )
    settings = MonitorSettings(view_check_seconds=1, quiet_seconds=1, pre_seconds=0, post_seconds=0)
    data = tmp_path / "data"
    Monitor([Camera("fence", str(clip), monitor=settings)], data).run(duration=60)
    db = sqlite3.connect(data / "monitor.db")
    score, since = db.execute("SELECT view_score, view_moved_since FROM camera_status").fetchone()
    assert since is not None and score < 0.5, (score, since)


def test_one_small_bird_is_motion_but_scattered_noise_is_not():
    detector = MotionDetector(MonitorSettings())
    still = np.full((180, 320), 100, np.uint8)
    detector.update(still)
    detector.update(still)
    bird = still.copy()
    bird[90:95, 200:205] = 30  # a 5x5 dark patch: 0.04% of the frame, far under min_changed
    changed = detector.update(bird)
    assert changed < detector.min_changed and detector.is_motion(changed)

    detector = MotionDetector(MonitorSettings())
    detector.update(still)
    rng = np.random.default_rng(0)
    noisy = still.copy()
    ys, xs = rng.integers(0, 180, 50), rng.integers(0, 320, 50)
    noisy[ys, xs] = 255  # 50 lone hot pixels scattered across the frame
    changed = detector.update(noisy)
    assert not detector.is_motion(changed) and detector.largest_blob == 0.0


def test_watermark_labels_the_corner_without_resizing():
    from camera_mcp.monitor import watermark
    frame = np.full((180, 320, 3), 120, np.uint8)
    out = watermark(frame, "de")
    assert out.shape == frame.shape
    assert (out[:120, :] == 120).all()  # top untouched
    assert (out[150:, :40] != 120).any()  # bottom-left corner has the label
    assert (watermark(frame, "") == frame).all()  # empty label: no change


def test_camera_watermark_field_overrides_the_name():
    assert Camera("east_deck", "rtsp://h/stream1", watermark="de").short == "de"
    assert Camera("east_deck", "rtsp://h/stream1", aliases=("deck east",)).short == "east_deck"  # alias != watermark
