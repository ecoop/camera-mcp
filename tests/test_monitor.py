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
from camera_mcp.monitor import Monitor
from camera_mcp.recorder import stop_stale_recorder
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
    from camera_mcp.zones import load_zone_overlay
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
    a._thinned["east_fence"] = 3
    a._infer_seconds.update(east_fence=1.5, backyard=0.5)
    a._max_lag.update(east_fence=42.0, backyard=5.0)
    a._flush_stats(force=True)
    db = sqlite3.connect(tmp_path / "monitor.db")
    agg = db.execute("SELECT queue_depth, max_lag, analyzed, dropped, infer_seconds, thinned"
                     " FROM analysis_stats").fetchone()
    assert agg == (0, 42.0, 14, 2, 2.0, 3)  # the aggregate is the sum (max for lag) across cameras
    percam = {r[0]: (r[1], r[2], r[3], r[4]) for r in
              db.execute("SELECT camera, analyzed, dropped, max_lag, thinned FROM analysis_camera_stats")}
    assert percam == {"east_fence": (10, 2, 42.0, 3), "backyard": (4, 0, 5.0, 0)}
    assert not (a._analyzed or a._dropped or a._thinned or a._max_lag or a._infer_seconds)  # all reset after flush


def test_analysis_behind_the_backlog_is_counted_as_thinned(tmp_path):
    # Past BACKLOG queued segments, each segment gets one frame; the stats sample and
    # analysis_health say so (they used to reach only the log).
    from camera_mcp.config import Site
    from camera_mcp.history import History
    from camera_mcp.monitor import Analyzer
    from camera_mcp.ringbuffer import Segment
    from camera_mcp.store import Store
    store = Store(tmp_path / "monitor.db")
    a = Analyzer(detector=None, stop=threading.Event(), store=store)
    camera = SimpleNamespace(name="east_fence", tags=(), roi=None, settings_at=lambda now: MonitorSettings())
    worker = SimpleNamespace(camera=camera, tz=None, frame_size=lambda seg: (160, 90))
    seg = Segment(tmp_path / "missing.mp4", time.time() - 4, time.time() - 2)  # decodes to no frames
    depth = [a.BACKLOG]
    a.sched.qsize = lambda: depth[0]
    a._analyze(worker, seg)  # at the backlog: every frame
    depth[0] = a.BACKLOG + 1
    a._analyze(worker, seg)  # past it: one frame per segment
    a._analyze(worker, seg)
    a._last_stats = time.time() - 61
    a._flush_stats(force=True)
    store.close()
    db = sqlite3.connect(tmp_path / "monitor.db")
    assert db.execute("SELECT thinned FROM analysis_stats").fetchone() == (2,)
    assert db.execute("SELECT camera, thinned FROM analysis_camera_stats").fetchall() == [("east_fence", 2)]
    history = History(tmp_path / "monitor.db", {"east_fence": Camera("east_fence", "x")}, Site())
    assert history.analysis_health(since="1h")["segments_thinned"] == 2
    assert [p["thinned"] for p in history.analysis_series(since="1h")["points"]] == [2]


def test_a_log_from_before_thinning_reads_as_none_thinned(tmp_path):
    # The server opens the log read-only and never adds columns: until the monitor restarts on
    # this build, the stats tables have no `thinned`, and every answer reads it as 0.
    from camera_mcp.config import Site
    from camera_mcp.history import History
    from camera_mcp.store import Store
    store = Store(tmp_path / "monitor.db")
    store.add_analysis_stats(time.time() - 60, 1, 5.0, 10, 0, 2.0)
    store.add_camera_stats(time.time() - 60, [("east_fence", 1, 5.0, 10, 0, 2.0, 0)])
    store.close()
    db = sqlite3.connect(tmp_path / "monitor.db")
    db.execute("ALTER TABLE analysis_stats DROP COLUMN thinned")
    db.execute("ALTER TABLE analysis_camera_stats DROP COLUMN thinned")
    db.commit()
    db.close()
    history = History(tmp_path / "monitor.db", {"east_fence": Camera("east_fence", "x")}, Site())
    assert history.analysis_health(since="1h")["segments_thinned"] == 0
    assert [p["thinned"] for p in history.analysis_series(since="1h")["points"]] == [0]
    assert [p["thinned"] for p in history.analysis_series(since="1h", camera="east_fence")["points"]] == [0]
    assert [p["thinned"] for p in history.analysis_series_all(since="1h")["cameras"]["east_fence"]] == [0]


def test_alerter_fires_once_per_cooldown_for_alert_species(monkeypatch):
    from zoneinfo import ZoneInfo
    import camera_mcp.mailer as mailer
    from camera_mcp.alerts import Alerter
    sent = []
    monkeypatch.setattr(mailer, "send_email", lambda subj, body, settings, *a, **k: sent.append((body, settings["SMTP_TO"])))
    a = Alerter({"coyote"}, {"SMTP_HOST": "h", "SMTP_FROM": "f", "SMTP_TO": "brief@x"},
                "15551234567@txt.att.net", ZoneInfo("America/Los_Angeles"))
    assert a.active
    a.maybe_alert("backyard", "cat", 1_700_000_000.0)      # not an alert species -> nothing
    a.maybe_alert("backyard", "coyote", 1_700_000_000.0)   # fires
    a.maybe_alert("backyard", "coyote", 1_700_000_100.0)   # within cooldown -> suppressed
    assert len(sent) == 1
    assert sent[0][1] == "15551234567@txt.att.net" and sent[0][0].startswith("coyote at backyard")
    assert not Alerter({"coyote"}, {"SMTP_TO": "x"}, None, ZoneInfo("UTC")).active  # no recipient -> inactive
    a = Alerter(set(), {"SMTP_TO": "x"}, "r", ZoneInfo("UTC"))  # no species: active for alert areas only
    a.maybe_alert("backyard", "coyote", 1_700_000_000.0)
    assert len(sent) == 1


def test_alerter_texts_once_per_cooldown_per_alert_area(monkeypatch):
    from zoneinfo import ZoneInfo
    import camera_mcp.mailer as mailer
    from camera_mcp.alerts import Alerter
    sent = []
    monkeypatch.setattr(mailer, "send_email", lambda subj, body, settings, *a, **k: sent.append(subj))
    a = Alerter({"cat"}, {"SMTP_HOST": "h"}, "r", ZoneInfo("America/Los_Angeles"))
    a.maybe_area_alert("west_alley", "cat on fence", "bird", 1_700_000_000.0)    # fires, whatever the label
    a.maybe_area_alert("west_alley", "cat on fence", "animal", 1_700_000_060.0)  # same area, within cooldown
    a.maybe_area_alert("west_alley", "branch", "animal", 1_700_000_060.0)        # another area has its own
    a.maybe_alert("west_alley", "cat", 1_700_000_060.0)                          # so does the species alert
    assert sent == ["cat on fence: bird at west_alley, 2:13pm", "branch: animal at west_alley, 2:14pm",
                    "cat at west_alley, 2:14pm"]
    a._last = {k: v - Alerter.COOLDOWN for k, v in a._last.items()}  # the cooldown passes
    a.maybe_area_alert("west_alley", "cat on fence", "domestic cat", 1_700_000_400.0)
    assert sent[-1].startswith("cat on fence: domestic cat")


def test_alert_email_carries_a_close_up_the_full_frame_and_a_link_to_the_visit(monkeypatch):
    from zoneinfo import ZoneInfo
    import camera_mcp.mailer as mailer
    from camera_mcp.alerts import Alerter
    sent, made = [], []
    monkeypatch.setattr(mailer, "send_email", lambda subj, body, settings, image=None, **k: sent.append(
        (subj, body, image, k["link"], k["attachments"])))

    def photo():
        made.append(1)
        return b"close", b"full"
    a = Alerter(set(), {"SMTP_HOST": "h"}, "r", ZoneInfo("America/Los_Angeles"), "http://studio:8080")
    a.maybe_area_alert("west_alley", "cat on fence", "bird", 1_700_000_000.0, photo=photo, visit=15779)
    a.maybe_area_alert("west_alley", "cat on fence", "bird", 1_700_000_060.0, photo=photo, visit=15779)
    assert made == [1]  # pictures are made only for an alert that's sent, not one the cooldown holds back
    subj, body, image, link, attachments = sent[0]
    assert subj == "cat on fence: bird at west_alley, 2:13pm" and body == subj + " (visit 15779)"
    assert image == b"close" and link == "http://studio:8080/visit/15779/image"
    assert attachments == (("west_alley-2023-11-14-141320.jpg", b"full"),)

    def broken():
        raise RuntimeError("no frame")
    b = Alerter(set(), {"SMTP_HOST": "h"}, "r", ZoneInfo("UTC"))  # no dashboard address: no link
    b.maybe_area_alert("west_alley", "cat on fence", "bird", 1_700_000_000.0, photo=broken, visit=7)
    assert sent[-1][2:] == (None, None, ())  # a failed picture still sends the alert


def test_alert_photo_crops_a_close_up_around_the_box_at_native_resolution():
    import numpy as np
    from camera_mcp.monitor import alert_photo
    full = np.zeros((1440, 2560, 3), np.uint8)

    def size(jpeg):
        import cv2
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    close, whole = alert_photo(full, (0.10, 0.50, 0.05, 0.05))  # a 128x72 box: 640x480 around it
    assert size(close) == (480, 640) and size(whole) == (1440, 2560)
    close, _ = alert_photo(full, (0.97, 0.97, 0.03, 0.03))  # at the corner: kept inside the frame
    assert size(close) == (480, 640)
    close, _ = alert_photo(full, (0.0, 0.0, 0.5, 0.5))  # a big box: the close-up is at most the frame
    assert size(close) == (1440, 2560)
    assert alert_photo(full, None)[0] is None  # no box: just the frame


def test_alerts_link_to_the_visit_the_detection_belongs_to():
    from types import SimpleNamespace
    import numpy as np
    from camera_mcp.config import Camera
    from camera_mcp.monitor import Analyzer
    from camera_mcp.zones import AlertArea
    calls = []

    class Fake:
        def maybe_area_alert(self, camera, area, label, t, photo=None, visit=None):
            calls.append(("area", area, label, visit, photo()[0] is not None))

        def maybe_alert(self, camera, species, t, photo=None, visit=None):
            calls.append(("species", species, visit, photo()[0] is not None))
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.alerter = Fake()
    cam = Camera("west_alley", "rtsp://x", roi=(0.08, 0.13, 0.48, 0.43),
                 alert=(AlertArea((0.0, 0.0, 1.0, 1.0), "yard", 0.0),))
    still_open = {"cat": SimpleNamespace(db_id=42), "coyote": SimpleNamespace(db_id=43)}
    analyzer._alert(cam, 1.0, np.zeros((360, 640, 3), np.uint8), [_det(0.4, 0.4, 0.1, 0.1, "domestic cat", "cat")],
                    still_open)
    assert calls == [("area", "yard", "domestic cat", 42, True),
                     ("species", "cat", 42, True), ("species", "coyote", 43, False)]  # coyote: no box this frame


def _det(x, y, w, h, name="animal", key=None):
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection
    return Detection(Box(x, y, w, h, 0.8), Label(name, "", 0.8), key or name)


def test_alert_areas_trust_place_and_size_not_the_label():
    from camera_mcp.config import Camera
    from camera_mcp.zones import AlertArea, alert_hits
    fence = AlertArea((0.1, 0.45, 0.2, 0.12), "cat on fence", 0.06, ("squirrel", "rodent"))
    cam = Camera("west_alley", "rtsp://x", alert=(fence,))
    hits = alert_hits(cam, [
        _det(0.15, 0.47, 0.08, 0.06, "bird", "songbirds"),   # a cat called a bird: alerts
        _det(0.15, 0.47, 0.08, 0.06, "person"),               # ... or a person: alerts
        _det(0.15, 0.47, 0.03, 0.02, "bird", "songbirds"),   # songbird-sized: no
        _det(0.15, 0.47, 0.08, 0.06, "eastern gray squirrel", "squirrel"),  # skipped by group
        _det(0.15, 0.47, 0.08, 0.06, "rodent"),               # skipped by label
        _det(0.60, 0.20, 0.10, 0.10, "domestic cat", "cat"),  # outside the area: no
    ])
    assert [d.label.name for _, d in hits] == ["bird", "person"]
    assert alert_hits(Camera("b", "rtsp://x"), [_det(0.15, 0.47, 0.08, 0.06)]) == []  # opt-in: none drawn
    # Boxes are fractions of the roi crop; the area is in full-frame fractions.
    cropped = Camera("west_alley", "rtsp://x", roi=(0.08, 0.13, 0.48, 0.43), alert=(fence,))
    centre_x, centre_y = (0.20 - 0.08) / 0.48, (0.51 - 0.13) / 0.43  # full frame (0.20, 0.51): on the fence
    assert len(alert_hits(cropped, [_det(centre_x - 0.08, centre_y - 0.07, 0.16, 0.14)])) == 1  # 0.077 of the frame
    assert alert_hits(cropped, [_det(centre_x - 0.04, centre_y - 0.03, 0.08, 0.06)]) == []      # 0.038: too small


def test_alert_areas_load_from_the_zone_overlay(tmp_path):
    import json
    from camera_mcp.zones import AlertArea, alert_json
    cfg = write(tmp_path, "cameras: [{name: a, source: a.mp4}]")
    zones = tmp_path / "zones.json"
    area = {"poly": [[0.1, 0.4], [0.3, 0.4], [0.3, 0.6]], "name": "cat on fence", "min_size": 0.06,
            "skip": ["Squirrel", "rodent"]}
    zones.write_text(json.dumps({"cameras": {"a": {"alert": [area]}}}))
    (got,) = load_cameras(cfg, zones=zones)["a"].alert
    assert got == AlertArea(((0.1, 0.4), (0.3, 0.4), (0.3, 0.6)), "cat on fence", 0.06, ("squirrel", "rodent"))
    assert alert_json(got) == {**area, "skip": ["squirrel", "rodent"]}
    assert load_cameras(cfg)["a"].alert == ()
    zones.write_text(json.dumps({"cameras": {"a": {"alert": [{**area, "min_size": 3}]}}}))
    with pytest.raises(ConfigError):
        load_cameras(cfg, zones=zones)


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
    Monitor([camera], data).run(duration=60)  # a file source: returns when the file has played

    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    windows = db.execute("SELECT * FROM motion_windows").fetchall()
    assert len(windows) == 1
    w = windows[0]
    # Time zero is the first frame the watcher received (the uptime span's start), not when the
    # test began: probing the file and starting ffmpeg take a variable few seconds on a loaded CI
    # runner, which once pushed a correct start past a wall-clock bound. From the first frame on,
    # `-re` paces delivery at the file's own rate, so the box's 5 s offset holds.
    ((first_frame,),) = db.execute("SELECT started_at FROM camera_uptime WHERE camera = 'demo'").fetchall()
    zero = datetime.fromisoformat(first_frame).timestamp()
    start, end = (datetime.fromisoformat(w[k]).timestamp() - zero for k in ("started_at", "ended_at"))
    assert 4.5 <= start <= 6.5, start  # the box appears 5 s into the file (pre-roll would make it 4)
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
