# Copyright (c) 2026 Eric Cooper.
"""The unnamed-visit gate: its rules, what the monitor stores, what history shows, the backtest."""

from __future__ import annotations

import threading
import time
from datetime import timezone
from types import SimpleNamespace

import pytest

from camera_mcp.config import Camera, MonitorSettings, Site, load_cameras
from camera_mcp.gate import GateRules, backtest, judge, main, travel
from camera_mcp.history import History
from camera_mcp.store import Store
from camera_mcp.taxa import Label
from camera_mcp.times import iso
from camera_mcp.visits import Box, Detection, Visit

NOW = time.time()
CAT = ("domestic cat", "mammalia;carnivora;felidae;felis;catus")
RULES = GateRules(min_conf=0.3, min_travel=0.5, travel_conf=0.5)


def swaying(n=6, conf=0.4):
    """A leaf: the same spot, jittering by a few percent of its size."""
    return [Box(0.40 + 0.005 * (i % 2), 0.60, 0.10, 0.10, conf) for i in range(n)]


def crossing(n=6, conf=0.4):
    """An animal walking across the frame."""
    return [Box(0.10 + 0.12 * i, 0.60, 0.10, 0.10, conf) for i in range(n)]


# ---- rules -------------------------------------------------------------------------------

def test_travel_is_spread_in_box_sizes():
    assert travel([Box(0.1, 0.1, 0.1, 0.1, 0.5)]) is None  # one sighting can't show movement
    assert travel(swaying()) == pytest.approx(0.05)
    assert travel(crossing()) == pytest.approx(6.0)


def test_low_confidence_unnamed_visits_are_suppressed():
    assert judge("animal", swaying(conf=0.25), RULES) == "low_conf"
    assert judge("animal", [Box(0.4, 0.6, 0.1, 0.1, 0.25)], RULES) == "low_conf"  # even a single frame


def test_stationary_unnamed_visits_are_suppressed_unless_confident():
    assert judge("animal", swaying(conf=0.4), RULES) == "stationary"
    assert judge("animal", swaying(conf=0.8), RULES) is None  # a confident box is kept, even if still
    assert judge("animal", crossing(conf=0.4), RULES) is None  # it moved: kept
    assert judge("animal", [Box(0.4, 0.6, 0.1, 0.1, 0.4)], RULES) is None  # one frame: can't judge travel


def test_named_visits_and_inactive_rules_are_never_suppressed():
    assert judge("cat", swaying(conf=0.21), RULES) is None
    assert judge("person", swaying(conf=0.21), RULES) is None
    assert judge("animal", swaying(conf=0.21), GateRules()) is None  # off by default
    assert not GateRules.of(MonitorSettings()).active


def test_gate_settings_parse_per_camera(tmp_path):
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("cameras:\n  - {name: fence, source: a.mp4, unnamed_min_conf: 0.3, unnamed_min_travel: 0.5}\n"
                   "  - {name: deck, source: b.mp4}\n")
    cams = load_cameras(cfg, require_login=False)
    assert GateRules.of(cams["fence"].monitor) == GateRules(0.3, 0.5, 0.5)
    assert not GateRules.of(cams["deck"].monitor).active


# ---- the monitor's close path -----------------------------------------------------------

def _visit(camera, key, boxes, label=("animal", ""), start=-3600.0):
    v = Visit(camera, key, start=NOW + start, last_seen=NOW + start + 10 * len(boxes), frames=len(boxes), max_count=1)
    v.votes[label] = 1.0
    v.log = [(NOW + start + 10 * i, [Detection(b, Label(*label, 0.9), key)]) for i, b in enumerate(boxes)]
    return v


def test_monitor_judges_a_closed_visit_with_the_cameras_rules(monkeypatch):
    from camera_mcp.monitor import CameraWorker
    cam = Camera("fence", "x", monitor=MonitorSettings(unnamed_min_conf=0.3, unnamed_min_travel=0.5))
    worker = SimpleNamespace(camera=cam, tz=timezone.utc, lock=threading.Lock(), pending_visits=[])
    leaf, fox = _visit("fence", "animal", swaying()), _visit("fence", "animal", crossing(conf=0.6))
    for v in (leaf, fox):
        CameraWorker._visit_closed(worker, v)
    assert (leaf.suppressed, fox.suppressed) == ("stationary", None)
    assert worker.pending_visits == [leaf, fox]  # suppressed visits are still stored, media and all


# ---- storage and history -----------------------------------------------------------------

@pytest.fixture
def log(tmp_path):
    data = tmp_path / "data"
    store = Store(data / "monitor.db")
    leaf = _visit("fence", "animal", swaying())
    leaf.suppressed = judge(leaf.key, [d.box for _, ds in leaf.log for d in ds], RULES)
    leaf_id = store.add_visit(leaf, model="m", thumbnail=None, clip=None, suppressed=leaf.suppressed)
    store.add_visit(_visit("fence", "animal", crossing()), model="m", thumbnail=None, clip=None)
    cat_id = store.add_visit(_visit("fence", "cat", swaying(conf=0.25), CAT), model="m", thumbnail=None, clip=None)
    store.set_visit_label(leaf_id, "false")
    store.close()
    cams = {"fence": Camera("fence", "x")}
    return History(data / "monitor.db", cams, Site(timezone="UTC")), data, leaf_id, cat_id


def test_history_hides_suppressed_visits_but_counts_them(log):
    history, _, leaf_id, _ = log
    r = history.find_visits(since="2h")
    assert (r["total"], r["hidden"]) == (2, 1)
    assert leaf_id not in [v["id"] for v in r["visits"]]
    every = history.find_visits(since="2h", include_suppressed=True)
    assert every["total"] == 3 and every["hidden"] == 0
    assert {v["id"]: v["suppressed"] for v in every["visits"]}[leaf_id] == "stationary"
    summary = history.activity_summary(since="2h", group_by="species")
    assert (summary["total_visits"], summary["hidden"]) == (2, 1)
    assert history.visits_timeline(since="2h")["visits"] == 2


def test_history_reads_a_log_from_before_the_gate(tmp_path):
    import sqlite3
    db = tmp_path / "monitor.db"
    Store(db).close()
    with sqlite3.connect(db) as c:  # an older monitor's log: no suppressed column
        c.execute("ALTER TABLE visits DROP COLUMN suppressed")
        c.execute("INSERT INTO visits (camera, started_at, ended_at, label, taxonomy, grp, score, frames, max_count,"
                  " model) VALUES ('fence', ?, ?, 'animal', '', 'animal', 1, 1, 1, 'm')", (iso(NOW - 60), iso(NOW)))
    history = History(db, {"fence": Camera("fence", "x")}, Site(timezone="UTC"))
    r = history.find_visits(since="1h")
    assert (r["total"], r["hidden"], r["visits"][0]["suppressed"]) == (1, 0, None)


def test_backtest_replays_the_rules_over_the_log(log, capsys):
    _, data, _, _ = log
    since = iso(NOW - 7200)
    out = backtest(data / "monitor.db", RULES, since)["fence"]
    assert (out["unnamed"], out["suppressed"], out["stationary"], out["low_conf"]) == (2, 1, 1, 0)
    assert (out["named"], out["named_hit"]) == (1, 1)  # the low-confidence cat: the rule would have lost it
    assert (out["labelled_false"], out["labelled_real"]) == (1, 0)
    main(["--data-dir", str(data), "--since", "2h", "--min-conf", "0.3", "--min-travel", "0.5"])
    printed = capsys.readouterr().out
    assert "fence" in printed and "min_conf=0.3" in printed


def test_backtest_needs_a_rule(log):
    _, data, _, _ = log
    with pytest.raises(SystemExit):
        main(["--data-dir", str(data)])


# ---- exclusion zones ------------------------------------------------------------------------

ROOF = (0.35, 0.40, 0.15, 0.15)  # a feeder roof, as a fraction of the full frame


def roof_box(conf=0.5):
    return Box(0.40, 0.45, 0.04, 0.04, conf)  # centred at (0.42, 0.47): on the roof


def test_a_visit_whose_boxes_all_sit_in_a_zone_is_hidden():
    from camera_mcp.gate import in_zones
    unrecognized, ignore = ((ROOF, "hide_unrecognized"),), ((ROOF, "ignore"),)
    assert in_zones("animal", [roof_box(), roof_box()], unrecognized) == "zone"
    assert in_zones("songbirds", [roof_box()], unrecognized) is None  # an identified bird on the roof stays
    assert in_zones("songbirds", [roof_box()], ignore) == "zone"
    assert in_zones("animal", [roof_box(), Box(0.8, 0.8, 0.05, 0.05, 0.5)], ignore) is None  # one box outside
    assert in_zones("animal", [roof_box()], ()) is None


def test_zones_are_full_frame_while_boxes_are_fractions_of_the_roi():
    from camera_mcp.gate import in_zones
    roi = (0.0, 0.25, 1.0, 0.75)  # boxes are fractions of this band
    on_roof = Box(0.40, (0.47 - 0.25) / 0.75 - 0.02, 0.04, 0.04, 0.5)  # maps back to (0.42, 0.47)
    assert in_zones("animal", [on_roof], ((ROOF, "hide_unrecognized"),), roi) == "zone"
    assert in_zones("animal", [roof_box()], ((ROOF, "hide_unrecognized"),), roi) is None  # unmapped, it's elsewhere


def test_exclusion_zones_parse_with_a_cautious_default_and_read_the_earlier_names(tmp_path):
    import json

    from camera_mcp.config import ConfigError, load_cameras
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("cameras:\n  - {name: backyard, source: b.mp4}\n")
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps({"cameras": {"backyard": {"exclude": [
        {"rect": list(ROOF)}, {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore"},
        {"rect": [0.2, 0.2, 0.1, 0.1], "mode": "all"}, {"rect": [0.3, 0.3, 0.1, 0.1], "mode": "unnamed"}]}}}))
    cam = load_cameras(cfg, require_login=False, zones=zones)["backyard"]
    assert cam.exclude == ((ROOF, "hide_unrecognized"), ((0.1, 0.1, 0.1, 0.1), "ignore"),
                           ((0.2, 0.2, 0.1, 0.1), "ignore"), ((0.3, 0.3, 0.1, 0.1), "hide_unrecognized"))  # old names read
    zones.write_text(json.dumps({"cameras": {"backyard": {"exclude": [{"rect": list(ROOF), "mode": "some"}]}}}))
    with pytest.raises(ConfigError, match="exclude mode"):
        load_cameras(cfg, require_login=False, zones=zones)


def test_monitor_hides_a_visit_in_an_exclusion_zone():
    from camera_mcp.monitor import CameraWorker
    cam = Camera("backyard", "x", exclude=((ROOF, "hide_unrecognized"),))
    worker = SimpleNamespace(camera=cam, tz=timezone.utc, lock=threading.Lock(), pending_visits=[])
    roof, fox = _visit("backyard", "animal", [roof_box()] * 3), _visit("backyard", "animal", crossing(conf=0.6))
    for v in (roof, fox):
        CameraWorker._visit_closed(worker, v)
    assert (roof.suppressed, fox.suppressed) == ("zone", None)


# ---- polygon zones -------------------------------------------------------------------------

HEDGE = ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.3, 0.25), (0.1, 0.5))  # concave: a notch cut from the bottom


def test_polygons_parse_round_trip_and_reject_bad_input():
    from camera_mcp.config import ConfigError
    from camera_mcp.zones import parse_shape, parse_shapes, shape_json
    assert parse_shape({"poly": [list(p) for p in HEDGE]}, "t", "k") == HEDGE
    assert parse_shape([0.1, 0.2, 0.3, 0.4], "t", "k") == (0.1, 0.2, 0.3, 0.4)  # rectangles still work
    mixed = parse_shapes([[0.1, 0.2, 0.3, 0.4], {"poly": [list(p) for p in HEDGE]}], "t", "k")
    assert [shape_json(s) for s in mixed] == [[0.1, 0.2, 0.3, 0.4], {"poly": [list(p) for p in HEDGE]}]
    for bad, msg in (({"poly": [[0.1, 0.1], [0.2, 0.2]]}, "at least 3"), ({"poly": [[0.1, 0.1], [1.2, 0.2], [0.3, 0.3]]}, "outside"),
                     ({"pts": []}, "poly"), ({"poly": "nope"}, "polygon")):
        with pytest.raises(ConfigError, match=msg):
            parse_shape(bad, "t", "k")


def test_point_in_a_concave_polygon():
    from camera_mcp.zones import shape_contains
    assert shape_contains(HEDGE, 0.2, 0.2) and shape_contains(HEDGE, 0.49, 0.45)  # either side of the notch
    assert not shape_contains(HEDGE, 0.3, 0.4)  # in the notch: inside the bounding box, outside the shape
    assert not shape_contains(HEDGE, 0.7, 0.2)


def test_exclusion_polygon_hides_only_what_is_inside_it():
    from camera_mcp.gate import in_zones
    zone = ((HEDGE, "hide_unrecognized"),)
    in_hedge, in_notch = Box(0.18, 0.18, 0.04, 0.04, 0.5), Box(0.28, 0.38, 0.04, 0.04, 0.5)
    assert in_zones("animal", [in_hedge], zone) == "zone"
    assert in_zones("animal", [in_notch], zone) is None  # a rectangle zone would have hidden this one


def test_motion_mask_polygon_is_rasterised_once_and_costs_the_same_per_frame():
    import time

    import numpy as np

    from camera_mcp.motion import MotionDetector
    s = MonitorSettings()
    poly = MotionDetector(s, ignore=(HEDGE,))
    keep = poly._build_keep((180, 320))
    assert keep[int(0.2 * 180), int(0.2 * 320)] == 0  # inside the hedge: ignored
    assert keep[int(0.4 * 180), int(0.3 * 320)] == 1  # in the notch: still counts
    assert keep[int(0.8 * 180), int(0.8 * 320)] == 1
    frames = [np.random.default_rng(i).integers(0, 255, (180, 320), np.uint8) for i in range(50)]

    def per_frame(det):
        det.update(frames[0])  # first frame builds the mask
        t = time.perf_counter()
        for f in frames:
            det.update(f)
        return (time.perf_counter() - t) / len(frames)

    p, r = min(per_frame(MotionDetector(s, ignore=(HEDGE,))) for _ in range(3)), \
        min(per_frame(MotionDetector(s, ignore=((0.1, 0.1, 0.4, 0.4),))) for _ in range(3))
    assert p < r * 1.5 + 0.0005, (p, r)  # same work per frame: one multiply by the prebuilt mask


# ---- min_size: tell a fixed object from a small bird at the same spot ----------------------------

def test_min_size_keeps_small_boxes_out_of_a_zone():
    from camera_mcp.gate import in_zones
    roof = Box(0.40, 0.45, 0.07, 0.045, 0.6)  # a feeder roof: centre (0.435, 0.4725), ~0.07 wide
    dove = Box(0.433, 0.465, 0.006, 0.017, 0.6)  # a small bird crossing the same spot
    zone = ((0.38, 0.40, 0.12, 0.12), "ignore", 0.03)
    assert in_zones("songbirds", [roof], (zone,)) == "zone"
    assert in_zones("songbirds", [dove], (zone,)) is None  # below min_size: not in the zone
    assert in_zones("songbirds", [dove], ((zone[0], "ignore"),)) == "zone"  # without min_size it would be
    roi = (0.0, 0.25, 1.0, 0.5)  # sizes are measured on the full frame: h 0.09 in the roi is 0.045 of the frame
    in_roi = Box(0.40, (0.4725 - 0.25) / 0.5 - 0.045, 0.02, 0.09, 0.6)
    assert in_zones("animal", [in_roi], (((0.38, 0.40, 0.12, 0.12), "ignore", 0.05),), roi) is None
    assert in_zones("animal", [in_roi], (((0.38, 0.40, 0.12, 0.12), "ignore", 0.04),), roi) == "zone"


def test_min_size_parses_round_trips_and_is_optional(tmp_path):
    import json

    from camera_mcp.config import ConfigError, load_cameras
    from camera_mcp.zones import exclude_json
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("cameras:\n  - {name: backyard, source: b.mp4}\n")
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps({"cameras": {"backyard": {"exclude": [
        {"rect": list(ROOF), "mode": "ignore", "min_size": 0.03}, {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore"}]}}}))
    cam = load_cameras(cfg, require_login=False, zones=zones)["backyard"]
    assert cam.exclude == ((ROOF, "ignore", 0.03), ((0.1, 0.1, 0.1, 0.1), "ignore"))  # no size: unchanged shape
    assert [exclude_json(z) for z in cam.exclude] == [
        {"rect": list(ROOF), "mode": "ignore", "min_size": 0.03}, {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore"}]
    for bad in (1.5, -0.1, "big"):
        zones.write_text(json.dumps({"cameras": {"backyard": {"exclude": [{"rect": list(ROOF), "min_size": bad}]}}}))
        with pytest.raises(ConfigError, match="min_size"):
            load_cameras(cfg, require_login=False, zones=zones)


# ---- max_size: tell a small fixed thing from a bigger animal at the same spot ---------------------

def test_max_size_keeps_big_boxes_out_of_a_zone():
    from camera_mcp.gate import in_zones
    bolt = Box(0.400, 0.145, 0.012, 0.028, 0.25)  # a bolt-loop on a squirrel shield, ~0.03 tall
    rat = Box(0.380, 0.130, 0.05, 0.035, 0.53)  # a rat on the shield, centred on the bolt, 0.05 wide
    zone = ((0.396, 0.145, 0.011, 0.019), "ignore", 0.0, 0.04)
    assert in_zones("animal", [bolt], (zone,)) == "zone"
    assert in_zones("animal", [rat], (zone,)) is None  # not below max_size: not in the zone
    assert in_zones("animal", [rat], ((zone[0], "ignore"),)) == "zone"  # without max_size it would be
    ranged = ((0.396, 0.145, 0.011, 0.019), "ignore", 0.02, 0.04)  # both ends: 0.02 <= size < 0.04
    assert in_zones("animal", [bolt], (ranged,)) == "zone"
    assert in_zones("animal", [Box(0.4, 0.15, 0.01, 0.01, 0.3)], (ranged,)) is None  # 0.01: below the range


def test_max_size_parses_round_trips_and_must_be_above_min_size(tmp_path):
    import json

    from camera_mcp.config import ConfigError, load_cameras
    from camera_mcp.zones import exclude_json
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("cameras:\n  - {name: west_alley, source: w.mp4}\n")
    zones = tmp_path / "zones.json"
    bolt, ranged = {"rect": [0.396, 0.145, 0.011, 0.019], "mode": "ignore", "max_size": 0.04}, \
        {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore", "min_size": 0.01, "max_size": 0.04}
    zones.write_text(json.dumps({"cameras": {"west_alley": {"exclude": [bolt, ranged]}}}))
    cam = load_cameras(cfg, require_login=False, zones=zones)["west_alley"]
    assert cam.exclude == (((0.396, 0.145, 0.011, 0.019), "ignore", 0.0, 0.04), ((0.1, 0.1, 0.1, 0.1), "ignore", 0.01, 0.04))
    assert [exclude_json(z) for z in cam.exclude] == [bolt, ranged]  # no min_size written when it has none
    for bad in ({"max_size": 1.5}, {"max_size": "small"}, {"min_size": 0.04, "max_size": 0.04}, {"min_size": 0.05, "max_size": 0.03}):
        zones.write_text(json.dumps({"cameras": {"west_alley": {"exclude": [{"rect": [0.1, 0.1, 0.1, 0.1], **bad}]}}}))
        with pytest.raises(ConfigError, match="max_size"):
            load_cameras(cfg, require_login=False, zones=zones)
