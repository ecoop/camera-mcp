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
from camera_mcp.store import Store, iso
from camera_mcp.taxa import Label
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
