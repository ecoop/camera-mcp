# Copyright (c) 2026 Eric Cooper.
"""Phase 2c: questions about the past, from a visit log built with the monitor's own store."""

from __future__ import annotations

import base64
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from mcp import Client

from camera_mcp.config import Camera, Site
from camera_mcp.history import History, HistoryError, parse_when
from camera_mcp.server import build_server
from camera_mcp.store import Store
from camera_mcp.visits import Visit

pytestmark = pytest.mark.anyio
LA = ZoneInfo("America/Los_Angeles")
NOW = time.time()
H = 3600.0

FINCH = ("house finch", "aves;passeriformes;fringillidae;haemorhous;mexicanus")
CAT = ("domestic cat", "mammalia;carnivora;felidae;felis;catus")
BIRD = ("bird", "aves;;;;")
SKUNK = ("striped skunk", "mammalia;carnivora;mephitidae;mephitis;mephitis")
RACCOON = ("northern raccoon", "mammalia;carnivora;procyonidae;procyon;lotor")
RAT = ("brown rat", "mammalia;rodentia;muridae;rattus;norvegicus")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def visit(camera, key, label, start, minutes, count=1):
    v = Visit(camera, key, start=NOW + start, last_seen=NOW + start + minutes * 60, frames=minutes * 6, max_count=count)
    v.votes[label] = 1.0
    return v


@pytest.fixture
def log(tmp_path):
    """Four cameras, a known timeline (times relative to NOW)."""
    data = tmp_path / "data"
    store = Store(data / "monitor.db")
    thumb = data / "media" / "backyard" / "cat.jpg"
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"\xff\xd8 a jpeg")
    store.add_visit(visit("backyard", "songbirds", FINCH, -2 * H, 10, count=6), model="m", thumbnail=None, clip="c1.mp4")
    store.add_visit(visit("backyard", "cat", CAT, -2 * H + 300, 2), model="m",
                    thumbnail="media/backyard/cat.jpg", clip=None)
    store.add_visit(visit("west_alley", "jays", BIRD, -3 * H, 5), model="m", thumbnail=None, clip=None)
    store.add_visit(visit("backyard", "bird", BIRD, -1 * H, 3), model="m", thumbnail=None, clip=None)
    store.add_visit(visit("east_deck", "raccoon", RACCOON, -26 * H, 4), model="m", thumbnail=None, clip=None)
    store.add_visit(visit("east_fence", "rat", RAT, -5 * H, 1), model="m", thumbnail=None, clip=None)
    skunk = visit("east_deck", "skunk", SKUNK, -0.25 * H, 5)
    skunk.db_id = store.open_visit(skunk, model="m")  # still there
    for cam in ("backyard", "east_deck"):
        store.extend_uptime(cam, NOW - 48 * H, NOW)
    store.extend_uptime("west_alley", NOW - 24 * H, NOW - 2 * H)  # dropped out two hours ago...
    store.extend_uptime("west_alley", NOW - 1 * H, NOW - 0.5 * H)  # ...back for half an hour, then gone
    for cam, last in (("backyard", NOW), ("east_deck", NOW), ("west_alley", NOW - 0.5 * H), ("east_fence", NOW - 9 * H)):
        store.set_status(cam, "watching", now=last, last_frame=last)
    store.set_view("east_fence", 0.04, True, now=NOW - 8 * H)
    store.add_skipped("backyard", NOW - 3 * H, NOW - 3 * H + 40)  # the detector fell behind for a while
    store.add_skipped("backyard", NOW - 3 * H + 40.5, NOW - 3 * H + 60)  # touching: merged into one span
    store.close()
    cameras = {
        "backyard": Camera("backyard", "x", tags=("feeder", "dog_yard", "wide_view")),
        "west_alley": Camera("west_alley", "x", tags=("feeder", "dog_yard")),
        "east_deck": Camera("east_deck", "x", tags=("under_deck", "dog_yard")),
        "east_fence": Camera("east_fence", "x", tags=("thoroughfare",)),
    }
    return History(data / "monitor.db", cameras, Site(timezone="America/Los_Angeles")), data


def labels(result):
    return [v["label"] for v in result["visits"]]


# ---- times ----------------------------------------------------------------------------

def test_parse_when():
    now = datetime(2026, 10, 9, 21, 30, tzinfo=LA).timestamp()
    assert parse_when("15m", now, LA) == now - 900
    assert parse_when("2w", now, LA) == now - 14 * 86400
    assert parse_when(None, now, LA) == now
    assert parse_when("2026-10-09T06:00", now, LA) == datetime(2026, 10, 9, 6, 0, tzinfo=LA).timestamp()
    assert parse_when("2026-10-09T13:00:00Z", now, LA) == datetime(2026, 10, 9, 6, 0, tzinfo=LA).timestamp()
    assert parse_when("today", now, LA) == datetime(2026, 10, 9, tzinfo=LA).timestamp()
    assert parse_when("yesterday", now, LA) == datetime(2026, 10, 8, tzinfo=LA).timestamp()
    with pytest.raises(HistoryError, match="15m"):
        parse_when("last tuesday", now, LA)


# ---- find_visits ----------------------------------------------------------------------

def test_groups_are_exact_and_classes_are_broad(log):
    history, _ = log
    assert labels(history.find_visits("24h", taxon="songbirds")) == ["house finch"]  # not the jays, not "bird"
    assert set(labels(history.find_visits("24h", taxon="bird"))) == {"house finch", "bird"}  # jays (bird label) too:
    assert history.find_visits("24h", taxon="bird")["total"] == 3
    assert labels(history.find_visits("24h", taxon="striped skunk")) == ["striped skunk"]
    assert labels(history.find_visits("48h", taxon="night-critters")) == ["striped skunk", "northern raccoon"]


def test_is_the_dog_yard_clear(log):
    history, _ = log
    result = history.find_visits("15m", tag="dog_yard", taxon="night-critters")
    (skunk,) = result["visits"]
    assert skunk["camera"] == "east_deck" and skunk["still_there"] is True
    coverage = {c["camera"]: c["watched"] for c in result["coverage"]}
    assert coverage == pytest.approx({"backyard": 1.0, "west_alley": 0.0, "east_deck": 1.0}, abs=0.02)  # west alley isn't watched


def test_cats_among_the_birds(log):
    history, _ = log
    result = history.find_visits("24h", tag="feeder", taxon="cat", overlapping="bird")
    assert labels(result) == ["domestic cat"]
    assert history.find_visits("24h", taxon="cat", overlapping="rat")["total"] == 0


def test_cameras_match_loosely_and_unknown_ones_are_errors(log):
    history, _ = log
    assert history.find_visits("24h", cameras=["East Fence"])["total"] == 1
    with pytest.raises(HistoryError, match="known: backyard"):
        history.find_visits("24h", cameras=["driveway"])
    with pytest.raises(HistoryError, match="tags in use"):
        history.find_visits("24h", tag="driveway")


def test_coverage_counts_gaps_and_moved_cameras(log):
    history, _ = log
    coverage = {c["camera"]: c for c in history.find_visits("24h")["coverage"]}
    assert coverage["west_alley"]["watched"] == pytest.approx(22.5 / 24, abs=0.01)
    assert coverage["east_fence"]["watched"] == 0.0 and coverage["east_fence"]["looks_moved_since"]


# ---- activity_summary -----------------------------------------------------------------

def test_activity_by_species_and_by_local_hour(log):
    history, _ = log
    by_species = history.activity_summary("48h", group_by="species")
    assert by_species["total_visits"] == 7
    assert {r["key"]: r["visits"] for r in by_species["rows"]}["bird"] == 2
    finch = next(r for r in by_species["rows"] if r["key"] == "house finch")
    assert finch["max_at_once"] == 6
    by_hour = history.activity_summary("24h", group_by="hour", tag="feeder")
    expected = sorted({f"{datetime.fromtimestamp(NOW + s, LA):%H}:00" for s in (-2 * H, -2 * H + 300, -3 * H, -1 * H)})
    assert [r["key"] for r in by_hour["rows"]] == expected


def test_bad_group_by(log):
    history, _ = log
    with pytest.raises(HistoryError, match="group_by"):
        history.activity_summary(group_by="weekday")


# ---- media and status -----------------------------------------------------------------

def test_visit_thumbnail(log):
    history, data = log
    cat_id = history.find_visits("24h", taxon="cat")["visits"][0]["id"]
    image, caption = history.visit_thumbnail(cat_id, data)
    assert image.startswith(b"\xff\xd8") and "domestic cat" in caption
    finch_id = history.find_visits("24h", taxon="songbirds")["visits"][0]["id"]
    with pytest.raises(HistoryError, match="no thumbnail"):
        history.visit_thumbnail(finch_id, data)


def test_visit_labels_and_counts(log):
    history, data = log
    assert history.label_counts() == {"total": 0, "by_verdict": {}}  # nothing labelled yet
    ids = [v["id"] for v in history.find_visits("48h")["visits"]]
    store = Store(data / "monitor.db")
    store.set_visit_label(ids[0], "false", note="clothesline")
    store.set_visit_label(ids[1], "false")
    store.set_visit_label(ids[2], "real")
    store.set_visit_label(ids[0], "wrong_species")  # re-label replaces the whole row, doesn't add
    store.close()
    assert history.label_counts() == {"total": 3, "by_verdict": {"false": 1, "real": 1, "wrong_species": 1}}
    first = history.visit_label(ids[0])
    assert first["verdict"] == "wrong_species" and first["note"] is None  # the clothesline note was replaced
    assert history.visit_label(ids[1])["note"] is None
    assert history.visit_label(ids[3]) is None  # unlabelled visit
    with pytest.raises(ValueError, match="verdict"):
        Store(data / "monitor.db").set_visit_label(ids[0], "nope")
    with pytest.raises(ValueError, match="no visit"):
        Store(data / "monitor.db").set_visit_label(999999, "real")


def test_monitor_status(log):
    history, _ = log
    status = {s["camera"]: s for s in history.monitor_status()}
    assert status["backyard"]["watching_now"] and status["east_deck"]["open_visits"] == 1
    assert not status["west_alley"]["watching_now"]
    assert status["east_fence"]["looks_moved_since"] is not None


def test_analysis_health(log):
    import sqlite3

    from camera_mcp.store import iso

    history, data = log
    db = sqlite3.connect(data / "monitor.db")
    now = time.time()
    db.execute("INSERT INTO analysis_stats VALUES (?, ?, ?, ?, ?, ?)", (iso(now - 120), 3, 10.0, 50, 0, 20.0))
    db.execute("INSERT INTO analysis_stats VALUES (?, ?, ?, ?, ?, ?)", (iso(now - 60), 12, 45.0, 30, 2, 18.0))
    db.commit()
    db.close()
    h = history.analysis_health(since="1h")
    assert h["samples"] == 2
    assert h["frames_analyzed"] == 80 and h["segments_dropped"] == 2
    assert h["peak_lag_seconds"] == 45.0 and h["peak_queue_depth"] == 12
    assert h["mean_seconds_per_frame"] == round(38.0 / 80, 3)
    assert history.analysis_health(since="1s")["samples"] == 0  # window excludes both samples


def test_analysis_series_aggregate_and_per_camera(tmp_path):
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    store.add_analysis_stats(now - 120, 5, 30.0, 80, 1, 40.0)  # interval 1 aggregate
    store.add_camera_stats(now - 120, [("east_fence", 4, 30.0, 20, 1, 30.0),
                                       ("backyard", 1, 10.0, 60, 0, 10.0)])
    store.add_analysis_stats(now - 60, 2, 12.0, 50, 0, 20.0)   # interval 2 aggregate
    store.add_camera_stats(now - 60, [("east_fence", 2, 12.0, 10, 0, 12.0),
                                      ("backyard", 0, 5.0, 40, 0, 8.0)])
    store.close()
    cams = {"east_fence": Camera("east_fence", "x"), "backyard": Camera("backyard", "x")}
    history = History(tmp_path / "monitor.db", cams, Site(timezone="America/Los_Angeles"))
    allpts = history.analysis_series(since="1h")["points"]
    assert len(allpts) == 2 and allpts[0]["frames"] == 80 and allpts[0]["depth"] == 5  # the aggregate row
    ef = history.analysis_series(since="1h", camera="east fence")  # loose camera match
    assert ef["camera"] == "east fence"
    assert [p["frames"] for p in ef["points"]] == [20, 10] and [p["depth"] for p in ef["points"]] == [4, 2]
    assert history.analysis_series(since="1h", camera="nope")["points"] == []  # unknown camera: empty


def test_queries_close_their_db_connections(log):
    # Regression: each query opens a fresh connection and closes it. A leaked connection kept a
    # long-lived server pinned to a replaced monitor.db, so queries read a stale (deleted) file.
    import sqlite3

    history, _ = log
    orig, conns = history._db, []
    history._db = lambda: (conns.append(c := orig()), c)[1]
    history.find_visits(since="1d")
    history.activity_summary(since="1d")
    history.monitor_status()
    history.coverage(list(history.cameras), 0.0, time.time())  # standalone coverage opens its own too
    assert len(conns) >= 4
    for c in conns:  # a closed sqlite3 connection raises on use
        with pytest.raises(sqlite3.ProgrammingError):
            c.execute("SELECT 1")


def test_no_log_yet(tmp_path):
    history = History(tmp_path / "monitor.db", {"a": Camera("a", "x")})
    with pytest.raises(HistoryError, match="monitor hasn't run"):
        history.find_visits()


# ---- over MCP -------------------------------------------------------------------------

async def test_history_tools_over_mcp(log):
    history, data = log
    cameras = history.cameras
    async with Client(build_server(cameras, history, data)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert {"find_visits", "activity_summary", "visit_media", "monitor_status"} <= set(tools)
        assert all(tools[n].annotations.read_only_hint for n in ("find_visits", "visit_media"))
        found = await client.call_tool("find_visits", {"since": "15m", "tag": "dog_yard", "taxon": "night-critters"})
        assert found.structured_content["visits"][0]["label"] == "striped skunk"
        cat_id = (await client.call_tool("find_visits", {"taxon": "cat"})).structured_content["visits"][0]["id"]
        media = await client.call_tool("visit_media", {"visit_id": cat_id})
        image, caption = media.content
        assert image.mime_type == "image/jpeg" and base64.b64decode(image.data).startswith(b"\xff\xd8")
        bad = await client.call_tool("find_visits", {"since": "last tuesday"})
        assert bad.is_error and "15m" in bad.content[0].text


def test_logs_from_older_monitors(tmp_path):
    """The server opens the log read-only, so it never adds new columns: older logs must still work."""
    import sqlite3

    db_path = tmp_path / "monitor.db"
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE camera_status (camera TEXT PRIMARY KEY, state TEXT NOT NULL, last_frame_at TEXT,
          last_motion_at TEXT, error TEXT, updated_at TEXT NOT NULL);
    """)  # 2a: motion only, no visits
    db.commit()
    cams = {"a": Camera("a", "x")}
    with pytest.raises(HistoryError, match="no visits yet"):
        History(db_path, cams).find_visits()
    db.executescript("""
        CREATE TABLE visits (id INTEGER PRIMARY KEY, camera TEXT NOT NULL, started_at TEXT NOT NULL,
          ended_at TEXT NOT NULL, label TEXT NOT NULL, taxonomy TEXT NOT NULL, grp TEXT NOT NULL,
          score REAL NOT NULL, frames INTEGER NOT NULL, max_count INTEGER NOT NULL, thumbnail TEXT,
          clip TEXT, model TEXT NOT NULL);
    """)  # 2b: visits, but no `open` column and no uptime table
    db.execute("INSERT INTO visits VALUES (1, 'a', ?, ?, 'domestic cat', ?, 'cat', 1, 3, 1, NULL, NULL, 'm')",
               (iso_ago(600), iso_ago(500), CAT[1]))
    db.commit()
    result = History(db_path, cams).find_visits("1h")
    assert result["visits"][0]["still_there"] is False
    assert result["coverage"][0]["watched"] is None  # unknown, not "not watched"
    assert History(db_path, cams).monitor_status()[0]["open_visits"] == 0


def iso_ago(seconds: float) -> str:
    from camera_mcp.store import iso

    return iso(time.time() - seconds)


def test_coverage_reports_motion_that_was_never_analysed(log):
    history, _ = log
    coverage = {c["camera"]: c for c in history.find_visits("24h")["coverage"]}
    assert coverage["backyard"]["unanalyzed_motion_seconds"] == 60
    assert coverage["east_deck"]["unanalyzed_motion_seconds"] == 0
    import sqlite3

    rows = sqlite3.connect(history.db_path).execute("SELECT COUNT(*) FROM analysis_skipped").fetchone()[0]
    assert rows == 1  # the two touching spans were merged


def test_visits_timeline_bins_by_group(log):
    history, data = log
    d = history.visits_timeline(since="30h", bins=30)
    assert d["visits"] >= 1 and len(d["bins"]) == 30
    # the known fixture visits (songbirds, cat, jays, bird, raccoon, rat) show up in the group totals
    assert "songbirds" in d["total"] or "cat" in d["total"]
    assert sum(sum(b["grps"].values()) for b in d["bins"]) == d["visits"]


def test_label_map_places_each_visit_where_it_was_boxed(tmp_path):
    """One entry per visit at its mean box centre, mapped from the camera's roi crop to the full
    frame, with its verdict; labelled visits only unless asked."""
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection
    store = Store(tmp_path / "monitor.db")
    ids = []
    for x in (0.1, 0.6):  # two visits, boxes centred at x+0.05 in the roi crop
        v = Visit("backyard", "animal", start=NOW - 600, last_seen=NOW - 590, frames=2)
        v.votes[("animal", "")] = 1.0
        v.log = [(NOW - 600 + i, [Detection(Box(x, 0.5, 0.1, 0.2, 0.6), Label("animal", "", 0.5), "animal")]) for i in range(2)]
        ids.append(store.add_visit(v, model="m", thumbnail="t.jpg", clip="c.mp4"))
    store.set_visit_label(ids[0], "false")
    store.close()
    roi = (0.2, 0.0, 0.5, 1.0)  # analysis crop: x 0.2-0.7 of the full frame
    h = History(tmp_path / "monitor.db", {"backyard": Camera("backyard", "x", roi=roi)}, Site(timezone="UTC"))
    (only,) = h.label_map("backyard", since="1h")["visits"]  # unlabelled left out by default
    assert only["id"] == ids[0] and only["verdict"] == "false"
    assert only["x"] == pytest.approx(0.2 + 0.15 * 0.5) and only["y"] == pytest.approx(0.6)
    assert only["size"] == pytest.approx(0.2)  # larger side on the full frame: h 0.2 vs w 0.1*0.5
    both = h.label_map("backyard", since="1h", unlabelled=True)["visits"]
    assert {v["id"]: v["verdict"] for v in both} == {ids[0]: "false", ids[1]: None}
    with pytest.raises(HistoryError):
        h.label_map("garage", since="1h")


def test_label_map_places_a_scattered_visit_at_its_most_confident_box(tmp_path):
    """An unnamed visit can collect boxes on different things; the dot goes where the viewer's box
    is (the most confident one), never at an average of places where nothing was."""
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection
    store = Store(tmp_path / "monitor.db")
    v = Visit("west_alley", "animal", start=NOW - 600, last_seen=NOW - 590, frames=3)
    v.votes[("animal", "")] = 1.0
    boxes = [Box(0.40, 0.40, 0.08, 0.10, 0.34), Box(0.10, 0.10, 0.10, 0.10, 0.44), Box(0.40, 0.40, 0.08, 0.10, 0.28)]
    v.log = [(NOW - 600 + i, [Detection(b, Label("animal", "", 0.5), "animal")]) for i, b in enumerate(boxes)]
    vid = store.add_visit(v, model="m", thumbnail="t.jpg", clip="c.mp4")
    store.set_visit_label(vid, "false")
    store.close()
    h = History(tmp_path / "monitor.db", {"west_alley": Camera("west_alley", "x")}, Site(timezone="UTC"))
    (dot,) = h.label_map("west_alley", since="1h")["visits"]
    assert (dot["x"], dot["y"]) == (pytest.approx(0.15), pytest.approx(0.15))  # the 0.44 box, not the mean
    assert dot["spread"] == pytest.approx(0.30)  # centres ranged x 0.15-0.44, y 0.15-0.45: the larger range


def test_day_and_night_are_recorded_per_visit_backfilled_and_on_the_map(tmp_path):
    """Whether a visit was seen under infrared: set for new visits from their thumbnail, filled in for
    earlier ones from the saved thumbnail (once: one that can't be read isn't retried), and returned by
    the labels map so the dashboard can show day or night apart."""
    import cv2
    import numpy as np

    from camera_mcp.monitor import backfill_night
    from camera_mcp.taxa import Label
    from camera_mcp.view import is_infrared
    from camera_mcp.visits import Box, Detection
    grey = np.full((90, 90, 3), 120, np.uint8)
    dull = grey.copy(); dull[..., 0] += 3  # a faint warm cast, as on grey bark in daylight: still colour
    assert is_infrared(grey) and not is_infrared(dull) and not is_infrared(np.zeros((0, 0, 3), np.uint8))
    (tmp_path / "media").mkdir()
    cv2.imwrite(str(tmp_path / "media" / "night.jpg"), grey)
    cv2.imwrite(str(tmp_path / "media" / "day.jpg"), cv2.cvtColor(dull, cv2.COLOR_RGB2BGR))
    store = Store(tmp_path / "monitor.db")
    ids = []
    for i, (thumb, night) in enumerate((("media/night.jpg", None), ("media/day.jpg", None), ("media/gone.jpg", None),
                                         ("media/day.jpg", True))):  # the last one was flagged when it was saved
        v = Visit("backyard", "animal", start=NOW - 600 + i, last_seen=NOW - 590 + i, frames=1)
        v.votes[("animal", "")] = 1.0
        v.log = [(NOW - 600 + i, [Detection(Box(0.1 * i, 0.5, 0.1, 0.1, 0.6), Label("animal", "", 0.5), "animal")])]
        ids.append(store.add_visit(v, model="m", thumbnail=thumb, clip=None, night=night))
    assert backfill_night(store, tmp_path) == 3 and backfill_night(store, tmp_path) == 0  # once; the flagged one untouched
    store.close()
    h = History(tmp_path / "monitor.db", {"backyard": Camera("backyard", "x")}, Site(timezone="UTC"))
    night = {v["id"]: v["night"] for v in h.label_map("backyard", since="1h", unlabelled=True)["visits"]}
    assert night == {ids[0]: True, ids[1]: False, ids[2]: None, ids[3]: True}


# ---- tags per box, and species on tags ---------------------------------------------------------

def _tagged_visit(store, cam, boxes_per_frame, start):
    """A stored visit whose frames hold the given boxes (Box tuples); returns its id."""
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection
    v = Visit(cam, "animal", start=start, last_seen=start + len(boxes_per_frame), frames=len(boxes_per_frame))
    v.votes[("animal", "")] = 1.0
    v.log = [(start + i, [Detection(Box(*b), Label("animal", "", 0.5), "animal") for b in boxes])
             for i, boxes in enumerate(boxes_per_frame)]
    return store.add_visit(v, model="m", thumbnail=None, clip=None)


def test_box_tags_inherit_the_visit_tag_only_where_it_must_hold(tmp_path):
    """A 'false' or 'wrong_species' visit's tag holds for every box, and so does a 'real' one whose
    boxes stayed in one place; a 'real' visit whose boxes were in several places may hold junk boxes
    too, so they stay untagged until a human tags them. A box's own tag always wins."""
    store = Store(tmp_path / "monitor.db")
    still_bird = _tagged_visit(store, "backyard", [[(0.40, 0.40, 0.05, 0.05, 0.7)]] * 3, NOW - 900)
    roaming = _tagged_visit(store, "backyard", [[(0.10, 0.10, 0.05, 0.05, 0.4)],  # a box on fruit...
                                                [(0.60, 0.50, 0.05, 0.05, 0.8), (0.30, 0.70, 0.04, 0.04, 0.3)]],
                            NOW - 800)  # ...then the bird elsewhere, plus another box, on the still
    junk = _tagged_visit(store, "backyard", [[(0.10, 0.10, 0.05, 0.05, 0.3)], [(0.70, 0.70, 0.05, 0.05, 0.3)]], NOW - 700)
    store.set_visit_label(still_bird, "real", species="house finch")
    store.set_visit_label(roaming, "real")
    store.set_visit_label(junk, "false")
    with pytest.raises(ValueError, match="species"):
        store.set_visit_label(junk, "false", species="rat")  # nothing real was there
    store.close()
    h = History(tmp_path / "monitor.db", {"backyard": Camera("backyard", "x")}, Site(timezone="UTC"))
    bird = h.visit_boxes(still_bird)
    assert {t["verdict"] for f in bird["frames"] for t in f["tags"]} == {"real"}
    assert bird["frames"][0]["tags"][0] == {"verdict": "real", "species": "house finch", "source": "visit"}
    assert h.visit_label(still_bird)["species"] == "house finch"
    assert all(t["verdict"] == "false" for f in h.visit_boxes(junk)["frames"] for t in f["tags"])
    roam = h.visit_boxes(roaming)
    assert roam["best"] == 1 and roam["frames"][1]["tags"] == [None, None]  # several places: unknown
    assert h.needs_box_tags() == [roaming] and h.needs_box_tags("backyard") == [roaming]

    store = Store(tmp_path / "monitor.db")
    ts = roam["frames"][1]["ts"]
    store.set_box_label(roaming, ts, 0, "real", species="dark-eyed junco")
    assert h.needs_box_tags() == [roaming]  # one box on the still is still unknown
    store.set_box_label(roaming, ts, 1, "false")
    assert h.needs_box_tags() == []  # every box on the still is known now
    tags = h.visit_boxes(roaming)["frames"][1]["tags"]
    assert tags == [{"verdict": "real", "species": "dark-eyed junco", "source": "box"},
                    {"verdict": "false", "species": None, "source": "box"}]
    store.set_box_label(roaming, ts, 1, None)  # cleared: unknown again
    assert h.needs_box_tags() == [roaming]
    for bad in ((ts, 5, "real"), ("1999-01-01T00:00:00Z", 0, "real"), (ts, 0, "maybe")):
        with pytest.raises(ValueError):
            store.set_box_label(roaming, *bad)
    with pytest.raises(ValueError, match="species"):
        store.set_box_label(roaming, ts, 0, "false", species="rat")
    store.close()
