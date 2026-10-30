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


def test_counts_follow_the_owners_tags(log):
    """Counts leave out visits tagged false (counted in tagged_false), and a visit the owner named
    counts under that name and its group, in the taxon filter too, for activity_summary and the
    Analytics timeline alike."""
    history, data = log
    by_label = {(v["camera"], v["label"]): v["id"] for v in history.find_visits("48h")["visits"]}
    store = Store(data / "monitor.db")
    store.set_visit_label(by_label["backyard", "house finch"], "false")
    # a name the log knows, a group, a new name; and a tag with no name, which counts as logged
    store.set_visit_label(by_label["backyard", "domestic cat"], "wrong_species", species="northern raccoon")
    store.set_visit_label(by_label["backyard", "bird"], "real", species="jays")
    store.set_visit_label(by_label["east_fence", "brown rat"], "wrong_species", species="fox squirrel")
    store.set_visit_label(by_label["east_deck", "northern raccoon"], "real")
    store.close()

    by_species = history.activity_summary("48h", group_by="species")
    assert by_species["total_visits"] == 6 and by_species["tagged_false"] == 1
    assert {r["key"]: r["visits"] for r in by_species["rows"]} == {
        "northern raccoon": 2, "jays": 1, "bird": 1, "fox squirrel": 1, "striped skunk": 1}
    by_group = {r["key"]: r["visits"] for r in history.activity_summary("48h", group_by="group")["rows"]}
    assert by_group == {"raccoon": 2, "jays": 2, "fox squirrel": 1, "skunk": 1}
    assert history.activity_summary("48h", taxon="cat")["total_visits"] == 0
    assert history.activity_summary("48h", taxon="raccoon")["total_visits"] == 2
    assert history.activity_summary("48h", taxon="night-critters")["total_visits"] == 3
    assert history.activity_summary("48h", taxon="songbirds")["tagged_false"] == 1

    timeline = history.visits_timeline(since="30h", bins=30)
    assert timeline["visits"] == 6 and timeline["tagged_false"] == 1
    assert timeline["total"] == {"raccoon": 2, "jays": 2, "fox squirrel": 1, "skunk": 1}
    assert history.visits_timeline(since="30h", taxon="raccoon")["visits"] == 2

    assert len(history.find_visits("48h")["visits"]) == 7  # find_visits still lists them all, with their tags


def test_monitor_status(log):
    history, _ = log
    status = {s["camera"]: s for s in history.monitor_status()}
    assert status["backyard"]["watching_now"] and status["east_deck"]["open_visits"] == 1
    assert not status["west_alley"]["watching_now"]
    assert status["east_fence"]["looks_moved_since"] is not None


def test_analysis_health(log):
    import sqlite3

    from camera_mcp.times import iso

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
    from camera_mcp.times import iso

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
