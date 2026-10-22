# Copyright (c) 2026 Eric Cooper.
"""Tags in the visit log: visit and box tags with species, how box tags inherit, the labels map
(with day and night apart), and favourites."""

from __future__ import annotations

import time

import pytest

from camera_mcp.config import Camera, Site
from camera_mcp.history import History, HistoryError
from camera_mcp.store import Store
from camera_mcp.visits import Visit

NOW = time.time()


def test_label_map_places_each_visit_where_it_was_boxed(tmp_path):
    """One entry per visit at its most confident box, mapped from the camera's roi crop to the full
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


def test_tag_species_and_visits_by_id(tmp_path):
    """The picker's suggestions are the species named in tags, most used first; visits_by_id returns
    visits in the order asked, shaped as find_visits returns them (the revisit list opens them)."""
    store = Store(tmp_path / "monitor.db")
    a = _tagged_visit(store, "backyard", [[(0.4, 0.4, 0.05, 0.05, 0.7)]], NOW - 900)
    b = _tagged_visit(store, "backyard", [[(0.4, 0.4, 0.05, 0.05, 0.7)], [(0.8, 0.8, 0.05, 0.05, 0.9)]], NOW - 800)
    store.set_visit_label(a, "wrong_species", species="norway rat")
    store.set_visit_label(b, "real", species="house finch")
    frames = History(tmp_path / "monitor.db", {"backyard": Camera("backyard", "x")}, Site(timezone="UTC")).visit_boxes(b)["frames"]
    store.set_box_label(b, frames[1]["ts"], 0, "real", species="norway rat")
    store.close()
    h = History(tmp_path / "monitor.db", {"backyard": Camera("backyard", "x")}, Site(timezone="UTC"))
    assert h.tagged_species() == ["norway rat", "house finch"]
    got = h.visits_by_id([b, 999, a])
    assert [v["id"] for v in got] == [b, a] and got[0]["verdict"] == "real" and "clip" in got[0]
    assert set(got[0]) == set(h.find_visits(since="1h")["visits"][0])


def test_favourites_are_starred_apart_from_tags_and_listed_from_any_time(tmp_path):
    """A star keeps whether it opens on the still or the clip (and the clip moment); the favourites
    filter lists starred visits from any time, hidden or not; unstarring removes it."""
    store = Store(tmp_path / "monitor.db")
    old = _tagged_visit(store, "east_deck", [[(0.4, 0.4, 0.1, 0.1, 0.9)]], NOW - 20 * 86400)  # weeks ago
    new = _tagged_visit(store, "east_deck", [[(0.4, 0.4, 0.1, 0.1, 0.9)]], NOW - 600)
    plain = _tagged_visit(store, "east_deck", [[(0.4, 0.4, 0.1, 0.1, 0.9)]], NOW - 500)
    store.set_favorite(old, True, "clip", clip_at=12.345)
    store.set_favorite(new, True, "still")
    store.set_visit_suppressed(new, "zone")  # hidden, but still a favourite
    with pytest.raises(ValueError):
        store.set_favorite(plain, True, "gif")
    with pytest.raises(ValueError):
        store.set_favorite(99999, True)
    store.close()
    h = History(tmp_path / "monitor.db", {"east_deck": Camera("east_deck", "x")}, Site(timezone="UTC"))
    favs = h.find_visits(since="1h", favorites_only=True)["visits"]
    assert [v["id"] for v in favs] == [new, old]  # any time, hidden or not, newest first
    assert favs[1]["favorite"] == {"kind": "clip", "clip_at": 12.35} and favs[0]["favorite"] == {"kind": "still", "clip_at": None}
    assert {v["id"]: v["favorite"] for v in h.find_visits(since="1h")["visits"]} == {plain: None}  # the normal grid: unchanged
    assert h.visits_by_id([old])[0]["favorite"]["kind"] == "clip"
    store = Store(tmp_path / "monitor.db")
    store.set_favorite(old, False)
    store.close()
    assert [v["id"] for v in h.find_visits(since="1h", favorites_only=True)["visits"]] == [new]
