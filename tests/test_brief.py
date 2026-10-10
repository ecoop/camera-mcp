# Copyright (c) 2026 Eric Cooper.
"""The morning brief summarises the overnight log into a short, phone-sized message."""

from __future__ import annotations

import time

from camera_mcp.brief import build_brief
from camera_mcp.config import Camera, Site
from camera_mcp.history import History
from camera_mcp.store import Store
from camera_mcp.visits import Visit

NOW = time.time()
H = 3600.0
RACCOON = ("northern raccoon", "mammalia;carnivora;procyonidae;procyon;lotor")
CAT = ("domestic cat", "mammalia;carnivora;felidae;felis;catus")
FINCH = ("house finch", "aves;passeriformes;fringillidae;haemorhous;mexicanus")


def _visit(store, camera, key, label, hours_ago):
    v = Visit(camera, key, start=NOW - hours_ago * H, last_seen=NOW - hours_ago * H + 120, frames=12, max_count=1)
    v.votes[label] = 1.0
    store.add_visit(v, model="m", thumbnail=None, clip=None)


def _history(tmp_path, with_critters=True):
    store = Store(tmp_path / "data" / "monitor.db")
    if with_critters:
        _visit(store, "east_deck", "raccoon", RACCOON, 6)
        _visit(store, "east_deck", "raccoon", RACCOON, 5)
    _visit(store, "backyard", "cat", CAT, 4)
    _visit(store, "west_alley", "songbirds", FINCH, 1)
    for cam in ("east_deck", "backyard", "west_alley"):
        store.extend_uptime(cam, NOW - 14 * H, NOW)
        store.set_status(cam, "watching", now=NOW, last_frame=NOW)
    store.close()
    cams = {
        "backyard": Camera("backyard", "rtsp://h/stream1", tags=("feeder", "dog_yard")),
        "west_alley": Camera("west_alley", "rtsp://h/stream1", tags=("feeder",)),
        "east_deck": Camera("east_deck", "rtsp://h/stream1", tags=("under_deck", "dog_yard")),
        "clip": Camera("clip", "/tmp/x.mp4"),  # a file stand-in: excluded from the gaps note
    }
    return History(tmp_path / "data" / "monitor.db", cams, Site(timezone="America/Los_Angeles"))


def test_brief_leads_with_dog_yard_and_cats(tmp_path):
    brief = build_brief(_history(tmp_path), since="14h")
    assert "Wildlife brief" in brief
    assert "Dog yard:" in brief and "2 visits" in brief  # the two raccoons under the deck
    assert "Cats at feeders:" in brief
    assert "raccoon 2" in brief and "songbirds 1" in brief
    assert "clip" not in brief  # the file stand-in isn't in the note


def test_brief_says_clear_when_no_night_critters(tmp_path):
    brief = build_brief(_history(tmp_path, with_critters=False), since="14h")
    assert "clear of raccoons/skunks/opossums" in brief


def test_write_montage_builds_an_image_of_notable_visitors(tmp_path):
    import cv2
    import numpy as np
    from camera_mcp.brief import write_montage

    data = tmp_path / "data"
    store = Store(data / "monitor.db")
    thumb_rel = "media/east_deck/raccoon.jpg"
    thumb = data / thumb_rel
    thumb.parent.mkdir(parents=True)
    cv2.imwrite(str(thumb), np.full((180, 320, 3), 90, np.uint8))
    v = Visit("east_deck", "raccoon", start=NOW - 5 * H, last_seen=NOW - 5 * H + 60, frames=6, max_count=1)
    v.votes[RACCOON] = 1.0
    store.add_visit(v, model="m", thumbnail=thumb_rel, clip=None)
    store.close()
    cams = {"east_deck": Camera("east_deck", "rtsp://h/stream1", tags=("under_deck", "dog_yard"))}
    history = History(data / "monitor.db", cams, Site())

    out = tmp_path / "montage.jpg"
    assert write_montage(history, "14h", data, out) is True
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] >= 180 and img.shape[1] >= 320


def test_write_montage_returns_false_when_nothing_notable(tmp_path):
    from camera_mcp.brief import write_montage

    store = Store(tmp_path / "data" / "monitor.db")
    store.close()
    cams = {"a": Camera("a", "rtsp://h/stream1")}
    out = tmp_path / "m.jpg"
    assert write_montage(History(tmp_path / "data" / "monitor.db", cams, Site()), "14h", tmp_path / "data", out) is False
    assert not out.exists()
