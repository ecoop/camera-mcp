# Copyright (c) 2026 Eric Cooper.
"""The tag model (camera_mcp.tagmodel): examples come from the tagged visits' stills with the tags
box_tags gives them, features are cached by box, the test set is the most recent days, and the
heads learn the owner's tags. SpeciesNet is a stand-in that reads a crop's mean colour."""

from __future__ import annotations

import json
import sqlite3
import time

import numpy as np
import pytest

from camera_mcp.config import Camera, Site
from camera_mcp.history import History, HistoryError
from camera_mcp.store import Store
from camera_mcp.visits import Visit

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
from camera_mcp import tagmodel  # noqa: E402

DAY = 86400
COLOURS = {"cat": (200, 40, 40), "songbirds": (40, 40, 200), "leaves": (40, 200, 40)}


def colour_featurizer(calls):
    """A stand-in for SpeciesNet: each crop's mean colour (plus a constant), counting the boxes it sees."""
    def featurize(image, boxes):
        a = np.asarray(image, dtype=np.float32) / 255
        out = []
        for x, y, w, h in boxes:
            crop = a[int(y * a.shape[0]):int((y + h) * a.shape[0]), int(x * a.shape[1]):int((x + w) * a.shape[1])]
            out.append([*crop.reshape(-1, 3).mean(axis=0), 1.0])
        calls.append(len(boxes))
        return np.array(out, dtype=np.float32)
    return lambda: featurize


def add_visit(store, db, data, kind, logged, name, t) -> int:
    """One visit with a still: a grey frame with a `kind`-coloured square, one box on it."""
    v = Visit("alley", logged, start=t, last_seen=t + 5, frames=1)
    v.votes[(name, "")] = 1.0
    n = db.execute("SELECT COUNT(*) FROM visits").fetchone()[0] + 1
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    img.paste(COLOURS[kind], (16, 16, 48, 48))
    img.save(data / "stills" / f"{n}.jpg", quality=95)
    vid = store.add_visit(v, model="m", thumbnail=None, clip=None, still=f"stills/{n}.jpg")
    db.execute("INSERT INTO visit_frames VALUES (?, ?, ?, '', 0.9, ?)",
               (vid, f"2026-01-01T00:00:{n % 60:02d}", name,
                json.dumps([{"conf": 0.4 + 0.01 * (n % 40), "bbox": [0.25, 0.25, 0.5, 0.5]}])))
    db.commit()
    return vid


@pytest.fixture
def log(tmp_path):
    """Two weeks of tagged visits on one camera: cats (one a day logged as "bird" and tagged wrong
    species: domestic cat), songbirds, and leaves the owner tagged false."""
    store = Store(tmp_path / "monitor.db")
    db = sqlite3.connect(tmp_path / "monitor.db")
    (tmp_path / "stills").mkdir()
    now = time.time() - 3600
    for day in range(14):
        for kind, logged, name, verdict, species in (
                ("cat", "cat", "domestic cat", "real", None), ("cat", "cat", "domestic cat", "real", None),
                ("cat", "bird", "bird", "wrong_species", "domestic cat"),
                ("songbirds", "songbirds", "house finch", "real", None),
                ("songbirds", "songbirds", "house finch", "real", None),
                ("leaves", "animal", "animal", "false", None), ("leaves", "animal", "animal", "false", None)):
            vid = add_visit(store, db, tmp_path, kind, logged, name, now - (13 - day) * DAY)
            store.set_visit_label(vid, verdict, species=species)
    db.close()
    store.close()
    return History(tmp_path / "monitor.db", {"alley": Camera("alley", "a.mp4")}, Site(timezone="UTC")), tmp_path


def test_examples_carry_the_owners_tags(log):
    history, data = log
    exs, stills, skipped = tagmodel.examples(history, data)
    assert len(exs) == 14 * 7 and len(stills) == 14 * 7 and not skipped
    by = {(e.logged, e.real, e.group) for e in exs}
    assert by == {("cat", True, "cat"), ("bird", True, "cat"), ("songbirds", True, "songbirds"), ("animal", False, None)}


def test_a_run_learns_the_tags_and_beats_the_log(log):
    history, data = log
    calls = []
    out = tagmodel.run(history, data, test_days=4, make_featurizer=colour_featurizer(calls))
    assert sum(calls) == 14 * 7
    js = json.loads((out / "report.json").read_text())
    assert js["test_boxes"] == 4 * 7 and js["group"]["groups"] == ["cat", "songbirds"]
    assert js["real"]["false_caught"] == js["real"]["false_total"] == 8 and js["real"]["real_called_false"] == 0
    assert js["group"]["right"] == js["group"]["total"] == 20
    assert js["group"]["log_right"] == 16 and js["group"]["fixed"] == 4  # the cats logged as birds
    assert js["group"]["by_log"] == {"same": [16, 16], "vague": [4, 4], "other": [0, 0]}  # "bird" is vague
    md = (out / "report.md").read_text()
    assert "tested from" in md and "| cat |" in md
    saved = torch.load(out / "model.pt", weights_only=False)
    assert saved["groups"] == ["cat", "songbirds"] and saved["mean"].shape == (4,)


def test_features_are_cached_by_box(log):
    history, data = log
    calls = []
    tagmodel.run(history, data, test_days=4, make_featurizer=colour_featurizer(calls))
    tagmodel.run(history, data, test_days=4, make_featurizer=colour_featurizer(calls))
    assert sum(calls) == 14 * 7  # the second run featurized nothing


def test_too_few_days_to_hold_a_week_out(log):
    history, data = log
    with pytest.raises(HistoryError, match="more than 20 days"):
        tagmodel.run(history, data, test_days=20, make_featurizer=colour_featurizer([]))


def test_suggest_scores_untagged_visits_and_ranks_them(log):
    """After training, today's untagged visits get a suggestion: a cat the log called "bird" (vague:
    accepting names it, real), a cat the log called songbirds (contradicted: most useful, accepting
    tags it wrong species), a songbird the log got right (accepting tags it real), and leaves the log
    called an animal (the real head says no animal: suggested false)."""
    history, data = log
    tagmodel.run(history, data, test_days=4, make_featurizer=colour_featurizer([]))
    store, db = Store(data / "monitor.db"), sqlite3.connect(data / "monitor.db")
    now = time.time()
    vague = add_visit(store, db, data, "cat", "bird", "bird", now - 300)
    wrong = add_visit(store, db, data, "cat", "songbirds", "house finch", now - 200)
    right = add_visit(store, db, data, "songbirds", "songbirds", "house finch", now - 100)
    leaves = add_visit(store, db, data, "leaves", "animal", "animal", now - 50)
    db.close()
    store.close()
    got = tagmodel.suggest(history, data, since="1h", make_featurizer=colour_featurizer([]))
    v = got["visits"]
    assert set(v) == {str(vague), str(wrong), str(right), str(leaves)}  # the tagged ones aren't scored
    assert v[str(leaves)]["group"] == "false" and tagmodel.accept_label(v[str(leaves)]) == ("false", "")
    assert (v[str(vague)]["group"], v[str(wrong)]["group"], v[str(right)]["group"]) == ("cat", "cat", "songbirds")
    assert v[str(wrong)]["useful"] > 1 > v[str(vague)]["useful"]  # contradicting the log ranks first
    assert tagmodel.accept_label(v[str(vague)]) == ("real", "cat")
    assert tagmodel.accept_label(v[str(wrong)]) == ("wrong_species", "cat")
    assert tagmodel.accept_label(v[str(right)]) == ("real", "")
    assert tagmodel.read_suggestions(data)["model"] == got["model"]


def test_suggest_needs_a_trained_model(log):
    history, data = log
    with pytest.raises(HistoryError, match="no trained model"):
        tagmodel.suggest(history, data, make_featurizer=colour_featurizer([]))
