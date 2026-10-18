# Copyright (c) 2026 Eric Cooper.
"""Phase 2b: labels and groups, the visit tracker, detector helpers, and the monitor with a
stand-in detector (CI doesn't install SpeciesNet). No network or hardware."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess

import cv2
import numpy as np
import pytest

from camera_mcp.config import Camera, MonitorSettings
from camera_mcp.detect import iou, merge_boxes, tile_windows
from camera_mcp.monitor import Monitor
from camera_mcp.taxa import (
    Label, allowed, bird_group_from_scores, expand_taxon, group_of, parse_prediction, visit_key,
)
from camera_mcp.visits import Box, Detection, VisitTracker

CAT = "mammalia;carnivora;felidae;felis;catus"
JAY = "aves;passeriformes;corvidae;aphelocoma;californica"
FINCH = "aves;passeriformes;fringillidae;haemorhous;mexicanus"
CROW = "aves;passeriformes;corvidae;corvus;brachyrhynchos"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")


# ---- labels and groups ----------------------------------------------------------------

def test_parse_prediction():
    assert parse_prediction(f"uuid;{CAT};domestic cat", 0.9) == Label("domestic cat", CAT, 0.9)
    assert parse_prediction("uuid;aves;;;;;bird", 0.7) == Label("bird", "aves;;;;", 0.7)
    assert parse_prediction("uuid;;;;;;blank", 0.9).name == "animal"
    assert parse_prediction("uuid;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result",
                            0.9).name == "animal"
    assert parse_prediction("uuid;mammalia;primates;hominidae;homo;sapiens;human", 0.8).name == "person"


@pytest.mark.parametrize("taxonomy, group", [
    (CROW, "crows"), (JAY, "jays"), (FINCH, "songbirds"), ("aves;columbiformes;columbidae;zenaida;macroura", "doves/pigeons"),
    (CAT, "cat"), ("mammalia;rodentia;muridae;rattus;norvegicus", "rat"), ("mammalia;rodentia;sciuridae;sciurus;griseus", "squirrel"),
    ("mammalia;carnivora;canidae;canis;familiaris", "dog"), ("mammalia;carnivora;canidae;canis;latrans", "coyote"),
    ("aves;;;;", None), ("reptilia;squamata;;;", None),
])
def test_groups_are_checked_in_order(taxonomy, group):
    assert group_of(taxonomy) == group


def test_unsorted_bird_is_sorted_by_summed_scores():
    assert bird_group_from_scores([("aves;;;;", 0.5), (JAY, 0.37), ("aves;passeriformes;corvidae;cyanocitta;stelleri", 0.05)]) == "jays"
    assert bird_group_from_scores([("aves;;;;", 0.74), ("aves;passeriformes;corvidae;pica;pica", 0.11), (JAY, 0.06)]) is None
    assert bird_group_from_scores([(CAT, 0.9)]) is None


def test_allowlist_rolls_unlisted_species_up_to_their_class():
    allow = ("domestic cat", "aves")
    assert allowed(Label("domestic cat", CAT, 0.9), allow).name == "domestic cat"
    assert allowed(Label("california scrub-jay", JAY, 0.8), allow).name == "california scrub-jay"  # under aves
    cattle = allowed(Label("domestic cattle", "mammalia;cetartiodactyla;bovidae;bos;taurus", 0.69), allow)
    assert (cattle.name, cattle.taxonomy, cattle.score) == ("mammal", "mammalia;;;;", 0.69)
    assert allowed(Label("domestic cattle", "mammalia;x;y;z;w", 0.5), ()).name == "domestic cattle"  # no list: anything


def test_visit_keys():
    assert visit_key(Label("domestic cat", CAT)) == "cat"
    assert visit_key(Label("bird", "aves;;;;")) == "bird"
    assert visit_key(Label("bird", "aves;;;;"), bird_group="jays") == "jays"
    assert visit_key(Label("mammal", "mammalia;;;;")) == "mammal"
    assert visit_key(Label("person")) == "person" and visit_key(Label("animal")) == "animal"


def test_expand_taxon():
    assert set(expand_taxon("night-critters")) == {"mammalia;carnivora;procyonidae", "mammalia;carnivora;mephitidae",
                                                   "mammalia;didelphimorphia"}
    assert expand_taxon("bird") == ["aves"] and expand_taxon("songbirds") == ["aves;passeriformes"]


# ---- visit tracker --------------------------------------------------------------------

def det(key, name="x", taxonomy="", score=0.9, conf=0.9, x=0.1):
    return Detection(Box(x, 0.1, 0.1, 0.1, conf), Label(name, taxonomy, score), key)


def test_a_flock_is_one_visit_and_a_cat_among_it_is_another():
    tracker = VisitTracker("feeder", close_after=10)
    birds = [det("songbirds", "house finch", FINCH, x=x / 10) for x in range(6)]
    tracker.feed(0, birds)
    tracker.feed(1, birds[:3] + [det("cat", "domestic cat", CAT)])
    assert set(tracker.open) == {"songbirds", "cat"}
    assert tracker.open["songbirds"].max_count == 6 and tracker.open["cat"].start == 1
    closed = tracker.feed(12, [])
    assert {v.key for v in closed} == {"songbirds", "cat"}


def test_visit_label_is_the_weighted_vote_and_ignores_animal():
    tracker = VisitTracker("c", close_after=5)
    tracker.feed(0, [det("cat", "domestic cat", CAT, score=0.9)])
    tracker.feed(1, [det("animal", "animal", score=0.9, conf=0.95)])  # unnamed: joins the cat visit
    tracker.feed(2, [det("cat", "domestic cat", CAT, score=0.8)])
    (visit,) = tracker.close_all()
    assert visit.key == "cat" and visit.frames == 3 and visit.label.name == "domestic cat" and visit.label.score == 1.0


def test_unsorted_birds_join_and_are_promoted_to_a_bird_group():
    tracker = VisitTracker("c", close_after=5)
    tracker.feed(0, [det("bird", "bird", "aves;;;;")])
    assert set(tracker.open) == {"bird"}
    tracker.feed(1, [det("jays", "california scrub-jay", JAY)])  # now we know: the bird visit becomes jays
    tracker.feed(2, [det("bird", "bird", "aves;;;;")])  # unsorted again: joins the jays
    (visit,) = tracker.close_all()
    assert visit.key == "jays" and visit.start == 0 and visit.frames == 3


def test_thumbnail_is_the_most_confident_box():
    tracker = VisitTracker("c", close_after=5)
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[10:30, 20:40] = 255
    tracker.feed(0, [Detection(Box(0.5, 0.5, 0.1, 0.1, 0.4), Label("domestic cat", CAT, 0.9), "cat")], frame)
    tracker.feed(1, [Detection(Box(0.1, 0.1, 0.1, 0.2, 0.95), Label("domestic cat", CAT, 0.9), "cat")], frame)
    (visit,) = tracker.close_all()
    assert visit.best_conf == 0.95 and visit.thumbnail.mean() > 100  # the white patch


# ---- detector helpers -----------------------------------------------------------------

def test_boxes_from_tiles_are_merged():
    a, b, c = Box(0.1, 0.1, 0.2, 0.2, 0.9), Box(0.11, 0.1, 0.2, 0.2, 0.7), Box(0.6, 0.6, 0.1, 0.1, 0.5)
    assert iou(a, a) == pytest.approx(1.0) and iou(a, c) == 0.0
    assert merge_boxes([(b, "animal"), (a, "animal"), (c, "animal")]) == [(a, "animal"), (c, "animal")]
    person, part = Box(0.5, 0.2, 0.1, 0.5, 0.9), Box(0.5, 0.2, 0.1, 0.2, 0.8)  # a tile cut the person in half
    assert merge_boxes([(person, "person"), (part, "person")]) == [(person, "person")]
    assert len(merge_boxes([(person, "person"), (part, "animal")])) == 2  # a cat at a person's feet stays


def test_tiles_cover_the_frame_with_overlap():
    tiles = tile_windows()
    assert len(tiles) == 4 and tiles[0][:2] == (0.0, 0.0)
    assert all(abs(x + w - 1) < 1e-9 for x, _, w, _ in tiles[1::2])  # right column reaches the edge
    assert tiles[0][2] > 0.5  # tiles overlap


# ---- monitor with a stand-in detector -------------------------------------------------

class WhiteThingDetector:
    """Stands in for SpeciesNet: anything near-white in the frame is `label`."""

    model = "stand-in"

    def __init__(self, label: Label, key: str):
        self.label, self.key = label, key

    def analyze(self, frame, *, tiles=False, min_conf=0.2, max_boxes=8, vehicles=False, drop=None):
        ys, xs = np.nonzero(frame.min(axis=2) > 230)
        if len(xs) < 50:
            return []
        h, w = frame.shape[:2]
        box = Box(xs.min() / w, ys.min() / h, (xs.max() - xs.min()) / w, (ys.max() - ys.min()) / h, 0.9)
        if drop and drop(box):  # like SpeciesNet: dropped where boxes are found, before any label
            return []
        return [Detection(box, self.label, self.key)]


@pytest.fixture
def box_crosses(tmp_path):
    """14 s at 10 fps: gray, a white box moving across from 5 s to 8 s, then gray."""
    path = tmp_path / "box.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=10:d=14",
         "-f", "lavfi", "-i", "color=c=white:s=60x60:r=10:d=14",
         "-filter_complex", "[0][1]overlay=x='100+120*(t-5)':y=150:enable='between(t,5,8)'",
         "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _run(tmp_path, clip, label, key, **settings):
    s = MonitorSettings(pre_seconds=1, post_seconds=1, quiet_seconds=2, visit_close_seconds=3, **settings)
    data = tmp_path / "data"
    Monitor([Camera("feeder", str(clip), monitor=s)], data, WhiteThingDetector(label, key)).run(duration=90)
    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    return data, db


@needs_ffmpeg
def test_monitor_turns_motion_into_a_visit(tmp_path, box_crosses, monkeypatch):
    # SpeciesNet's yolov5 replaces cv2.imwrite and cv2.imread with 2-argument versions on import;
    # thumbnails and reference views must not depend on them.
    def replaced(*args, **kwargs):
        raise TypeError("imwrite() takes 2 positional arguments but 3 were given")

    monkeypatch.setattr(cv2, "imwrite", replaced)
    monkeypatch.setattr(cv2, "imread", replaced)
    data, db = _run(tmp_path, box_crosses, Label("domestic cat", CAT, 0.9), "cat")
    (visit,) = db.execute("SELECT * FROM visits").fetchall()
    assert (visit["label"], visit["grp"], visit["max_count"], visit["model"]) == ("domestic cat", "cat", 1, "stand-in")
    assert visit["frames"] >= 2
    assert (data / visit["thumbnail"]).exists() and (data / visit["clip"]).exists()
    frames = db.execute("SELECT * FROM visit_frames WHERE visit_id = ?", (visit["id"],)).fetchall()
    assert len(frames) == visit["frames"] and frames[0]["label"] == "domestic cat"
    windows = db.execute("SELECT clip FROM motion_windows").fetchall()
    assert windows and all(w["clip"] is None for w in windows)  # with detection on, clips go with visits
    assert db.execute("SELECT state FROM camera_status").fetchone()["state"] == "stopped"
    assert visit["open"] == 0  # written when the cat arrived, completed (not duplicated) when it left
    (span,) = db.execute("SELECT * FROM camera_uptime").fetchall()  # the watched stretch, for coverage
    assert span["started_at"] < visit["started_at"] <= span["ended_at"]


@needs_ffmpeg
def test_people_images_can_be_turned_off_per_camera(tmp_path, box_crosses):
    _, db = _run(tmp_path, box_crosses, Label("person", "mammalia;primates;hominidae;homo;sapiens", 0.9), "person",
                 save_people_images=False)
    (visit,) = db.execute("SELECT * FROM visits").fetchall()
    assert visit["grp"] == "person" and visit["thumbnail"] is None and visit["clip"] is None


def test_an_unidentified_first_sighting_becomes_the_named_visit():
    tracker = VisitTracker("c", close_after=5)
    tracker.feed(0, [det("animal", "animal")])  # first frame: the classifier can't tell yet
    tracker.feed(1, [det("jays", "california scrub-jay", JAY)])
    (visit,) = tracker.close_all()
    assert visit.key == "jays" and visit.start == 0 and visit.label.name == "california scrub-jay"


class SlowDetector(WhiteThingDetector):
    """Like a slow machine: each frame takes seconds."""

    def analyze(self, frame, **kwargs):
        import time

        time.sleep(1.5)
        return super().analyze(frame, **kwargs)


@needs_ffmpeg
def test_a_detector_that_cant_keep_up_skips_old_motion_and_says_so(tmp_path, box_crosses):
    s = MonitorSettings(pre_seconds=1, post_seconds=1, quiet_seconds=2, visit_close_seconds=3, max_defer_seconds=1)
    data = tmp_path / "data"
    Monitor([Camera("feeder", str(box_crosses), monitor=s)], data, SlowDetector(Label("domestic cat", CAT, 0.9), "cat")).run(duration=90)
    db = sqlite3.connect(data / "monitor.db")
    spans = db.execute("SELECT started_at, ended_at FROM analysis_skipped WHERE camera = 'feeder'").fetchall()
    assert spans, "nothing was recorded as skipped"
    assert db.execute("SELECT state FROM camera_status").fetchone()[0] == "stopped"


# ---- learned scenery ------------------------------------------------------------------

def _unnamed(x=0.3, y=0.4):
    return Detection(Box(x, y, 0.03, 0.08, 0.6), Label("animal"), "animal")


def test_a_static_unidentified_box_is_learned_as_scenery_and_dropped(tmp_path):
    from camera_mcp.scenery import Scenery

    scenery = Scenery(tmp_path / "cam.json")
    for t in (0, 150, 300, 450):  # four sightings over 7.5 minutes: not yet scenery
        kept, newly = scenery.filter(t, [_unnamed()])
        assert len(kept) == 1 and not newly
    kept, newly = scenery.filter(660, [_unnamed()])  # fifth, 11 minutes after the first: scenery
    assert len(newly) == 1 and kept == []
    cat = Detection(Box(0.3, 0.4, 0.03, 0.08, 0.9), Label("domestic cat", CAT, 0.9), "cat")
    kept, _ = scenery.filter(700, [_unnamed(), cat, _unnamed(x=0.7)])  # a cat on the same spot stays
    assert [d.key for d in kept] == ["cat", "animal"] and kept[1].box.x == 0.7
    assert Scenery(tmp_path / "cam.json").filter(800, [_unnamed()])[0] == []  # kept across restarts
    assert Scenery(tmp_path / "cam.json").reset() == 1 and not (tmp_path / "cam.json").exists()


def test_scenery_roi_signature_and_legacy_format(tmp_path):
    import json

    from camera_mcp.scenery import Scenery

    path = tmp_path / "cam.json"
    s = Scenery(path, roi=(0.1, 0.1, 0.8, 0.8))
    for t in (0, 150, 300, 450, 660):  # learn a spot under this analysis crop
        s.filter(t, [_unnamed()])
    assert any(sp.learned for sp in s.spots) and path.exists()
    assert Scenery(path, roi=(0.1, 0.1, 0.8, 0.8)).spots  # same roi: spots persist across a restart
    assert Scenery(path, roi=(0.0, 0.0, 0.5, 0.5)).spots == []  # changed roi: a different region, forget them
    # A legacy bare-list file loads as whole-frame (roi None), but not once a crop is configured.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(json.loads(path.read_text())["spots"]))
    assert Scenery(legacy).spots
    assert Scenery(legacy, roi=(0.1, 0.1, 0.8, 0.8)).spots == []


def test_moving_or_brief_unidentified_boxes_are_not_scenery(tmp_path):
    from camera_mcp.scenery import Scenery

    scenery = Scenery(tmp_path / "cam.json")
    for i, t in enumerate(range(0, 1200, 120)):  # an animal wandering: a different place each time
        kept, newly = scenery.filter(t, [_unnamed(x=0.05 + i * 0.08)])
        assert kept and not newly
    brief = Scenery(tmp_path / "brief.json")
    for t in range(0, 60, 10):  # six sightings within one minute: an animal that paused
        kept, newly = brief.filter(t, [_unnamed()])
        assert kept and not newly


def test_a_static_person_box_is_scenery_but_a_person_walking_is_not(tmp_path):
    from camera_mcp.scenery import Scenery

    cloth = Detection(Box(0.8, 0.49, 0.03, 0.1, 0.4), Label("person"), "person")  # a cloth over the deck rail
    scenery = Scenery(tmp_path / "cam.json")
    for t in (0, 200, 400, 600, 800):
        kept, _ = scenery.filter(t, [cloth])
    assert kept == []
    walker = Scenery(tmp_path / "walker.json")
    for i, t in enumerate(range(0, 1200, 120)):
        kept, newly = walker.filter(t, [Detection(Box(0.1 + i * 0.05, 0.4, 0.05, 0.2, 0.9), Label("person"), "person")])
        assert kept and not newly
    bird = Detection(Box(0.8, 0.49, 0.03, 0.1, 0.9), Label("bird", "aves;;;;", 0.8), "bird")
    assert scenery.filter(900, [bird])[0] == [bird]  # a named animal on a scenery spot is never dropped


def test_something_sitting_on_learned_scenery_is_kept(tmp_path):
    """A learned feeder roof: dropped while it looks the same, kept when a bird sits on it."""
    from camera_mcp.scenery import Scenery

    rng = np.random.default_rng(2)
    base = np.clip(rng.normal(120, 30, (360, 640, 3)), 0, 255).astype(np.uint8)
    base[40:70, 230:300] = 40  # the dark roof
    base[40:44, 230:300] = 200  # its bright edge
    roof = Detection(Box(230 / 640, 40 / 360, 70 / 640, 30 / 360, 0.3), Label("animal"), "animal")
    scenery = Scenery(tmp_path / "cam.json")
    for t in (0, 200, 400, 600, 800):
        kept, _ = scenery.filter(t, [roof], base)
    assert kept == []  # learned, and still looks the same
    dimmer = (base * 0.7).astype(np.uint8)
    assert scenery.filter(900, [roof], dimmer)[0] == []  # evening light: the same look, still dropped
    with_bird = base.copy()
    with_bird[44:62, 250:268] = 230  # a pale finch perched on the roof
    assert scenery.filter(1000, [roof], with_bird)[0] == [roof]  # looks different: kept


@pytest.fixture
def busy(tmp_path):
    """60 s of a test pattern with white bars: the stand-in detector sees an animal in every frame."""
    path = tmp_path / "busy.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10:d=60",
                    "-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@needs_ffmpeg
def test_a_visit_open_at_shutdown_is_saved_with_a_capped_clip(tmp_path, busy):
    """Regression: the trial's 21-minute dove visit ended during shutdown and was never saved."""
    s = MonitorSettings(pre_seconds=0, post_seconds=0, quiet_seconds=2, max_clip_seconds=4)
    data = tmp_path / "data"
    Monitor([Camera("tree", str(busy), monitor=s)], data, WhiteThingDetector(Label("bird", "aves;;;;", 0.7), "bird")).run(duration=12)
    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    (visit,) = db.execute("SELECT * FROM visits").fetchall()
    assert visit["open"] == 0 and visit["thumbnail"] and visit["clip"]
    assert db.execute("SELECT COUNT(*) FROM visit_frames WHERE visit_id = ?", (visit["id"],)).fetchone()[0] > 0
    import json as _json
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(data / visit["clip"])],
                         capture_output=True, text=True).stdout
    assert float(_json.loads(dur)["format"]["duration"]) <= 4 + 2 * s.segment_seconds  # capped (segments are 2 s)


def test_a_species_label_on_learned_scenery_is_dropped_only_with_matching_appearance(tmp_path):
    """The crux fix: the feeder cap gets labelled 'opossum' sometimes. Drop it when it looks like
    the learned spot; keep a real animal (different look); keep when there's no frame to judge."""
    import cv2
    import numpy as np
    from camera_mcp.scenery import Scenery

    rng = np.random.default_rng(5)
    base = np.clip(rng.normal(110, 25, (360, 640, 3)), 0, 255).astype(np.uint8)
    base[40:70, 230:300] = np.clip(rng.normal(45, 12, (30, 70, 3)), 0, 255)  # a dark, textured cap
    cap_box = Box(230 / 640, 40 / 360, 70 / 640, 30 / 360, 0.4)
    cap_animal = Detection(cap_box, Label("animal"), "animal")
    opossum = Detection(cap_box, Label("virginia opossum", "mammalia;didelphimorphia;didelphidae;didelphis;virginiana", 0.5), "opossum")

    scenery = Scenery(tmp_path / "cam.json")
    for t in (0, 200, 400, 600, 800):  # learn the cap as scenery (unconfirmed "animal")
        scenery.filter(t, [cap_animal], base)
    assert any(s.learned for s in scenery.spots)
    assert scenery.filter(900, [opossum], base)[0] == []        # same look: the mislabelled cap -> dropped
    real = base.copy(); real[44:64, 250:272] = 210              # something pale on the cap
    assert scenery.filter(1000, [opossum], real)[0] == [opossum]  # different look: a real animal -> kept
    assert scenery.filter(1100, [opossum])[0] == [opossum]        # no frame: can't confirm -> kept (never miss)


def test_scenery_prunes_one_off_detections(tmp_path):
    from camera_mcp.scenery import Scenery, SINGLE_HIT_TTL
    scenery = Scenery(tmp_path / "cam.json")
    scenery.filter(0, [_unnamed(x=0.1)])  # a single stray detection
    assert len(scenery.spots) == 1
    scenery.filter(SINGLE_HIT_TTL + 10, [_unnamed(x=0.9)])  # much later, elsewhere: the stray is pruned
    assert all(abs(s.x - 0.1) > 0.05 for s in scenery.spots) and len(scenery.spots) == 1


def _still_under_box(data, db, roi=None):
    """The saved still's pixels under the visit's best box (roi-mapped to the full frame)."""
    (visit,) = db.execute("SELECT * FROM visits").fetchall()
    assert visit["still"], "every analysed visit gets a still"
    still = cv2.imdecode(np.frombuffer((data / visit["still"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    best = max((b for f in db.execute("SELECT boxes FROM visit_frames WHERE visit_id = ?", (visit["id"],))
                for b in json.loads(f["boxes"])), key=lambda b: b["conf"])
    x, y, w, h = best["bbox"]
    if roi:
        rx, ry, rw, rh = roi
        x, y, w, h = rx + x * rw, ry + y * rh, w * rw, h * rh
    H, W = still.shape[:2]
    return still, still[int((y + h * .25) * H):int((y + h * .75) * H), int((x + w * .25) * W):int((x + w * .75) * W)]


@needs_ffmpeg
def test_the_still_is_the_exact_frame_the_box_came_from(tmp_path, box_crosses):
    """The box moves 12 px per analysed tenth of a second; a still from any other moment misses it.
    (It used to be re-decoded by seeking the segment, whose timestamps don't start at 0: a seek past the
    first frame found nothing, leaving a stale still from an earlier moment, or none.)"""
    data, db = _run(tmp_path, box_crosses, Label("domestic cat", CAT, 0.9), "cat")
    still, under = _still_under_box(data, db)
    assert still.shape[:2] == (360, 640)  # the full frame
    assert under.size and under.min() > 200, "the box's middle on the still is the white square"


@needs_ffmpeg
def test_the_still_is_full_frame_and_aligned_with_a_cropped_scaled_analysis(tmp_path, box_crosses):
    roi = (0.0, 0.25, 1.0, 0.75)  # analyse a band, scaled down: boxes are fractions of that
    s = MonitorSettings(pre_seconds=1, post_seconds=1, quiet_seconds=2, visit_close_seconds=3, analysis_width=320)
    data = tmp_path / "data"
    Monitor([Camera("feeder", str(box_crosses), monitor=s, roi=roi)], data,
            WhiteThingDetector(Label("domestic cat", CAT, 0.9), "cat")).run(duration=90)
    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    still, under = _still_under_box(data, db, roi)
    assert still.shape[:2] == (360, 640)  # uncropped, unscaled
    assert under.size and under.min() > 200


def test_analysis_frame_crops_then_scales():
    from camera_mcp.monitor import analysis_frame
    full = np.zeros((360, 640, 3), np.uint8)
    full[90:, :, 0] = 255  # mark the roi band
    out = analysis_frame(full, (0, 90, 640, 270), (320, 134))
    assert out.shape == (134, 320, 3) and out[..., 0].min() == 255 and out.flags["C_CONTIGUOUS"]
    assert analysis_frame(full, None, None).shape == full.shape  # no roi, no scale: the frame as decoded


# ---- "ignore" exclusion zones act before classification -----------------------------------------

PATH = (0.1, 0.35, 0.8, 0.3)  # covers the white square's whole crossing (centre y = 0.5, x 0.2..0.77)


class IgnoresDrop(WhiteThingDetector):
    """A detector without the drop seam: the analyzer's backstop must still keep zoned boxes out."""

    def analyze(self, frame, *, drop=None, **kwargs):
        return super().analyze(frame, **kwargs)


def _zoned_run(tmp_path, clip, exclude, detector=None):
    s = MonitorSettings(pre_seconds=1, post_seconds=1, quiet_seconds=2, visit_close_seconds=3)
    data = tmp_path / "data"
    det = detector or WhiteThingDetector(Label("domestic cat", CAT, 0.9), "cat")
    Monitor([Camera("feeder", str(clip), monitor=s, exclude=exclude)], data, det).run(duration=90)
    db = sqlite3.connect(data / "monitor.db")
    db.row_factory = sqlite3.Row
    return db.execute("SELECT * FROM visits").fetchall()


@needs_ffmpeg
def test_a_box_in_an_ignore_zone_never_opens_a_visit(tmp_path, box_crosses):
    assert _zoned_run(tmp_path, box_crosses, ((PATH, "ignore"),)) == []  # nothing opened, nothing hidden: nothing at all


@needs_ffmpeg
def test_the_backstop_holds_for_a_detector_without_the_seam(tmp_path, box_crosses):
    det = IgnoresDrop(Label("domestic cat", CAT, 0.9), "cat")
    assert _zoned_run(tmp_path, box_crosses, ((PATH, "ignore"),), det) == []


@needs_ffmpeg
def test_zones_elsewhere_or_hide_unrecognized_leave_the_visit_alone(tmp_path, box_crosses):
    elsewhere = ((0.0, 0.0, 0.2, 0.2), "ignore")
    assert len(_zoned_run(tmp_path / "a", box_crosses, (elsewhere,))) == 1
    (visit,) = _zoned_run(tmp_path / "b", box_crosses, ((PATH, "hide_unrecognized"),))  # an identified cat: never hidden
    assert visit["suppressed"] is None


def test_zone_drop_maps_roi_boxes_to_the_full_frame():
    from camera_mcp.monitor import zone_drop
    assert zone_drop(Camera("c", "x", exclude=((PATH, "hide_unrecognized"),))) is None  # only "ignore" zones drop
    drop = zone_drop(Camera("c", "x", roi=(0.0, 0.25, 1.0, 0.75), exclude=((PATH, "ignore"),)))
    centre_y = (0.5 - 0.25) / 0.75  # full-frame y 0.5, as a fraction of the roi band
    assert drop(Box(0.45, centre_y - 0.02, 0.1, 0.04, 0.9))
    assert not drop(Box(0.45, 0.9, 0.1, 0.04, 0.9))  # near the bottom of the band: outside the zone


def test_speciesnet_drops_zoned_boxes_before_classifying(monkeypatch):
    """The seam itself, with stand-in models (CI doesn't install SpeciesNet): a box in the zone is
    never sent to the classifier; one outside is classified as usual."""
    import sys
    import types
    from collections import namedtuple

    from camera_mcp.config import Site
    from camera_mcp.detect import SpeciesNetDetector
    utils = types.ModuleType("speciesnet.utils")
    utils.BBox = namedtuple("BBox", "xmin ymin width height")
    pil, image = types.ModuleType("PIL"), types.ModuleType("PIL.Image")  # Pillow comes with SpeciesNet, not the base
    image.fromarray = lambda a: types.SimpleNamespace(size=(a.shape[1], a.shape[0]))
    pil.Image = image
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image)
    monkeypatch.setitem(sys.modules, "speciesnet", types.ModuleType("speciesnet"))
    monkeypatch.setitem(sys.modules, "speciesnet.utils", utils)
    classified = []

    class Mega:
        def preprocess(self, image):
            return image

        def predict(self, key, image):
            return {"detections": [{"label": "animal", "conf": 0.7, "bbox": [0.55, 0.18, 0.06, 0.06]},  # the roof
                                   {"label": "animal", "conf": 0.8, "bbox": [0.10, 0.60, 0.08, 0.08]}]}

    class Classifier:
        def preprocess(self, image, bboxes):
            return bboxes[0]

        def batch_predict(self, keys, crops):
            classified.extend(crops)
            return [{"classifications": {"classes": [], "scores": []}} for _ in keys]

    class Ensemble:
        def combine(self, *a):
            return [{"prediction": "1;mammalia;rodentia;sciuridae;sciurus;carolinensis;eastern gray squirrel",
                     "prediction_score": 0.9}]

    det = SpeciesNetDetector(Site())
    det._parts = (Mega(), Classifier(), Ensemble())
    roof = lambda b: 0.5 <= b.x + b.w / 2 <= 0.65 and 0.1 <= b.y + b.h / 2 <= 0.3
    found = det.analyze(np.zeros((90, 160, 3), np.uint8), drop=roof)
    assert [round(c.xmin, 2) for c in classified] == [0.10]  # only the squirrel went to the classifier
    assert [round(d.box.x, 2) for d in found] == [0.10]
    classified.clear()
    assert len(det.analyze(np.zeros((90, 160, 3), np.uint8))) == 2 and len(classified) == 2  # no zone: both


@needs_ffmpeg
def test_an_ignore_zone_with_min_size_drops_big_boxes_and_keeps_small_ones(tmp_path, box_crosses):
    # the white square is 60x60 px of 640x360: its larger side is 60/360 = 0.167 of the frame
    assert _zoned_run(tmp_path / "a", box_crosses, ((PATH, "ignore", 0.1),)) == []  # big enough: dropped
    assert len(_zoned_run(tmp_path / "b", box_crosses, ((PATH, "ignore", 0.2),))) == 1  # smaller than 0.2: kept


def test_zone_drop_measures_size_on_the_full_frame():
    from camera_mcp.monitor import zone_drop
    roi = (0.0, 0.25, 1.0, 0.5)
    drop = zone_drop(Camera("c", "x", roi=roi, exclude=((PATH, "ignore", 0.05),)))
    centre_y = (0.5 - 0.25) / 0.5
    assert drop(Box(0.45, centre_y - 0.06, 0.02, 0.12, 0.9))  # 0.12 of the roi band = 0.06 of the frame
    assert not drop(Box(0.45, centre_y - 0.04, 0.02, 0.08, 0.9))  # 0.04 of the frame: below min_size
    capped = zone_drop(Camera("c", "x", roi=roi, exclude=((PATH, "ignore", 0.0, 0.05),)))
    assert capped(Box(0.45, centre_y - 0.04, 0.02, 0.08, 0.9))  # 0.04 of the frame: below max_size, dropped
    assert not capped(Box(0.45, centre_y - 0.06, 0.02, 0.12, 0.9))  # 0.06: too big for the zone, kept
