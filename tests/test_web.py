# Copyright (c) 2026 Eric Cooper.
import pathlib

import pytest

from camera_mcp.config import load_cameras, load_site

starlette_testclient = pytest.importorskip("starlette.testclient")


@pytest.fixture
def client(tmp_path):
    from camera_mcp.web import build_app
    cams = load_cameras(_cfg(tmp_path), require_login=False)
    app = build_app(cams, tmp_path, load_site(_cfg(tmp_path)))
    return starlette_testclient.TestClient(app)


def _cfg(tmp_path):
    p = tmp_path / "cameras.yaml"
    p.write_text("cameras:\n  - {name: backyard, source: b.mp4}\n  - {name: alley, source: a.mp4}\n")
    return str(p)


def test_index_and_cameras(client):
    r = client.get("/")
    assert r.status_code == 200 and "Yard Sentry" in r.text
    cams = client.get("/api/cameras").json()
    assert {c["name"] for c in cams} == {"backyard", "alley"}


def test_history_endpoints_degrade_without_a_db(client):
    # no monitor.db in the temp dir -> history endpoints answer 503 with an error, not a crash
    assert client.get("/api/health").status_code == 503
    assert client.get("/api/visits").status_code == 503
    assert client.get("/api/stats").status_code == 503
    assert client.get("/media/visit/1/thumb").status_code == 404


def test_visit_boxes_maps_roi_to_full_frame(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit, Detection, Box
    from camera_mcp.taxa import Label
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    cam = Camera(name="fence", source="f.mp4", roi=(0.0, 0.2, 1.0, 0.6))  # analysis crop
    v = Visit("fence", "animal", start=time.time(), last_seen=time.time() + 5, frames=1)
    v.votes[("animal", "")] = 1.0
    v.log = [(v.start + 1, [Detection(Box(0.5, 0.5, 0.2, 0.2, 0.9), Label("animal", "", 0.5), "animal")])]
    store.add_visit(v, model="m", thumbnail=None, clip="c.mp4")
    c = starlette_testclient.TestClient(build_app({"fence": cam}, tmp_path, None))
    d = c.get("/api/visit/1/boxes").json()
    assert d["roi"] == [0.0, 0.2, 1.0, 0.6] and d["has_clip"] is True and d["best"] == 0
    b = d["frames"][0]["boxes"][0]
    # box fraction of the roi crop -> fraction of the full frame
    assert (b["x"], b["y"], b["w"], b["h"]) == (0.5, 0.5, 0.2, 0.12) and b["conf"] == 0.9
    assert c.get("/api/visit/999/boxes").status_code == 404


def _seed_visits(tmp_path):
    """A few closed visits with distinct groups, so /api/species has something to derive."""
    import time
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    store = Store(tmp_path / "monitor.db")
    now = time.time()

    def v(camera, grp, label, taxonomy):
        vis = Visit(camera, grp, start=now - 600, last_seen=now - 300, frames=5)
        vis.votes[(label, taxonomy)] = 1.0
        return vis

    # two squirrels, one raccoon, one finch -> squirrel is most frequent; raccoon pulls in the union
    store.add_visit(v("backyard", "squirrel", "fox squirrel", "mammalia;rodentia;sciuridae;sciurus;niger"),
                    model="m", thumbnail=None, clip=None)
    store.add_visit(v("backyard", "squirrel", "fox squirrel", "mammalia;rodentia;sciuridae;sciurus;niger"),
                    model="m", thumbnail=None, clip=None)
    store.add_visit(v("alley", "raccoon", "northern raccoon", "mammalia;carnivora;procyonidae;procyon;lotor"),
                    model="m", thumbnail=None, clip=None)
    store.add_visit(v("backyard", "songbirds", "house finch", "aves;passeriformes;fringillidae"),
                    model="m", thumbnail=None, clip=None)
    store.close()


def test_species_endpoint_derives_from_the_log(tmp_path):
    from camera_mcp.web import build_app
    _seed_visits(tmp_path)
    cams = load_cameras(_cfg(tmp_path), require_login=False)
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, load_site(_cfg(tmp_path))))
    species = c.get("/api/species").json()["species"]
    # the distinct observed groups, most frequent first, with the applicable union pulled in front
    assert species[0] == "night-critters"  # raccoon was seen
    assert set(species) == {"night-critters", "squirrel", "raccoon", "songbirds"}
    assert species.index("squirrel") < species.index("raccoon")  # squirrel seen twice, raccoon once
    # every value is a taxon the visit filter accepts
    for s in species:
        assert c.get("/api/visits?taxon=" + s).status_code == 200


def test_species_endpoint_empty_without_a_log(client):
    assert client.get("/api/species").json() == {"species": []}


def test_species_endpoint_honors_menu_species(tmp_path):
    from camera_mcp.web import build_app
    _seed_visits(tmp_path)  # observed: squirrel, raccoon, songbirds — but the pinned menu overrides
    p = tmp_path / "pinned.yaml"
    p.write_text("site:\n  menu_species: [cat, owls, person]\n"
                 "cameras:\n  - {name: backyard, source: b.mp4}\n  - {name: alley, source: a.mp4}\n")
    cams = load_cameras(str(p), require_login=False)
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, load_site(str(p))))
    assert c.get("/api/species").json()["species"] == ["cat", "owls", "person"]  # exactly these, in order


def test_derived_species_resolve_as_valid_taxa(tmp_path):
    from camera_mcp.history import History
    from camera_mcp.taxa import expand_taxon
    _seed_visits(tmp_path)
    cams = load_cameras(_cfg(tmp_path), require_login=False)
    history = History(tmp_path / "monitor.db", cams)
    for taxon in history.observed_taxa():
        assert expand_taxon(taxon)  # a group, union or class expands to at least one taxonomy prefix


def test_stats_endpoint_takes_a_camera(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    store.add_analysis_stats(now - 60, 3, 20.0, 70, 0, 30.0)
    store.add_camera_stats(now - 60, [("backyard", 1, 8.0, 50, 0, 10.0, 0), ("alley", 2, 20.0, 20, 0, 20.0, 0)])
    store.close()
    cams = {"backyard": Camera("backyard", "b.mp4"), "alley": Camera("alley", "a.mp4")}
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, None))
    allr = c.get("/api/stats").json()
    assert allr["camera"] is None and allr["points"][0]["frames"] == 70  # aggregate
    one = c.get("/api/stats?camera=alley").json()
    assert one["camera"] == "alley" and one["points"][0]["frames"] == 20 and one["points"][0]["depth"] == 2


def _one_visit(tmp_path):
    """A single stored visit, so label endpoints have something to point at."""
    import time
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit, Detection, Box
    from camera_mcp.taxa import Label
    store = Store(tmp_path / "monitor.db")
    v = Visit("fence", "animal", start=time.time(), last_seen=time.time() + 5, frames=1)
    v.votes[("animal", "")] = 1.0
    v.log = [(v.start + 1, [Detection(Box(0.5, 0.5, 0.2, 0.2, 0.9), Label("animal", "", 0.5), "animal")])]
    store.add_visit(v, model="m", thumbnail=None, clip=None)
    store.close()


def test_a_live_snapshot_is_never_cached(tmp_path):
    # The LIVE tiles refresh by re-requesting the snapshot: a cached picture would show daylight at night.
    import subprocess
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    clip = tmp_path / "f.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=640x360:rate=5:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", str(clip))}, tmp_path, None))
    r = c.get("/api/snapshot/fence?w=480&t=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"


def test_visit_label_stores_and_reads_back(tmp_path):
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    assert c.get("/api/visit/1/label").json() == {"label": None, "editable": True}
    r = c.post("/api/visit/1/label", json={"verdict": "false", "note": "swaying branch"})
    assert r.status_code == 200 and r.json()["ok"] is True
    got = c.get("/api/visit/1/label").json()
    assert got["label"]["verdict"] == "false" and got["label"]["note"] == "swaying branch"
    # re-labelling replaces (upsert on visit_id), and an empty note clears it
    assert c.post("/api/visit/1/label", json={"verdict": "real"}).status_code == 200
    again = c.get("/api/visit/1/label").json()["label"]
    assert again["verdict"] == "real" and again["note"] is None


def test_visit_label_rejects_bad_verdict_and_unknown_visit(tmp_path):
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    assert c.post("/api/visit/1/label", json={"verdict": "maybe"}).status_code == 400
    assert c.post("/api/visit/999/label", json={"verdict": "real"}).status_code == 404


def test_visit_label_blocked_when_readonly(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_MCP_WEB_READONLY", "1")
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    assert c.get("/api/visit/1/label").json()["editable"] is False
    assert c.post("/api/visit/1/label", json={"verdict": "real"}).status_code == 403


@pytest.fixture
def no_reload(monkeypatch):
    # the editor restarts the monitor LaunchAgent on save; stub it in tests
    import camera_mcp.web.zones as zones
    monkeypatch.setattr(zones, "_reload_monitor", lambda: (True, "stubbed"))


def test_zones_page_and_list(client):
    assert client.get("/zones").status_code == 200 and "zones" in client.get("/zones").text
    d = client.get("/api/zones").json()
    assert d["editable"] is True
    assert {c["name"] for c in d["cameras"]} == {"backyard", "alley"}
    assert all(c["roi"] is None and c["motion_mask"] == [] for c in d["cameras"])


def test_zone_save_writes_overlay_and_reloads(tmp_path, client, no_reload):
    import json
    body = {"roi": [0.2, 0.2, 0.5, 0.5], "motion_roi": None, "motion_mask": [[0.0, 0.0, 0.1, 0.1]]}
    r = client.post("/api/zones/backyard", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] and d["reloaded"] and d["scenery_reset"] is True
    assert d["zones"]["roi"] == [0.2, 0.2, 0.5, 0.5]
    # persisted to the app-owned overlay file, and it reloads cleanly through the config parser
    overlay = json.loads((tmp_path / "zones.json").read_text())
    assert overlay["cameras"]["backyard"]["roi"] == [0.2, 0.2, 0.5, 0.5]
    from camera_mcp.config import load_cameras
    cams = load_cameras(_cfg(tmp_path), require_login=False, zones=tmp_path / "zones.json")
    assert cams["backyard"].roi == (0.2, 0.2, 0.5, 0.5)
    # the live app reflects the change immediately
    assert next(c for c in client.get("/api/zones").json()["cameras"]
                if c["name"] == "backyard")["roi"] == [0.2, 0.2, 0.5, 0.5]


def test_zone_save_rejects_bad_rect_and_unknown_camera(client, no_reload):
    assert client.post("/api/zones/backyard", json={"roi": [0.5, 0.0, 0.8, 0.5]}).status_code == 400
    assert client.post("/api/zones/nope", json={"roi": None}).status_code == 404


def test_zone_save_blocked_when_readonly(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_MCP_WEB_READONLY", "1")
    from camera_mcp.web import build_app
    cams = load_cameras(_cfg(tmp_path), require_login=False)
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, load_site(_cfg(tmp_path))))
    assert c.get("/api/zones").json()["editable"] is False
    assert c.post("/api/zones/backyard", json={"roi": None}).status_code == 403


def test_frame_endpoint_404_without_clip(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    v = Visit("cam", "animal", start=time.time(), last_seen=time.time() + 1, frames=1)
    v.votes[("animal", "")] = 1.0
    store.add_visit(v, model="m", thumbnail=None, clip=None)  # no clip -> no full-res frame
    store.close()
    c = starlette_testclient.TestClient(build_app({"cam": Camera("cam", "c.mp4")}, tmp_path, None))
    assert c.get("/media/visit/1/frame").status_code == 404
    assert c.get("/media/visit/999/frame").status_code == 404  # unknown visit


def test_stats_all_returns_each_camera(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    store.add_camera_stats(now - 60, [("backyard", 1, 8.0, 50, 0, 10.0, 0), ("alley", 3, 90.0, 20, 2, 20.0, 0)])
    store.close()
    cams = {"backyard": Camera("backyard", "b.mp4"), "alley": Camera("alley", "a.mp4")}
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, None))
    d = c.get("/api/stats/all").json()
    assert set(d["cameras"]) == {"backyard", "alley"}
    assert d["cameras"]["alley"][0]["lag"] == 90 and d["cameras"]["alley"][0]["dropped"] == 2


def test_frame_prefers_saved_still(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    (tmp_path / "stills" / "cam").mkdir(parents=True)
    (tmp_path / "stills" / "cam" / "s.jpg").write_bytes(b"\xff\xd8 a jpeg")
    v = Visit("cam", "animal", start=time.time(), last_seen=time.time() + 1, frames=1)
    v.votes[("animal", "")] = 1.0
    store.add_visit(v, model="m", thumbnail=None, clip=None, still="stills/cam/s.jpg")
    store.close()
    c = starlette_testclient.TestClient(build_app({"cam": Camera("cam", "c.mp4")}, tmp_path, None))
    r = c.get("/media/visit/1/frame")
    assert r.status_code == 200 and r.content == b"\xff\xd8 a jpeg"   # served the aligned still, no ffmpeg


def test_heatmap_hour_by_camera(tmp_path):
    import time
    from camera_mcp.store import Store
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    store.add_camera_stats(now - 60, [("east_fence", 2, 100.0, 40, 5, 10.0, 0), ("backyard", 0, 5.0, 10, 0, 2.0, 0)])
    store.close()
    cams = {"east_fence": Camera("east_fence", "e.mp4"), "backyard": Camera("backyard", "b.mp4")}
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, None))
    d = c.get("/api/stats/heatmap?since=2h").json()
    assert set(d["cameras"]) == {"east_fence", "backyard"}
    assert len(d["cells"]["east_fence"]) == 24 and d["max_dropped"] == 5
    assert sum(cell["frames"] for cell in d["cells"]["east_fence"]) == 40


def test_visits_hide_gated_ones_unless_asked_and_carry_labels(tmp_path):
    """The labelling flow: gate-hidden visits are counted, listed on ?suppressed=1 with their reason,
    and every visit carries its human verdict so the dashboard can skip to the next unlabelled one."""
    import time

    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    ids = {}
    for name, reason in (("leaf", "stationary"), ("blur", "low_conf"), ("fox", None)):
        vis = Visit("fence", "animal", start=now - 600, last_seen=now - 300, frames=3)
        vis.votes[("animal", "")] = 1.0
        ids[name] = store.add_visit(vis, model="m", thumbnail=None, clip=None, suppressed=reason)
    store.set_visit_label(ids["leaf"], "false")
    store.close()
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    shown = c.get("/api/visits?since=1h").json()
    assert ([v["id"] for v in shown["visits"]], shown["hidden"]) == ([ids["fox"]], 2)
    every = c.get("/api/visits?since=1h&suppressed=1").json()
    got = {v["id"]: (v["suppressed"], v["verdict"]) for v in every["visits"]}
    assert got == {ids["leaf"]: ("stationary", "false"), ids["blur"]: ("low_conf", None), ids["fox"]: (None, None)}
    assert every["hidden"] == 0


def test_dashboard_has_the_hidden_toggle_and_label_queue(client):
    page = client.get("/").text
    assert 'id="showhidden"' in page and "p.set('suppressed','1')" in page
    assert 'id="nextbtn"' in page and "function nextUnlabelled" in page
    assert 'id="hidnote"' in page


def test_the_viewer_has_one_next_through_a_chosen_queue(client):
    """One "next" walks the queue picked beside it (grid / most useful / boxes to tag), remembered; a tag
    moves on through the same queue. The viewer's own tools sit together apart from tagging, and the
    visit-number box is in the filter row, not a block above the grid."""
    page = client.get("/").text
    assert page.count("next &rsaquo;") == 1 and 'id="usebtn"' not in page and 'id="boxqbtn"' not in page
    for q in ('value="grid"', 'value="useful"', 'value="boxes"'):
        assert q in page[page.index('id="qsel"'):page.index('id="nextbtn"')]
    assert "function advance(){if(qsel.value==='useful')nextUseful(); else if(qsel.value==='boxes')nextBoxToTag(); else nextUnlabelled();}" in page
    assert "$('#nextbtn').onclick=advance;" in page and "localStorage.setItem(QKEY,qsel.value)" in page
    vsec = page[page.index('class="vsec"'):page.index('id="boxtag"')]
    assert all(f'id="{b}"' in vsec for b in ("playbtn", "favbtn", "hidebtn", "boxbtn", "cliplink"))
    filters = page[page.index('class="filters"'):page.index('id="view-feeds"')]
    assert 'id="askform"' in filters and 'id="askans"' not in filters


def test_visits_come_before_live_on_the_feeds_view(client):
    page = client.get("/").text
    assert page.index('id="grid"') < page.index('id="live"')


def test_viewer_links_the_full_res_still_not_the_clip(client):
    page = client.get("/").text
    assert ">full-res &nearr;</a>" in page and "open raw" not in page
    assert "$('#cliplink').href='/visit/'+id+'/image'" in page  # a zoomable page over /frame/boxed, not the clip


def test_draw_boxes_outlines_each_box_in_red_at_full_resolution():
    import cv2
    import numpy as np

    from camera_mcp.web import draw_boxes
    grey = np.full((720, 1280, 3), 128, np.uint8)
    ok, jpeg = cv2.imencode(".jpg", grey)
    out = cv2.imdecode(np.frombuffer(draw_boxes(jpeg.tobytes(), [{"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2,
                                                                  "conf": 0.75}]), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == (720, 1280, 3)  # full resolution kept
    b, g, r = (int(v) for v in out[360 + 72, 640])  # on the box's left edge, halfway down
    assert r > 180 and b < 110 and g < 110  # red outline
    assert abs(int(out[100, 100][2]) - 128) < 10  # elsewhere untouched
    assert draw_boxes(b"not a jpeg", []) is None


def test_frame_boxed_draws_the_best_frames_boxes_on_the_saved_still(tmp_path):
    import time

    import cv2
    import numpy as np

    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection, Visit
    from camera_mcp.web import build_app
    (tmp_path / "stills").mkdir()
    ok, jpeg = cv2.imencode(".jpg", np.full((360, 640, 3), 128, np.uint8))
    (tmp_path / "stills" / "s.jpg").write_bytes(jpeg.tobytes())
    store = Store(tmp_path / "monitor.db")
    v = Visit("cam", "animal", start=time.time(), last_seen=time.time() + 5, frames=2)
    v.votes[("animal", "")] = 1.0
    weak, strong = Box(0.1, 0.1, 0.1, 0.1, 0.3), Box(0.6, 0.5, 0.2, 0.2, 0.8)
    v.log = [(v.start + i, [Detection(b, Label("animal", "", 0.5), "animal")]) for i, b in enumerate((weak, strong))]
    store.add_visit(v, model="m", thumbnail=None, clip=None, still="stills/s.jpg")
    store.close()
    c = starlette_testclient.TestClient(build_app({"cam": Camera("cam", "c.mp4")}, tmp_path, None))
    r = c.get("/media/visit/1/frame/boxed")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (360, 640, 3)
    assert img[int(0.6 * 360), int(0.6 * 640)][2] > 180  # the best (0.8) frame's box is drawn...
    assert abs(int(img[int(0.15 * 360), int(0.1 * 640)][2]) - 128) < 15  # ...not the weaker frame's
    assert (tmp_path / "frames" / "1-boxed.jpg").exists()  # cached: boxes never change
    assert c.get("/media/visit/99/frame/boxed").status_code == 404


def test_draw_boxes_keeps_crowded_labels_apart():
    """A flock: four boxes bunched together get four labels, none overprinting another."""
    import cv2
    import numpy as np

    import camera_mcp.web as web
    calls = []
    real = cv2.putText
    ok, jpeg = cv2.imencode(".jpg", np.full((1440, 2560, 3), 128, np.uint8))
    boxes = [{"x": 0.354, "y": 0.4215, "w": 0.03, "h": 0.045, "conf": 0.68},
             {"x": 0.4287, "y": 0.4209, "w": 0.03, "h": 0.045, "conf": 0.5},
             {"x": 0.3798, "y": 0.4535, "w": 0.02, "h": 0.07, "conf": 0.28},
             {"x": 0.4302, "y": 0.4459, "w": 0.015, "h": 0.06, "conf": 0.25}]
    try:
        cv2.putText = lambda img, text, org, *a, **k: (calls.append((text, org, a[1])), real(img, text, org, *a, **k))
        assert web.draw_boxes(jpeg.tobytes(), boxes)
    finally:
        cv2.putText = real
    labels = {}
    for text, (x, y), scale in calls:  # each label is drawn twice (outline, then fill) at one spot
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        labels[text] = (x, y - th, x + tw, y + base)
    assert set(labels) == {"68%", "50%", "28%", "25%"}
    rects = list(labels.values())
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]), labels


def test_labelled_tiles_are_styled_from_their_verdict(client):
    page = client.get("/").text
    assert '<span class="vpill"></span>' in page  # every tile carries the badge slot...
    for v in ("real", "false", "wrong_species"):  # ...shown, outlined and dimmed per verdict
        assert f".tile[data-verdict={v}] .vpill::after" in page
    assert "t.dataset.verdict=v;" in page  # labelling restyles the tile in place


def test_a_visit_can_be_shown_or_hidden_by_hand(tmp_path):
    """The case-by-case undo: show a visit the gate or a zone hid, or hide one, from the viewer."""
    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    store = Store(tmp_path / "monitor.db")
    store.set_visit_suppressed(1, "zone")
    store.close()
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    assert c.get("/api/visits?since=1h").json()["hidden"] == 1
    assert c.post("/api/visit/1/hidden", json={"hidden": False}).json() == {"ok": True, "suppressed": None}
    assert [v["id"] for v in c.get("/api/visits?since=1h").json()["visits"]] == [1]  # shown again
    assert c.post("/api/visit/1/hidden", json={"hidden": True}).json()["suppressed"] == "manual"
    assert c.get("/api/visits?since=1h").json()["hidden"] == 1
    assert c.post("/api/visit/1/hidden", json={"hidden": "yes"}).status_code == 400
    assert c.post("/api/visit/99/hidden", json={"hidden": True}).status_code == 404
    page = c.get("/").text
    assert 'id="hidebtn"' in page and "function toggleHidden" in page


def test_visit_hide_is_blocked_when_readonly(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_MCP_WEB_READONLY", "1")
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    assert c.post("/api/visit/1/hidden", json={"hidden": True}).status_code == 403


def test_zone_editor_saves_exclusion_zones_with_their_mode(tmp_path, client, no_reload):
    import json
    zones = {"roi": None, "motion_roi": None, "motion_mask": [],
             "exclude": [{"rect": [0.35, 0.4, 0.15, 0.15], "mode": "hide_unrecognized"}, {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore"}]}
    r = client.post("/api/zones/backyard", json=zones)
    assert r.status_code == 200 and r.json()["zones"]["exclude"] == zones["exclude"]
    assert json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]["exclude"] == zones["exclude"]
    listed = {c["name"]: c for c in client.get("/api/zones").json()["cameras"]}
    assert listed["backyard"]["exclude"] == zones["exclude"]
    bad = client.post("/api/zones/backyard", json={**zones, "exclude": [{"rect": [0.9, 0.9, 0.5, 0.5]}]})
    assert bad.status_code == 400  # out of bounds: never written
    page = client.get("/zones").text
    assert 'value="exclude"' in page and 'id="l-exclude"' in page


def test_zone_editor_saves_polygons_and_keeps_the_roi_a_rectangle(tmp_path, client, no_reload):
    import json
    poly = {"poly": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.4]]}
    zones = {"roi": [0.0, 0.25, 1.0, 0.75], "motion_roi": poly, "motion_mask": [[0.6, 0.6, 0.1, 0.1], poly],
             "exclude": [{"poly": poly["poly"], "mode": "ignore"}]}
    r = client.post("/api/zones/backyard", json=zones)
    assert r.status_code == 200
    saved = json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]
    assert saved == zones
    listed = {c["name"]: c for c in client.get("/api/zones").json()["cameras"]}["backyard"]
    assert {k: listed[k] for k in zones} == zones
    bad = client.post("/api/zones/backyard", json={**zones, "roi": poly})
    assert bad.status_code == 400  # the analysis roi must stay a rectangle
    page = client.get("/zones").text
    assert 'name="shape" value="poly"' in page and 'id="zin"' in page and "w=1920" in page


def test_zone_modes_saved_under_the_earlier_names_come_back_renamed(client, no_reload):
    old = {"roi": None, "motion_roi": None, "motion_mask": [],
           "exclude": [{"rect": [0.1, 0.1, 0.1, 0.1], "mode": "all"}, {"rect": [0.3, 0.3, 0.1, 0.1], "mode": "unnamed"}]}
    r = client.post("/api/zones/backyard", json=old)
    assert [z["mode"] for z in r.json()["zones"]["exclude"]] == ["ignore", "hide_unrecognized"]
    page = client.get("/zones").text
    assert "hide unrecognized" in page and "<b>Ignore</b>" in page and "everything" not in page


LAUNCHCTL_LIST = """PID\tStatus\tLabel
123\t0\texample.camera-mcp.web
456\t0\texample.camera-mcp
-\t0\texample.camera-mcp.brief
-\t0\texample.camera-mcp.health
789\t0\tcom.apple.something
"""


def test_the_monitor_service_is_found_among_the_loaded_agents(monkeypatch):
    """"apply & reload" must restart the monitor whatever prefix it was installed under: the one
    agent named camera-mcp or *.camera-mcp, not .web/.brief/.health."""
    import subprocess
    from types import SimpleNamespace

    import camera_mcp.web.zones as web
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        out = LAUNCHCTL_LIST if cmd[:2] == ["launchctl", "list"] else ""
        return SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.delenv("CAMERA_MCP_LABEL", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    assert web._monitor_label() == "example.camera-mcp"
    ok, detail = web._reload_monitor()
    assert ok and detail == "restarted example.camera-mcp"
    assert calls[-1][:3] == ["launchctl", "kickstart", "-k"] and calls[-1][3].endswith("/example.camera-mcp")
    monkeypatch.setenv("CAMERA_MCP_LABEL", "set.by.installer")  # an explicit label wins
    assert web._monitor_label() == "set.by.installer"


def test_reload_says_what_to_do_when_there_is_no_monitor_service(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    import camera_mcp.web.zones as web
    monkeypatch.delenv("CAMERA_MCP_LABEL", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: SimpleNamespace(
        returncode=0, stdout="PID\tStatus\tLabel\n-\t0\tcom.apple.something\n", stderr=""))
    ok, detail = web._reload_monitor()
    assert not ok and "CAMERA_MCP_LABEL" in detail and "restart the monitor by hand" in detail


def test_zone_editor_saves_min_size_and_leaves_zones_without_it_alone(tmp_path, client, no_reload):
    import json
    zones = {"roi": None, "motion_roi": None, "motion_mask": [],
             "exclude": [{"poly": [[0.39, 0.15], [0.43, 0.15], [0.43, 0.18]], "mode": "ignore", "min_size": 0.03},
                         {"rect": [0.1, 0.1, 0.1, 0.1], "mode": "hide_unrecognized"}]}
    r = client.post("/api/zones/backyard", json=zones)
    assert r.status_code == 200 and r.json()["zones"]["exclude"] == zones["exclude"]
    saved = json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]["exclude"]
    assert saved == zones["exclude"] and "min_size" not in saved[1]
    bad = client.post("/api/zones/backyard", json={**zones, "exclude": [{**zones["exclude"][0], "min_size": 2}]})
    assert bad.status_code == 400
    page = client.get("/zones").text
    assert 'class="minsz"' in page and 'class="maxsz"' in page and "max_size" in page
    capped = {**zones, "exclude": [{"rect": [0.396, 0.145, 0.011, 0.019], "mode": "ignore", "max_size": 0.04}]}
    r = client.post("/api/zones/backyard", json=capped)
    assert r.status_code == 200 and r.json()["zones"]["exclude"] == capped["exclude"]
    assert json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]["exclude"] == capped["exclude"]
    inverted = {**zones, "exclude": [{"rect": [0.1, 0.1, 0.1, 0.1], "mode": "ignore", "min_size": 0.05, "max_size": 0.03}]}
    assert client.post("/api/zones/backyard", json=inverted).status_code == 400


def test_zone_editor_saves_alert_areas_and_an_older_page_keeps_them(tmp_path, client, no_reload):
    import json
    area = {"rect": [0.1, 0.45, 0.2, 0.12], "name": "cat on fence", "min_size": 0.06, "skip": ["squirrel", "rodent"]}
    zones = {"roi": None, "motion_roi": None, "motion_mask": [], "exclude": [], "alert": [area]}
    r = client.post("/api/zones/backyard", json=zones)
    assert r.status_code == 200 and r.json()["zones"]["alert"] == [area]
    assert json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]["alert"] == [area]
    # a page loaded before alert areas existed sends none: the drawn areas stay
    older = {k: v for k, v in zones.items() if k != "alert"}
    assert client.post("/api/zones/backyard", json=older).json()["zones"]["alert"] == [area]
    assert client.post("/api/zones/backyard", json={**zones, "alert": [{**area, "skip": "squirrel"}]}).status_code == 400
    # deleting the last one is saved as none, not left out (left out would mean "keep cameras.yaml's")
    assert client.post("/api/zones/backyard", json={**zones, "alert": []}).json()["zones"]["alert"] == []
    assert json.loads((tmp_path / "zones.json").read_text())["cameras"]["backyard"]["alert"] == []
    assert "alert" not in json.loads((tmp_path / "zones.json").read_text())["cameras"].get("alley", {})
    assert 'value="alert"' in client.get("/zones").text and 'id="l-alert"' in client.get("/zones").text


def test_zone_outlines_stay_thin_at_any_zoom(client):
    """The zoom scales the whole drawing layer, so outline width (and dashes) divide by the zoom
    level the editor sets on it: 2px on screen whether at 100% or 750%."""
    page = client.get("/zones").text
    assert "stroke-width:calc(2px / var(--z))" in page
    assert "$('#ov').style.setProperty('--z',zs)" in page


def test_analytics_can_show_paused_cameras_and_hides_the_visit_filter(client):
    """Cameras with no data in the window (e.g. paused outside their watch hours) can be shown as
    blank tiles and heatmap rows, remembered per browser; "+ hidden" only filters the feeds grid."""
    page = client.get("/analytics").text
    assert 'id="showidle"' in page and "localStorage.setItem('showIdle'" in page
    assert "CAMS.filter(n=>!names.includes(n))" in page  # grid: idle cameras appended as blank tiles
    assert "CAMS.filter(n=>!cams.includes(n))" in page  # heatmap: idle cameras appended as blank rows
    assert "$('#hidtoglbl').style.display=VIEW==='feeds'?'':'none'" in page  # only the feeds grid uses it


def test_the_map_view_and_its_api(client):
    assert client.get("/map").status_code == 200  # the same single page, opened on the map view
    page = client.get("/").text
    assert 'data-view="map"' in page and 'id="view-map"' in page and "async function loadMap" in page
    assert client.get("/api/labels/map").status_code == 400  # a camera is required
    r = client.get("/api/labels/map?camera=backyard")
    assert r.status_code in (200, 503)  # 503: no visit log yet in this fixture


def test_the_map_zooms_with_dots_that_keep_their_size(client):
    page = client.get("/map").text
    assert 'id="mpz"' in page and "function mapZoomAt" in page and 'id="mzfit"' in page
    assert "transform:scale(var(--iz))" in page  # dots counter-scaled: tight clusters spread apart when zoomed
    assert "if(mJustPanned){ev.stopPropagation()" in page  # a drag to pan isn't a click on a dot
    assert "?w=2560" in page  # the full-resolution picture, so zooming stays sharp


def test_the_map_shows_dots_only_with_their_cameras_picture(client):
    """Dots appear with the picture they belong to: while a new camera's picture loads, the stage keeps
    the picture's shape and shows neither (no squashed dots, no dots over the previous camera); a
    filter change keeps the picture; a slower, superseded request is dropped."""
    page = client.get("/map").text
    assert ".mapstage.loading img,.mapstage.loading #mapdots{visibility:hidden}" in page
    assert "aspect-ratio:var(--ar,16/9)" in page and "loading picture\\2026" in page
    assert "if(seq!==mapSeq)return;" in page  # a newer camera or filter has taken over
    assert "if(mapPic!==pic){mapPic=pic;" in page  # fetched once per camera (and day/night still), not per filter change
    assert "stage.classList.remove('loading'); mapZoomReset();" in page  # shown once the picture is in


def test_the_full_res_image_opens_in_its_own_zoomable_page(client):
    """"full-res image" opens a page that fills the window with the visit's boxed still and zooms on
    scroll or pinch at the pointer, pans on drag, zooms in or fits on a double click or tap (the browser's
    own image view only has a click-to-100% magnifier); the boxes switch off, and the JPEG itself is a
    link away."""
    page = client.get("/").text
    assert "$('#cliplink').href='/visit/'+id+'/image'" in page and 'id="cliplink" target="_blank"' in page
    r = client.get("/visit/42/image")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    img = r.text
    assert "stage.addEventListener('wheel'" in img and "function zoomAt" in img and "fitAll();" in img
    assert "'/frame/boxed':'/frame'" in img and "/thumb'" in img  # boxes switchable; thumbnail-only visits still show
    assert 'id="raw"' in img and "$('#raw').href=img.src" in img
    # touch: two fingers pinch and pan; a double tap zooms in at the spot, or fits once zoomed
    assert "zoomAt(m.d/(pd||m.d),m.x,m.y)" in img and "zoomAt(2.5,e.clientX,e.clientY)" in img
    assert client.get("/visit/x/image").status_code == 404


def test_the_viewer_opens_on_the_full_frame_zooms_by_touch_and_fills_the_screen(client):
    """A visit opens on its full frame whatever its tile says (the grid may have loaded before the clip
    was kept), and the visit's record says whether there's a clip; only with no frame at all does it
    show the thumbnail, scaled up and without boxes. A pinch zooms and pans, a double tap zooms in or
    back out; full screen keeps the tags, and a phone always gets it."""
    page = client.get("/").text
    assert "img.src='/media/visit/'+id+'/frame';" in page and "curClip=!!boxData.has_clip;" in page
    assert "$('#still').classList.add('small');" in page and ".pz img.small{" in page
    assert "if(pts.size===2){pinched=true;" in page and "zoomTo(2.5,...local(e.clientX,e.clientY))" in page
    assert "if(moved>5||pinched||mode!=='image'" in page  # a pinch isn't a tap on a box
    assert 'id="fullbtn"' in page and "$('#clip').classList.toggle('full')" in page
    assert "@media(max-width:760px){.clipx.clipfull{display:none}" in page


def test_python_m_camera_mcp_web_starts_the_dashboard():
    """The web agent runs `python -m camera_mcp.web`: the package's __main__ only calls main(), and no
    module in the package has a main guard of its own (one in the middle of web.py once left a page
    undefined at run time while imports, as in these tests, never noticed). And it really starts."""
    import ast
    import subprocess
    import sys
    from importlib import resources
    pkg = resources.files("camera_mcp.web")
    main_body = ast.parse((pkg / "__main__.py").read_text()).body
    assert ast.unparse(main_body[-1]) == "main()"
    for f in pkg.iterdir():
        if f.name.endswith(".py") and f.name != "__main__.py":
            assert "__main__" not in f.read_text(), f"{f.name} has its own main guard"
    r = subprocess.run([sys.executable, "-m", "camera_mcp.web", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "python -m camera_mcp.web" in r.stdout and "read-only web dashboard" in r.stdout


def test_the_map_shows_day_or_night_apart(client):
    """Day / night (IR) checkboxes beside the label ones; with only one ticked, unknown visits drop out
    and the picture is a recent still in that light (the live view may be the other one)."""
    page = client.get("/map").text
    assert 'class="mdn" value="day" checked' in page and 'class="mdn" value="night" checked' in page
    assert "dn.length===2||dn.includes(lit(v))" in page
    assert "only?[...stills.map(" in page and "'a '+only+' still, #'+v.id" in page
    assert "document.querySelectorAll('.mv,.mdn')" in page


def test_map_dots_grow_a_little_when_zoomed_and_are_easy_to_hit(client):
    """Dots grow far less than the picture (~2x at 800%, so clusters still separate), not a fixed
    on-screen size that looks tiny on a magnified picture, and each has a margin that takes clicks."""
    page = client.get("/map").text
    assert "setProperty('--iz',Math.pow(mz,-0.65))" in page
    assert ".mdot::after{content:'';position:absolute;inset:-6px" in page


def test_tags_carry_a_species_and_boxes_can_be_tagged(tmp_path):
    """The tag API takes a species (not on 'false'); a box is named by its frame time and stored index,
    both in /api/visit/{id}/boxes, and its tag shows there; the revisit list is served."""
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    r = c.post("/api/visit/1/label", json={"verdict": "wrong_species", "species": "norway rat"})
    assert r.status_code == 200 and c.get("/api/visit/1/label").json()["label"]["species"] == "norway rat"
    assert c.post("/api/visit/1/label", json={"verdict": "false", "species": "rat"}).status_code == 400
    b = c.get("/api/visit/1/boxes").json()
    frame = b["frames"][b["best"]]
    box = frame["boxes"][0]
    assert box["i"] == 0 and box["tag"]["source"] == "visit" and b["visit_tag"]["verdict"] == "wrong_species"
    r = c.post("/api/visit/1/box-label", json={"ts": frame["ts"], "box": 0, "verdict": "real", "species": "norway rat"})
    assert r.status_code == 200
    assert c.get("/api/visit/1/boxes").json()["frames"][0]["boxes"][0]["tag"] == \
        {"verdict": "real", "species": "norway rat", "source": "box"}
    assert c.post("/api/visit/1/box-label", json={"ts": frame["ts"], "box": 3, "verdict": "real"}).status_code == 404
    assert c.post("/api/visit/1/box-label", json={"ts": frame["ts"], "box": 0, "verdict": "false",
                                                  "species": "rat"}).status_code == 400
    assert c.post("/api/visit/1/box-label", json={"box": 0}).status_code == 400
    assert c.post("/api/visit/1/box-label", json={"ts": frame["ts"], "box": 0, "verdict": None}).status_code == 200
    assert c.get("/api/labels/needs-box-tags").json() == {"visits": [], "count": 0}



def test_tagging_every_box_on_the_still_tags_the_visit(tmp_path):
    """Two boxes on the still: tagging one leaves the visit alone; tagging the other too tags the
    visit from them (noted "from boxes") and says the visit is complete. A rat and a bird make it
    real with no species (each box keeps its own); both false make it false."""
    import time
    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.taxa import Label
    from camera_mcp.visits import Box, Detection, Visit
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    v = Visit("fence", "animal", start=time.time(), last_seen=time.time() + 5, frames=1)
    v.votes[("animal", "")] = 1.0
    v.log = [(v.start + 1, [Detection(Box(0.1, 0.1, 0.1, 0.1, 0.9), Label("animal", "", 0.5), "animal"),
                            Detection(Box(0.7, 0.7, 0.1, 0.1, 0.6), Label("animal", "", 0.5), "animal")])]
    store.add_visit(v, model="m", thumbnail=None, clip=None)
    store.close()
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    b = c.get("/api/visit/1/boxes").json()
    ts = b["frames"][b["best"]]["ts"]
    r = c.post("/api/visit/1/box-label", json={"ts": ts, "box": 0, "verdict": "real", "species": "rat"}).json()
    assert r["complete"] is False and r["label"] is None and c.get("/api/visit/1/label").json()["label"] is None
    r = c.post("/api/visit/1/box-label", json={"ts": ts, "box": 1, "verdict": "wrong_species", "species": "songbirds"}).json()
    assert r["complete"] is True and (r["label"]["verdict"], r["label"]["species"]) == ("real", None)
    assert c.get("/api/visit/1/label").json()["label"]["note"] == "from boxes"
    assert [t["tag"]["species"] for t in c.get("/api/visit/1/boxes").json()["frames"][0]["boxes"]] == ["rat", "songbirds"]
    c.post("/api/visit/1/box-label", json={"ts": ts, "box": 0, "verdict": "false"})
    r = c.post("/api/visit/1/box-label", json={"ts": ts, "box": 1, "verdict": "false"}).json()
    assert r["complete"] is True and r["label"]["verdict"] == "false"
    assert c.get("/api/visit/1/label").json()["label"]["verdict"] == "false"

def test_the_viewer_tags_single_boxes_and_names_species(client):
    """One set of tag buttons tags the target: the whole visit, or a box picked from the chips or
    clicked on the still (a click, not a drag; the smallest box under the pointer). An untagged
    visit with several boxes starts on its first box; each tag moves to the next, and once all are
    tagged the visit is and the viewer moves on. Boxes are coloured by their tag, dashed when it
    comes from the visit's; wrong sp. waits for its species; the "boxes to tag" queue walks the
    revisit list; suggestions come from /api/labels/species."""
    page = client.get("/").text
    assert 'id="boxtag"' in page and 'id="tgts"' in page and 'id="vsp"' in page and "function untaggedBox" in page
    assert 'id="qopt-boxes"' in page and "function nextBoxToTag" in page and "/api/labels/needs-box-tags" in page
    assert "if(moved>5||pinched||mode!=='image'||!boxesOn||!boxData)return;" in page
    assert ".sort((a,b)=>a.w*a.h-b.w*b.h)[0]" in page
    assert ".boxov rect.inh{stroke-dasharray:6 4}" in page and "t.source==='visit'?' inh'" in page
    assert "if(v!=='wrong_species'||keep)setTimeout(advance,350)" in page
    assert "if(d.complete){" in page and "else if(named&&v)selBox=untaggedBox()||selBox;" in page  # every box tagged: move on
    assert "if(!curLabel&&stillBoxes().boxes.length>1)selBox=untaggedBox();" in page  # several boxes: start on the first
    assert '<datalist id="spplist">' in page and client.get("/api/labels/species").json() == {"species": []}


def test_favourites_can_be_starred_from_the_viewer_and_filtered(tmp_path):
    """POST /api/visit/{id}/favorite stars (still, or clip at a moment) and unstars; ?favorites=1 lists
    them; the page has the star, the tile mark, the filter, and opens a favourite clip at its moment."""
    from camera_mcp.config import Camera
    from camera_mcp.web import build_app
    _one_visit(tmp_path)
    c = starlette_testclient.TestClient(build_app({"fence": Camera("fence", "f.mp4")}, tmp_path, None))
    r = c.post("/api/visit/1/favorite", json={"on": True, "kind": "clip", "clip_at": 7.5})
    assert r.status_code == 200 and r.json()["favorite"] == {"kind": "clip", "clip_at": 7.5}
    (v,) = c.get("/api/visits?favorites=1&since=2h").json()["visits"]
    assert v["id"] == 1 and v["favorite"] == {"kind": "clip", "clip_at": 7.5}
    assert c.post("/api/visit/1/favorite", json={"on": True, "kind": "gif"}).status_code == 400
    assert c.post("/api/visit/999/favorite", json={"on": True}).status_code == 404
    assert c.post("/api/visit/1/favorite", json={"on": False}).status_code == 200
    assert c.get("/api/visits?favorites=1").json()["visits"] == []
    page = c.get("/").text
    assert 'id="favbtn"' in page and 'id="favonly"' in page and "p.set('favorites','1')" in page
    assert 'data-fav="${x.favorite?x.favorite.kind:\'\'}"' in page
    assert "if(curClip&&t.dataset.fav==='clip'){seekTo=" in page and "if(seekTo!=null){try{vid.pause();vid.currentTime=seekTo;}" in page


def test_pages_are_html_files_shipped_with_the_package(client):
    """The dashboard, zone editor and image pages are plain HTML files in camera_mcp/web/pages/ (package
    data, so they ship in the wheel), served as they are: copyright line first, then the doctype."""
    from importlib import resources
    pages = resources.files("camera_mcp.web") / "pages"
    for route, name in (("/", "dashboard.html"), ("/zones", "zones.html"), ("/visit/1/image", "image.html")):
        html = (pages / name).read_text(encoding="utf-8")
        assert html.startswith("<!-- Copyright") and html.splitlines()[1].startswith("<!doctype html>")
        assert client.get(route).text == html


def test_no_page_defines_a_function_twice():
    """In a page's script a second `function f` silently replaces the first (the species menu went
    empty that way when the tag picker's loader took its name)."""
    import re
    from collections import Counter
    from pathlib import Path

    import camera_mcp.web
    pages = Path(camera_mcp.web.__file__).parent / "pages"
    for page in sorted(pages.glob("*.html")):
        names = Counter(re.findall(r"\bfunction\s+(\w+)\s*\(", page.read_text()))
        assert not [n for n, k in names.items() if k > 1], f"{page.name} defines {[n for n, k in names.items() if k > 1]} twice"


def test_tagged_thumbnails_are_not_dimmed():
    """A tagged visit is marked by its outline and badge; its picture stays clear (dimming it made
    the visits the owner had looked at hardest to see)."""
    import re
    from pathlib import Path

    import camera_mcp.web
    css = (Path(camera_mcp.web.__file__).parent / "pages" / "dashboard.html").read_text()
    assert ".tile[data-verdict=real]{outline" in css
    assert not re.search(r"data-verdict[^{]*img\s*\{[^}]*(opacity|filter)", css)


def test_visits_page_back_and_take_a_from_to_range(tmp_path):
    """The feeds grid shows LIMIT visits at a time: ?before=<id> is the next page (no repeats, none
    skipped, `total` still the whole window), and ?until= closes the window at a time in the past."""
    import time
    from datetime import datetime

    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    for i in range(7):  # one a minute, i minutes ago (two share a start, so id breaks the tie)
        vis = Visit("alley", "cat", start=now - 60 * min(i, 5) - 30, last_seen=now - 60 * min(i, 5) - 25, frames=3)
        vis.votes[("domestic cat", "")] = 1.0
        store.add_visit(vis, model="m", thumbnail=None, clip=None)
    store.close()
    c = starlette_testclient.TestClient(build_app({"alley": Camera("alley", "a.mp4")}, tmp_path, None))
    seen, before = [], ""
    while True:
        r = c.get("/api/visits?since=1h&limit=3" + before).json()
        assert r["total"] == 7
        if not r["visits"]:
            break
        seen += [v["id"] for v in r["visits"]]
        before = f"&before={seen[-1]}"
    assert sorted(seen) == list(range(1, 8)) and len(seen) == 7
    until = datetime.fromtimestamp(now - 120).isoformat(timespec="seconds")  # local, as the page sends it
    r = c.get(f"/api/visits?since=1h&until={until}").json()
    assert r["total"] == 5 and r["to"].startswith(until[:16])  # the 2 newest started after `until`


def test_the_feeds_window_can_be_a_range_with_load_more(client):
    html = client.get("/").text
    assert '<option value="range">' in html and 'type="datetime-local" id="from"' in html
    assert 'id="morebtn"' in html and "p.set('before',lastId)" in html
    assert "?since='+" not in html and "since:$('#since').value" not in html  # every view asks via win()


def test_range_dates_show_only_for_from_to(client):
    """The from/until fields are hidden unless the window is "from–to…": their own display rule
    mustn't beat `hidden` (it did, so they showed, unused, beside "24h")."""
    page = client.get("/").text
    assert '<span class="range" id="range" hidden>' in page and ".range[hidden]{display:none}" in page
    assert page.count("$('#range').hidden=!ranged()") == 2  # on every change, and once at load


def test_zone_editor_resizes_rectangles_and_moves_any_zone(client):
    """Rectangles get corner and edge handles (polygons already had point handles), and every zone a
    centre grip that moves it whole, kept inside the frame; all of it edits in place until apply."""
    page = client.get("/zones").text
    assert "function rectHandles(s)" in page and "else rectHandles(s); grip(s);" in page
    assert "function grip(s)" in page and "clamp(p[0]-start[0],-b[0],1-b[0]-b[2])" in page


def test_every_visit_tag_moves_on_to_the_next_visit(client):
    """Real and false move on at once, even when the visit was already tagged (a real box tag makes
    it real first); wrong sp. moves on once it's named."""
    html = client.get("/").text
    assert "if(v!=='wrong_species'||keep)setTimeout(advance" in html and "!had&&" not in html
    assert "v==='wrong_species'&&sp&&VIEW!=='map')setTimeout(advance" in html


def test_the_viewer_shows_the_tag_models_suggestion_and_its_queue(tmp_path):
    """/api/visit/<id>/suggestion gives a visit's suggestion and the tag accepting it records;
    /api/labels/suggested lists the untagged ones, most useful first, leaving out what's been tagged."""
    import json
    import time

    from camera_mcp.config import Camera
    from camera_mcp.store import Store
    from camera_mcp.visits import Visit
    from camera_mcp.web import build_app
    store = Store(tmp_path / "monitor.db")
    ids = []
    for logged in ("bird", "songbirds", "cat"):
        v = Visit("alley", logged, start=time.time() - 600, last_seen=time.time() - 500, frames=1)
        v.votes[(logged, "")] = 1.0
        ids.append(store.add_visit(v, model="m", thumbnail=None, clip=None))
    store.set_visit_label(ids[2], "real")
    store.close()
    (tmp_path / "tagmodel").mkdir()
    s = lambda g, logged, useful: {"group": g, "p": 0.9, "second": "rat", "p2": 0.05, "logged": logged,  # noqa: E731
                                   "boxes": 1, "useful": useful}
    (tmp_path / "tagmodel" / "suggestions.json").write_text(json.dumps({"model": "20261105-1303", "visits": {
        str(ids[0]): s("cat", "bird", 0.2), str(ids[1]): s("cat", "songbirds", 1.2), str(ids[2]): s("cat", "cat", 0.1)}}))
    c = starlette_testclient.TestClient(build_app({"alley": Camera("alley", "a.mp4")}, tmp_path, None))
    one = c.get(f"/api/visit/{ids[1]}/suggestion").json()
    assert one["suggestion"]["group"] == "cat" and one["accept"] == {"verdict": "wrong_species", "species": "cat"}
    assert c.get(f"/api/visit/{ids[0]}/suggestion").json()["accept"] == {"verdict": "real", "species": "cat"}
    assert c.get("/api/visit/999/suggestion").json()["suggestion"] is None
    q = c.get("/api/labels/suggested").json()
    assert [v["id"] for v in q["visits"]] == [ids[1], ids[0]] and q["count"] == 2  # the tagged one is gone
    assert q["visits"][0]["suggestion"]["logged"] == "songbirds"
    html = c.get("/").text
    assert 'id="suggok"' in html and 'id="qopt-useful"' in html and "function advance()" in html


def test_visits_open_by_number_from_the_ask_box_and_a_link(tmp_path):
    """#148: `#3` (or `#3 #1`) in the ask box shows those visits, with no model call, whether or not
    asking is on; /?visit=3 (an alert's link, and the image page's "tag") opens it ready to tag."""
    from camera_mcp.web import build_app
    _seed_visits(tmp_path)
    cams = load_cameras(_cfg(tmp_path), require_login=False)
    c = starlette_testclient.TestClient(build_app(cams, tmp_path, load_site(_cfg(tmp_path))))
    d = c.get("/api/visits/by-id?ids=3,999,1").json()
    assert [v["id"] for v in d["visits"]] == [3, 1]  # in the order asked; unknown ones left out
    assert d["visits"][0]["label"] == "northern raccoon"
    assert c.get("/api/visits/by-id?ids=").json()["visits"] == []
    page = c.get("/").text
    assert '<div class="ask" id="askbox">' in page  # shown even where asking is off
    assert "const ids=visitIds(q); if(ids){await lookUpVisits(ids); return;}" in page  # before any model call
    assert "new URLSearchParams(location.search).get('visit')" in page and "openLinkedVisit();" in page
    assert "$('#tag').href='/?visit='+id" in c.get("/visit/3/image").text
