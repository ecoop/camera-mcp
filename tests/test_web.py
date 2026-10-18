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
    store.add_camera_stats(now - 60, [("backyard", 1, 8.0, 50, 0, 10.0), ("alley", 2, 20.0, 20, 0, 20.0)])
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
    import camera_mcp.web as web
    monkeypatch.setattr(web, "_reload_monitor", lambda: (True, "stubbed"))


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
    store.add_camera_stats(now - 60, [("backyard", 1, 8.0, 50, 0, 10.0), ("alley", 3, 90.0, 20, 2, 20.0)])
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
    store.add_camera_stats(now - 60, [("east_fence", 2, 100.0, 40, 5, 10.0), ("backyard", 0, 5.0, 10, 0, 2.0)])
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


def test_visits_come_before_live_on_the_feeds_view(client):
    page = client.get("/").text
    assert page.index('id="grid"') < page.index('id="live"')


def test_viewer_links_the_full_res_still_not_the_clip(client):
    page = client.get("/").text
    assert ">full-res image</a>" in page and "open raw" not in page
    assert "$('#cliplink').href=(curClip?'/media/visit/'+id+'/frame/boxed'" in page


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

    import camera_mcp.web as web
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

    import camera_mcp.web as web
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
    assert 'class="minsz"' in client.get("/zones").text


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
