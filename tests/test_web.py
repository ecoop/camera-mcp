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
