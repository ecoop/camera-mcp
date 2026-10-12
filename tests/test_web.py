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
    assert client.get("/media/visit/1/thumb").status_code == 404
