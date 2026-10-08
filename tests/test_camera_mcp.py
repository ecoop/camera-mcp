# Copyright (c) 2026 Eric Cooper. All rights reserved.
"""Tests run against a generated video file standing in for a camera: no network, no
hardware."""

from __future__ import annotations

import base64
import shutil
import subprocess

import pytest
from mcp import Client

from camera_mcp.capture import grab_frame, reachable, to_jpeg
from camera_mcp.config import Camera, ConfigError, load_cameras, redact
from camera_mcp.server import build_server

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary"),
]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def clip(tmp_path):
    """A 2-second 640x360 test-pattern clip, made with ffmpeg (the same binary capture uses)."""
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10",
         "-t", "2", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture
def config(tmp_path, clip, monkeypatch):
    monkeypatch.setenv("CAMERA_USER", "u")
    monkeypatch.setenv("CAMERA_PASS", "secret")
    path = tmp_path / "cameras.yaml"
    path.write_text(
        f"""
cameras:
  - name: demo
    label: Generated clip
    source: {clip}
  - name: fence
    host: 127.0.0.1:9
  - name: parked
    source: {clip}
    enabled: false
"""
    )
    return path


# ---- config ---------------------------------------------------------------------------

def test_host_builds_rtsp_url_and_endpoint(config):
    cams = load_cameras(config)
    assert cams["fence"].source == "rtsp://u:secret@127.0.0.1:9/stream1"
    assert cams["fence"].endpoint() == ("127.0.0.1", 9)
    assert cams["demo"].endpoint() is None and not cams["parked"].enabled


def test_host_quotes_special_characters_in_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_USER", "cam")
    monkeypatch.setenv("CAMERA_PASS", "p@ss/w:rd")
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    host: 10.0.0.2\n")
    cam = load_cameras(path)["x"]
    assert cam.source == "rtsp://cam:p%40ss%2Fw%3Ard@10.0.0.2/stream1"
    assert cam.endpoint() == ("10.0.0.2", 554)


def test_host_without_credentials_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMERA_USER", raising=False)
    monkeypatch.delenv("CAMERA_PASS", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    host: 10.0.0.2\n")
    with pytest.raises(ConfigError, match="CAMERA_USER and CAMERA_PASS"):
        load_cameras(path)


def test_host_and_source_together_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    host: 10.0.0.2\n    source: rtsp://h/s\n")
    with pytest.raises(ConfigError, match="not both"):
        load_cameras(path)


def test_missing_env_var_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    source: rtsp://${NOPE}@h/s\n")
    with pytest.raises(ConfigError, match="NOPE"):
        load_cameras(path)


def test_wildlife_pipeline_rtsp_url_key_is_accepted(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    rtsp_url: rtsp://h/s\n")
    assert load_cameras(path)["x"].source == "rtsp://h/s"


def test_unquoted_yaml_boolean_name_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: off\n    source: rtsp://h/s\n")  # YAML reads `off` as False
    with pytest.raises(ConfigError, match="quote"):
        load_cameras(path)


def test_redact():
    assert redact("failed rtsp://u:p@10.0.0.2/stream1 x") == "failed rtsp://***@10.0.0.2/stream1 x"


# ---- capture --------------------------------------------------------------------------

def test_grab_and_downscale(clip):
    frame = grab_frame(Camera("demo", str(clip)))
    assert frame.image.shape == (360, 640, 3)
    data, w, h = to_jpeg(frame.image, max_width=320)
    assert (w, h) == (320, 180) and data[:2] == b"\xff\xd8"


def test_reachable(clip):
    assert reachable(Camera("demo", str(clip)))
    assert not reachable(Camera("gone", "rtsp://127.0.0.1:9/s"), timeout=0.5)


# ---- server, in-process over the MCP client -------------------------------------------

async def test_list_cameras_hides_sources(config):
    async with Client(build_server(load_cameras(config))) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["snapshot"].annotations.read_only_hint is True
        result = await client.call_tool("list_cameras", {})
        text = result.content[0].text
        assert "secret" not in text and "rtsp://" not in text
        by_name = {c["name"]: c for c in result.structured_content["result"]}
        assert by_name["demo"]["reachable"] is True
        assert by_name["fence"]["reachable"] is False
        assert by_name["parked"]["reachable"] is False


async def test_snapshot_returns_an_image(config):
    async with Client(build_server(load_cameras(config))) as client:
        result = await client.call_tool("snapshot", {"camera": "demo", "max_width": 320})
        assert not result.is_error
        image, caption = result.content
        assert image.mime_type == "image/jpeg"
        assert base64.b64decode(image.data)[:2] == b"\xff\xd8"
        assert "320x180" in caption.text


async def test_snapshot_errors_are_tool_errors_without_secrets(config):
    async with Client(build_server(load_cameras(config))) as client:
        unknown = await client.call_tool("snapshot", {"camera": "nope"})
        assert unknown.is_error and "known: demo, fence, parked" in unknown.content[0].text
        down = await client.call_tool("snapshot", {"camera": "fence"})
        assert down.is_error and "secret" not in down.content[0].text
        off = await client.call_tool("snapshot", {"camera": "parked"})
        assert off.is_error and "disabled" in off.content[0].text
