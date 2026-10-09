# Copyright (c) 2026 Eric Cooper.
"""Tests run against a generated video file standing in for a camera: no network, no
hardware."""

from __future__ import annotations

import base64
import shutil
import subprocess

import pytest
from mcp import Client

from camera_mcp import config as camera_config
from camera_mcp.capture import grab_frame, reachable, to_jpeg
from camera_mcp.config import Camera, ConfigError, load_cameras, redact
from camera_mcp.probe import ProbeError, probe_stream
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
  - name: west_alley
    source: {clip}
    tags: [feeder, dog_yard]
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


def _no_env_login(monkeypatch, keychain=None):
    monkeypatch.delenv("CAMERA_USER", raising=False)
    monkeypatch.delenv("CAMERA_PASS", raising=False)
    monkeypatch.setattr(camera_config, "_keychain_login", lambda: keychain)


def test_host_without_credentials_is_an_error(tmp_path, monkeypatch):
    _no_env_login(monkeypatch)
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    host: 10.0.0.2\n")
    with pytest.raises(ConfigError, match="CAMERA_USER and CAMERA_PASS, or store it in the macOS Keychain"):
        load_cameras(path)


def test_login_comes_from_the_keychain_when_the_environment_has_none(tmp_path, monkeypatch):
    _no_env_login(monkeypatch, keychain=("cam", "p@ss"))
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: x\n    host: 10.0.0.2\n")
    assert load_cameras(path)["x"].source == "rtsp://cam:p%40ss@10.0.0.2/stream1"
    monkeypatch.setenv("CAMERA_USER", "envuser")
    monkeypatch.setenv("CAMERA_PASS", "envpass")
    assert load_cameras(path)["x"].source == "rtsp://envuser:envpass@10.0.0.2/stream1"  # the environment wins


def test_keychain_lookup_parses_security_output(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "-w" in args:
            return subprocess.CompletedProcess(args, 0, stdout="s3cret pass\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout='keychain: "login.keychain-db"\nattributes:\n'
                                           '    "acct"<blob>="tapo-admin"\n    "svce"<blob>="camera-mcp"\n', stderr="")

    camera_config._keychain_login.cache_clear()
    monkeypatch.setattr(camera_config.shutil, "which", lambda name: "/usr/bin/security")
    monkeypatch.setattr(camera_config.subprocess, "run", fake_run)
    try:
        assert camera_config._keychain_login() == ("tapo-admin", "s3cret pass")
        assert all("-s" in c and "camera-mcp" in c for c in calls)
        camera_config._keychain_login.cache_clear()
        monkeypatch.setattr(camera_config.subprocess, "run",
                            lambda args, **kw: subprocess.CompletedProcess(args, 44, stdout="", stderr="not found"))
        assert camera_config._keychain_login() is None
    finally:
        camera_config._keychain_login.cache_clear()


async def test_server_without_a_login_still_answers_but_snapshot_says_why(tmp_path, monkeypatch):
    _no_env_login(monkeypatch)
    path = tmp_path / "c.yaml"
    path.write_text("cameras:\n  - name: fence\n    host: 127.0.0.1:9\n")
    cams = load_cameras(path, require_login=False)
    assert cams["fence"].needs_login and cams["fence"].source == "rtsp://127.0.0.1:9/stream1"
    async with Client(build_server(cams)) as client:
        assert not (await client.call_tool("list_cameras", {})).is_error
        snap = await client.call_tool("snapshot", {"camera": "fence"})
        assert snap.is_error and "macOS Keychain" in snap.content[0].text


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


def test_substream_only_for_tapo_rtsp(config):
    cams = load_cameras(config)
    assert cams["fence"].substream().source == "rtsp://u:secret@127.0.0.1:9/stream2"
    assert cams["demo"].substream() is None


# ---- probe ----------------------------------------------------------------------------

def test_probe_reads_stream_header(clip):
    info = probe_stream(Camera("demo", str(clip)))
    assert (info.codec, info.width, info.height, info.fps) == ("h264", 640, 360, 10.0)


def test_probe_failure_is_redacted(config):
    fence = load_cameras(config)["fence"]  # port 9: nothing listens
    with pytest.raises(ProbeError) as e:
        probe_stream(fence, timeout=2)
    assert "secret" not in str(e.value)


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
        assert unknown.is_error and "known: demo, fence, parked, west_alley" in unknown.content[0].text
        down = await client.call_tool("snapshot", {"camera": "fence"})
        assert down.is_error and "secret" not in down.content[0].text
        off = await client.call_tool("snapshot", {"camera": "parked"})
        assert off.is_error and "disabled" in off.content[0].text


async def test_snapshot_matches_names_loosely_and_lists_tags(config):
    async with Client(build_server(load_cameras(config))) as client:
        result = await client.call_tool("snapshot", {"camera": "West Alley", "max_width": 320})
        assert not result.is_error and result.content[1].text.startswith("west_alley ")
        cams = (await client.call_tool("list_cameras", {})).structured_content["result"]
        assert {c["name"]: c["tags"] for c in cams}["west_alley"] == ["feeder", "dog_yard"]
