# Copyright (c) 2026 Eric Cooper.
"""The launchd service template renders to a valid, fully-substituted plist."""

from __future__ import annotations

import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "deploy" / "install-service.sh"


def render(tmp_path, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CAMERA_MCP_DATA": "/srv/cam/data", **(env_extra or {})}
    out = subprocess.run(["bash", str(SCRIPT), "--dry-run"], capture_output=True, text=True, env=env, check=True)
    return out.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_template_renders_with_no_placeholders_left(tmp_path):
    rendered = render(tmp_path)
    assert "@@" not in rendered
    plist = plistlib.loads(rendered.encode())
    assert plist["Label"] == "nu.cooper.camera-mcp"
    args = plist["ProgramArguments"]
    assert args[0] == "/usr/bin/caffeinate" and args[2].endswith("/.venv/bin/camera-monitor")
    assert "/srv/cam/data" in args and plist["StandardOutPath"] == "/srv/cam/data/monitor.log"
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] == {"SuccessfulExit": False}
    assert "bin" in plist["EnvironmentVariables"]["PATH"]  # Homebrew/ffmpeg reachable under launchd


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_label_and_data_are_overridable(tmp_path):
    plist = plistlib.loads(render(tmp_path, {"CAMERA_MCP_LABEL": "test.monitor"}).encode())
    assert plist["Label"] == "test.monitor"


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS only")
def test_rendered_plist_passes_plutil(tmp_path):
    out = tmp_path / "r.plist"
    out.write_text(render(tmp_path))
    subprocess.run(["plutil", "-lint", str(out)], check=True, capture_output=True)


BRIEF_SCRIPT = REPO / "deploy" / "install-brief.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_brief_plist_renders_with_time_and_recipient(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CAMERA_MCP_DATA": "/srv/cam/data",
           "CAMERA_MCP_BRIEF_TIME": "06:30"}
    out = subprocess.run(["bash", str(BRIEF_SCRIPT), "--dry-run", "+15551234567"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "@@" not in out
    plist = plistlib.loads(out.encode())
    assert plist["StartCalendarInterval"] == {"Hour": 6, "Minute": 30}
    assert plist["ProgramArguments"][1] == "+15551234567"
    assert plist["ProgramArguments"][0].endswith("deploy/morning-brief.sh")
    assert plist["EnvironmentVariables"]["CAMERA_MCP_DATA"] == "/srv/cam/data"
    assert plist["EnvironmentVariables"]["CAMERA_MCP_BRIEF_SINCE"] == "14h"  # default window
