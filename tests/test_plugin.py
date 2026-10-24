# Copyright (c) 2026 Eric Cooper.
"""The Claude Code plugin (plugin/): its manifest, MCP launcher and Skills stay true to the code, so
the Skill never names a tool, argument or taxon the server doesn't have."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import anyio
import pytest

from camera_mcp.config import Camera
from camera_mcp.history import History
from camera_mcp.server import build_server
from camera_mcp.taxa import reference_markdown

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin"
SKILLS = sorted((PLUGIN / "skills").glob("*/SKILL.md"))
LAUNCHER = PLUGIN / "scripts" / "launch-server"


def frontmatter(path: Path) -> dict:
    text = path.read_text()
    assert text.startswith("---\n"), path
    head = text.split("---\n", 2)[1]
    return {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in head.splitlines() if ":" in line)}


def test_manifest_marketplace_and_mcp_config_agree():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    mcp = json.loads((PLUGIN / ".mcp.json").read_text())
    assert manifest["name"] == "camera-mcp" == market["plugins"][0]["name"]
    assert (ROOT / market["plugins"][0]["source"]).resolve() == PLUGIN
    server = mcp["mcpServers"]["cameras"]
    assert server["command"] == "${CLAUDE_PLUGIN_ROOT}/scripts/launch-server"
    used = set(re.findall(r"\$\{user_config\.(\w+)\}", json.dumps(server)))
    assert used == set(manifest["userConfig"]) == {"host", "repo_dir"}  # every setting used, none undeclared
    assert os.access(LAUNCHER, os.X_OK)
    assert not (PLUGIN / "bin").exists()  # a top-level bin/ keeps claude.ai and Cowork from installing it


def test_the_taxon_reference_is_generated_from_taxa():
    assert (PLUGIN / "skills" / "camera-history" / "reference.md").read_text() == reference_markdown()


def test_skills_name_only_real_tools_and_arguments(tmp_path):
    cams = {"backyard": Camera("backyard", "b.mp4")}
    tools = {t.name: t for t in anyio.run(build_server(cams, History(tmp_path / "monitor.db", cams), tmp_path).list_tools)}
    calls = 0
    for skill in SKILLS:
        text = skill.read_text()
        for name, args in re.findall(r"`(\w+)\(([^)`]*)\)`", text):
            assert name in tools, f"{skill.parent.name} calls {name}, which the server doesn't have"
            for arg in re.findall(r"(\w+)=", args):
                assert arg in tools[name].input_schema["properties"], f"{skill.parent.name}: {name} has no {arg}"
            calls += 1
        for name in re.findall(r"`(\w+)`", text):
            if name.endswith(("_visits", "_summary", "_status", "_health", "_cameras", "_media")):
                assert name in tools, f"{skill.parent.name} mentions {name}"
    assert calls >= 8  # the examples are actually being checked


def test_skill_frontmatter():
    meta = {p.parent.name: frontmatter(p) for p in SKILLS}
    assert set(meta) == {"camera-history", "overnight", "yard-check", "weekly-digest"}
    for name, m in meta.items():
        assert m["name"] == name and len(m["description"]) > 20
    for name in ("overnight", "yard-check", "weekly-digest"):  # run only when asked (or scheduled)
        assert meta[name]["disable-model-invocation"] == "true"
    assert "disable-model-invocation" not in meta["camera-history"]  # Claude loads this one when it fits


@pytest.fixture
def fake_bin(tmp_path):
    """A stand-in ssh and a stand-in local camera-mcp, each printing what it was run with."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text('#!/bin/sh\necho "ssh $*"\n')
    (bin_dir / "ssh").chmod(0o755)
    home = tmp_path / "home"
    local = home / "projects" / "camera-mcp" / ".venv" / "bin"
    local.mkdir(parents=True)
    (local / "camera-mcp").write_text('#!/bin/sh\necho "local $*"\n')
    (local / "camera-mcp").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(home)}
    return env, home


def launch(env, *args):
    return subprocess.run([str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=10).stdout.strip()


def test_launcher_runs_over_ssh_or_locally(fake_bin):
    env, home = fake_bin
    out = launch(env, "me@studio", "~/projects/camera-mcp")
    assert out == ("ssh -o BatchMode=yes me@studio ~/projects/camera-mcp/.venv/bin/camera-mcp "
                   "--config ~/projects/camera-mcp/cameras.yaml --data-dir ~/projects/camera-mcp/data")  # ~ left for the remote shell
    cam = home / "projects" / "camera-mcp"
    assert launch(env, "", "") == f"local --config {cam}/cameras.yaml --data-dir {cam}/data"
    # settings never saved arrive as their literal ${user_config.*} references: run locally, default folder
    assert launch(env, "${user_config.host}", "${user_config.repo_dir}").startswith("local --config")
    # the environment overrides the settings
    assert launch(env | {"CAMERA_MCP_HOST": "you@box"}, "", "").startswith("ssh -o BatchMode=yes you@box ")
