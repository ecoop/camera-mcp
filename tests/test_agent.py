# Copyright (c) 2026 Eric Cooper.
"""Scheduled agents (camera_mcp.agent): a job's prompt is its plugin skill, the agent is locked down
like the ask box, and a run writes its report, mails it if asked and counts its spend. The SDK is a
stand-in (no network)."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from camera_mcp import agent
from camera_mcp.store import Store
from camera_mcp.visits import Visit

sdk = pytest.importorskip("claude_agent_sdk")
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A config and a visit log with one skunk, and no key, mail settings or ledger from the host."""
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("site: {timezone: UTC}\ncameras:\n  - {name: east_deck, host: 10.0.0.1, tags: [under_deck]}\n")
    store = Store(tmp_path / "monitor.db")
    v = Visit("east_deck", "skunk", start=time.time() - 3600, last_seen=time.time() - 3500, frames=5)
    v.votes[("striped skunk", "mammalia;carnivora;mephitidae;mephitis;mephitis")] = 1.0
    store.add_visit(v, model="m", thumbnail=None, clip=None)
    store.close()
    for name in ("camera_mcp.ask", "camera_mcp.mailer", "camera_mcp.agent"):
        monkeypatch.setattr(f"{name}.read_credentials_file", dict)
    for k in ("ANTHROPIC_API_KEY", "AGENT_DAILY_USD", "SMTP_HOST", "SMTP_FROM", "SMTP_TO"):
        monkeypatch.delenv(k, raising=False)
    return cfg, tmp_path


def fake_sdk(text="A quiet week: 1 skunk.", cost=0.05, tools=None, subtype="success", is_error=False):
    """A stand-in for claude_agent_sdk.query that records what it was started with."""
    seen = {}

    async def query(prompt, options):
        seen.update(prompt=prompt, options=options)
        yield sdk.SystemMessage(subtype="init", data={"tools": sorted(agent.TOOL_NAMES) if tools is None else tools})
        yield sdk.ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error, num_turns=4,
                                session_id="s", total_cost_usd=cost, result=text)
    return query, seen


def test_a_job_runs_its_skill_locked_down_and_writes_its_report(site):
    cfg, data = site
    query, seen = fake_sdk()
    out = agent.run_job("weekly-digest", cfg, data, query=query)
    assert out.parent == data / "agent" / "weekly-digest" and out.read_text().startswith("A quiet week: 1 skunk.")
    assert seen["prompt"].startswith("(It is now ") and "compared with the 7 days before" in seen["prompt"]
    o = seen["options"]
    assert "Reading the answers honestly" in o.system_prompt  # the camera-history skill guides it
    assert "- east_deck: watched around the clock; tags: under_deck" in o.system_prompt
    assert o.model == "claude-sonnet-5-5" and o.max_budget_usd == pytest.approx(0.25)
    assert o.tools == [] and o.strict_mcp_config and o.setting_sources == [] and o.permission_mode == "dontAsk"
    assert set(o.allowed_tools) == agent.TOOL_NAMES and o.extra_args == {"bare": None, "no-session-persistence": None}
    assert "--metadata-only" in o.mcp_servers["cameras"]["args"]
    assert (data / "agent_usage.json").exists() and '"usd": 0.05' in (data / "agent_usage.json").read_text()


def test_it_fails_closed_and_respects_its_budget(site, monkeypatch):
    cfg, data = site
    query, _ = fake_sdk(tools=[*sorted(agent.TOOL_NAMES), "mcp__cameras__snapshot"])
    with pytest.raises(agent.AgentError, match="snapshot"):
        agent.run_job("weekly-digest", cfg, data, query=query)
    assert not (data / "agent").exists()  # nothing written
    monkeypatch.setenv("AGENT_DAILY_USD", "0.10")
    query, seen = fake_sdk(cost=0.10)
    agent.run_job("weekly-digest", cfg, data, query=query)
    assert seen["options"].max_budget_usd == pytest.approx(0.10)  # what's left today, under the job's own cap
    with pytest.raises(agent.AgentError, match="used up"):
        agent.run_job("weekly-digest", cfg, data, query=fake_sdk()[0])


def test_it_mails_the_report_when_asked(site, monkeypatch):
    cfg, data = site
    sent = []
    monkeypatch.setattr("camera_mcp.mailer.send_email", lambda subject, body, settings: sent.append((subject, body)))
    with pytest.raises(agent.AgentError, match="mail settings"):  # checked before spending anything
        agent.run_job("weekly-digest", cfg, data, email=True, query=fake_sdk()[0])
    for k, v in (("SMTP_HOST", "h"), ("SMTP_FROM", "a@b"), ("SMTP_TO", "c@d")):
        monkeypatch.setenv(k, v)
    agent.run_job("weekly-digest", cfg, data, email=True, query=fake_sdk()[0])
    assert sent and sent[0][0].startswith("Camera weekly digest — ") and sent[0][1] == "A quiet week: 1 skunk."


def test_a_failed_or_cut_short_run(site):
    cfg, data = site
    with pytest.raises(agent.AgentError, match="failed"):
        agent.run_job("weekly-digest", cfg, data, query=fake_sdk(text="", is_error=True, subtype="error_during_execution")[0])
    out = agent.run_job("weekly-digest", cfg, data, query=fake_sdk(text="Partly.", subtype="error_max_budget_usd",
                                                                  is_error=True)[0])
    assert "Stopped early" in out.read_text()


def test_without_a_key_it_does_not_run(site, capsys):
    cfg, data = site
    with pytest.raises(SystemExit) as e:
        agent.main(["weekly-digest", "--config", str(cfg), "--data-dir", str(data)])
    assert e.value.code == 1 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_every_job_has_a_skill():
    for name in agent.JOBS:
        assert agent.skill_text(name) and not agent.skill_text(name).startswith("---")


def test_the_launch_agent_installer_renders_a_weekly_schedule():
    out = subprocess.run([str(ROOT / "deploy" / "install-agent.sh"), "--dry-run", "weekly-digest"],
                         capture_output=True, text=True, check=True).stdout
    assert "<string>weekly-digest</string>" in out and "<string>--email</string>" in out
    assert "<key>Weekday</key><integer>0</integer>" in out and "<key>Hour</key><integer>8</integer>" in out
    assert "@@" not in out
