# Copyright (c) 2026 Eric Cooper.
"""The health checks beside the monitor (camera_mcp.watchdog): a camera with no video inside its
watch hours alerts once and again when it's back; paused hours and a just-opened window don't
count; and the midday report keeps its verdicts. Mail is a stand-in."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from camera_mcp import watchdog
from camera_mcp.config import Site
from camera_mcp.history import History
from camera_mcp.store import Store

UTC = ZoneInfo("UTC")
NOON = datetime(2026, 10, 24, 12, 0, tzinfo=UTC).timestamp()  # the decks (19:00-08:00) are paused
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def site(tmp_path, monkeypatch):
    """Two all-day cameras and a night-only deck, with mail settings and an SMS gateway."""
    cfg = tmp_path / "cameras.yaml"
    cfg.write_text("site: {timezone: UTC}\ncameras:\n"
                   "  - {name: backyard, host: 10.0.0.1}\n  - {name: west_alley, host: 10.0.0.2}\n"
                   "  - {name: east_deck, host: 10.0.0.3, active: '19:00-08:00'}\n")
    Store(tmp_path / "monitor.db").close()
    monkeypatch.setattr("camera_mcp.mailer.read_credentials_file", dict)
    for k, v in (("SMTP_HOST", "h"), ("SMTP_FROM", "a@b"), ("SMTP_TO", "me@x"), ("SMS_ALERT_TO", "555@sms")):
        monkeypatch.setenv(k, v)
    sent = []
    monkeypatch.setattr("camera_mcp.mailer.send_email", lambda subject, body, s: sent.append((subject, body, s["SMTP_TO"])))
    return cfg, tmp_path, sent


def frames(data: Path, at: float, **cams):
    """Record each camera's last frame (seconds before `at`) and state, as the monitor would."""
    store = Store(data / "monitor.db")
    for name, (ago, state, error) in cams.items():
        store.set_status(name, state, now=at, last_frame=None if ago is None else at - ago, error=error)
    store.close()


def test_an_outage_alerts_once_and_again_when_its_back(site):
    cfg, data, sent = site
    frames(data, NOON, backyard=(25 * 60, "retrying", "connection refused"), west_alley=(5, "watching", None),
           east_deck=(4 * 3600, "paused", None))  # the deck's silence is its paused hours
    assert watchdog.check_offline(cfg, data, now=NOON) == "No video from backyard since 11:35am (retrying: connection refused)"
    assert sent == [("", "No video from backyard since 11:35am (retrying: connection refused)", "555@sms")]
    assert watchdog.check_offline(cfg, data, now=NOON + 300) == "" and len(sent) == 1  # once per outage
    assert "backyard" in json.loads((data / "watchdog.json").read_text())["offline"]
    frames(data, NOON + 600, backyard=(3, "watching", None))
    assert watchdog.check_offline(cfg, data, now=NOON + 600) == "backyard is back"
    assert json.loads((data / "watchdog.json").read_text())["offline"] == {}


def test_paused_hours_and_a_just_opened_window_do_not_count(site):
    cfg, data, _ = site
    opened = datetime(2026, 10, 24, 19, 10, tzinfo=UTC).timestamp()  # the deck's window opened 10 min ago
    frames(data, opened, backyard=(5, "watching", None), west_alley=(5, "watching", None),
           east_deck=(11 * 3600, "paused", None))
    assert watchdog.check_offline(cfg, data, now=opened) == ""
    later = opened + 30 * 60  # 40 min into its window and still nothing
    frames(data, later, backyard=(5, "watching", None), west_alley=(5, "watching", None))
    assert watchdog.check_offline(cfg, data, now=later).startswith("No video from east_deck since 8:10am")


def test_every_camera_silent_points_at_the_monitor(site):
    cfg, data, _ = site
    frames(data, NOON, backyard=(30 * 60, "watching", None), west_alley=(30 * 60, "watching", None),
           east_deck=(4 * 3600, "paused", None))  # the deck is paused: "every camera" means those due
    assert watchdog.check_offline(cfg, data, now=NOON) == "No video from any camera for 20+ min: is the monitor running?"


def test_a_camera_the_monitor_never_saw(site):
    cfg, data, _ = site
    frames(data, NOON, backyard=(5, "watching", None))
    assert "No video from west_alley yet (never seen by the monitor)" in watchdog.check_offline(cfg, data, now=NOON)


def test_dry_run_sends_and_remembers_nothing(site):
    cfg, data, sent = site
    frames(data, NOON, backyard=(25 * 60, "watching", None), west_alley=(5, "watching", None))
    assert watchdog.check_offline(cfg, data, dry_run=True, now=NOON).startswith("No video from backyard")
    assert not sent and not (data / "watchdog.json").exists()


def test_without_an_sms_gateway_it_emails(site, monkeypatch):
    cfg, data, sent = site
    monkeypatch.delenv("SMS_ALERT_TO")
    frames(data, NOON, backyard=(25 * 60, "watching", None), west_alley=(5, "watching", None))
    watchdog.check_offline(cfg, data, now=NOON)
    assert sent[0][0] == "Cameras: offline" and sent[0][2] == "me@x"


@pytest.mark.parametrize(("dropped", "lag", "verdict"), [(0, 30, "keeping up"), (0, 600, "behind, not dropping"),
                                                         (12, 1800, "DROPPING")])
def test_the_midday_report(tmp_path, dropped, lag, verdict):
    import time
    store = Store(tmp_path / "monitor.db")
    store.add_analysis_stats(time.time() - 600, 5, lag, 1000, dropped, 500.0)
    store.close()
    subject, body = watchdog.midday_report(History(tmp_path / "monitor.db", {}, Site(timezone="UTC")))
    assert subject == f"Yard Sentry: midday health — {verdict}" and "frames analysed: 1,000" in body


def test_the_installer_renders_both_checks():
    out = subprocess.run([str(ROOT / "deploy" / "install-watchdog.sh"), "--dry-run"], capture_output=True,
                         text=True, check=True, env={"PATH": "/usr/bin:/bin", "HOME": "/tmp",
                                                     "CAMERA_MCP_LABEL_PREFIX": "nu.example"}).stdout
    assert "<string>nu.example.watchdog</string>" in out and "<string>offline</string>" in out
    assert "<key>StartInterval</key><integer>300</integer>" in out
    assert "<string>nu.example.health</string>" in out and "<key>Hour</key><integer>12</integer>" in out
    assert "@@" not in out
