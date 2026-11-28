# Copyright (c) 2026 Eric Cooper.
"""ntfy push alerts: settings, the publish request, and the Alerter sending push and email."""

from zoneinfo import ZoneInfo

import pytest

import camera_mcp.push as push
from camera_mcp.alerts import Alerter


@pytest.fixture
def no_file(monkeypatch):
    monkeypatch.setattr(push, "read_credentials_file", lambda: {})
    for k in push.SETTINGS:
        monkeypatch.delenv(k, raising=False)


def test_push_is_off_without_a_topic_and_the_environment_overrides_the_file(no_file, monkeypatch):
    assert push.push_settings() is None
    monkeypatch.setattr(push, "read_credentials_file", lambda: {"NTFY_TOPIC": "from-file", "NTFY_PRIORITY": "max"})
    monkeypatch.setenv("NTFY_TOPIC", "from-env")
    assert push.push_settings() == {"NTFY_TOPIC": "from-env", "NTFY_PRIORITY": "max"}


def test_a_push_with_a_picture_attaches_it_and_carries_the_text_and_link_in_headers():
    req = push.push_request("cat on fence: bird at west_alley, 9:02am", "… (visit 7)",
                            {"NTFY_TOPIC": "yard-x1", "NTFY_TOKEN": "tk_1"}, b"jpeg", "w.jpg", "http://s/?visit=7")
    h = {k.lower(): v for k, v in req.header_items()}
    assert (req.full_url, req.get_method(), req.data) == ("https://ntfy.sh/yard-x1", "PUT", b"jpeg")
    assert h["title"] == "cat on fence: bird at west_alley, 9:02am" and h["priority"] == "high"
    assert h["filename"] == "w.jpg" and h["click"] == "http://s/?visit=7" and h["authorization"] == "Bearer tk_1"
    assert h["message"].startswith("=?UTF-8?B?")  # non-ASCII text survives as an encoded word
    req = push.push_request("t", "plain", {"NTFY_TOPIC": "a/b", "NTFY_SERVER": "https://n.example/"})
    h = {k.lower(): v for k, v in req.header_items()}
    assert (req.full_url, req.get_method(), req.data) == ("https://n.example/a%2Fb", "POST", b"plain")
    assert "click" not in h and "authorization" not in h and "filename" not in h


def test_the_alerter_pushes_and_emails_and_one_failing_doesnt_stop_the_other(monkeypatch):
    import camera_mcp.mailer as mailer
    pushed, mailed = [], []

    def fake_push(title, text, settings, image=None, filename="", link=None):
        pushed.append((title, text, image, filename, link))
        raise push.PushError("down")
    monkeypatch.setattr(push, "send_push", fake_push)
    monkeypatch.setattr(mailer, "send_email", lambda subj, body, settings, image=None, **k: mailed.append(subj))
    a = Alerter(set(), {"SMTP_HOST": "h"}, "r", ZoneInfo("America/Los_Angeles"), "http://s",
                push={"NTFY_TOPIC": "t"})
    a.maybe_area_alert("west_alley", "cat on fence", "bird", 1_700_000_000.0, photo=lambda: (b"c", b"f"), visit=7)
    assert pushed == [("cat on fence: bird at west_alley, 2:13pm", "cat on fence: bird at west_alley, 2:13pm (visit 7)",
                       b"c", "west_alley-2023-11-14-141320.jpg", "http://s/?visit=7&at=2023-11-14T22%3A13%3A20.000Z")]
    assert mailed == ["cat on fence: bird at west_alley, 2:13pm"]  # the push failed; the email still went

    only_push = Alerter({"cat"}, None, None, ZoneInfo("UTC"), push={"NTFY_TOPIC": "t"})
    assert only_push.active and not only_push.emails
    only_push.maybe_alert("west_alley", "cat", 1_700_000_000.0)
    assert len(pushed) == 2 and len(mailed) == 1
