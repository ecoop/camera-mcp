# Copyright (c) 2026 Eric Cooper.
"""The dashboard's ask box: Claude over the read-only history tools, with a stand-in client (no network)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace as NS

import pytest

from camera_mcp.ask import FALLBACK_BETA, MODEL, Asker, AskError, Budget, cost_of
from camera_mcp.config import Camera, Site
from camera_mcp.history import History
from camera_mcp.store import Store
from camera_mcp.visits import Visit

SKUNK = ("striped skunk", "mammalia;carnivora;mephitidae;mephitis;mephitis")
CAT = ("domestic cat", "mammalia;carnivora;felidae;felis;catus")


def usage(inp=1000, out=100, read=0, write=0):
    return NS(input_tokens=inp, output_tokens=out, cache_read_input_tokens=read, cache_creation_input_tokens=write)


def tool_call(name, args, tid="t1"):
    return NS(content=[NS(type="tool_use", id=tid, name=name, input=args)], stop_reason="tool_use",
              usage=usage(write=3000), model=MODEL)


def answer(text, stop="end_turn"):
    return NS(content=[NS(type="text", text=text)], stop_reason=stop, usage=usage(read=3000), model=MODEL)


class FakeClient:
    """Plays back scripted responses and records every request."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: o.__dict__)))
        return self.responses.pop(0)


@pytest.fixture
def log(tmp_path):
    store = Store(tmp_path / "monitor.db")
    now = time.time()
    ids = []
    for cam, key, label, ago in (("east_deck", "skunk", SKUNK, 7200), ("west_deck", "skunk", SKUNK, 3600),
                                 ("backyard", "cat", CAT, 1800)):
        v = Visit(cam, key, start=now - ago, last_seen=now - ago + 60, frames=5)
        v.votes[label] = 1.0
        ids.append(store.add_visit(v, model="m", thumbnail=f"media/{cam}/x.jpg", clip=f"media/{cam}/x.mp4"))
    store.close()
    cams = {n: Camera(n, "x", tags=t) for n, t in (("east_deck", ("under_deck",)), ("west_deck", ()), ("backyard", ("feeder",)))}
    return History(tmp_path / "monitor.db", cams, Site(timezone="UTC")), ids, tmp_path


def asker(log, client, cap=1.0):
    history, _, tmp = log
    return Asker(history, Budget(tmp / "ask_usage.json", cap), client=client)


def test_a_question_is_answered_with_the_history_tools_and_its_visits_linked(log):
    _, (sk1, sk2, cat), _ = log
    client = FakeClient(tool_call("find_visits", {"since": "24h", "taxon": "skunk"}),
                        answer(f"Two skunks: [#{sk2}] on west_deck and [#{sk1}] on east_deck. Also [#99999]."))
    a = asker(log, client).ask("all skunks in the last 24 hours?")
    assert [v["id"] for v in a.visits] == [sk2, sk1]  # in the answer's order; an id it never saw is dropped
    assert a.visits[0]["has_thumbnail"] and a.visits[0]["label"] == "striped skunk"
    first, second = client.calls
    assert first["model"] == MODEL and first["output_config"] == {"effort": "low"}
    assert first["fallbacks"] == "default" and first["betas"] == [FALLBACK_BETA]
    assert {t["name"] for t in first["tools"]} == {"find_visits", "activity_summary", "monitor_status", "analysis_health"}
    assert "east_deck (tags: under_deck)" in first["system"][0]["text"]  # the cameras are in the cached prompt
    result = second["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and not result["is_error"]
    payload = json.loads(result["content"])
    assert {v["id"] for v in payload["visits"]} == {sk1, sk2}  # only skunks; the cat isn't asked about
    assert "media/" not in result["content"]  # metadata only: no thumbnail or clip paths leave the machine


def test_cost_is_counted_and_the_daily_cap_stops_a_question(log):
    one_call = cost_of(usage(write=3000), MODEL)
    assert one_call == pytest.approx((1000 * 4 + 100 * 20 + 3000 * 5) / 1e6)
    client = FakeClient(tool_call("monitor_status", {}), answer("never reached"))
    a = asker(log, client, cap=one_call / 2).ask("is everything being watched?")
    assert len(client.calls) == 1  # over the cap after the first call: no second one
    assert "budget is used up" in a.text and a.spent_today == pytest.approx(one_call, rel=1e-3)
    with pytest.raises(AskError) as e:
        asker(log, FakeClient(), cap=one_call / 2).ask("again?")
    assert e.value.status == 429


def test_refusals_tool_errors_and_bad_questions(log):
    a = asker(log, FakeClient(answer("", stop="refusal"))).ask("something it declines")
    assert a.refused and "can't help" in a.text
    client = FakeClient(tool_call("find_visits", {"since": "24h", "cameras": ["no_such_camera"]}), answer("No such camera."))
    asker(log, client).ask("visits on the garage camera?")
    err = client.calls[1]["messages"][-1]["content"][0]
    assert err["is_error"] and "unknown camera" in err["content"]
    for bad in ("", "x" * 501):
        with pytest.raises(AskError) as e:
            asker(log, FakeClient()).ask(bad)
        assert e.value.status == 400


def test_without_a_key_asking_is_off(log, monkeypatch):
    import camera_mcp.ask as ask
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ask, "read_credentials_file", lambda: {})
    a = asker(log, None)
    assert not a.enabled
    with pytest.raises(AskError) as e:
        a.ask("anything?")
    assert e.value.status == 503


def test_the_dashboard_ask_endpoint(log, monkeypatch):
    starlette_testclient = pytest.importorskip("starlette.testclient")
    import camera_mcp.ask as ask
    from camera_mcp.web import build_app
    history, (sk1, _, _), tmp = log
    fake = FakeClient(tool_call("find_visits", {"since": "24h", "taxon": "skunk"}), answer(f"One I'd point at: [#{sk1}]."))
    monkeypatch.setattr(ask.Asker, "client", lambda self: fake)
    monkeypatch.setattr(ask, "api_key", lambda: "test-key")
    c = starlette_testclient.TestClient(build_app(history.cameras, tmp, Site(timezone="UTC")))
    status = c.get("/api/ask").json()
    assert status["enabled"] and status["cap"] == 1.0
    r = c.post("/api/ask", json={"question": "skunks?"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["answer"].endswith(f"[#{sk1}].") and [v["id"] for v in d["visits"]] == [sk1] and d["cost"] > 0
    assert c.post("/api/ask", json={"question": ""}).status_code == 400
    page = c.get("/").text
    assert 'id="askbox"' in page and "function tileHTML" in page
