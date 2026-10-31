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
from camera_mcp.server import VISIT_FIELDS
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
    cams = {n: Camera(n, "x", tags=t, active=w) for n, t, w in (("east_deck", ("under_deck",), "19:00-07:00"),
                                                                 ("west_deck", (), "19:00-07:00"), ("backyard", ("feeder",), ""))}
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
    system = first["system"][0]["text"]  # the cameras, with their watch hours, are in the cached prompt
    assert "- east_deck: watched 19:00-07:00 local, paused otherwise; tags: under_deck" in system
    assert "- backyard: watched around the clock; tags: feeder" in system
    assert "isn't offline" in system and "weak hint" in system  # paused-by-design and the moved flag
    result = second["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and not result["is_error"]
    payload = json.loads(result["content"])
    assert {v["id"] for v in payload["visits"]} == {sk1, sk2}  # only skunks; the cat isn't asked about
    assert "media/" not in result["content"]  # metadata only: no thumbnail or clip paths leave the machine


def test_the_prompts_are_markdown_files_shipped_with_the_package(log):
    """The server's instructions and the ask box's style are Markdown files in camera_mcp/prompts/
    (package data, so they ship in the wheel): copyright line first, which the model never sees.
    The server sends the one, and the ask box's cached prompt carries both."""
    from importlib import resources

    from camera_mcp.ask import STYLE, guidance
    from camera_mcp.server import INSTRUCTIONS, build_server

    history, _, _ = log
    prompts = resources.files("camera_mcp") / "prompts"
    for name, text in (("server_instructions.md", INSTRUCTIONS), ("ask_style.md", STYLE)):
        raw = (prompts / name).read_text(encoding="utf-8")
        assert raw.startswith("<!-- Copyright") and raw.endswith(text) and text.strip()
        assert "Copyright" not in text and not text.startswith("\n")
    assert INSTRUCTIONS.startswith("Home cameras watched") and STYLE.startswith("You answer questions")
    assert build_server(history.cameras, history).instructions == INSTRUCTIONS
    assert guidance(history).endswith("tags: feeder\n\n" + STYLE)  # the camera list, a blank line, then the style
    client = FakeClient(answer("Nothing yet."))
    asker(log, client).ask("anything?")
    assert client.calls[0]["system"][0]["text"] == INSTRUCTIONS + "\n" + guidance(history)


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


def test_the_box_renders_a_safe_subset_of_markdown(log, monkeypatch):
    starlette_testclient = pytest.importorskip("starlette.testclient")
    from camera_mcp.web import build_app
    history, _, tmp = log
    page = starlette_testclient.TestClient(build_app(history.cameras, tmp, Site(timezone="UTC"))).get("/").text
    assert "function renderAnswer" in page and "esc(text)" in page  # escaped first, then bold/bullets/links


def test_the_ask_box_uses_the_mcp_servers_own_tools(log):
    """One definition: the ask box offers the MCP server's metadata-only tools with the server's own
    names, descriptions and schemas (group_by's five choices included), calls them through it, gets
    list results whole (the server wraps them), and passes errors on without the server's prefix."""
    import anyio

    from camera_mcp.ask import ASK_TOOLS
    a = asker(log, FakeClient())
    server = {t.name: t for t in anyio.run(a.server.list_tools)}
    assert [t["name"] for t in a.tools()] == list(ASK_TOOLS) and "snapshot" not in ASK_TOOLS
    assert set(server) == set(ASK_TOOLS) and not anyio.run(a.server.list_resources)  # metadata-only: nothing else
    for t in a.tools():
        assert t["description"] == server[t["name"]].description and t["input_schema"] == server[t["name"]].input_schema
    assert a.tools()[1]["input_schema"]["properties"]["group_by"]["enum"] == ["species", "group", "camera", "hour", "day"]
    status = json.loads(a.run_tool("monitor_status", {}, {}))
    assert sorted(c["camera"] for c in status) == ["backyard", "east_deck", "west_deck"]  # every camera, not the first
    with pytest.raises(Exception) as e:
        a.run_tool("find_visits", {"since": "24h", "cameras": ["garage"]}, {})
    assert str(e.value).startswith("unknown camera") and "Error executing tool" not in str(e.value)


def test_the_ask_box_keeps_its_own_visit_limit(log, monkeypatch):
    """find_visits here (the server built metadata-only) defaults to 20 visits and never asks for more
    than 50, whatever Claude sends; its visits carry metadata only."""
    a = asker(log, FakeClient())
    sent = []
    real = a.history.find_visits

    def find_visits(*args):
        sent.append(args[7])  # limit
        return real(*args)
    monkeypatch.setattr(a.history, "find_visits", find_visits)
    seen = {}
    for limit in (None, 5, 500):
        out = json.loads(a.run_tool("find_visits", {"since": "24h"} | ({"limit": limit} if limit else {}), seen))
    assert sent == [20, 5, 50]
    assert out["visits"] and set(out["visits"][0]) == {*VISIT_FIELDS, "clip"} and seen


# The same box on the Agent SDK (ASK_BACKEND=agent), with a stand-in for claude_agent_sdk.query.

def agent_run(log, *script, cap=1.0, tools=None, result=None, spent=0.0):
    """An AgentAsker whose SDK plays `script` (assistant turns and tool results), offering `tools`
    at init, then ending with `result`; returns the asker and the options it was started with."""
    sdk = pytest.importorskip("claude_agent_sdk")
    from camera_mcp.ask import AgentAsker
    history, _, tmp = log
    seen = {}

    async def query(prompt, options):
        seen.update(prompt=prompt, options=options)
        names = tools if tools is not None else [f"mcp__cameras__{n}" for n in
                                                 ("find_visits", "activity_summary", "monitor_status", "analysis_health")]
        yield sdk.SystemMessage(subtype="init", data={"tools": names})
        for m in script:
            yield m
        yield result or sdk.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                                          num_turns=2, session_id="s", total_cost_usd=0.03, result="done")
    budget = Budget(tmp / "ask_usage.json", cap)
    budget.add(spent)
    return AgentAsker(history, budget, tmp / "cameras.yaml", tmp, query=query), seen


def find_turn(log, args, tid="t1"):
    """find_visits called with `args`, and its result as the metadata-only server returns it."""
    import anyio
    sdk = pytest.importorskip("claude_agent_sdk")
    from camera_mcp.server import build_server
    history, _, tmp = log
    r = anyio.run(build_server(history.cameras, history, tmp, metadata_only=True).call_tool, "find_visits", args)
    return (sdk.AssistantMessage(content=[sdk.ToolUseBlock(id=tid, name="mcp__cameras__find_visits", input=args)],
                                 model=MODEL),
            sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id=tid, content=[
                {"type": "text", "text": json.dumps(r.structured_content)}], is_error=False)]))


def done(text, cost=0.03, subtype="success", stop="end_turn", is_error=False):
    sdk = pytest.importorskip("claude_agent_sdk")
    return sdk.ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error, num_turns=2,
                             session_id="s", total_cost_usd=cost, result=text, stop_reason=stop)


def test_the_agent_backend_answers_and_links_its_visits(log):
    _, (sk1, sk2, _), _ = log
    a, seen = agent_run(log, *find_turn(log, {"since": "24h", "taxon": "skunk"}),
                        result=done(f"Two skunks: [#{sk2}] and [#{sk1}]. Also [#99999].", cost=0.04))
    ans = a.ask("all skunks in the last 24 hours?")
    assert [v["id"] for v in ans.visits] == [sk2, sk1] and ans.visits[0]["has_thumbnail"]  # unseen ids dropped
    assert ans.cost == 0.04 and a.budget.spent() == pytest.approx(0.04) and ans.rounds == 2
    assert seen["prompt"].startswith("(It is now ") and seen["prompt"].endswith("all skunks in the last 24 hours?")


def test_the_agent_backend_is_locked_down(log):
    """No built-in tools, no other MCP servers, settings, plugins or stored logins; only the four
    history tools pre-approved and everything else refused; the server metadata-only; and a
    question may spend only what's left of today's cap."""
    a, seen = agent_run(log, cap=1.0, spent=0.25)
    a.ask("anything?")
    o = seen["options"]
    assert o.tools == [] and o.strict_mcp_config and o.setting_sources == [] and o.permission_mode == "dontAsk"
    assert set(o.allowed_tools) == a.tool_names and len(a.tool_names) == 4
    assert o.extra_args == {"bare": None, "no-session-persistence": None}
    assert "--metadata-only" in o.mcp_servers["cameras"]["args"]
    assert o.max_budget_usd == pytest.approx(0.75) and o.max_turns == 6 and o.model == MODEL
    assert "isn't offline" in o.system_prompt and "Home cameras watched" not in o.system_prompt  # the server sends that


def test_the_agent_backend_fails_closed_on_any_other_tool(log):
    a, _ = agent_run(log, tools=["mcp__cameras__find_visits", "mcp__cameras__snapshot", "Bash"])
    with pytest.raises(AskError) as e:
        a.ask("anything?")
    assert e.value.status == 503 and "snapshot" in str(e.value)


def test_the_agent_backend_reports_stops_and_refusals(log):
    a, _ = agent_run(log, result=done("", subtype="error_max_turns", is_error=True))
    assert "more steps" in a.ask("q").text
    a, _ = agent_run(log, result=done("Partly:", subtype="error_max_budget_usd", is_error=True))
    assert a.ask("q").text == "Partly:\n\n(Stopped: today's ask budget is used up.)"
    a, _ = agent_run(log, result=done("no", stop="refusal"))
    r = a.ask("q")
    assert r.refused and r.text == "I can't help with that one."
    a, _ = agent_run(log, result=done("", is_error=True))
    with pytest.raises(RuntimeError):
        a.ask("q")
    a, _ = agent_run(log, cap=1.0, spent=1.0)
    with pytest.raises(AskError) as e:  # the cap is checked before the agent starts
        a.ask("q")
    assert e.value.status == 429


def test_ask_backend_picks_the_loop(log, monkeypatch):
    from camera_mcp import ask
    history, _, tmp = log
    monkeypatch.setattr(ask, "read_credentials_file", dict)
    budget = Budget(tmp / "u.json", 1.0)
    monkeypatch.delenv("ASK_BACKEND", raising=False)
    assert type(ask.make_asker(history, budget, tmp / "c.yaml", tmp)) is ask.Asker
    monkeypatch.setenv("ASK_BACKEND", "Agent")
    assert type(ask.make_asker(history, budget, tmp / "c.yaml", tmp)) is ask.AgentAsker
