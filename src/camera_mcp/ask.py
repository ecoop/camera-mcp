# Copyright (c) 2026 Eric Cooper.
"""Ask the visit log in plain English, from the dashboard: "all skunks in the last 24 hours?"

Claude answers with the MCP server's own read-only history tools (find_visits, activity_summary,
monitor_status, analysis_health), listed and called in process, so there is one definition of each.
Only metadata leaves the machine — labels, times, counts, camera names — never images or clips. Visits it mentions are written as [#1234]
so the dashboard can show their thumbnails.

Spend is capped per day (ASK_DAILY_USD, default $1): the running total is kept in
data/ask_usage.json and checked before every model call, so a question that would go over the
cap stops instead of running on. The API key is ANTHROPIC_API_KEY, from the environment or the
host's git-ignored credentials file; without it the feature is off.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import anyio
from mcp.server.mcpserver.exceptions import ToolError

from camera_mcp import agent, prompts
from camera_mcp.credentials import read_credentials_file
from camera_mcp.history import History, HistoryError
from camera_mcp.server import INSTRUCTIONS, METADATA_TOOLS, build_server

MODEL = "claude-opus-5-5"
EFFORT = "low"  # short Q&A over a few tool calls: low effort answers well and keeps cost and latency down
MAX_TOKENS = 4000
MAX_ROUNDS = 6  # model calls per question: plenty for a few tool calls, a hard stop otherwise
MAX_QUESTION = 500  # characters
DEFAULT_DAILY_USD = 1.0
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # with fallbacks="default": a declined request is retried server-side
# $ per million tokens: input, output, cache read, cache write (5-minute). A fallback model may serve
# a turn; it's priced at its own rates.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.50, 6.25),
}
VISIT_REF = re.compile(r"\[#(\d+)\]")

# The history tools Claude may use here: the MCP server's metadata-only ones (no images or clips), from
# a server built metadata-only, which also keeps find_visits' pages small and its visits trimmed.
ASK_TOOLS = METADATA_TOOLS

STYLE = prompts.load("ask_style.md")


class AskError(Exception):
    """A question that can't be answered now; `status` is the HTTP status to report."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or read_credentials_file().get("ANTHROPIC_API_KEY") or None


def daily_cap() -> float:
    raw = os.environ.get("ASK_DAILY_USD") or read_credentials_file().get("ASK_DAILY_USD")
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_DAILY_USD
    except ValueError:
        return DEFAULT_DAILY_USD


def cost_of(usage, model: str) -> float:
    """Dollars for one response's usage, at the serving model's rates (Opus 5.5's if unknown)."""
    inp, out, read, write = PRICES.get(model, PRICES[MODEL])
    g = lambda k: getattr(usage, k, 0) or 0  # noqa: E731
    return (g("input_tokens") * inp + g("output_tokens") * out + g("cache_read_input_tokens") * read
            + g("cache_creation_input_tokens") * write) / 1e6


class Budget:
    """Today's spend, persisted in a small JSON file so the cap survives restarts."""

    def __init__(self, path: Path, cap: float):
        self.path, self.cap = Path(path), cap
        self._lock = threading.Lock()

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _load(self) -> dict:
        try:
            d = json.loads(self.path.read_text())
            if d.get("date") == self._today():
                return d
        except (OSError, ValueError):
            pass
        return {"date": self._today(), "usd": 0.0, "questions": 0}

    def spent(self) -> float:
        with self._lock:
            return float(self._load()["usd"])

    def add(self, usd: float, question: bool = False) -> float:
        with self._lock:
            d = self._load()
            d["usd"] = round(float(d["usd"]) + usd, 6)
            d["questions"] = int(d["questions"]) + (1 if question else 0)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(self.path)
            return d["usd"]


@dataclass
class Answer:
    text: str
    visits: list[dict]  # metadata of the visits the answer names, in order
    cost: float
    spent_today: float
    cap: float
    rounds: int
    refused: bool = False
    seen: dict = field(default_factory=dict, repr=False)


TOO_MANY_STEPS = "That took more steps than a quick question should; try asking more specifically."
BUDGET_STOP = "(Stopped: today's ask budget is used up.)"
REFUSED = "I can't help with that one."


def guidance(history: History) -> str:
    """The cameras and their watch hours, then how to answer: the prompt both backends add to the
    MCP server's own instructions."""
    return agent.cameras_text(history) + "\n\n" + STYLE


def opening(question: str, budget: Budget) -> str:
    """The first message for a question, or AskError if it can't be asked now."""
    question = (question or "").strip()
    if not question:
        raise AskError("ask a question")
    if len(question) > MAX_QUESTION:
        raise AskError(f"keep questions under {MAX_QUESTION} characters")
    if budget.spent() >= budget.cap:
        raise AskError(f"today's ask budget (${budget.cap:.2f}) is used up; it resets at midnight", 429)
    now = datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z")
    return f"(It is now {now}.)\n\n{question}"


def remember(found: dict, seen: dict) -> None:
    """Keep the visits a find_visits answer listed, so an answer's [#id] references can be shown."""
    for v in found.get("visits", []):
        seen[v["id"]] = v


def answer(text: str, seen: dict, cost: float, budget: Budget, rounds: int, refused: bool) -> Answer:
    ids = []
    for m in VISIT_REF.finditer(text):
        vid = int(m.group(1))
        if vid in seen and vid not in ids:
            ids.append(vid)
    return Answer(text=text or "(no answer)", visits=[seen[i] for i in ids], cost=round(cost, 4),
                  spent_today=round(budget.spent(), 4), cap=budget.cap, rounds=rounds, refused=refused, seen=seen)


class Asker:
    def __init__(self, history: History, budget: Budget, client=None):
        self.history, self.budget = history, budget
        self._client = client  # injected in tests; otherwise built from the API key on first use
        self.server = build_server(history.cameras, history, metadata_only=True)  # the MCP tools, in process
        self._tools: list[dict] | None = None
        self._wrapped: set[str] = set()  # tools whose result the server wraps as {"result": ...} (lists)

    @property
    def enabled(self) -> bool:
        return self._client is not None or api_key() is not None

    def client(self):
        if self._client is None:
            key = api_key()
            if not key:
                raise AskError("asking is off: no ANTHROPIC_API_KEY on this host", 503)
            try:
                import anthropic
            except ImportError:
                raise AskError("asking is off: install the ask extra (pip install 'camera-mcp[ask]')", 503) from None
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def tools(self) -> list[dict]:
        """ASK_TOOLS as the MCP server defines them (name, description, input schema), in that order."""
        if self._tools is None:
            listed = {t.name: t for t in anyio.run(self.server.list_tools)}
            for n in ASK_TOOLS:
                out = listed[n].output_schema or {}
                if out.get("required") == ["result"] and set(out.get("properties", {})) == {"result"}:
                    self._wrapped.add(n)
            self._tools = [{"name": n, "description": listed[n].description, "input_schema": listed[n].input_schema}
                           for n in ASK_TOOLS]
        return self._tools

    def system(self) -> list[dict]:
        """The stable part of the prompt (cached): the MCP server's own guidance and the cameras."""
        return [{"type": "text", "text": INSTRUCTIONS + "\n" + guidance(self.history)}]

    def run_tool(self, name: str, args: dict, seen: dict) -> str:
        """One read-only history call through the MCP server, as compact JSON. The visits find_visits
        returns are remembered (`seen`) so the answer's [#id] references can be shown."""
        if name not in ASK_TOOLS:
            raise HistoryError(f"unknown tool {name!r}")
        self.tools()  # learns which results come wrapped
        try:
            result = anyio.run(self.server.call_tool, name, dict(args or {}))
        except ToolError as e:  # the server's message, without its "Error executing tool …:" prefix
            raise HistoryError(str(e).removeprefix(f"Error executing tool {name}: ")) from None
        r = result.structured_content
        if name in self._wrapped:
            r = r["result"]
        if name == "find_visits":
            remember(r, seen)
        return json.dumps(r, default=str, separators=(",", ":"))

    def ask(self, question: str) -> Answer:
        prompt = opening(question, self.budget)
        client = self.client()
        messages: list = [{"role": "user", "content": prompt}]
        seen: dict = {}
        cost, rounds, refused, text = 0.0, 0, False, ""
        self.budget.add(0.0, question=True)
        while True:
            if rounds >= MAX_ROUNDS:
                text = text or TOO_MANY_STEPS
                break
            if self.budget.spent() >= self.budget.cap:
                text = (text + "\n\n" if text else "") + BUDGET_STOP
                break
            response = client.beta.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=self.system(), tools=self.tools(), messages=messages,
                output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA], fallbacks="default",
            )
            rounds += 1
            spent = cost_of(response.usage, getattr(response, "model", MODEL))
            cost += spent
            self.budget.add(spent)
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
            if response.stop_reason == "refusal":
                refused, text = True, REFUSED
                break
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})  # append-only: thinking blocks kept
            results = []
            for block in response.content:
                if getattr(block, "type", "") != "tool_use":
                    continue
                try:
                    out, err = self.run_tool(block.name, block.input, seen), False
                except (HistoryError, TypeError, ValueError) as e:
                    out, err = f"Error: {e}", True
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out, "is_error": err})
            messages.append({"role": "user", "content": results})
        return answer(text, seen, cost, self.budget, rounds, refused)


class AgentAsker:
    """The same ask box on the Claude Agent SDK (ASK_BACKEND=agent), locked down as every agent in
    camera_mcp.agent is: one Claude Code process per question, whose only tools are the four history
    tools of camera-mcp --metadata-only. Spend goes into the same daily ledger, and a question may
    spend at most what's left of today's cap."""

    def __init__(self, history: History, budget: Budget, config: Path, data_dir: Path, query=None):
        self.history, self.budget = history, budget
        self.config, self.data_dir = Path(config), Path(data_dir)
        self._query = query  # injected in tests; otherwise claude_agent_sdk.query
        self.tool_names = agent.TOOL_NAMES

    @property
    def enabled(self) -> bool:
        return self._query is not None or api_key() is not None

    def options(self):
        return agent.locked_options(self.config, self.data_dir, model=MODEL, effort=EFFORT,
                                    system_prompt=guidance(self.history), max_turns=MAX_ROUNDS,
                                    max_budget_usd=self.budget.cap - self.budget.spent(), api_key=api_key() or "")

    def ask(self, question: str) -> Answer:
        prompt = opening(question, self.budget)
        if not self.enabled:
            raise AskError("asking is off: no ANTHROPIC_API_KEY on this host", 503)
        self.budget.add(0.0, question=True)
        seen: dict = {}
        finds: set[str] = set()  # ids of find_visits calls, whose results list visits

        def watch(m) -> None:  # remember the visits find_visits returned, for the [#id] links
            for b in m.content if isinstance(getattr(m, "content", None), list) else []:
                if getattr(b, "name", "") == f"mcp__{agent.SERVER}__find_visits":
                    finds.add(b.id)
                elif getattr(b, "tool_use_id", None) in finds and not b.is_error:
                    remember(tool_json(b.content), seen)
        try:
            result = anyio.run(agent.run, prompt, self.options(), self._query, watch)
        except agent.AgentError as e:
            raise AskError(f"asking is off: {e}", 503) from None
        cost = result.total_cost_usd or 0.0
        self.budget.add(cost)
        text, refused = (result.result or "").strip(), result.stop_reason == "refusal"
        if refused:
            text = REFUSED
        elif result.subtype == "error_max_turns":
            text = text or TOO_MANY_STEPS
        elif result.subtype == "error_max_budget_usd":
            text = (text + "\n\n" if text else "") + BUDGET_STOP
        elif result.is_error:
            raise RuntimeError(f"the agent failed ({result.subtype}, {'; '.join(result.errors or []) or result.api_error_status})")
        return answer(text, seen, cost, self.budget, result.num_turns, refused)


def tool_json(content) -> dict:
    """A tool result's JSON, from the text the MCP server sent ({} if it isn't JSON)."""
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    try:
        r = json.loads(content or "")
    except ValueError:
        return {}
    return r if isinstance(r, dict) else {}


def ask_backend() -> str:
    """"api" (the hand-written loop, the default) or "agent" (the Agent SDK), from ASK_BACKEND."""
    raw = os.environ.get("ASK_BACKEND") or read_credentials_file().get("ASK_BACKEND") or "api"
    return "agent" if raw.strip().lower() == "agent" else "api"


def make_asker(history: History, budget: Budget, config: Path, data_dir: Path):
    """The ask box's backend, by ASK_BACKEND."""
    return AgentAsker(history, budget, config, data_dir) if ask_backend() == "agent" else Asker(history, budget)
