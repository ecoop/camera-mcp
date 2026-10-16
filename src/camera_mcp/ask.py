# Copyright (c) 2026 Eric Cooper.
"""Ask the visit log in plain English, from the dashboard: "all skunks in the last 24 hours?"

Claude answers with the same read-only history tools the MCP server offers (find_visits,
activity_summary, monitor_status, analysis_health). Only metadata leaves the machine — labels,
times, counts, camera names — never images or clips. Visits it mentions are written as [#1234]
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

from camera_mcp.config import read_credentials_file
from camera_mcp.history import History, HistoryError

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

TOOLS = [
    {
        "name": "find_visits",
        "description": "Animal visits in a stretch of time, newest first, with how much of it each camera was "
                       "watched (coverage). Use for 'which/when/show me' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "start: ISO time (local) or a duration ago (15m, 4h, 3d, 2w), today, yesterday"},
                "until": {"type": "string", "description": "end (default now)"},
                "cameras": {"type": "array", "items": {"type": "string"}, "description": "camera names (default all)"},
                "tag": {"type": "string", "description": "only cameras with this tag"},
                "taxon": {"type": "string", "description": "group, union, class or common name, e.g. skunk, night-critters, songbirds, bird"},
                "limit": {"type": "integer", "description": "at most this many visits (1-50, default 20); 'total' says how many matched"},
                "include_suppressed": {"type": "boolean", "description": "also list visits the gate or a zone hid as likely junk"},
            },
            "required": ["since"],
        },
    },
    {
        "name": "activity_summary",
        "description": "Count visits in a stretch of time, grouped by species, group, camera, hour (local hour of "
                       "day) or day. Use for 'how many / when most / busiest' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "start, as for find_visits"},
                "until": {"type": "string", "description": "end (default now)"},
                "group_by": {"type": "string", "enum": ["species", "group", "camera", "hour", "day"]},
                "cameras": {"type": "array", "items": {"type": "string"}},
                "tag": {"type": "string"},
                "taxon": {"type": "string"},
            },
            "required": ["since"],
        },
    },
    {
        "name": "monitor_status",
        "description": "Which cameras are being watched right now, when each last saw motion, and open visits.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "analysis_health",
        "description": "How the detector kept up over a window: frames analysed, backlog/lag, anything dropped.",
        "input_schema": {
            "type": "object",
            "properties": {"since": {"type": "string"}, "until": {"type": "string"}},
            "required": ["since"],
        },
    },
]

STYLE = """
You answer questions about this visit log for its owner, from a box on their camera dashboard.

- Use the tools; never guess at what the log holds. Times you receive are local.
- Answer briefly in plain text (no markdown tables). Lead with the answer: a count, a time, a yes/no.
- When you mention specific visits, write each one's id as [#1234] (the dashboard turns these into
  thumbnails). Mention at most 12.
- Be honest about absence, as the tool notes say: if coverage was low or motion went unanalysed,
  say which camera wasn't fully watched.
- You can only read the log. You can't see images or clips, change settings, or control cameras;
  say so if asked.
"""


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


class Asker:
    def __init__(self, history: History, budget: Budget, client=None):
        self.history, self.budget = history, budget
        self._client = client  # injected in tests; otherwise built from the API key on first use

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

    def system(self) -> list[dict]:
        """The stable part of the prompt (cached): the MCP server's own guidance and the cameras."""
        from camera_mcp.server import INSTRUCTIONS
        cams = "\n".join(f"- {c.name}" + (f" (tags: {', '.join(c.tags)})" if c.tags else "")
                         for c in self.history.cameras.values() if c.enabled)
        return [{"type": "text", "text": INSTRUCTIONS + "\nCameras:\n" + cams + "\n" + STYLE}]

    def run_tool(self, name: str, args: dict, seen: dict) -> str:
        """One read-only history call, as compact JSON. Visits it returns are remembered (`seen`)
        so the answer's [#id] references can be shown."""
        h, a = self.history, dict(args or {})
        if name == "find_visits":
            limit = max(1, min(int(a.get("limit") or 20), 50))
            r = h.find_visits(since=a.get("since", "24h"), until=a.get("until"), cameras=a.get("cameras"),
                              tag=a.get("tag"), taxon=a.get("taxon"), limit=limit,
                              include_suppressed=bool(a.get("include_suppressed")))
            keep = ("id", "camera", "started", "last_seen", "still_there", "label", "group", "score",
                    "max_at_once", "frames", "suppressed", "verdict")
            trimmed = []
            for v in r["visits"]:
                meta = {k: v.get(k) for k in keep}
                seen[meta["id"]] = meta | {"has_thumbnail": v.get("has_thumbnail"), "clip": bool(v.get("clip"))}
                trimmed.append(meta)
            r["visits"] = trimmed
        elif name == "activity_summary":
            r = h.activity_summary(since=a.get("since", "7d"), until=a.get("until"),
                                   group_by=a.get("group_by", "species"), cameras=a.get("cameras"),
                                   tag=a.get("tag"), taxon=a.get("taxon"))
        elif name == "monitor_status":
            r = h.monitor_status()
        elif name == "analysis_health":
            r = h.analysis_health(since=a.get("since", "24h"), until=a.get("until"))
        else:
            raise HistoryError(f"unknown tool {name!r}")
        return json.dumps(r, default=str, separators=(",", ":"))

    def ask(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            raise AskError("ask a question")
        if len(question) > MAX_QUESTION:
            raise AskError(f"keep questions under {MAX_QUESTION} characters")
        if self.budget.spent() >= self.budget.cap:
            raise AskError(f"today's ask budget (${self.budget.cap:.2f}) is used up; it resets at midnight", 429)
        client = self.client()
        now = datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z")
        messages: list = [{"role": "user", "content": f"(It is now {now}.)\n\n{question}"}]
        seen: dict = {}
        cost, rounds, refused, text = 0.0, 0, False, ""
        self.budget.add(0.0, question=True)
        while True:
            if rounds >= MAX_ROUNDS:
                text = text or "That took more steps than a quick question should; try asking more specifically."
                break
            if self.budget.spent() >= self.budget.cap:
                text = (text + "\n\n" if text else "") + "(Stopped: today's ask budget is used up.)"
                break
            response = client.beta.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=self.system(), tools=TOOLS, messages=messages,
                output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA], fallbacks="default",
            )
            rounds += 1
            spent = cost_of(response.usage, getattr(response, "model", MODEL))
            cost += spent
            self.budget.add(spent)
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
            if response.stop_reason == "refusal":
                refused, text = True, "I can't help with that one."
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
        ids = []
        for m in VISIT_REF.finditer(text):
            vid = int(m.group(1))
            if vid in seen and vid not in ids:
                ids.append(vid)
        return Answer(text=text or "(no answer)", visits=[seen[i] for i in ids], cost=round(cost, 4),
                      spent_today=round(self.budget.spent(), 4), cap=self.budget.cap, rounds=rounds,
                      refused=refused, seen=seen)
