# Copyright (c) 2026 Eric Cooper.
"""Locked-down Claude agents over the visit log, on the Claude Agent SDK: the dashboard's ask box
(ask.AgentAsker) and scheduled jobs, run as `camera-agent <job>` (e.g. the weekly digest).

Every agent here is locked down the same way: the SDK runs the tool loop in a Claude Code process
with no built-in tools, settings, plugins, stored logins or saved sessions (--bare), and its only
tools are the four metadata-only history tools of camera-mcp started --metadata-only, so no image
or clip ever leaves the machine. If it is offered anything else, it stops before answering.

A job's prompt is the plugin's skill of the same name (plugin/skills/<job>/SKILL.md), so the same
text runs from Claude Code (/camera-mcp:<job>) and on a schedule here, guided by the
camera-history skill. A run writes data/agent/<job>/<date>.md and, with --email, mails it. Spend
goes into its own daily ledger (data/agent_usage.json, AGENT_DAILY_USD, default $0.50).

    camera-agent weekly-digest [--email] [--config cameras.yaml] [--data-dir data]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anyio

from camera_mcp.config import ConfigError, default_config_path, default_data_dir, load_cameras, load_site
from camera_mcp.credentials import read_credentials_file
from camera_mcp.history import History
from camera_mcp.server import METADATA_TOOLS

SERVER = "cameras"
TOOL_NAMES = frozenset(f"mcp__{SERVER}__{n}" for n in METADATA_TOOLS)
PLUGIN = Path(__file__).resolve().parents[2] / "plugin"  # the repo's plugin: jobs' prompts are its skills
DEFAULT_DAILY_USD = 0.50


class AgentError(RuntimeError):
    """An agent that can't run now (no SDK, no key, budget spent) or wasn't locked down."""


@dataclass(frozen=True)
class Job:
    title: str  # the email's subject, before the date
    model: str = "claude-sonnet-5-5"  # a scheduled summary doesn't need the top model
    effort: str = "medium"
    max_usd: float = 0.25  # per run
    max_turns: int = 12


JOBS = {"weekly-digest": Job("Camera weekly digest")}


def cameras_text(history: History) -> str:
    """The cameras with their watch hours and tags, for an agent's prompt."""
    def line(c) -> str:
        hours = f"watched {c.active} local, paused otherwise" if c.active else "watched around the clock"
        return f"- {c.name}: {hours}" + (f"; tags: {', '.join(c.tags)}" if c.tags else "")
    return "Cameras:\n" + "\n".join(line(c) for c in history.cameras.values() if c.enabled)


def skill_text(name: str) -> str:
    """A plugin skill's instructions, without its frontmatter."""
    path = PLUGIN / "skills" / name / "SKILL.md"
    try:
        text = path.read_text()
    except OSError:
        raise AgentError(f"no skill {name!r} at {path}") from None
    return text.split("---\n", 2)[2].strip() if text.startswith("---\n") else text.strip()


def locked_options(config: Path, data_dir: Path, *, model: str, effort: str, system_prompt: str, max_turns: int,
                   max_budget_usd: float, api_key: str):
    """SDK options for a locked-down agent over camera-mcp --metadata-only (see the module docstring)."""
    from claude_agent_sdk import ClaudeAgentOptions

    server = {"command": sys.executable, "args": ["-m", "camera_mcp.server", "--config", str(config),
                                                  "--data-dir", str(data_dir), "--metadata-only"]}
    return ClaudeAgentOptions(
        model=model, effort=effort, system_prompt=system_prompt,  # the server sends its own guide too
        tools=[], mcp_servers={SERVER: server}, strict_mcp_config=True, setting_sources=[],
        allowed_tools=sorted(TOOL_NAMES), permission_mode="dontAsk",
        max_turns=max_turns, max_budget_usd=max(0.0, max_budget_usd),
        cwd=str(data_dir), env={"ANTHROPIC_API_KEY": api_key},
        extra_args={"bare": None, "no-session-persistence": None},
    )


async def run(prompt: str, options, query=None, on_message=None):
    """Run one agent to its result (the SDK's ResultMessage), passing every message to on_message.
    Raises AgentError if the SDK isn't installed or the agent is offered any tool but the four."""
    try:
        from claude_agent_sdk import ResultMessage, SystemMessage
        from claude_agent_sdk import query as sdk_query
    except ImportError:
        raise AgentError("the Agent SDK isn't installed (pip install 'camera-mcp[agent]')") from None
    result = None
    async for m in (query or sdk_query)(prompt=prompt, options=options):
        if isinstance(m, SystemMessage) and m.subtype == "init":
            offered = set(m.data.get("tools", []))
            if offered != TOOL_NAMES:  # fail closed: exactly the four, nothing else
                raise AgentError(f"the agent was offered {sorted(offered)}, not just the history tools")
        elif isinstance(m, ResultMessage):
            result = m
        if on_message:
            on_message(m)
    if result is None:
        raise RuntimeError("the agent ended without a result")
    return result


def daily_cap() -> float:
    raw = os.environ.get("AGENT_DAILY_USD") or read_credentials_file().get("AGENT_DAILY_USD")
    try:
        return max(0.0, float(raw)) if raw else DEFAULT_DAILY_USD
    except ValueError:
        return DEFAULT_DAILY_USD


def run_job(name: str, config: Path, data_dir: Path, email: bool = False, query=None) -> Path:
    """Run a scheduled job once: write its report to data/agent/<job>/<date>.md, and mail it if asked.
    Returns the report's path."""
    from camera_mcp.ask import Budget, api_key
    from camera_mcp.mailer import email_settings, send_email

    job = JOBS[name]
    config, data_dir = Path(config), Path(data_dir)
    budget = Budget(data_dir / "agent_usage.json", daily_cap())
    left = budget.cap - budget.spent()
    if left <= 0:
        raise AgentError(f"today's agent budget (${budget.cap:.2f}) is used up")
    key = api_key()
    if not key and query is None:
        raise AgentError("no ANTHROPIC_API_KEY on this host")
    settings = email_settings() if email else None
    if email and not settings:
        raise AgentError("--email needs the mail settings (SMTP_HOST, SMTP_FROM, SMTP_TO)")
    cameras = load_cameras(str(config), require_login=False)
    history = History(data_dir / "monitor.db", cameras, load_site(str(config)))
    system = ("You write a scheduled report about the home cameras for their owner, delivered by email "
              "as plain text. Use the tools; never guess at what the log holds. Times you receive are "
              "local.\n\n" + skill_text("camera-history") + "\n\n" + cameras_text(history))
    now = datetime.now().astimezone()
    prompt = f"(It is now {now:%A %Y-%m-%d %H:%M %Z}.)\n\n" + skill_text(name)
    options = locked_options(config, data_dir, model=job.model, effort=job.effort, system_prompt=system,
                             max_turns=job.max_turns, max_budget_usd=min(job.max_usd, left), api_key=key or "")
    result = anyio.run(run, prompt, options, query)
    cost = result.total_cost_usd or 0.0
    budget.add(cost, question=True)
    text = (result.result or "").strip()
    if result.subtype == "error_max_budget_usd":
        text += "\n\n(Stopped early: the agent reached its spending limit.)"
    elif result.is_error and not text:
        raise AgentError(f"the agent failed ({result.subtype}, {'; '.join(result.errors or [])})")
    out = data_dir / "agent" / name / f"{now:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{text}\n\n<!-- {job.model}, {result.num_turns} turns, ${cost:.4f} -->\n")
    if settings:
        send_email(f"{job.title} — {now:%a %b %-d}", text, settings)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="camera-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--email", action="store_true", help="also mail the report (SMTP settings)")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    from camera_mcp.mailer import MailError
    try:
        out = run_job(args.job, Path(args.config), args.data_dir or default_data_dir(), args.email)
    except (AgentError, ConfigError, MailError) as e:
        print(f"camera-agent: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"camera-agent: {args.job} written to {out}")


if __name__ == "__main__":
    main()
