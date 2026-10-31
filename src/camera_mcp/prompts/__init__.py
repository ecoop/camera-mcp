# Copyright (c) 2026 Eric Cooper.
"""The prose we send the model, as plain Markdown files beside this module (package data):
server_instructions.md (what the MCP server tells every client when it connects) and ask_style.md
(how the dashboard's ask box answers). Edit them as Markdown: there is no Python string escaping
in between. Tool descriptions stay docstrings in server.py, next to the parameters they describe."""

from __future__ import annotations

from importlib import resources


def load(name: str) -> str:
    """A prompt file's text, without its copyright comment (the first line), which the model never sees."""
    text = (resources.files(__package__) / name).read_text(encoding="utf-8")
    if text.startswith("<!--"):
        text = text.split("\n", 1)[1].lstrip("\n")
    return text
