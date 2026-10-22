# Copyright (c) 2026 Eric Cooper.
"""The dashboard's ask box: whether asking is on (and today's spend), and a plain-English question
answered by Claude over the read-only history tools (see camera_mcp.ask)."""

from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from camera_mcp.web.context import Context, to_thread


def routes(ctx: Context) -> list[Route]:
    from camera_mcp.ask import Asker, AskError, Budget, daily_cap
    asker = Asker(ctx.history, Budget(ctx.data_dir / "ask_usage.json", daily_cap()))

    async def api_ask_status(request: Request) -> JSONResponse:
        """Whether asking is on here, and today's spend against the cap (the dashboard shows the box)."""
        return JSONResponse({"enabled": asker.enabled, "spent_today": round(asker.budget.spent(), 4),
                             "cap": asker.budget.cap})

    async def api_ask(request: Request) -> JSONResponse:
        """A plain-English question about the visit log, answered by Claude with read-only history
        tools (see ask.py). Only metadata leaves the machine; spend is capped per day."""
        try:
            question = (await request.json()).get("question", "")
        except (ValueError, json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        try:
            a = await to_thread(asker.ask, str(question))
        except AskError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)
        except Exception as e:  # the API or network failed: say so, don't crash the dashboard
            return JSONResponse({"error": f"couldn't get an answer ({type(e).__name__}: {e})"}, status_code=502)
        return JSONResponse({"answer": a.text, "visits": a.visits, "cost": a.cost, "spent_today": a.spent_today,
                             "cap": a.cap, "refused": a.refused})

    return [
        Route("/api/ask", api_ask_status, methods=["GET"]),
        Route("/api/ask", api_ask, methods=["POST"]),
    ]
