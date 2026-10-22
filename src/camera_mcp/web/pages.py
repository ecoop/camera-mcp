# Copyright (c) 2026 Eric Cooper.
"""The HTML pages. Each is a plain file in camera_mcp/web/pages/ (package data) with its CSS and JS
inline: dashboard.html (feeds, analytics, map, the visit viewer), zones.html (the zone editor) and
image.html (one visit's full-res still, filling the window: scroll or pinch to zoom at the pointer,
drag to pan, double-click to fit; the browser's own image view only offers a click-to-100%
magnifier). Edit them as HTML: there is no Python string escaping in between."""

from __future__ import annotations

from importlib import resources

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route

from camera_mcp.web.context import Context


def _page(name: str) -> str:
    return (resources.files("camera_mcp.web") / "pages" / name).read_text(encoding="utf-8")


PAGE = _page("dashboard.html")
ZONES_PAGE = _page("zones.html")
IMAGE_PAGE = _page("image.html")


def routes(ctx: Context) -> list[Route]:
    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(PAGE)

    async def zones_page(request: Request) -> HTMLResponse:
        return HTMLResponse(ZONES_PAGE)

    async def image_page(request: Request) -> HTMLResponse:
        return HTMLResponse(IMAGE_PAGE)  # the page reads the visit id from its own path

    return [
        Route("/", index),
        Route("/analytics", index),  # same single-page doc; the client shows the analytics view
        Route("/map", index),  # ...or the labels map
        Route("/zones", zones_page),
        Route("/visit/{vid:int}/image", image_page),
    ]
