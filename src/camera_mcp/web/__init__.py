# Copyright (c) 2026 Eric Cooper.
"""A read-only web dashboard for the Studio: browse visits and watch how the monitor is doing,
over the same History API the MCP tools use. Starlette + uvicorn, no new framework.

    python -m camera_mcp.web [--config cameras.yaml] [--data-dir data] [--host 0.0.0.0] [--port 8080]

The visit browser, analytics and live snapshots are read-only. Two surfaces are read-write: the
/zones editor draws per-camera ROI / motion-ROI / motion-mask on a snapshot and saves them to an
app-owned overlay (data/zones.json, merged by load_cameras), then restarts the monitor to apply;
and the visit modal captures a one-click correctness verdict per visit (real / false /
wrong_species) into the monitor.db visit_labels table, as training data for a later re-scorer.
Both are gated by the same flag: editing is LAN-only and unauthenticated for now
(CAMERA_MCP_WEB_READONLY=1 disables it); real auth arrives with external access (Cloudflare).

Layout: app.py assembles the app from one module per area, each adding its routes over a shared
Context (context.py): pages (the HTML pages, in pages/), visits (visits, stats, the labels map),
tags (visit and box tags, favourites, hide/show), media (thumbnails, stills, clips, snapshots),
zones (the zone editor's API) and askbox (the ask box).
"""

from camera_mcp.web.app import build_app, main
from camera_mcp.web.media import draw_boxes
from camera_mcp.web.pages import IMAGE_PAGE, PAGE, ZONES_PAGE

__all__ = ["build_app", "main", "draw_boxes", "PAGE", "ZONES_PAGE", "IMAGE_PAGE"]
