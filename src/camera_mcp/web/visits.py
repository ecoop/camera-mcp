# Copyright (c) 2026 Eric Cooper.
"""Read-only JSON over the visit log: cameras, status and health, visits, the labels map, the
species menu, and the analytics (timeline, summary, stats, heatmap)."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from camera_mcp.web.context import Context


def routes(ctx: Context) -> list[Route]:
    history = ctx.history

    def reply(r):
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_cameras(request: Request) -> JSONResponse:
        return JSONResponse([{"name": c.name, "label": c.label, "tags": list(c.tags),
                              "stream": c.is_stream, "enabled": c.enabled} for c in ctx.cameras.values()])

    async def api_status(request: Request):
        return reply(ctx.query(history.monitor_status))

    async def api_health(request: Request):
        return reply(ctx.query(history.analysis_health, request.query_params.get("since", "24h")))

    async def api_visits(request: Request):
        q = request.query_params
        cams = [q["camera"]] if q.get("camera") else None
        return reply(ctx.query(history.find_visits, since=q.get("since", "24h"), cameras=cams,
                               taxon=q.get("taxon") or None, limit=int(q.get("limit", 60)),
                               include_suppressed=q.get("suppressed") == "1",  # ?suppressed=1: the gate's hidden visits too
                               favorites_only=q.get("favorites") == "1"))  # ?favorites=1: starred ones, from any time

    async def api_label_map(request: Request):
        """Past visits on one camera, placed where they were boxed, with their labels (see
        History.label_map): the dashboard's map view plots them over the camera's picture."""
        q = request.query_params
        if not q.get("camera"):
            return JSONResponse({"error": "pick a camera"}, status_code=400)
        return reply(ctx.query(history.label_map, q["camera"], since=q.get("since", "30d"),
                               taxon=q.get("taxon") or None, unlabelled=q.get("unlabelled") == "1"))

    async def api_species(request: Request) -> JSONResponse:
        # The filter menu, derived from what's been seen (or the site's pinned menu_species).
        # Degrades to an empty list when there's no visit log yet, so the page still loads.
        menu = ctx.site.menu_species if ctx.site else ()
        return JSONResponse({"species": history.observed_taxa(menu)})

    async def api_visits_timeline(request: Request):
        q = request.query_params
        cams = [q["camera"]] if q.get("camera") else None
        return reply(ctx.query(history.visits_timeline, since=q.get("since", "24h"), cameras=cams,
                               taxon=q.get("taxon") or None))

    async def api_summary(request: Request):
        q = request.query_params
        return reply(ctx.query(history.activity_summary, since=q.get("since", "24h"), group_by=q.get("group_by", "group")))

    async def api_stats(request: Request):
        return reply(ctx.query(history.analysis_series, since=request.query_params.get("since", "24h"),
                               camera=request.query_params.get("camera") or None))

    async def api_stats_all(request: Request):
        return reply(ctx.query(history.analysis_series_all, since=request.query_params.get("since", "24h")))

    async def api_stats_heatmap(request: Request):
        return reply(ctx.query(history.stats_heatmap, since=request.query_params.get("since", "24h")))

    return [
        Route("/api/cameras", api_cameras),
        Route("/api/status", api_status),
        Route("/api/health", api_health),
        Route("/api/visits", api_visits),
        Route("/api/labels/map", api_label_map),
        Route("/api/visits/timeline", api_visits_timeline),
        Route("/api/species", api_species),
        Route("/api/summary", api_summary),
        Route("/api/stats", api_stats),
        Route("/api/stats/all", api_stats_all),
        Route("/api/stats/heatmap", api_stats_heatmap),
    ]
