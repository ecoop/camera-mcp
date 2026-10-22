# Copyright (c) 2026 Eric Cooper.
"""What a person records about visits: correctness tags on a visit or a single box (with the
species, if they say), favourites, and hiding or showing a visit by hand. Every write is refused
when editing is off (CAMERA_MCP_WEB_READONLY)."""

from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from camera_mcp.store import VERDICTS, Store
from camera_mcp.web.context import Context


def routes(ctx: Context) -> list[Route]:
    history = ctx.history

    async def api_visit_label_get(request: Request):
        vid = int(request.path_params["vid"])
        r = ctx.query(history.visit_label, vid)
        if isinstance(r, Response):
            return r
        return JSONResponse({"label": r, "editable": ctx.editable})

    async def api_visit_label_set(request: Request):
        if not ctx.editable:
            return JSONResponse({"error": "labeling is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        vid = int(request.path_params["vid"])
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        verdict = (body.get("verdict") or "").strip()
        if verdict not in VERDICTS:
            return JSONResponse({"error": f"verdict must be one of {', '.join(VERDICTS)}"}, status_code=400)
        note = body.get("note")
        note = str(note).strip() or None if note is not None else None
        species = (str(body.get("species") or "").strip() or None)
        if species and verdict == "false":
            return JSONResponse({"error": "a 'false' verdict can't name a species"}, status_code=400)
        if not ctx.db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(ctx.db_path)
        try:
            store.set_visit_label(vid, verdict, note, species)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "label": {"verdict": verdict, "note": note, "species": species}})

    async def api_box_label_set(request: Request):
        """Tag one box: {"ts": its frame's time, "box": its index (both from /api/visit/{id}/boxes),
        "verdict": real | false | wrong_species | null to clear, "species": optional}."""
        if not ctx.editable:
            return JSONResponse({"error": "labeling is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        vid = int(request.path_params["vid"])
        try:
            body = await request.json()
            ts, box = str(body["ts"]), int(body["box"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return JSONResponse({"error": "send {ts, box, verdict}"}, status_code=400)
        verdict = body.get("verdict") or None
        species = (str(body.get("species") or "").strip() or None)
        if verdict is not None and verdict not in VERDICTS:
            return JSONResponse({"error": f"verdict must be one of {', '.join(VERDICTS)}, or null"}, status_code=400)
        if species and verdict in (None, "false"):
            return JSONResponse({"error": "only a real or wrong_species box can name a species"}, status_code=400)
        if not ctx.db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(ctx.db_path)
        try:
            store.set_box_label(vid, ts, box, verdict, species)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "box": {"ts": ts, "i": box, "verdict": verdict, "species": species}})

    async def api_needs_box_tags(request: Request):
        """Visits tagged real whose still has a box with no tag yet (history.needs_box_tags), newest
        first, as /api/visits returns them, so the viewer can open them in turn."""
        ids = ctx.query(history.needs_box_tags, request.query_params.get("camera") or None)
        if isinstance(ids, Response):
            return ids
        return JSONResponse({"visits": history.visits_by_id(ids), "count": len(ids)})

    async def api_favorite_set(request: Request):
        """Star or unstar a visit: {"on": true|false, "kind": "still"|"clip", "clip_at": seconds}."""
        if not ctx.editable:
            return JSONResponse({"error": "labeling is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        vid = int(request.path_params["vid"])
        try:
            body = await request.json()
            on, kind = bool(body.get("on", True)), str(body.get("kind") or "still")
            clip_at = float(body["clip_at"]) if body.get("clip_at") is not None else None
        except (ValueError, TypeError, json.JSONDecodeError):
            return JSONResponse({"error": "send {on, kind: still|clip, clip_at}"}, status_code=400)
        if on and kind not in ("still", "clip"):
            return JSONResponse({"error": "kind must be still or clip"}, status_code=400)
        if not ctx.db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(ctx.db_path)
        try:
            store.set_favorite(vid, on, kind, clip_at, body.get("note"))
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "favorite": {"kind": kind, "clip_at": clip_at if kind == "clip" else None} if on else None})

    async def api_tag_species(request: Request):
        """Suggestions for the tag species picker: names used in tags so far (most used first), then
        the groups seen on this site (rat, raccoon, songbirds, ...). Any other name can be typed."""
        used = ctx.query(history.tagged_species)
        used = [] if isinstance(used, Response) else used
        seen = [t for t in history.observed_taxa() if t not in used]
        return JSONResponse({"species": used + seen})

    async def api_visit_hidden_set(request: Request):
        """Show or hide one visit by hand ({"hidden": true|false}), overriding the gate or an exclusion
        zone: the case-by-case undo. Hiding records the reason "manual"."""
        if not ctx.editable:
            return JSONResponse({"error": "editing is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        vid = int(request.path_params["vid"])
        try:
            hidden = (await request.json()).get("hidden")
        except (ValueError, json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(hidden, bool):
            return JSONResponse({"error": "hidden must be true or false"}, status_code=400)
        if not ctx.db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        reason = "manual" if hidden else None
        store = Store(ctx.db_path)
        try:
            store.set_visit_suppressed(vid, reason)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "suppressed": reason})

    return [
        Route("/api/visit/{vid:int}/label", api_visit_label_get, methods=["GET"]),
        Route("/api/visit/{vid:int}/label", api_visit_label_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/hidden", api_visit_hidden_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/box-label", api_box_label_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/favorite", api_favorite_set, methods=["POST"]),
        Route("/api/labels/needs-box-tags", api_needs_box_tags),
        Route("/api/labels/species", api_tag_species),
    ]
