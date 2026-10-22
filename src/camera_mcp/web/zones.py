# Copyright (c) 2026 Eric Cooper.
"""The zone editor's API: each camera's zones as JSON, and saving a camera's zones to the app-owned
overlay (data/zones.json), validated with the same parsers load_cameras uses, then restarting the
monitor so it applies them."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from camera_mcp.config import (ConfigError, exclude_json, find_camera, load_zone_overlay, parse_excludes,
                               parse_rect, parse_shape, parse_shapes, save_zone_overlay, shape_json)
from camera_mcp.web.context import Context


def _monitor_label() -> str | None:
    """The monitor's launchd label, kicked to reload cameras.yaml + the zone overlay after an edit.
    CAMERA_MCP_LABEL if set (install-webapp.sh writes it); else found among the loaded services: the
    one named "camera-mcp" or ending in ".camera-mcp" (deploy/install-service.sh installs it under
    CAMERA_MCP_LABEL, which can carry a reverse-domain prefix; the .web/.brief/... agents don't
    match). None if there's no such service (e.g. off the monitor host)."""
    if label := os.environ.get("CAMERA_MCP_LABEL"):
        return label
    try:
        r = subprocess.run(["launchctl", "list"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    labels = [line.split()[-1] for line in r.stdout.splitlines()[1:] if line.split()]
    found = [lb for lb in labels if lb == "camera-mcp" or lb.endswith(".camera-mcp")]
    return found[0] if len(found) == 1 else None


def _reload_monitor() -> tuple[bool, str]:
    """Restart the monitor LaunchAgent so it re-reads the zone overlay. Best-effort: returns
    (ok, detail); the detail says what to do when it couldn't."""
    label = _monitor_label()
    if label is None:
        return False, ("couldn't find the monitor's service (no loaded launchd agent named camera-mcp or "
                       "*.camera-mcp); set CAMERA_MCP_LABEL for the web app, or restart the monitor by hand")
    try:
        r = subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run launchctl ({e})"
    if r.returncode == 0:
        return True, f"restarted {label}"
    return False, (r.stderr or r.stdout or f"launchctl exit {r.returncode}").strip()


def zone_state(cam) -> dict:
    return {"name": cam.name, "label": cam.label, "stream": cam.is_stream,
            "roi": list(cam.roi) if cam.roi else None,
            "motion_roi": shape_json(cam.motion_roi) if cam.motion_roi else None,
            "motion_mask": [shape_json(z) for z in cam.motion_mask],
            "exclude": [exclude_json(z) for z in cam.exclude]}


def routes(ctx: Context) -> list[Route]:
    async def api_zones(request: Request) -> JSONResponse:
        return JSONResponse({"editable": ctx.editable,
                             "cameras": [zone_state(c) for c in ctx.cameras.values() if c.enabled]})

    async def api_zones_save(request: Request) -> JSONResponse:
        if not ctx.editable:
            return JSONResponse({"error": "editing is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        cam = find_camera(ctx.cameras, request.path_params["camera"])
        if cam is None:
            return JSONResponse({"error": "unknown camera"}, status_code=404)
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        where = f"zones for {cam.name!r}"
        try:  # validate with the same parsers load_cameras uses, so a bad zone can never be written
            roi = parse_rect(body["roi"], where, "roi") if body.get("roi") else None
            motion_roi = parse_shape(body["motion_roi"], where, "motion_roi") if body.get("motion_roi") else None
            motion_mask = parse_shapes(body.get("motion_mask"), where, "motion_mask")
            exclude = parse_excludes(body.get("exclude"), where)
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        new_roi = tuple(roi) if roi else None
        old_roi = tuple(cam.roi) if cam.roi else None
        scenery_reset = new_roi != old_roi  # a changed analysis ROI makes the monitor relearn scenery

        # Merge this camera's entry into the overlay file and persist atomically.
        raw = {"cameras": {}}
        if ctx.zones_path.exists():
            try:
                raw = json.loads(ctx.zones_path.read_text())
                raw.setdefault("cameras", {})
            except (ValueError, OSError):
                raw = {"cameras": {}}
        raw["cameras"][cam.name] = {"roi": list(roi) if roi else None,
                                    "motion_roi": shape_json(motion_roi) if motion_roi else None,
                                    "motion_mask": [shape_json(z) for z in motion_mask],
                                    "exclude": [exclude_json(z) for z in exclude]}
        try:
            save_zone_overlay(ctx.zones_path, raw)
            load_zone_overlay(ctx.zones_path)  # re-validate the file as written
        except (OSError, ConfigError) as e:
            return JSONResponse({"error": f"could not save: {e}"}, status_code=500)

        ctx.cameras[cam.name] = replace(cam, roi=roi, motion_roi=motion_roi, motion_mask=motion_mask, exclude=exclude)
        reloaded, detail = _reload_monitor()
        return JSONResponse({"ok": True, "reloaded": reloaded, "reload_detail": detail,
                             "scenery_reset": bool(scenery_reset), "zones": zone_state(ctx.cameras[cam.name])})

    return [
        Route("/api/zones", api_zones),
        Route("/api/zones/{camera}", api_zones_save, methods=["POST"]),
    ]
