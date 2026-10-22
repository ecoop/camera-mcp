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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from camera_mcp.capture import CaptureError, grab_any
from camera_mcp.config import (ConfigError, default_config_path, default_data_dir, find_camera,
                               exclude_json, load_cameras, load_site, load_zone_overlay, parse_excludes,
                               parse_rect, parse_shape, parse_shapes, save_zone_overlay, shape_json)
from camera_mcp.history import History, HistoryError
from camera_mcp.store import VERDICTS, Store

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


def build_app(cameras: dict, data_dir: Path, site=None) -> Starlette:
    db_path = data_dir / "monitor.db"
    history = History(db_path, cameras, site)
    from camera_mcp.ask import Asker, AskError, Budget, daily_cap
    asker = Asker(history, Budget(data_dir / "ask_usage.json", daily_cap()))
    zones_path = data_dir / "zones.json"
    editable = os.environ.get("CAMERA_MCP_WEB_READONLY", "").lower() not in ("1", "true", "yes")

    def _zone_state(cam):
        return {"name": cam.name, "label": cam.label, "stream": cam.is_stream,
                "roi": list(cam.roi) if cam.roi else None,
                "motion_roi": shape_json(cam.motion_roi) if cam.motion_roi else None,
                "motion_mask": [shape_json(z) for z in cam.motion_mask],
                "exclude": [exclude_json(z) for z in cam.exclude]}

    def _ask(fn, *a, **k):
        try:
            return fn(*a, **k)
        except HistoryError as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(PAGE)

    async def api_cameras(request: Request) -> JSONResponse:
        return JSONResponse([{"name": c.name, "label": c.label, "tags": list(c.tags),
                              "stream": c.is_stream, "enabled": c.enabled} for c in cameras.values()])

    async def api_status(request: Request):
        r = _ask(history.monitor_status)
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_health(request: Request):
        r = _ask(history.analysis_health, request.query_params.get("since", "24h"))
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_visits(request: Request):
        q = request.query_params
        cams = [q["camera"]] if q.get("camera") else None
        r = _ask(history.find_visits, since=q.get("since", "24h"), cameras=cams,
                 taxon=q.get("taxon") or None, limit=int(q.get("limit", 60)),
                 include_suppressed=q.get("suppressed") == "1",  # ?suppressed=1: the gate's hidden visits too
                 favorites_only=q.get("favorites") == "1")  # ?favorites=1: starred ones, from any time
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_label_map(request: Request):
        """Past visits on one camera, placed where they were boxed, with their labels (see
        History.label_map): the dashboard's map view plots them over the camera's picture."""
        q = request.query_params
        if not q.get("camera"):
            return JSONResponse({"error": "pick a camera"}, status_code=400)
        r = _ask(history.label_map, q["camera"], since=q.get("since", "30d"), taxon=q.get("taxon") or None,
                 unlabelled=q.get("unlabelled") == "1")
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_species(request: Request) -> JSONResponse:
        # The filter menu, derived from what's been seen (or the site's pinned menu_species).
        # Degrades to an empty list when there's no visit log yet, so the page still loads.
        menu = site.menu_species if site else ()
        return JSONResponse({"species": history.observed_taxa(menu)})

    async def api_visits_timeline(request: Request):
        q = request.query_params
        cams = [q["camera"]] if q.get("camera") else None
        r = _ask(history.visits_timeline, since=q.get("since", "24h"), cameras=cams, taxon=q.get("taxon") or None)
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_summary(request: Request):
        q = request.query_params
        r = _ask(history.activity_summary, since=q.get("since", "24h"), group_by=q.get("group_by", "group"))
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_stats(request: Request):
        r = _ask(history.analysis_series, since=request.query_params.get("since", "24h"),
                 camera=request.query_params.get("camera") or None)
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_stats_all(request: Request):
        r = _ask(history.analysis_series_all, since=request.query_params.get("since", "24h"))
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_stats_heatmap(request: Request):
        r = _ask(history.stats_heatmap, since=request.query_params.get("since", "24h"))
        return r if isinstance(r, Response) else JSONResponse(r)

    def mapped_boxes(vid: int) -> dict:
        """A visit's per-frame boxes as fractions of the FULL frame (roi-mapped), each frame's clip
        offset and time (`ts`, which with a box's stored index `i` names it for tagging), each box's
        `tag` (history.box_tags), and `best`: the frame with the most confident box. Raises HistoryError."""
        d = history.visit_boxes(vid)
        cam = find_camera(cameras, d["camera"])
        roi = cam.roi if cam else None  # boxes are fractions of the roi crop; map to the full frame
        pre = cam.monitor.pre_seconds if cam else 5.0
        try:
            start = datetime.fromisoformat(d["started_at"]).timestamp()
        except (ValueError, TypeError):
            start = None

        def mapbox(b, i, tag):
            try:
                x, y, w, h = b["bbox"]
            except (KeyError, ValueError, TypeError):
                return None
            if roi:
                rx, ry, rw, rh = roi
                x, y, w, h = rx + x * rw, ry + y * rh, w * rw, h * rh
            return {"i": i, "conf": b.get("conf"), "x": round(x, 4), "y": round(y, 4),
                    "w": round(w, 4), "h": round(h, 4), "tag": tag}

        frames = []
        for f in d["frames"]:
            boxes = [m for m in (mapbox(b, i, t) for i, (b, t) in enumerate(zip(f["boxes"], f["tags"]))) if m]
            try:
                offset = datetime.fromisoformat(f["ts"]).timestamp() - start + pre if start else 0.0
            except (ValueError, TypeError):
                offset = 0.0
            frames.append({"ts": f["ts"], "offset": round(max(0.0, offset), 2), "label": f["label"],
                           "taxonomy": f["taxonomy"], "score": f["score"], "boxes": boxes})
        return {"camera": d["camera"], "has_clip": d["has_clip"], "roi": list(roi) if roi else None,
                "label": d["label"], "taxonomy": d["taxonomy"], "best": d["best"], "frames": frames,
                "visit_tag": d["visit_tag"]}

    async def api_visit_boxes(request: Request):
        try:
            return JSONResponse(mapped_boxes(int(request.path_params["vid"])))
        except HistoryError as e:
            return JSONResponse({"error": str(e)}, status_code=404)

    async def api_visit_label_get(request: Request):
        vid = int(request.path_params["vid"])
        r = _ask(history.visit_label, vid)
        if isinstance(r, Response):
            return r
        return JSONResponse({"label": r, "editable": editable})

    async def api_visit_label_set(request: Request):
        if not editable:
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
        if not db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(db_path)
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
        if not editable:
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
        if not db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(db_path)
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
        ids = _ask(history.needs_box_tags, request.query_params.get("camera") or None)
        if isinstance(ids, Response):
            return ids
        return JSONResponse({"visits": history.visits_by_id(ids), "count": len(ids)})

    async def api_favorite_set(request: Request):
        """Star or unstar a visit: {"on": true|false, "kind": "still"|"clip", "clip_at": seconds}."""
        if not editable:
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
        if not db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(db_path)
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
        used = _ask(history.tagged_species)
        used = [] if isinstance(used, Response) else used
        seen = [t for t in history.observed_taxa() if t not in used]
        return JSONResponse({"species": used + seen})

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
            a = await _to_thread(asker.ask, str(question))
        except AskError as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)
        except Exception as e:  # the API or network failed: say so, don't crash the dashboard
            return JSONResponse({"error": f"couldn't get an answer ({type(e).__name__}: {e})"}, status_code=502)
        return JSONResponse({"answer": a.text, "visits": a.visits, "cost": a.cost, "spent_today": a.spent_today,
                             "cap": a.cap, "refused": a.refused})

    async def api_visit_hidden_set(request: Request):
        """Show or hide one visit by hand ({"hidden": true|false}), overriding the gate or an exclusion
        zone: the case-by-case undo. Hiding records the reason "manual"."""
        if not editable:
            return JSONResponse({"error": "editing is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        vid = int(request.path_params["vid"])
        try:
            hidden = (await request.json()).get("hidden")
        except (ValueError, json.JSONDecodeError, AttributeError):
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(hidden, bool):
            return JSONResponse({"error": "hidden must be true or false"}, status_code=400)
        if not db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        reason = "manual" if hidden else None
        store = Store(db_path)
        try:
            store.set_visit_suppressed(vid, reason)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "suppressed": reason})

    async def media_thumb(request: Request):
        try:
            data, _ = history.visit_thumbnail(int(request.path_params["vid"]), data_dir)
        except HistoryError:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def media_clip(request: Request):
        from starlette.responses import FileResponse
        try:
            path = history.visit_clip(int(request.path_params["vid"]), data_dir)
        except HistoryError:
            return Response(status_code=404)
        return FileResponse(path, media_type="video/mp4")

    async def frame_path(vid: int) -> Path | int:
        """The full-resolution still of the visit's best-box moment, or an HTTP error status.
        Prefers the still the monitor saved at detection time (exactly aligned to the box); falls back
        to extracting a frame from the clip (approximate) for older visits that have no saved still."""
        from camera_mcp.capture import FFMPEG
        try:  # the aligned still the monitor saved at detection time
            return history.visit_still(vid, data_dir)
        except HistoryError:
            pass
        cache = data_dir / "frames" / f"{vid}.jpg"
        if cache.exists():
            return cache
        try:
            d = history.visit_boxes(vid)
            clip = history.visit_clip(vid, data_dir)
        except HistoryError:
            return 404
        cam = find_camera(cameras, d["camera"])  # clip time of the best-box frame (same math as /boxes)
        pre = cam.monitor.pre_seconds if cam else 5.0
        try:
            start = datetime.fromisoformat(d["started_at"]).timestamp()
        except (ValueError, TypeError):
            start = None
        best = None
        for f in d["frames"]:
            mc = max((b.get("conf", 0) or 0 for b in f["boxes"]), default=0.0)
            if best is None or mc > best[0]:
                best = (mc, f["ts"])
        off = 0.0
        if best and start:
            try:
                off = max(0.0, datetime.fromisoformat(best[1]).timestamp() - start + pre)
            except (ValueError, TypeError):
                off = 0.0
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = await _to_thread(lambda: subprocess.run(
            [FFMPEG, "-y", "-ss", f"{off:.2f}", "-i", str(clip), "-frames:v", "1", "-q:v", "2", str(cache)],
            capture_output=True, timeout=30))
        if r.returncode != 0 or not cache.exists():
            return 502
        return cache

    async def media_frame(request: Request):
        """The full-resolution still, for zooming/panning in the viewer (see frame_path)."""
        from starlette.responses import FileResponse
        path = await frame_path(int(request.path_params["vid"]))
        if isinstance(path, int):
            return Response(status_code=path)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def media_frame_boxed(request: Request):
        """The full-resolution still with the best frame's detector boxes and confidences drawn on it:
        a plain image to open on its own and zoom into, so what was boxed can be inspected closely."""
        from starlette.responses import FileResponse
        vid = int(request.path_params["vid"])
        cache = data_dir / "frames" / f"{vid}-boxed.jpg"  # a visit's boxes never change once it's stored
        if not cache.exists():
            path = await frame_path(vid)
            if isinstance(path, int):
                return Response(status_code=path)
            try:
                boxes = mapped_boxes(vid)
            except HistoryError:
                return Response(status_code=404)
            frames = boxes["frames"]
            best = frames[boxes["best"]]["boxes"] if frames else []
            jpeg = await _to_thread(draw_boxes, path.read_bytes(), best)
            if jpeg is None:
                return Response(status_code=502)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(jpeg)
        return FileResponse(cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def zones_page(request: Request) -> HTMLResponse:
        return HTMLResponse(ZONES_PAGE)

    async def image_page(request: Request) -> HTMLResponse:
        return HTMLResponse(IMAGE_PAGE)  # the page reads the visit id from its own path

    async def api_zones(request: Request) -> JSONResponse:
        return JSONResponse({"editable": editable,
                             "cameras": [_zone_state(c) for c in cameras.values() if c.enabled]})

    async def api_zones_save(request: Request) -> JSONResponse:
        if not editable:
            return JSONResponse({"error": "editing is disabled (CAMERA_MCP_WEB_READONLY)"}, status_code=403)
        cam = find_camera(cameras, request.path_params["camera"])
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
        if zones_path.exists():
            try:
                raw = json.loads(zones_path.read_text())
                raw.setdefault("cameras", {})
            except (ValueError, OSError):
                raw = {"cameras": {}}
        raw["cameras"][cam.name] = {"roi": list(roi) if roi else None,
                                    "motion_roi": shape_json(motion_roi) if motion_roi else None,
                                    "motion_mask": [shape_json(z) for z in motion_mask],
                                    "exclude": [exclude_json(z) for z in exclude]}
        try:
            save_zone_overlay(zones_path, raw)
            load_zone_overlay(zones_path)  # re-validate the file as written
        except (OSError, ConfigError) as e:
            return JSONResponse({"error": f"could not save: {e}"}, status_code=500)

        from dataclasses import replace
        cameras[cam.name] = replace(cam, roi=roi, motion_roi=motion_roi, motion_mask=motion_mask, exclude=exclude)
        reloaded, detail = _reload_monitor()
        return JSONResponse({"ok": True, "reloaded": reloaded, "reload_detail": detail,
                             "scenery_reset": bool(scenery_reset), "zones": _zone_state(cameras[cam.name])})

    async def api_snapshot(request: Request):
        cam = find_camera(cameras, request.path_params["camera"])
        if cam is None or not cam.enabled:
            return Response(status_code=404)
        try:
            frame, _ = await _to_thread(grab_any, cam)
        except CaptureError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        img = frame.image
        w = int(request.query_params.get("w", 640))
        if img.shape[1] > w:
            img = cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])))
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return Response(status_code=500)
        return Response(jpeg.tobytes(), media_type="image/jpeg")

    return Starlette(routes=[
        Route("/", index),
        Route("/analytics", index),  # same single-page doc; the client shows the analytics view
        Route("/map", index),  # ...or the labels map
        Route("/zones", zones_page),
        Route("/visit/{vid:int}/image", image_page),
        Route("/api/cameras", api_cameras),
        Route("/api/status", api_status),
        Route("/api/health", api_health),
        Route("/api/visits", api_visits),
        Route("/api/labels/map", api_label_map),
        Route("/api/ask", api_ask_status, methods=["GET"]),
        Route("/api/ask", api_ask, methods=["POST"]),
        Route("/api/visits/timeline", api_visits_timeline),
        Route("/api/species", api_species),
        Route("/api/summary", api_summary),
        Route("/api/stats", api_stats),
        Route("/api/stats/all", api_stats_all),
        Route("/api/stats/heatmap", api_stats_heatmap),
        Route("/api/zones", api_zones),
        Route("/api/zones/{camera}", api_zones_save, methods=["POST"]),
        Route("/api/snapshot/{camera}", api_snapshot),
        Route("/api/visit/{vid:int}/boxes", api_visit_boxes),
        Route("/api/visit/{vid:int}/label", api_visit_label_get, methods=["GET"]),
        Route("/api/visit/{vid:int}/label", api_visit_label_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/hidden", api_visit_hidden_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/box-label", api_box_label_set, methods=["POST"]),
        Route("/api/visit/{vid:int}/favorite", api_favorite_set, methods=["POST"]),
        Route("/api/labels/needs-box-tags", api_needs_box_tags),
        Route("/api/labels/species", api_tag_species),
        Route("/media/visit/{vid:int}/thumb", media_thumb),
        Route("/media/visit/{vid:int}/frame", media_frame),
        Route("/media/visit/{vid:int}/frame/boxed", media_frame_boxed),
        Route("/media/visit/{vid:int}/clip", media_clip),
    ])


def draw_boxes(jpeg: bytes, boxes: list[dict]) -> bytes | None:
    """`jpeg` with each box (fractions of the full frame, with `conf`) outlined in red and labelled
    with its confidence; line and text scale with the image so they read at full resolution. Labels
    go where they don't cover another label or box (a flock's boxes crowd together)."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    thick = max(2, round(w / 640))
    scale, tthick, gap = max(0.5, w / 2400), max(1, thick - 1), thick + 2
    rects = [(round(b["x"] * w), round(b["y"] * h), round((b["x"] + b["w"]) * w), round((b["y"] + b["h"]) * h))
             for b in boxes]
    for r in rects:
        cv2.rectangle(img, r[:2], r[2:], (40, 40, 230), thick)

    def overlaps(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    placed: list[tuple] = []
    for i in sorted(range(len(boxes)), key=lambda i: -(boxes[i].get("conf") or 0)):  # most confident picks first
        if boxes[i].get("conf") is None:
            continue
        text = f"{round(boxes[i]['conf'] * 100)}%"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, tthick)
        x0, y0, x1, y1 = rects[i]
        spots = [(x0, y0 - gap - th - base), (x0, y1 + gap), (x1 - tw, y0 - gap - th - base), (x1 - tw, y1 + gap),
                 (x1 + gap, y0), (x0 - gap - tw, y0)]  # top-left corners: above, below, right-aligned, beside
        inside = [(x, y, x + tw, y + th + base) for x, y in spots if x >= 0 and y >= 0 and x + tw <= w and y + th + base <= h]
        others = [r for j, r in enumerate(rects) if j != i]
        spot = next((r for r in inside if not any(overlaps(r, o) for o in placed + others)), None) \
            or next((r for r in inside if not any(overlaps(r, o) for o in placed)), None) \
            or (inside or [(x0, max(0, y0 - th - base), x0 + tw, max(0, y0 - th - base) + th + base)])[0]
        placed.append(spot)
        org = (spot[0], spot[1] + th)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tthick + 3, cv2.LINE_AA)  # outline
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (60, 60, 255), tthick, cv2.LINE_AA)
    # imencode, not imwrite: SpeciesNet's yolov5 replaces cv2.imwrite with a 2-argument version.
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


async def _to_thread(fn, *a):
    import anyio
    return await anyio.to_thread.run_sync(lambda: fn(*a))


# ---- the pages ---------------------------------------------------------------
# Each page is a plain HTML file in camera_mcp/pages/ (package data), with its CSS and JS inline:
# dashboard.html (feeds, analytics, map, the visit viewer), zones.html (the zone editor) and
# image.html (one visit's full-res still, filling the window: scroll or pinch to zoom at the pointer,
# drag to pan, double-click to fit; the browser's own image view only offers a click-to-100%
# magnifier). Edit them as HTML: there is no Python string escaping in between.


def _page(name: str) -> str:
    return (resources.files("camera_mcp") / "pages" / name).read_text(encoding="utf-8")


PAGE = _page("dashboard.html")
ZONES_PAGE = _page("zones.html")
IMAGE_PAGE = _page("image.html")


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m camera_mcp.web", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    try:
        cameras = load_cameras(args.config, require_login=False, zones=data_dir / "zones.json")
        site = load_site(args.config)
    except ConfigError as e:
        raise SystemExit(f"web: {e}")
    app = build_app(cameras, data_dir, site)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
