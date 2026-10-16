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
                 include_suppressed=q.get("suppressed") == "1")  # ?suppressed=1: the gate's hidden visits too
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
        offset, and `best`: the frame with the most confident box. Raises HistoryError."""
        d = history.visit_boxes(vid)
        cam = find_camera(cameras, d["camera"])
        roi = cam.roi if cam else None  # boxes are fractions of the roi crop; map to the full frame
        pre = cam.monitor.pre_seconds if cam else 5.0
        try:
            start = datetime.fromisoformat(d["started_at"]).timestamp()
        except (ValueError, TypeError):
            start = None

        def mapbox(b):
            try:
                x, y, w, h = b["bbox"]
            except (KeyError, ValueError, TypeError):
                return None
            if roi:
                rx, ry, rw, rh = roi
                x, y, w, h = rx + x * rw, ry + y * rh, w * rw, h * rh
            return {"conf": b.get("conf"), "x": round(x, 4), "y": round(y, 4),
                    "w": round(w, 4), "h": round(h, 4)}

        frames, best_i, best_conf = [], 0, -1.0
        for i, f in enumerate(d["frames"]):
            boxes = [m for m in (mapbox(b) for b in f["boxes"]) if m]
            try:
                offset = datetime.fromisoformat(f["ts"]).timestamp() - start + pre if start else 0.0
            except (ValueError, TypeError):
                offset = 0.0
            fc = max((b["conf"] or 0 for b in boxes), default=0.0)
            if fc > best_conf:
                best_conf, best_i = fc, i
            frames.append({"offset": round(max(0.0, offset), 2), "label": f["label"],
                           "taxonomy": f["taxonomy"], "score": f["score"], "boxes": boxes})
        return {"camera": d["camera"], "has_clip": d["has_clip"], "roi": list(roi) if roi else None,
                "label": d["label"], "taxonomy": d["taxonomy"], "best": best_i, "frames": frames}

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
        if not db_path.exists():
            return JSONResponse({"error": "there's no visit log yet: the monitor hasn't run"}, status_code=503)
        store = Store(db_path)
        try:
            store.set_visit_label(vid, verdict, note)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        finally:
            store.close()
        return JSONResponse({"ok": True, "label": {"verdict": verdict, "note": note}})

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
        Route("/zones", zones_page),
        Route("/api/cameras", api_cameras),
        Route("/api/status", api_status),
        Route("/api/health", api_health),
        Route("/api/visits", api_visits),
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


# ---- the single page --------------------------------------------------------

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Yard Sentry</title>
<style>
:root{--paper:#f5f4ef;--frame:#fff;--ink:#2c2b27;--muted:#6d6a60;--line:#d3d0c6;--chrome:#eceae2;
  --accent:#5b6b8c;--accent-soft:#dde3ee;--good:#4f7a52;--warn:#a5514a;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#12141a;--frame:#191c22;
  --ink:#d6d9e0;--muted:#868b95;--line:#353a45;--chrome:#20242c;--accent:#8fa6cf;--accent-soft:#28324a;
  --good:#7fae82;--warn:#d98b83;}}
*{box-sizing:border-box}body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  margin:0;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:1.4rem 20px 3rem}
.bar{display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem}
.brand{font-family:var(--mono);font-weight:600;font-size:1.2rem}
.who{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.tabs{display:flex;gap:1rem;flex:1;margin-left:1.3rem}
.tabs a{font-family:var(--mono);font-size:.82rem;color:var(--muted);text-decoration:none;text-transform:lowercase;letter-spacing:.02em;padding:.1rem 0;border-bottom:2px solid transparent}
.tabs a.active{color:var(--accent);border-bottom-color:var(--accent)}
.tabs a:hover{color:var(--ink)}
.morelink{font-family:var(--mono);font-size:.72rem;color:var(--accent);text-decoration:none;white-space:nowrap}
.morelink:hover{text-decoration:underline}
#featsel{font-family:var(--mono);font-size:.72rem;background:var(--frame);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:.2rem .4rem;text-transform:none;letter-spacing:0}
.health{font-family:var(--mono);font-size:.74rem;color:var(--muted);border:1px solid var(--line);
  border-radius:8px;padding:.5rem .7rem;background:var(--frame);margin-bottom:1rem}
.health b{color:var(--ink)} .ok{color:var(--good)} .bad{color:var(--warn)}
.cams{display:flex;gap:.4rem;flex-wrap:wrap;margin:.4rem 0 0}
.chip{font-family:var(--mono);font-size:.68rem;border:1px solid var(--line);border-radius:999px;
  padding:.1rem .5rem;color:var(--muted)} .chip.live{color:var(--good);border-color:var(--good)}
.chip.off{opacity:.55}
.chart{border:1px solid var(--line);border-radius:11px;background:var(--frame);padding:.6rem;margin-bottom:1rem}
.chart svg{width:100%;height:132px;display:block}
.cgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:.55rem}
.cmini{border:1px solid var(--line);border-radius:8px;background:var(--chrome);padding:.3rem .4rem}
.cmini svg{width:100%;height:52px;display:block}
.cmh{display:flex;justify-content:space-between;gap:.5rem;font-family:var(--mono);font-size:.63rem;color:var(--muted);margin-bottom:.15rem}
.cmh b{color:var(--ink)}
/* validated categorical palette for the visits timeline (dark steps swap in dark mode) */
:root{--c-birds:#2a78d6;--c-night:#eb6834;--c-rodents:#1baf7a;--c-pets:#eda100;--c-canids:#e87ba4;--c-person:#4a3aa7;--c-other:#9a988f}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--c-birds:#3987e5;--c-night:#d95926;--c-rodents:#199e70;--c-pets:#c98500;--c-canids:#d55181;--c-person:#9085e9;--c-other:#8a887f}}
:root[data-theme=dark]{--c-birds:#3987e5;--c-night:#d95926;--c-rodents:#199e70;--c-pets:#c98500;--c-canids:#d55181;--c-person:#9085e9;--c-other:#8a887f}
.vtl svg{width:100%;height:120px;display:block}
.vleg{display:flex;flex-wrap:wrap;gap:.1rem .8rem;font-family:var(--mono);font-size:.64rem;color:var(--muted);margin-top:.35rem}
.vleg span{display:inline-flex;align-items:center;gap:.28rem}
.vleg i{width:9px;height:9px;border-radius:2px;display:inline-block}
.vtip{position:fixed;z-index:50;pointer-events:none;background:var(--ink);color:var(--frame);font-family:var(--mono);font-size:.66rem;padding:.3rem .45rem;border-radius:6px;opacity:0;transition:opacity .08s;max-width:220px;line-height:1.45}
#hmap{overflow-x:auto}
.hgrid{display:grid;gap:2px;font-family:var(--mono);font-size:.6rem;min-width:520px}
.hcell{height:15px;border-radius:2px;background:var(--chrome)}
.hlab{color:var(--muted);display:flex;align-items:center;justify-content:center}
.hcamlab{color:var(--ink);padding-right:.45rem;white-space:nowrap;justify-content:flex-end}
.hmode{display:inline-flex;gap:.3rem}
.clabel{display:flex;justify-content:space-between;gap:1rem;font-family:var(--mono);font-size:.66rem;color:var(--muted);margin-top:.35rem;flex-wrap:wrap}
.xax{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;color:var(--muted);margin-top:.15rem;padding:0 3px}
.filters{display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;margin-bottom:1rem}
.filters select,.filters input{font-family:var(--mono);font-size:.78rem;background:var(--frame);
  color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:.3rem .5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:.8rem}
.tile{border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--frame);
  text-decoration:none;color:inherit;display:flex;flex-direction:column}
.tile img{width:100%;aspect-ratio:16/9;object-fit:cover;background:var(--chrome);display:block}
.tile .meta{padding:.45rem .6rem;font-family:var(--mono);font-size:.72rem}
.tile .sp{color:var(--ink);font-weight:600} .tile .sub{color:var(--muted)}
.empty{color:var(--muted);font-family:var(--mono);font-size:.8rem;padding:2rem 0}
.sect{font-family:var(--mono);font-size:.74rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;margin:1.4rem 0 .6rem}
dialog{border:1px solid var(--line);border-radius:12px;background:var(--frame);color:var(--ink);
  max-width:92vw;padding:0}
.stage{position:relative;overflow:hidden;max-width:86vw;max-height:76vh;background:#000;border-radius:12px 12px 0 0;touch-action:none}
.stage.pan{cursor:grab} .stage.panning{cursor:grabbing}
.pz{position:relative;transform-origin:0 0;line-height:0;will-change:transform}
.pz img,.pz video{display:block;max-width:86vw;max-height:76vh}
dialog .vmeta{font-family:var(--mono);font-size:.74rem;padding:.6rem .8rem;display:flex;justify-content:space-between;gap:.8rem;align-items:center;flex-wrap:wrap}
dialog .vmeta a{color:var(--accent);cursor:pointer}
.vact{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.clipx{position:absolute;top:.4rem;right:.4rem;z-index:3;width:2rem;height:2rem;border-radius:50%;
  border:1px solid var(--line);background:rgba(0,0,0,.55);color:#fff;font-size:1rem;line-height:1;cursor:pointer}
.lblset{display:inline-flex;gap:.3rem}
.lblbtn{border:1px solid var(--line);border-radius:6px;padding:.15rem .5rem;color:var(--muted);background:none;font-family:var(--mono);font-size:.72rem;cursor:pointer}
.lblbtn:hover:not(:disabled){border-color:var(--ink);color:var(--ink)}
.lblbtn.on[data-v=real]{color:var(--good);border-color:var(--good)}
.lblbtn.on[data-v=false],.lblbtn.on[data-v=wrong_species]{color:var(--warn);border-color:var(--warn)}
.lblbtn:disabled{opacity:.5;cursor:default}
/* gate-hidden visits: shown only with "+ hidden"; a badge says why, a mark says how they were labelled */
.hidtog{font-family:var(--mono);font-size:.78rem;color:var(--muted);display:inline-flex;align-items:center;gap:.3rem;cursor:pointer}
.tile{position:relative}
.tile.sup img{opacity:.72}
.tile .badge{position:absolute;top:.4rem;left:.4rem;font-family:var(--mono);font-size:.62rem;padding:.05rem .4rem;
  border-radius:999px;background:var(--frame);color:var(--warn);border:1px solid var(--warn)}
/* labelled tiles: outlined, dimmed and badged from data-verdict, so what's left to label stands out */
.tile[data-verdict=real]{outline:2px solid var(--good);outline-offset:-1px}
.tile[data-verdict=false],.tile[data-verdict=wrong_species]{outline:2px solid var(--warn);outline-offset:-1px}
.tile[data-verdict=real] img,.tile[data-verdict=false] img,.tile[data-verdict=wrong_species] img{opacity:.45}
.tile .vpill{display:none;position:absolute;top:.4rem;right:.4rem;font-family:var(--mono);font-size:.66rem;font-weight:600;
  padding:.05rem .45rem;border-radius:999px;background:var(--frame);border:1px solid}
.tile[data-verdict=real] .vpill{display:block;color:var(--good)} .tile[data-verdict=real] .vpill::after{content:"\\2713  real"}
.tile[data-verdict=false] .vpill,.tile[data-verdict=wrong_species] .vpill{display:block;color:var(--warn)}
.tile[data-verdict=false] .vpill::after{content:"\\2717  false"} .tile[data-verdict=wrong_species] .vpill::after{content:"\\2717  wrong sp."}
#hidnote{text-transform:none;letter-spacing:0}
/* the ask box: a question about the visit log, answered with linked visits */
.ask{margin:0 0 1rem}
.ask form{display:flex;gap:.5rem}
.ask input{flex:1;min-width:0;font-family:var(--sans);font-size:.9rem;background:var(--frame);color:var(--ink);
  border:1px solid var(--line);border-radius:9px;padding:.5rem .7rem}
.ask button{font-family:var(--mono);font-size:.8rem;background:var(--accent);color:#fff;border:1px solid var(--accent);
  border-radius:9px;padding:.45rem .9rem;cursor:pointer}
.ask button:disabled{opacity:.6;cursor:default}
.askans{font-size:.92rem;line-height:1.5;margin:.6rem 0 0;white-space:normal}
.askans:empty{display:none}
.askans .vref{font-family:var(--mono);font-size:.8rem;color:var(--accent);cursor:pointer;text-decoration:none;
  border-bottom:1px dotted var(--accent)}
.askans.err{color:var(--warn)}
.askmeta{font-family:var(--mono);font-size:.66rem;color:var(--muted);margin-top:.35rem}
.askgrid{margin-top:.6rem}
.askgrid:empty{display:none}
.boxov{position:absolute;top:0;left:0;pointer-events:none;display:none}
.boxov rect{fill:none;stroke:#ff3b30;stroke-width:2;vector-effect:non-scaling-stroke}
.boxov text{fill:#ff3b30;font-family:var(--mono);font-size:3.2px;paint-order:stroke;
  stroke:rgba(0,0,0,.55);stroke-width:.6px;dominant-baseline:text-after-edge}
.boxbtn{border:1px solid var(--line);border-radius:6px;padding:.15rem .5rem;color:var(--muted);background:none;font-family:var(--mono);font-size:.72rem;cursor:pointer}
.boxbtn.on{color:var(--warn);border-color:var(--warn)}
</style></head><body><div class="wrap">
<div class="bar"><div class="brand">&#129437; Yard Sentry</div>
  <nav class="tabs"><a href="/" data-view="feeds">feeds</a><a href="/analytics" data-view="analytics">analytics</a>
    <a href="/zones">zones &rarr;</a></nav>
  <div class="who"><span id="clock"></span></div></div>
<div class="health" id="health">loading&hellip;</div>
<div class="filters">
  <select id="since"><option value="2h">last 2h</option><option value="12h">12h</option>
    <option value="24h" selected>24h</option><option value="3d">3 days</option><option value="7d">7 days</option></select>
  <select id="taxon"><option value="">all species</option></select>
  <select id="camera"><option value="">all cameras</option></select>
  <label class="hidtog" title="also show unnamed visits the gate hid as likely leaves (to review and label)">
    <input type="checkbox" id="showhidden"> + hidden</label>
</div>

<section id="view-feeds">
  <div class="ask" id="askbox" hidden>
    <form id="askform"><input id="askq" maxlength="500" autocomplete="off"
      placeholder="Ask about your cameras, e.g. all skunks in the last 24 hours?"><button id="askbtn">ask</button></form>
    <div class="askans" id="askans"></div>
    <div class="grid askgrid" id="askgrid"></div>
    <div class="askmeta" id="askmeta"></div>
  </div>
  <div class="sect" style="margin-top:0">Visits <span id="hidnote"></span></div>
  <div class="grid" id="grid"></div>
  <div class="empty" id="empty" style="display:none">No visits in this window.</div>
  <div class="sect">Live</div>
  <div class="grid" id="live"></div>
  <div class="sect" style="display:flex;justify-content:space-between;align-items:center;gap:1rem">
    <span>At a glance</span>
    <span style="display:flex;gap:.6rem;align-items:center">
      <select id="featsel"><option value="timeline">visits over time</option>
        <option value="throughput">throughput &amp; backlog</option>
        <option value="heatmap">hourly load</option></select>
      <a class="morelink" href="/analytics" data-view="analytics">full analytics &rarr;</a>
    </span>
  </div>
  <div class="chart" id="feat"></div>
</section>

<section id="view-analytics" hidden>
  <div class="sect" style="margin-top:0;display:flex;justify-content:space-between;align-items:center;gap:1rem">
    <span>Throughput &amp; backlog</span>
    <select id="chartcam" style="font-family:var(--mono);font-size:.72rem;background:var(--frame);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:.2rem .4rem;text-transform:none;letter-spacing:0">
      <option value="">all cameras</option><option value="__each__">per camera (grid)</option></select>
  </div>
  <div class="chart" id="chart"></div>
  <div class="sect" style="display:flex;justify-content:space-between;align-items:center;gap:1rem">
    <span>Hourly load &middot; by camera</span>
    <span class="hmode"><button class="boxbtn on" data-m="dropped">drops</button><button class="boxbtn" data-m="frames">activity</button></span>
  </div>
  <div class="chart" id="hmap"></div>
  <div class="sect">Visits over time</div>
  <div class="chart vtl" id="vtl"></div>
</section>
<div class="vtip" id="vtip"></div>
<dialog id="clip"><div class="stage pan" id="stage"><button class="clipx" id="clipclose" aria-label="close">&times;</button><div class="pz" id="pz">
    <img id="still" alt="frame">
    <video controls style="display:none"></video>
    <svg class="boxov" id="boxov" viewBox="0 0 100 100" preserveAspectRatio="none"></svg></div></div>
  <div class="vmeta"><span id="clipmeta"></span>
  <span class="vact">
    <button class="boxbtn" id="playbtn">watch clip &#9654;</button>
    <span class="lblset" id="lblset" title="was this detection correct?">
      <button class="lblbtn" data-v="real">&check; real</button>
      <button class="lblbtn" data-v="false">&times; false</button>
      <button class="lblbtn" data-v="wrong_species">&times; wrong sp.</button></span>
    <button class="boxbtn" id="nextbtn" title="the next visit in the grid you haven't labelled">next unlabelled &rsaquo;</button>
    <button class="boxbtn" id="hidebtn" title="hide this visit, or show one the gate or a zone hid (undo a mistake)">hide</button>
    <button class="boxbtn on" id="boxbtn">hide boxes</button> &middot; <a id="cliplink" title="the full-resolution still with the detector's boxes, on its own (pinch or zoom to inspect)">full-res image</a>
  </span></div></dialog>
<script>
const $=s=>document.querySelector(s), params=()=>{const p=new URLSearchParams();
  p.set('since',$('#since').value); if($('#taxon').value)p.set('taxon',$('#taxon').value);
  if($('#camera').value)p.set('camera',$('#camera').value);
  if($('#showhidden').checked)p.set('suppressed','1'); return p.toString();};
function tick(){$('#clock').textContent=new Date().toLocaleString();}
async function loadHealth(){try{const win=$('#since').value;
  const h=await (await fetch('/api/health?since='+win)).json();
  const st=await (await fetch('/api/status')).json();
  const watching=st.filter(c=>c.watching_now).length, total=st.filter(c=>c.state!=='never seen by the monitor').length;
  let s;
  if(h.samples){const dr=h.segments_dropped; s=`<b>Analysis (${win}):</b> ${h.frames_analyzed.toLocaleString()} frames &middot; `+
    `<span class="${dr?'bad':'ok'}">${dr?dr+' dropped':'none dropped'}</span> &middot; peak lag ${Math.round(h.peak_lag_seconds)}s`;}
  else s='<b>Analysis:</b> no stats yet';
  s+=`<div class="cams">`+st.map(c=>`<span class="chip ${c.watching_now?'live':(c.state==='paused'?'off':'')}">`+
    `${c.camera}${c.state==='paused'?' (paused)':''}</span>`).join('')+`</div>`;
  $('#health').innerHTML=s;}catch(e){$('#health').textContent='health unavailable';}}
async function loadCameras(){const cs=await (await fetch('/api/cameras')).json();
  const opts=cs.filter(c=>c.enabled).map(c=>`<option value="${c.name}">${c.name}</option>`).join('');
  $('#camera').insertAdjacentHTML('beforeend',opts);
  $('#chartcam').insertAdjacentHTML('beforeend',opts);
  $('#live').innerHTML=cs.filter(c=>c.enabled&&c.stream).map(c=>`<a class="tile"><img loading="lazy" `+
    `src="/api/snapshot/${c.name}?w=480" alt="${c.name}"><div class="meta"><span class="sp">${c.name}</span></div></a>`).join('');}
async function loadSpecies(){try{const r=await (await fetch('/api/species')).json();
  $('#taxon').insertAdjacentHTML('beforeend',(r.species||[]).map(s=>`<option value="${s}">${s}</option>`).join(''));
  }catch(e){}}
async function loadVisits(){const r=await (await fetch('/api/visits?'+params())).json();
  const v=(r.visits||[]).filter(x=>x.has_thumbnail);
  $('#empty').style.display=v.length?'none':'block';
  $('#hidnote').textContent=r.hidden?`· ${r.hidden} hidden as likely leaves`:'';
  $('#grid').innerHTML=v.map(tileHTML).join('');
  document.querySelectorAll('#grid .tile').forEach(t=>t.onclick=ev=>{ev.preventDefault();openVisit(t);});}
function tileHTML(x){return `<a class="tile${x.suppressed?' sup':''}" href="#" data-id="${x.id}" data-clip="${x.clip?1:0}"`+
    ` data-verdict="${x.verdict||''}" data-sup="${x.suppressed||''}">`+
    `<img loading="lazy" src="/media/visit/${x.id}/thumb" alt="${esc(x.label)}">`+
    (x.suppressed?`<span class="badge">hidden: ${x.suppressed.replace('_',' ')}</span>`:'')+`<span class="vpill"></span>`+
    `<div class="meta"><div class="sp">${esc(x.label)}${x.max_at_once>1?' ×'+x.max_at_once:''}`+
    `</div>`+
    `<div class="sub">${x.camera} &middot; ${(x.started||'').slice(5,16).replace('T',' ')}</div></div></a>`;}
const esc=s=>String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// ---- the ask box ----
async function askInit(){try{const d=await (await fetch('/api/ask')).json(); if(!d.enabled)return;
  $('#askbox').hidden=false; askMeta(d.spent_today,d.cap);}catch(e){}}
function askMeta(spent,cap,cost){$('#askmeta').textContent=(cost!=null?`this answer ≈ $${cost.toFixed(3)} · `:'')+
  `$${(+spent).toFixed(2)} of $${(+cap).toFixed(2)} used today · answers use the visit log only (no images)`;}
$('#askform').addEventListener('submit',async ev=>{ev.preventDefault(); const q=$('#askq').value.trim(); if(!q)return;
  $('#askbtn').disabled=true; $('#askans').className='askans'; $('#askans').textContent='looking…'; $('#askgrid').innerHTML='';
  try{const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const d=await res.json();
    if(!res.ok){$('#askans').className='askans err';$('#askans').textContent=d.error||'something went wrong';return;}
    const byId={}; (d.visits||[]).forEach(v=>byId[v.id]=v);
    $('#askans').innerHTML=esc(d.answer).replace(/\\[#(\\d+)\\]/g,(m,id)=>byId[id]?`<a class="vref" data-id="${id}">#${id}</a>`:m)
      .replace(/\\n/g,'<br>');
    $('#askgrid').innerHTML=(d.visits||[]).filter(v=>v.has_thumbnail).map(tileHTML).join('');
    $('#askgrid').querySelectorAll('.tile').forEach(t=>t.onclick=e=>{e.preventDefault();openVisit(t);});
    $('#askans').querySelectorAll('.vref').forEach(a=>a.onclick=()=>{const t=$('#askgrid .tile[data-id="'+a.dataset.id+'"]');if(t)openVisit(t);});
    askMeta(d.spent_today,d.cap,d.cost);
  }catch(e){$('#askans').className='askans err';$('#askans').textContent='network error: '+e;}
  finally{$('#askbtn').disabled=false;}});
askInit();
let labelEditable=true;
function markLabel(v){$('#lblset').querySelectorAll('.lblbtn').forEach(b=>b.classList.toggle('on',b.dataset.v===v));}
async function loadLabel(id){markLabel(null);
  $('#lblset').querySelectorAll('.lblbtn').forEach(b=>b.disabled=true);
  try{const d=await (await fetch('/api/visit/'+id+'/label')).json();
    labelEditable=d.editable; markLabel(d.label&&d.label.verdict);
    $('#lblset').querySelectorAll('.lblbtn').forEach(b=>b.disabled=!d.editable);
  }catch(e){}}
async function setLabel(v){if(!labelEditable||!curVisit)return;
  try{const res=await fetch('/api/visit/'+curVisit+'/label',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict:v})});
    if(res.ok){markLabel(v); const t=tileOf(curVisit);
      if(t){const had=t.dataset.verdict; t.dataset.verdict=v;  // CSS restyles the tile from this
        if(!had)setTimeout(nextUnlabelled,350);}}}catch(e){}}  // first label on a visit: move on to the next
function tileOf(id){return document.querySelector('#grid .tile[data-id="'+id+'"]')||document.querySelector('#askgrid .tile[data-id="'+id+'"]');}
function nextUnlabelled(){const ts=[...document.querySelectorAll('#grid .tile')], i=ts.indexOf(tileOf(curVisit));
  const t=[...ts.slice(i+1),...ts.slice(0,Math.max(i,0))].find(x=>!x.dataset.verdict);
  if(t)openVisit(t); else $('#clipmeta').textContent='every visit in the grid is labelled';}
$('#lblset').querySelectorAll('.lblbtn').forEach(b=>b.onclick=()=>setLabel(b.dataset.v));
// A visit opens on a full-res still (zoom/pan); "watch clip" swaps to the video. Boxes on by default.
let curVisit=null, curClip=false, mode='image', boxesOn=true, boxData=null, stillFB=false;
let zs=1, ztx=0, zty=0;
function applyZoom(){$('#pz').style.transform='translate('+ztx+'px,'+zty+'px) scale('+zs+')';}
function resetZoom(){zs=1;ztx=0;zty=0;applyZoom();}
function activeMedia(){return mode==='video'?$('#clip video'):$('#still');}
// Show/hide by hand: the case-by-case override of the gate and exclusion zones.
function markHidden(t){const sup=t&&t.dataset.sup; $('#hidebtn').textContent=sup?'show ('+sup.replace('_',' ')+')':'hide';
  $('#hidebtn').classList.toggle('on',!!sup);}
async function toggleHidden(){const t=tileOf(curVisit); if(!t||!labelEditable)return; const hide=!t.dataset.sup;
  try{const res=await fetch('/api/visit/'+curVisit+'/hidden',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({hidden:hide})}); if(!res.ok)return; const d=await res.json();
    t.dataset.sup=d.suppressed||''; t.classList.toggle('sup',!!d.suppressed);
    let b=t.querySelector('.badge'); if(d.suppressed){if(!b){b=document.createElement('span');b.className='badge';
      t.insertBefore(b,t.querySelector('.vpill'));} b.textContent='hidden: '+d.suppressed;} else if(b)b.remove();
    markHidden(t); $('#clipmeta').textContent=$('#clipmeta').textContent.replace(/ · hidden: .*$/,'')+
      (d.suppressed?' · hidden: '+d.suppressed:'');}catch(e){}}
function openVisit(t){const id=t.dataset.id; curVisit=id; curClip=t.dataset.clip==='1'; markHidden(t);
  boxData=null; mode='image'; boxesOn=curClip; stillFB=false; resetZoom();
  const img=$('#still'), vid=$('#clip video');
  img.style.display=''; vid.pause(); vid.removeAttribute('src'); vid.style.display='none';
  img.src=(curClip?'/media/visit/'+id+'/frame':'/media/visit/'+id+'/thumb');
  $('#stage').classList.add('pan');
  $('#playbtn').style.display=curClip?'':'none'; $('#playbtn').textContent='watch clip ▶';
  $('#boxbtn').style.display=curClip?'':'none';
  $('#boxbtn').classList.toggle('on',boxesOn); $('#boxbtn').textContent=boxesOn?'hide boxes':'show boxes';
  $('#clipmeta').textContent=t.querySelector('.sp').firstChild.textContent+' — '+t.querySelector('.sub').textContent+
    (t.dataset.sup?' · hidden: '+t.dataset.sup.replace('_',' '):'');
  $('#cliplink').href=(curClip?'/media/visit/'+id+'/frame/boxed':'/media/visit/'+id+'/thumb');  // the clip has "watch clip"
  loadLabel(id); if(!$('#clip').open)$('#clip').showModal();
  if(curClip) loadBoxData().then(drawBoxes);}
async function loadBoxData(){if(boxData)return; try{boxData=await (await fetch('/api/visit/'+curVisit+'/boxes')).json();}catch(e){boxData={frames:[]};}}
function drawBoxes(){const svg=$('#boxov'), m=activeMedia();
  svg.style.width=(m.clientWidth||m.offsetWidth)+'px'; svg.style.height=(m.clientHeight||m.offsetHeight)+'px';
  if(!boxesOn||!boxData){svg.style.display='none';return;}
  if(!boxData.frames||!boxData.frames.length){svg.innerHTML='<text x="2" y="7">no boxes stored</text>';svg.style.display='block';return;}
  const f=boxData.frames[boxData.best]||boxData.frames[0]; let s='';
  (f.boxes||[]).forEach(b=>{const x=b.x*100,y=b.y*100,w=b.w*100,h=b.h*100;
    s+='<rect x="'+x.toFixed(2)+'" y="'+y.toFixed(2)+'" width="'+w.toFixed(2)+'" height="'+h.toFixed(2)+'"/>';
    if(b.conf!=null)s+='<text x="'+x.toFixed(2)+'" y="'+Math.max(3.2,y).toFixed(2)+'">'+Math.round(b.conf*100)+'%</text>';});
  svg.innerHTML=s; svg.style.display='block';}
function toggleBoxes(){boxesOn=!boxesOn; $('#boxbtn').classList.toggle('on',boxesOn);
  $('#boxbtn').textContent=boxesOn?'hide boxes':'show boxes'; drawBoxes();}
function playClip(){if(!curClip)return; const img=$('#still'), vid=$('#clip video');
  if(mode==='image'){mode='video'; resetZoom(); $('#stage').classList.remove('pan');
    img.style.display='none'; vid.style.display=''; vid.src='/media/visit/'+curVisit+'/clip';
    $('#playbtn').textContent='◀ back to image';}
  else{mode='image'; resetZoom(); $('#stage').classList.add('pan');
    vid.pause(); vid.removeAttribute('src'); vid.style.display='none'; img.style.display='';
    $('#playbtn').textContent='watch clip ▶'; drawBoxes();}}
$('#playbtn').onclick=playClip; $('#boxbtn').onclick=toggleBoxes; $('#nextbtn').onclick=nextUnlabelled; $('#hidebtn').onclick=toggleHidden;
$('#still').addEventListener('load',()=>{if(mode==='image')drawBoxes();});
$('#still').addEventListener('error',()=>{if(curClip&&!stillFB){stillFB=true;$('#still').src='/media/visit/'+curVisit+'/thumb';}});
$('#clip video').addEventListener('loadeddata',()=>{if(mode!=='video')return;
  const vid=$('#clip video'), f=boxData&&boxData.frames&&boxData.frames[boxData.best];
  if(f){try{vid.pause();if(f.offset)vid.currentTime=f.offset;}catch(e){}} drawBoxes();});
(function(){const stage=$('#stage'); let drag=false,lx=0,ly=0;
  stage.addEventListener('wheel',e=>{if(mode!=='image')return; e.preventDefault();
    const r=stage.getBoundingClientRect(),cx=e.clientX-r.left,cy=e.clientY-r.top;
    const f=e.deltaY<0?1.15:1/1.15,ns=Math.min(8,Math.max(1,zs*f));
    ztx=cx-(cx-ztx)*(ns/zs); zty=cy-(cy-zty)*(ns/zs); zs=ns; if(zs===1){ztx=0;zty=0;} applyZoom();},{passive:false});
  stage.addEventListener('pointerdown',e=>{if(mode!=='image'||zs===1)return; drag=true;lx=e.clientX;ly=e.clientY;
    stage.classList.add('panning'); try{stage.setPointerCapture(e.pointerId);}catch(_){}});
  stage.addEventListener('pointermove',e=>{if(!drag)return; ztx+=e.clientX-lx; zty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; applyZoom();});
  stage.addEventListener('pointerup',()=>{drag=false;stage.classList.remove('panning');});
  stage.addEventListener('dblclick',()=>{if(mode==='image')resetZoom();});})();
$('#clip').onclick=e=>{if(e.target.id==='clip')$('#clip').close();};
$('#clipclose').addEventListener('pointerdown',e=>e.stopPropagation());  // don't start a pan
$('#clipclose').onclick=e=>{e.stopPropagation();$('#clip').close();};
$('#clip').addEventListener('close',()=>{const v=$('#clip video');try{v.pause();}catch(_){}} );  // stop audio on close
// one lag-area chart as an SVG string; shared by the big chart and the per-camera minis
function lagSVG(pts,W,H,maxLag,grid){const pad=4,n=pts.length,cap=1800;
  const X=i=>pad+i*(W-2*pad)/(n-1||1), Y=v=>H-pad-(v/maxLag)*(H-2*pad);
  let area='M'+X(0).toFixed(1)+','+(H-pad);
  pts.forEach((p,i)=>area+=' L'+X(i).toFixed(1)+','+Y(p.lag).toFixed(1));
  area+=' L'+X(n-1).toFixed(1)+','+(H-pad)+' Z';
  // activity: frames analysed per sample, on its own scale (shape, not comparable to lag)
  const maxF=pts.reduce((m,p)=>Math.max(m,p.frames||0),0);
  let act='';
  if(maxF>0){const YF=v=>H-pad-(v/maxF)*(H-2*pad);
    act='<polyline fill="none" stroke="var(--good)" stroke-width="1.2" opacity=".85" points="'+
        pts.map((p,i)=>X(i).toFixed(1)+','+YF(p.frames||0).toFixed(1)).join(' ')+'"/>';}
  const dh=Math.max(5,H*0.06);
  const drops=pts.map((p,i)=>p.dropped>0?`<rect x="${X(i).toFixed(1)}" y="${(H-dh).toFixed(1)}" width="2" height="${dh.toFixed(1)}" fill="var(--warn)"/>`:'').join('');
  let g='';
  if(grid){const tix=[0,.25,.5,.75,1].map(f=>Math.round(f*(n-1)));
    g=tix.map(i=>`<line x1="${X(i).toFixed(1)}" y1="0" x2="${X(i).toFixed(1)}" y2="${(H-9).toFixed(1)}" stroke="var(--line)" stroke-width="1" opacity=".45"/>`).join('');}
  const capLine=cap<=maxLag?`<line x1="0" y1="${Y(cap).toFixed(1)}" x2="${W}" y2="${Y(cap).toFixed(1)}" stroke="var(--warn)" stroke-dasharray="5 5" stroke-width="1" opacity=".55"/>`:'';
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`+g+capLine+
    `<path d="${area}" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.5"/>${act}${drops}</svg>`;}
const peakFrames=pts=>pts.reduce((m,p)=>Math.max(m,p.frames||0),0);
const sumFrames=pts=>pts.reduce((s,p)=>s+(p.frames||0),0);
const fmtK=n=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+'k':''+n;
function xAxis(pts){const n=pts.length,tix=[0,.25,.5,.75,1].map(f=>Math.round(f*(n-1)));
  return tix.map((i,k)=>`<span style="${k===0?'':(k===tix.length-1?'text-align:right':'text-align:center')}">${(pts[i].at||'').slice(11,16)}</span>`).join('');}
const peakLag=pts=>pts.reduce((m,p)=>Math.max(m,p.lag),0);
async function loadStats(sel='#chart',forceAll=false){const el=$(sel); try{const cc=forceAll?'':$('#chartcam').value;
  if(cc==='__each__') return loadStatsGrid(el);
  const d=await (await fetch('/api/stats?since='+$('#since').value+(cc?'&camera='+encodeURIComponent(cc):''))).json();
  const pts=d.points||[];
  if(!pts.length){el.innerHTML='<div class="empty" style="padding:1rem">'+(d.camera?'no per-camera stats for '+d.camera+' yet (recorded from this build on)':'no stats yet')+'</div>';return;}
  const maxLag=Math.max(30,peakLag(pts));  // fit the data; the 30-min cap line shows only when neared
  el.innerHTML=lagSVG(pts,1000,132,maxLag,true)+
    `<div class="xax">${xAxis(pts)}</div>`+
    `<div class="clabel"><span>${d.camera?'<b>'+d.camera+'</b> &middot; ':''}<b style="color:var(--accent)">lag</b> (area) &middot; <b style="color:var(--good)">activity</b> (line) &middot; <span class="bad">drops</span> (ticks)${peakLag(pts)>=1800?' &middot; dashed = 30-min cap':''}</span>`+
    `<span>${fmtK(sumFrames(pts))} frames &middot; ${(d.from||'').slice(0,10)}</span></div>`;
}catch(e){el.innerHTML='<div class="empty" style="padding:1rem">stats unavailable</div>';}}
async function loadStatsGrid(el){const d=await (await fetch('/api/stats/all?since='+$('#since').value)).json();
  const cams=d.cameras||{}, names=Object.keys(cams).filter(n=>cams[n].length);
  if(!names.length){el.innerHTML='<div class="empty" style="padding:1rem">no per-camera stats yet (recorded from the latest build on)</div>';return;}
  let maxLag=30; names.forEach(n=>{const p=peakLag(cams[n]); if(p>maxLag)maxLag=p;});  // shared lag scale, fit to data
  const totDrop=n=>cams[n].reduce((s,p)=>s+p.dropped,0);
  const busy=n=>totDrop(n)*1e7+sumFrames(cams[n]);  // worst first: drops, then total activity
  names.sort((a,b)=>busy(b)-busy(a));
  let html='<div class="cgrid">';
  names.forEach(n=>{const pts=cams[n], pk=Math.round(peakLag(pts)), dr=totDrop(n);
    html+=`<div class="cmini"><div class="cmh"><b>${n}</b><span>${fmtK(sumFrames(pts))}f &middot; peak ${pk}s${dr?' &middot; <span class="bad">'+dr+' drop</span>':''}</span></div>`+
      lagSVG(pts,500,52,maxLag,false)+'</div>';});
  const capNote=maxLag>=1800?' &middot; dashed = 30-min cap':'';
  html+=`</div><div class="clabel"><span><b style="color:var(--accent)">lag</b> (area, shared scale) &middot; <b style="color:var(--good)">activity</b> (line, per-camera) &middot; <span class="bad">drops</span>${capNote}</span>`+
    `<span>${(d.from||'').slice(0,10)} &middot; ${(d.from||'').slice(11,16)}&ndash;${(d.to||'').slice(11,16)}</span></div>`;
  el.innerHTML=html;}
// ---- visits timeline (what visited, when) ----
const GROUP_CLASS={songbirds:'birds',jays:'birds',crows:'birds','doves/pigeons':'birds',hawks:'birds',owls:'birds',bird:'birds',
  raccoon:'night',skunk:'night',opossum:'night', rat:'rodents',squirrel:'rodents',rodent:'rodents',
  cat:'pets',dog:'pets', fox:'canids',coyote:'canids', person:'person'};
const TL_CLASSES=[['birds','birds','--c-birds'],['night','night-critters','--c-night'],['rodents','rodents','--c-rodents'],
  ['pets','cat/dog','--c-pets'],['canids','fox/coyote','--c-canids'],['person','person','--c-person'],['other','other','--c-other']];
const classOf=g=>GROUP_CLASS[g]||'other';
function showTip(e,html){const t=$('#vtip');t.innerHTML=html;t.style.opacity='1';
  t.style.left=Math.min(e.clientX+12,innerWidth-230)+'px';t.style.top=(e.clientY+14)+'px';}
function hideTip(){$('#vtip').style.opacity='0';}
async function loadTimeline(sel='#vtl'){const el=$(sel); try{
  const p=new URLSearchParams();p.set('since',$('#since').value);if($('#camera').value)p.set('camera',$('#camera').value);
  const d=await (await fetch('/api/visits/timeline?'+p.toString())).json();
  const bins=d.bins||[];
  if(!d.visits){el.innerHTML='<div class="empty" style="padding:1rem">no visits in this window</div>';return;}
  const data=bins.map(b=>{const c={};for(const[g,v]of Object.entries(b.grps||{}))c[classOf(g)]=(c[classOf(g)]||0)+v;return {t:b.t,c,raw:b.grps};});
  const present=TL_CLASSES.filter(([k])=>data.some(b=>b.c[k]));
  const maxTot=Math.max(1,...data.map(b=>Object.values(b.c).reduce((s,v)=>s+v,0)));
  const W=1000,H=120,pad=4,n=data.length,bw=(W-2*pad)/n;
  let bars='';
  data.forEach((b,i)=>{let y=H-pad;const x=pad+i*bw;
    present.forEach(([k,,v])=>{const cnt=b.c[k]||0;if(!cnt)return;const h=(cnt/maxTot)*(H-2*pad);
      bars+=`<rect x="${(x+0.6).toFixed(1)}" y="${(y-h).toFixed(1)}" width="${Math.max(0.5,bw-1.2).toFixed(1)}" height="${h.toFixed(1)}" fill="var(${v})"></rect>`;
      y-=h+(h>3?0.8:0);});
    const tot=Object.values(b.c).reduce((s,v)=>s+v,0);
    const lines=Object.entries(b.raw).sort((a,b)=>b[1]-a[1]).map(([g,c])=>g+' '+c).join('<br>');
    bars+=`<rect x="${x.toFixed(1)}" y="0" width="${bw.toFixed(1)}" height="${H}" fill="transparent" class="vhit" data-tip="${(b.t||'').slice(11,16)} &middot; ${tot} visit${tot==1?'':'s'}<br>${lines}"></rect>`;});
  const tix=[0,.25,.5,.75,1].map(f=>Math.round(f*(bins.length-1)));
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${bars}</svg>`+
    `<div class="xax">`+tix.map((i,k)=>`<span style="${k===0?'':(k===4?'text-align:right':'text-align:center')}">${(bins[i].t||'').slice(11,16)}</span>`).join('')+`</div>`+
    `<div class="vleg">`+present.map(([,label,v])=>`<span><i style="background:var(${v})"></i>${label}</span>`).join('')+
    `<span style="margin-left:auto">${d.visits} visits &middot; ${(d.from||'').slice(0,10)}</span></div>`;
  el.querySelectorAll('.vhit').forEach(r=>{r.addEventListener('mousemove',e=>showTip(e,r.dataset.tip));r.addEventListener('mouseleave',hideTip);});
}catch(e){el.innerHTML='<div class="empty" style="padding:1rem">timeline unavailable</div>';}}
// ---- hourly load heatmap (hour-of-day x camera) ----
async function loadHeatmap(sel='#hmap',forceMode=null){const el=$(sel),mode=forceMode||($('.hmode .on')&&$('.hmode .on').dataset.m)||'dropped'; try{
  const d=await (await fetch('/api/stats/heatmap?since='+$('#since').value)).json();
  const cams=d.cameras||[];
  if(!cams.length){el.innerHTML='<div class="empty" style="padding:1rem">no per-camera stats yet (recorded from the latest build on)</div>';return;}
  const max=mode==='dropped'?d.max_dropped:d.max_frames, hue=mode==='dropped'?'--warn':'--accent';
  let html='<div class="hgrid" style="grid-template-columns:auto repeat(24,1fr)"><div></div>';
  for(let h=0;h<24;h++)html+=`<div class="hlab">${h%6===0?h:''}</div>`;
  cams.forEach(cam=>{html+=`<div class="hlab hcamlab">${cam}</div>`;
    d.cells[cam].forEach((c,h)=>{const v=mode==='dropped'?c.dropped:c.frames,intl=max?v/max:0;
      const bg=v?`color-mix(in srgb, var(${hue}) ${Math.round(12+intl*88)}%, transparent)`:'var(--chrome)';
      html+=`<div class="hcell" style="background:${bg}" data-tip="<b>${cam}</b> &middot; ${h}:00&ndash;${h+1}:00<br>${c.frames.toLocaleString()} frames &middot; ${c.dropped} dropped"></div>`;});});
  html+=`</div><div class="clabel"><span>${mode==='dropped'?'<span class="bad">segments dropped</span>':'frames analysed'} per hour of day &middot; darker = more &middot; hover a cell</span><span>${(d.from||'').slice(0,10)} &rarr; ${(d.to||'').slice(0,10)}</span></div>`;
  el.innerHTML=html;
  el.querySelectorAll('.hcell').forEach(c=>{c.addEventListener('mousemove',e=>showTip(e,c.dataset.tip));c.addEventListener('mouseleave',hideTip);});
}catch(e){el.innerHTML='<div class="empty" style="padding:1rem">heatmap unavailable</div>';}}
// ---- the one featured chart on the feeds view (user-selectable, persisted per browser) ----
function loadFeatured(){const k=$('#featsel').value;
  if(k==='throughput')loadStats('#feat',true);        // always all-cameras in the compact slot
  else if(k==='heatmap')loadHeatmap('#feat','dropped');
  else loadTimeline('#feat');}
try{const f=localStorage.getItem('feat'); if(f)$('#featsel').value=f;}catch(e){}
$('#featsel').onchange=()=>{try{localStorage.setItem('feat',$('#featsel').value);}catch(e){} loadFeatured();};

// ---- two views (feeds / analytics), bookmarkable at / and /analytics ----
let VIEW='feeds';
function renderView(){if(VIEW==='analytics'){loadStats();loadTimeline();loadHeatmap();}else{loadVisits();loadFeatured();}}
function setView(v,push){VIEW=v==='analytics'?'analytics':'feeds';
  $('#view-feeds').hidden=VIEW!=='feeds'; $('#view-analytics').hidden=VIEW!=='analytics';
  document.querySelectorAll('.tabs a[data-view]').forEach(a=>a.classList.toggle('active',a.dataset.view===VIEW));
  if(push)history.pushState({v:VIEW},'',VIEW==='analytics'?'/analytics':'/');
  renderView();}
document.querySelectorAll('a[data-view]').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();setView(a.dataset.view,true);}));
addEventListener('popstate',()=>setView(location.pathname.indexOf('analytics')>=0?'analytics':'feeds',false));

$('.hmode').querySelectorAll('button').forEach(b=>b.onclick=()=>{$('.hmode .on').classList.remove('on');b.classList.add('on');loadHeatmap();});
// filters are global; a change refreshes the header and whichever view is showing
['since','taxon','camera','showhidden'].forEach(id=>$('#'+id).addEventListener('change',()=>{loadHealth();renderView();}));
$('#chartcam').onchange=()=>{if(VIEW==='analytics')loadStats();};
tick();setInterval(tick,1000);loadCameras();loadSpecies();loadHealth();
setView(location.pathname.indexOf('analytics')>=0?'analytics':'feeds',false);
setInterval(()=>{loadHealth();renderView();},60000);
</script></div></body></html>"""


ZONES_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Zones &middot; Yard Sentry</title>
<style>
:root{--paper:#f5f4ef;--frame:#fff;--ink:#2c2b27;--muted:#6d6a60;--line:#d3d0c6;--chrome:#eceae2;
  --accent:#5b6b8c;--accent-soft:#dde3ee;--good:#4f7a52;--warn:#a5514a;--excl:#9a6a12;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--paper:#12141a;--frame:#191c22;
  --ink:#d6d9e0;--muted:#868b95;--line:#353a45;--chrome:#20242c;--accent:#8fa6cf;--accent-soft:#28324a;
  --good:#7fae82;--warn:#d98b83;--excl:#d9a441;}}
*{box-sizing:border-box}body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  margin:0;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:1.4rem 20px 3rem}
.bar{display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap;margin-bottom:1rem}
.brand{font-family:var(--mono);font-weight:600;font-size:1.2rem}
.brand a{color:inherit;text-decoration:none}
.who{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.row{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin-bottom:.8rem}
select,button{font-family:var(--mono);font-size:.78rem;background:var(--frame);color:var(--ink);
  border:1px solid var(--line);border-radius:7px;padding:.35rem .6rem;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button:disabled{opacity:.5;cursor:default}
.tools{display:flex;gap:.3rem;flex-wrap:wrap}
.tools label{font-family:var(--mono);font-size:.72rem;border:1px solid var(--line);border-radius:999px;
  padding:.2rem .6rem;cursor:pointer;color:var(--muted);display:inline-flex;gap:.3rem;align-items:center}
.tools input{accent-color:var(--accent)}
.tools label.roi.on{color:var(--accent);border-color:var(--accent)}
.tools label.motion_roi.on{color:var(--good);border-color:var(--good)}
.tools label.motion_mask.on{color:var(--warn);border-color:var(--warn)}
.tools label.exclude.on{color:var(--excl);border-color:var(--excl)}
.tools label.shape.on{color:var(--ink);border-color:var(--ink)}
.tools label.dis{opacity:.4;cursor:default}
.stage{position:relative;border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--chrome);
  max-width:960px;aspect-ratio:16/9;min-height:200px;touch-action:none;user-select:none}
.pz{position:absolute;inset:0;transform-origin:0 0}
.pz img{width:100%;height:100%;object-fit:contain;display:block;pointer-events:none}
.stage .msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:.74rem;color:var(--muted);text-align:center;padding:1rem;pointer-events:none}
.stage.noshot .msg{display:flex}
#ov{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;overflow:visible}
.stage.panning #ov,.stage.pan #ov{cursor:grab}
#ov .z{stroke-width:2;vector-effect:non-scaling-stroke}
#ov .roi{stroke:var(--accent);fill:rgba(91,107,140,.14)}
#ov .motion_roi{stroke:var(--good);fill:rgba(79,122,82,.16)}
#ov .motion_mask{stroke:var(--warn);fill:rgba(165,81,74,.16)}
#ov .exclude{stroke:var(--excl);fill:rgba(154,106,18,.16);stroke-dasharray:6 4}
#ov .exclude.ignore{stroke-dasharray:none;fill:rgba(154,106,18,.3)}
#ov .draft{fill:none;stroke:var(--ink);stroke-dasharray:4 3}
.tags,.handles{position:absolute;inset:0;pointer-events:none}
.tag{position:absolute;font-family:var(--mono);font-size:.6rem;padding:0 .25rem;color:#fff;background:var(--accent);
  transform-origin:0 0;white-space:nowrap}
.tag.motion_roi{background:var(--good)} .tag.motion_mask{background:var(--warn)} .tag.exclude{background:var(--excl)}
.hd{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--frame);
  border:2px solid var(--ink);pointer-events:auto;cursor:move;transform-origin:50% 50%}
.hd.first{background:var(--ink)}
.zoombar{position:absolute;right:.5rem;bottom:.5rem;display:flex;gap:.25rem;z-index:2}
.zoombar button{padding:.15rem .5rem;font-size:.8rem;opacity:.9}
.zoombar button.on{background:var(--ink);color:var(--frame)}
.zoombar .lvl{font-family:var(--mono);font-size:.68rem;color:var(--muted);background:var(--frame);border:1px solid var(--line);
  border-radius:7px;padding:.15rem .4rem;min-width:3.2rem;text-align:center}
.list button.mode{color:var(--excl);border-color:var(--excl)}
.side{font-family:var(--mono);font-size:.72rem}
.zdesc{font-family:var(--sans);font-size:.72rem;color:var(--muted);line-height:1.4;margin:.2rem 0 .1rem}
.list{margin:.3rem 0 0;padding:0;list-style:none}
.list li{display:flex;justify-content:space-between;gap:.5rem;padding:.2rem 0;border-bottom:1px solid var(--line)}
.list .k{color:var(--muted)} .list button{padding:.05rem .4rem;font-size:.66rem}
.hint{color:var(--muted);font-family:var(--mono);font-size:.7rem;margin:.5rem 0}
.status{font-family:var(--mono);font-size:.74rem;margin-top:.8rem;min-height:1.2rem}
.status.ok{color:var(--good)} .status.err{color:var(--warn)}
.cols{display:grid;grid-template-columns:minmax(0,960px) 240px;gap:1.2rem;align-items:start}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="bar"><div class="brand"><a href="/">&#129437; Yard Sentry</a> &middot; zones</div>
  <div class="who"><a href="/" style="color:var(--accent);text-decoration:none">&larr; dashboard</a></div></div>
<div class="row">
  <select id="camera"></select>
  <div class="tools">
    <label class="roi on"><input type="radio" name="tool" value="roi" checked>analysis ROI</label>
    <label class="motion_roi"><input type="radio" name="tool" value="motion_roi">motion ROI</label>
    <label class="motion_mask"><input type="radio" name="tool" value="motion_mask">motion mask</label>
    <label class="exclude"><input type="radio" name="tool" value="exclude">exclude</label>
  </div>
  <div class="tools">
    <label class="shape on" data-s="rect"><input type="radio" name="shape" value="rect" checked>&#9645; rect</label>
    <label class="shape" data-s="poly"><input type="radio" name="shape" value="poly">&#11040; polygon</label>
  </div>
  <button id="snap">reload snapshot</button>
</div>
<div class="hint" id="hint"></div>
<div class="cols">
  <div class="stage" id="stage">
    <div class="pz" id="pz"><img id="shot" alt="snapshot">
      <svg id="ov" viewBox="0 0 1600 900" preserveAspectRatio="none"></svg>
      <div class="tags" id="tags"></div><div class="handles" id="handles"></div></div>
    <div class="msg">snapshot unavailable (camera busy or offline) &mdash; you can still draw and save; zones are fractions of the frame</div>
    <div class="zoombar"><button id="zout" title="zoom out">&minus;</button><span class="lvl" id="zlvl">100%</span>
      <button id="zin" title="zoom in">+</button><button id="zfit" title="fit the whole frame">fit</button>
      <button id="zpan" title="pan mode: drag to move around (or hold Space, or middle-drag)">&#9995;</button></div>
  </div>
  <div class="side">
    <div><b>Analysis ROI</b> <span class="k">tier-2 crop</span>
      <div class="zdesc">Crops the frame before SpeciesNet looks, so it only analyses this region &mdash; cheaper, and sharper on what matters (e.g. a feeder). Doesn&rsquo;t change how <i>often</i> analysis runs. Always a rectangle (the detector needs one). Changing it makes the camera relearn its scenery.</div>
      <ul class="list" id="l-roi"></ul></div>
    <div style="margin-top:.6rem"><b>Motion ROI</b> <span class="k">tier-1 gate</span>
      <div class="zdesc">Only motion <i>inside</i> this zone wakes the detector, so fewer frames get analysed at all. The biggest lever on load &mdash; use it to point a camera at just the active area.</div>
      <ul class="list" id="l-motion_roi"></ul></div>
    <div style="margin-top:.6rem"><b>Motion mask</b> <span class="k">tier-1 ignore</span>
      <div class="zdesc">Motion <i>inside</i> these zones is ignored. Mute known false motion &mdash; a swaying branch, a street, a neighbour&rsquo;s window &mdash; so it never wakes the detector. Add as many as you need.</div>
      <ul class="list" id="l-motion_mask"></ul></div>
    <div style="margin-top:.6rem"><b>Exclude</b> <span class="k">per-zone mode</span>
      <div class="zdesc">For spots that only ever produce junk (a feeder roof&rsquo;s underside, a hedge). Click a zone&rsquo;s mode to switch it.<br><b>Ignore</b>: anything boxed here is dropped as soon as it&rsquo;s found, before it&rsquo;s classified &mdash; it never becomes a visit and can&rsquo;t keep the detector busy. For spots no animal can be.<br><b>Hide unrecognized</b>: a visit here that SpeciesNet couldn&rsquo;t identify is hidden when it ends (an identified bird stays) &mdash; kept, and shown again with one click in the viewer.<br>Applies to new visits.</div>
      <ul class="list" id="l-exclude"></ul></div>
    <div class="row" style="margin-top:1rem">
      <button id="apply" class="primary">apply &amp; reload</button>
      <button id="revert">revert</button>
    </div>
    <div class="status" id="status"></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
// A zone shape is [x,y,w,h] (a rect) or {poly:[[x,y],...]}; all coordinates are fractions of the frame.
let all={}, editable=true, name=null, zones={roi:null,motion_roi:null,motion_mask:[],exclude:[]}, tool='roi', shapeMode='rect';
const MODE={hide_unrecognized:'hide unrecognized',ignore:'ignore'}, VW=1600, VH=900, NS='http://www.w3.org/2000/svg';
const f3=v=>(+v).toFixed(3), r4=v=>+(+v).toFixed(4), clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const isPoly=s=>s&&!Array.isArray(s), copy=s=>isPoly(s)?{poly:s.poly.map(p=>p.slice())}:s.slice();
const bboxOf=s=>{if(!isPoly(s))return s;const xs=s.poly.map(p=>p[0]),ys=s.poly.map(p=>p[1]);
  return [Math.min(...xs),Math.min(...ys),Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys)];};
function status(msg,cls){const s=$('#status');s.textContent=msg;s.className='status'+(cls?' '+cls:'');}
function hint(){$('#hint').innerHTML=shapeMode==='poly'
  ?'<b>Polygon:</b> click to add points; close by clicking the first point, double-clicking or pressing <b>Enter</b>. <b>Backspace</b> removes the last point, <b>Esc</b> cancels. Drag a point to move it; <b>Alt-click</b> a point to delete it.'
  :'<b>Rect:</b> drag on the frame to draw. Drag any polygon point to adjust it.';
  $('#hint').innerHTML+=' Zoom with the wheel / pinch or the buttons; pan with <b>Space</b>-drag, middle-drag or &#9995;. <b>Analysis ROI</b> and <b>motion ROI</b> are single zones (a new one replaces); <b>motion mask</b> and <b>exclude</b> are lists.';}

// ---- zoom & pan: the snapshot, zones, tags and handles all live in #pz, transformed together ----
let zs=1, tx=0, ty=0, panMode=false, spaceDown=false, pan=null;
function stageSize(){const r=$('#stage').getBoundingClientRect();return [r.width,r.height];}
function applyZoom(){const [W,H]=stageSize(); tx=clamp(tx,W-W*zs,0); ty=clamp(ty,H-H*zs,0);
  $('#pz').style.transform=`translate(${tx}px,${ty}px) scale(${zs})`; $('#zlvl').textContent=Math.round(zs*100)+'%';
  document.querySelectorAll('.tag').forEach(t=>t.style.transform=`scale(${1/zs})`);  // labels and handles keep their size
  document.querySelectorAll('.hd').forEach(h=>h.style.transform=`scale(${1/zs})`);}
function zoomAt(f,cx,cy){const n=clamp(zs*f,1,10); if(cx==null){const [W,H]=stageSize();cx=W/2;cy=H/2;}
  tx=cx-(cx-tx)*n/zs; ty=cy-(cy-ty)*n/zs; zs=n; applyZoom();}
$('#stage').addEventListener('wheel',ev=>{ev.preventDefault();const r=$('#stage').getBoundingClientRect();
  zoomAt(Math.exp(-ev.deltaY*(ev.ctrlKey?0.01:0.0015)),ev.clientX-r.left,ev.clientY-r.top);},{passive:false});
$('#zin').onclick=()=>zoomAt(1.5); $('#zout').onclick=()=>zoomAt(1/1.5);
$('#zfit').onclick=()=>{zs=1;tx=0;ty=0;applyZoom();};
$('#zpan').onclick=()=>{panMode=!panMode;$('#zpan').classList.toggle('on',panMode);$('#stage').classList.toggle('pan',panMode);};
addEventListener('keydown',ev=>{if(ev.code==='Space'&&ev.target===document.body){ev.preventDefault();spaceDown=true;$('#stage').classList.add('panning');}});
addEventListener('keyup',ev=>{if(ev.code==='Space'){spaceDown=false;$('#stage').classList.remove('panning');}});
addEventListener('resize',applyZoom);
const stage=$('#stage');
stage.addEventListener('pointerdown',ev=>{if(!(ev.button===1||panMode||spaceDown)||ev.target.closest('.zoombar'))return;
  ev.preventDefault();ev.stopPropagation();pan={x:ev.clientX,y:ev.clientY,tx,ty};stage.setPointerCapture(ev.pointerId);},true);
stage.addEventListener('pointermove',ev=>{if(!pan)return;tx=pan.tx+ev.clientX-pan.x;ty=pan.ty+ev.clientY-pan.y;applyZoom();});
stage.addEventListener('pointerup',()=>{pan=null;});

// ---- render ----
function svgShape(s,cls){const el=document.createElementNS(NS,isPoly(s)?'polygon':'rect');el.setAttribute('class','z '+cls);
  if(isPoly(s))el.setAttribute('points',s.poly.map(p=>p[0]*VW+','+p[1]*VH).join(' '));
  else{el.setAttribute('x',s[0]*VW);el.setAttribute('y',s[1]*VH);el.setAttribute('width',s[2]*VW);el.setAttribute('height',s[3]*VH);}
  return el;}
function each(fn){  // every drawn zone: (shape, css class, tag, the object holding it, key)
  if(zones.roi)fn(zones.roi,'roi','ROI',zones,'roi'); if(zones.motion_roi)fn(zones.motion_roi,'motion_roi','mROI',zones,'motion_roi');
  zones.motion_mask.forEach((s,i)=>fn(s,'motion_mask','m'+(i+1),zones.motion_mask,i));
  zones.exclude.forEach((z,i)=>fn(z.shape,'exclude'+(z.mode==='ignore'?' ignore':''),'x'+(i+1)+(z.mode==='ignore'?' ignore':''),z,'shape'));}
function render(){
  const ov=$('#ov'); ov.innerHTML=''; $('#tags').innerHTML=''; $('#handles').innerHTML='';
  each((s,cls,tag,holder,key)=>{ov.appendChild(svgShape(s,cls)); const b=bboxOf(s), t=document.createElement('div');
    t.className='tag '+cls.split(' ')[0]; t.textContent=tag; t.style.left=b[0]*100+'%'; t.style.top=b[1]*100+'%'; $('#tags').appendChild(t);
    if(isPoly(s))s.poly.forEach((p,i)=>handle(p[0],p[1],s,i));});
  drawDraft();
  const desc=s=>isPoly(s)?`polygon &middot; ${s.poly.length} pts`:`[${s.map(f3).join(', ')}]`;
  const li=(s,cls,label,idx)=>`<li><span class="k">${label}</span><span>${desc(s)} <button data-cls="${cls}" data-i="${idx}">&times;</button></span></li>`;
  $('#l-roi').innerHTML=zones.roi?li(zones.roi,'roi','ROI',-1):'<li class="k">none</li>';
  $('#l-motion_roi').innerHTML=zones.motion_roi?li(zones.motion_roi,'motion_roi','mROI',-1):'<li class="k">none</li>';
  $('#l-motion_mask').innerHTML=zones.motion_mask.length?zones.motion_mask.map((s,i)=>li(s,'motion_mask','m'+(i+1),i)).join(''):'<li class="k">none</li>';
  $('#l-exclude').innerHTML=zones.exclude.length?zones.exclude.map((z,i)=>`<li><span class="k">x${i+1}</span><span>${isPoly(z.shape)?z.shape.poly.length+' pts':'rect'} `+
    `<button class="mode" data-i="${i}" title="switch what this zone hides">${MODE[z.mode]}</button> `+
    `<button data-cls="exclude" data-i="${i}">&times;</button></span></li>`).join(''):'<li class="k">none</li>';
  document.querySelectorAll('.list button[data-cls]').forEach(b=>b.onclick=()=>{const c=b.dataset.cls,i=+b.dataset.i;
    if(c==='motion_mask'||c==='exclude')zones[c].splice(i,1);else zones[c]=null;render();});
  document.querySelectorAll('.list button.mode').forEach(b=>b.onclick=()=>{const z=zones.exclude[+b.dataset.i];
    z.mode=z.mode==='ignore'?'hide_unrecognized':'ignore';render();});
  applyZoom();}

// ---- polygon points: drag to move, Alt-click to delete ----
function handle(x,y,s,i){const h=document.createElement('div');h.className='hd';h.style.left=x*100+'%';h.style.top=y*100+'%';
  h.title='drag to move · Alt-click to delete';
  h.addEventListener('pointerdown',ev=>{if(!editable||ev.button!==0)return;ev.stopPropagation();ev.preventDefault();
    if(ev.altKey){if(s.poly.length>3){s.poly.splice(i,1);render();}else status('a polygon needs at least 3 points','err');return;}
    h.setPointerCapture(ev.pointerId);
    const move=e=>{const p=pt(e);s.poly[i]=[r4(p[0]),r4(p[1])];h.style.left=p[0]*100+'%';h.style.top=p[1]*100+'%';
      $('#ov').innerHTML='';each((sh,cls)=>$('#ov').appendChild(svgShape(sh,cls)));};
    const up=()=>{h.removeEventListener('pointermove',move);h.removeEventListener('pointerup',up);render();};
    h.addEventListener('pointermove',move);h.addEventListener('pointerup',up);});
  $('#handles').appendChild(h);}

// ---- drawing ----
let drag=null, draft=null, cursor=null;  // drag: a rect being dragged; draft: polygon points so far
function pt(ev){const r=$('#ov').getBoundingClientRect();  // the transformed box, so fractions hold at any zoom
  return [clamp((ev.clientX-r.left)/r.width,0,1),clamp((ev.clientY-r.top)/r.height,0,1)];}
function place(shape){if(tool==='motion_mask')zones.motion_mask.push(shape);
  else if(tool==='exclude')zones.exclude.push({shape,mode:'hide_unrecognized'});  // the cautious mode first
  else zones[tool]=shape; render();}
function drawDraft(){const ov=$('#ov'); ov.querySelectorAll('.draft').forEach(e=>e.remove()); $('#handles').querySelectorAll('.dh').forEach(e=>e.remove());
  if(drag&&drag.cur){const s=drag.start,p=drag.cur;ov.appendChild(Object.assign(svgShape([Math.min(s[0],p[0]),Math.min(s[1],p[1]),
    Math.abs(p[0]-s[0]),Math.abs(p[1]-s[1])],'draft'),{}));}
  if(draft&&draft.length){const pts=cursor?draft.concat([cursor]):draft;const pl=document.createElementNS(NS,'polyline');
    pl.setAttribute('class','z draft');pl.setAttribute('points',pts.map(p=>p[0]*VW+','+p[1]*VH).join(' '));ov.appendChild(pl);
    draft.forEach((p,i)=>{const h=document.createElement('div');h.className='hd dh'+(i===0?' first':'');h.style.left=p[0]*100+'%';
      h.style.top=p[1]*100+'%';h.style.pointerEvents='none';h.style.transform=`scale(${1/zs})`;$('#handles').appendChild(h);});}}
function closeDraft(){if(draft&&draft.length>=3)place({poly:draft.map(p=>[r4(p[0]),r4(p[1])])});
  else if(draft&&draft.length)status('a polygon needs at least 3 points','err');
  draft=null;cursor=null;drawDraft();}
const ov=$('#ov');
ov.addEventListener('pointerdown',ev=>{
  if(!editable||ev.button!==0)return; ev.preventDefault();
  const p=pt(ev), poly=shapeMode==='poly'&&tool!=='roi';
  if(poly){const [W]=stageSize(), near=draft&&draft.length>=3&&Math.hypot((p[0]-draft[0][0])*W*zs,(p[1]-draft[0][1])*W*9/16*zs)<10;
    if(near){closeDraft();return;} (draft=draft||[]).push(p); drawDraft(); return;}
  ov.setPointerCapture(ev.pointerId); drag={start:p,cur:null};});
ov.addEventListener('pointermove',ev=>{
  if(pan)return;
  if(drag){drag.cur=pt(ev);drawDraft();} else if(draft){cursor=pt(ev);drawDraft();}});
ov.addEventListener('pointerup',ev=>{
  if(!drag)return; const s=drag.start,p=pt(ev); drag=null; drawDraft();
  const x=Math.min(s[0],p[0]),y=Math.min(s[1],p[1]),w=Math.abs(p[0]-s[0]),h=Math.abs(p[1]-s[1]);
  if(w<0.005||h<0.005)return; place([r4(x),r4(y),r4(w),r4(h)]);});
ov.addEventListener('dblclick',ev=>{if(draft){ev.preventDefault();if(draft.length>3)draft.pop();closeDraft();}});  // the 2nd click added a point
ov.addEventListener('contextmenu',ev=>ev.preventDefault());
addEventListener('keydown',ev=>{if(!draft)return;
  if(ev.key==='Enter'){ev.preventDefault();closeDraft();}
  else if(ev.key==='Escape'){draft=null;cursor=null;drawDraft();}
  else if(ev.key==='Backspace'){ev.preventDefault();draft.pop();if(!draft.length)draft=null;drawDraft();}});

// ---- tool and shape pickers ----
function setTool(){tool=document.querySelector('input[name=tool]:checked').value;
  document.querySelectorAll('.tools label[class]').forEach(l=>{if(!l.classList.contains('shape'))l.classList.toggle('on',l.classList.contains(tool));});
  const polyLabel=document.querySelector('label.shape[data-s=poly]'), roi=tool==='roi';
  polyLabel.classList.toggle('dis',roi); polyLabel.querySelector('input').disabled=roi;
  polyLabel.title=roi?'the analysis ROI is always a rectangle: the detector needs one':'';
  setShape();}
function setShape(){if(tool==='roi')document.querySelector('input[name=shape][value=rect]').checked=true;
  shapeMode=document.querySelector('input[name=shape]:checked').value;
  document.querySelectorAll('label.shape').forEach(l=>l.classList.toggle('on',l.dataset.s===shapeMode));
  draft=null;cursor=null;drawDraft();hint();}
document.querySelectorAll('input[name=tool]').forEach(i=>i.onchange=setTool);
document.querySelectorAll('input[name=shape]').forEach(i=>i.onchange=setShape);

// ---- load / save ----
function snapURL(){return '/api/snapshot/'+name+'?w=1920&t='+Date.now();}  // full detail for zooming in
function loadCam(n){name=n;const c=all[n];
  zones={roi:c.roi?c.roi.slice():null,motion_roi:c.motion_roi?copy(c.motion_roi):null,
    motion_mask:(c.motion_mask||[]).map(copy),
    exclude:(c.exclude||[]).map(z=>({shape:z.poly?{poly:z.poly.map(p=>p.slice())}:z.rect.slice(),mode:z.mode}))};
  draft=null;drag=null;$('#shot').src=snapURL();zs=1;tx=0;ty=0;render();status('');}
function payload(){return {roi:zones.roi,motion_roi:zones.motion_roi,motion_mask:zones.motion_mask,
  exclude:zones.exclude.map(z=>isPoly(z.shape)?{poly:z.shape.poly,mode:z.mode}:{rect:z.shape,mode:z.mode})};}
$('#shot').onerror=()=>$('#stage').classList.add('noshot');
$('#shot').onload=()=>$('#stage').classList.remove('noshot');
$('#camera').onchange=()=>loadCam($('#camera').value);
$('#snap').onclick=()=>{$('#shot').src=snapURL();};
$('#revert').onclick=()=>loadCam(name);
$('#apply').onclick=async()=>{if(!editable){status('editing is disabled','err');return;}
  if(draft){status('finish the polygon first (Enter) or cancel it (Esc)','err');return;}
  $('#apply').disabled=true;status('saving & reloading the monitor…');
  try{const res=await fetch('/api/zones/'+encodeURIComponent(name),{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  const d=await res.json();
  if(!res.ok){status(d.error||'save failed','err');}
  else{all[name]=d.zones;let m='saved. '+(d.reloaded?'monitor restarting':'monitor NOT reloaded: '+d.reload_detail);
    if(d.scenery_reset)m+=' · scenery will relearn';status(m,d.reloaded?'ok':'err');}
  }catch(e){status('network error: '+e,'err');}finally{$('#apply').disabled=false;}};
async function init(){const d=await (await fetch('/api/zones')).json();editable=d.editable;
  d.cameras.filter(c=>c.stream).forEach(c=>all[c.name]=c);
  $('#camera').innerHTML=Object.keys(all).map(n=>`<option>${n}</option>`).join('');
  if(!editable){$('#apply').disabled=true;status('read-only (CAMERA_MCP_WEB_READONLY set)','err');}
  setTool(); const first=Object.keys(all)[0]; if(first)loadCam(first);}
init();
</script></div></body></html>"""


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
