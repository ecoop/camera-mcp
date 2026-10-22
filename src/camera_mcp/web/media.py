# Copyright (c) 2026 Eric Cooper.
"""Pictures and video: a visit's boxes mapped to the full frame, its thumbnail, full-res still
(plain or with the boxes drawn in) and clip, and a live snapshot from a camera."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from camera_mcp.capture import CaptureError, grab_any
from camera_mcp.config import find_camera
from camera_mcp.history import HistoryError
from camera_mcp.web.context import Context, to_thread


def routes(ctx: Context) -> list[Route]:
    history, data_dir = ctx.history, ctx.data_dir

    def mapped_boxes(vid: int) -> dict:
        """A visit's per-frame boxes as fractions of the FULL frame (roi-mapped), each frame's clip
        offset and time (`ts`, which with a box's stored index `i` names it for tagging), each box's
        `tag` (history.box_tags), and `best`: the frame with the most confident box. Raises HistoryError."""
        d = history.visit_boxes(vid)
        cam = find_camera(ctx.cameras, d["camera"])
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

    async def media_thumb(request: Request):
        try:
            data, _ = history.visit_thumbnail(int(request.path_params["vid"]), data_dir)
        except HistoryError:
            return Response(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def media_clip(request: Request):
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
        cam = find_camera(ctx.cameras, d["camera"])  # clip time of the best-box frame (same math as /boxes)
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
        r = await to_thread(lambda: subprocess.run(
            [FFMPEG, "-y", "-ss", f"{off:.2f}", "-i", str(clip), "-frames:v", "1", "-q:v", "2", str(cache)],
            capture_output=True, timeout=30))
        if r.returncode != 0 or not cache.exists():
            return 502
        return cache

    async def media_frame(request: Request):
        """The full-resolution still, for zooming/panning in the viewer (see frame_path)."""
        path = await frame_path(int(request.path_params["vid"]))
        if isinstance(path, int):
            return Response(status_code=path)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def media_frame_boxed(request: Request):
        """The full-resolution still with the best frame's detector boxes and confidences drawn on it:
        a plain image to open on its own and zoom into, so what was boxed can be inspected closely."""
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
            jpeg = await to_thread(draw_boxes, path.read_bytes(), best)
            if jpeg is None:
                return Response(status_code=502)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(jpeg)
        return FileResponse(cache, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    async def api_snapshot(request: Request):
        cam = find_camera(ctx.cameras, request.path_params["camera"])
        if cam is None or not cam.enabled:
            return Response(status_code=404)
        try:
            frame, _ = await to_thread(grab_any, cam)
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

    return [
        Route("/api/snapshot/{camera}", api_snapshot),
        Route("/api/visit/{vid:int}/boxes", api_visit_boxes),
        Route("/media/visit/{vid:int}/thumb", media_thumb),
        Route("/media/visit/{vid:int}/frame", media_frame),
        Route("/media/visit/{vid:int}/frame/boxed", media_frame_boxed),
        Route("/media/visit/{vid:int}/clip", media_clip),
    ]


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
