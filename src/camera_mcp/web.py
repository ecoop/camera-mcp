# Copyright (c) 2026 Eric Cooper.
"""A read-only web dashboard for the Studio: browse visits and watch how the monitor is doing,
over the same History API the MCP tools use. Starlette + uvicorn, no new framework.

    python -m camera_mcp.web [--config cameras.yaml] [--data-dir data] [--host 0.0.0.0] [--port 8080]

The visit browser, analytics and live snapshots are read-only. The /zones editor is read-write:
it draws per-camera ROI / motion-ROI / motion-mask on a snapshot and saves them to an app-owned
overlay (data/zones.json, merged by load_cameras), then restarts the monitor to apply. Editing is
LAN-only and unauthenticated for now (CAMERA_MCP_WEB_READONLY=1 disables it); real auth arrives
with external access (Cloudflare).
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
                               load_cameras, load_site, load_zone_overlay, parse_rect, parse_rects,
                               save_zone_overlay)
from camera_mcp.history import History, HistoryError

# The monitor's launchd label, kicked to reload cameras.yaml + the zone overlay after an edit.
MONITOR_LABEL = os.environ.get("CAMERA_MCP_LABEL", "nu.cooper.camera-mcp")


def _reload_monitor() -> tuple[bool, str]:
    """Restart the monitor LaunchAgent so it re-reads the zone overlay. Best-effort: returns
    (ok, detail). Off the Studio (no such service) this just reports it couldn't."""
    label = f"gui/{os.getuid()}/{MONITOR_LABEL}"
    try:
        r = subprocess.run(["launchctl", "kickstart", "-k", label],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run launchctl ({e})"
    if r.returncode == 0:
        return True, f"restarted {MONITOR_LABEL}"
    return False, (r.stderr or r.stdout or f"launchctl exit {r.returncode}").strip()


def build_app(cameras: dict, data_dir: Path, site=None) -> Starlette:
    history = History(data_dir / "monitor.db", cameras, site)
    zones_path = data_dir / "zones.json"
    editable = os.environ.get("CAMERA_MCP_WEB_READONLY", "").lower() not in ("1", "true", "yes")

    def _zone_state(cam):
        return {"name": cam.name, "label": cam.label, "stream": cam.is_stream,
                "roi": list(cam.roi) if cam.roi else None,
                "motion_roi": list(cam.motion_roi) if cam.motion_roi else None,
                "motion_mask": [list(r) for r in cam.motion_mask]}

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
                 taxon=q.get("taxon") or None, limit=int(q.get("limit", 60)))
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_summary(request: Request):
        q = request.query_params
        r = _ask(history.activity_summary, since=q.get("since", "24h"), group_by=q.get("group_by", "group"))
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_stats(request: Request):
        r = _ask(history.analysis_series, since=request.query_params.get("since", "24h"),
                 camera=request.query_params.get("camera") or None)
        return r if isinstance(r, Response) else JSONResponse(r)

    async def api_visit_boxes(request: Request):
        vid = int(request.path_params["vid"])
        try:
            d = history.visit_boxes(vid)
        except HistoryError as e:
            return JSONResponse({"error": str(e)}, status_code=404)
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
        return JSONResponse({"camera": d["camera"], "has_clip": d["has_clip"],
                             "roi": list(roi) if roi else None, "label": d["label"],
                             "taxonomy": d["taxonomy"], "best": best_i, "frames": frames})

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
            motion_roi = parse_rect(body["motion_roi"], where, "motion_roi") if body.get("motion_roi") else None
            motion_mask = parse_rects(body.get("motion_mask"), where, "motion_mask")
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
                                    "motion_roi": list(motion_roi) if motion_roi else None,
                                    "motion_mask": [list(r) for r in motion_mask]}
        try:
            save_zone_overlay(zones_path, raw)
            load_zone_overlay(zones_path)  # re-validate the file as written
        except (OSError, ConfigError) as e:
            return JSONResponse({"error": f"could not save: {e}"}, status_code=500)

        from dataclasses import replace
        cameras[cam.name] = replace(cam, roi=roi, motion_roi=motion_roi, motion_mask=motion_mask)
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
        Route("/zones", zones_page),
        Route("/api/cameras", api_cameras),
        Route("/api/status", api_status),
        Route("/api/health", api_health),
        Route("/api/visits", api_visits),
        Route("/api/summary", api_summary),
        Route("/api/stats", api_stats),
        Route("/api/zones", api_zones),
        Route("/api/zones/{camera}", api_zones_save, methods=["POST"]),
        Route("/api/snapshot/{camera}", api_snapshot),
        Route("/api/visit/{vid:int}/boxes", api_visit_boxes),
        Route("/media/visit/{vid:int}/thumb", media_thumb),
        Route("/media/visit/{vid:int}/clip", media_clip),
    ])


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
.health{font-family:var(--mono);font-size:.74rem;color:var(--muted);border:1px solid var(--line);
  border-radius:8px;padding:.5rem .7rem;background:var(--frame);margin-bottom:1rem}
.health b{color:var(--ink)} .ok{color:var(--good)} .bad{color:var(--warn)}
.cams{display:flex;gap:.4rem;flex-wrap:wrap;margin:.4rem 0 0}
.chip{font-family:var(--mono);font-size:.68rem;border:1px solid var(--line);border-radius:999px;
  padding:.1rem .5rem;color:var(--muted)} .chip.live{color:var(--good);border-color:var(--good)}
.chip.off{opacity:.55}
.chart{border:1px solid var(--line);border-radius:11px;background:var(--frame);padding:.6rem;margin-bottom:1rem}
.chart svg{width:100%;height:132px;display:block}
.clabel{display:flex;justify-content:space-between;gap:1rem;font-family:var(--mono);font-size:.66rem;color:var(--muted);margin-top:.35rem;flex-wrap:wrap}
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
dialog video{display:block;max-width:86vw;max-height:76vh;border-radius:12px 12px 0 0;background:#000}
dialog .vmeta{font-family:var(--mono);font-size:.74rem;padding:.6rem .8rem;display:flex;justify-content:space-between;gap:1rem;align-items:center}
dialog .vmeta a{color:var(--accent);cursor:pointer}
.stagewrap{position:relative;display:block;line-height:0}
.boxov{position:absolute;top:0;left:0;pointer-events:none;display:none}
.boxov rect{fill:none;stroke:#ff3b30;stroke-width:2;vector-effect:non-scaling-stroke}
.boxov text{fill:#ff3b30;font-family:var(--mono);font-size:3.2px;paint-order:stroke;
  stroke:rgba(0,0,0,.55);stroke-width:.6px;dominant-baseline:text-after-edge}
.boxbtn{border:1px solid var(--line);border-radius:6px;padding:.15rem .5rem;color:var(--muted);background:none;font-family:var(--mono);font-size:.72rem;cursor:pointer}
.boxbtn.on{color:var(--warn);border-color:var(--warn)}
</style></head><body><div class="wrap">
<div class="bar"><div class="brand">&#129437; Yard Sentry</div>
  <div class="who"><a href="/zones" style="color:var(--accent);text-decoration:none">zones &rarr;</a> &middot; <span id="clock"></span></div></div>
<div class="health" id="health">loading&hellip;</div>
<div class="sect" style="margin-top:0;display:flex;justify-content:space-between;align-items:center;gap:1rem">
  <span>Throughput &amp; backlog</span>
  <select id="chartcam" style="font-family:var(--mono);font-size:.72rem;background:var(--frame);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:.2rem .4rem;text-transform:none;letter-spacing:0">
    <option value="">all cameras</option></select>
</div>
<div class="chart" id="chart"></div>
<div class="filters">
  <select id="since"><option value="2h">last 2h</option><option value="12h">12h</option>
    <option value="24h" selected>24h</option><option value="3d">3 days</option><option value="7d">7 days</option></select>
  <select id="taxon"><option value="">all species</option><option>night-critters</option><option>cat</option>
    <option>rat</option><option>squirrel</option><option>raccoon</option><option>opossum</option>
    <option>skunk</option><option>fox</option><option>coyote</option><option>owl</option>
    <option>songbirds</option><option>person</option></select>
  <select id="camera"><option value="">all cameras</option></select>
</div>
<div class="sect">Visits</div>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" style="display:none">No visits in this window.</div>
<div class="sect">Live</div>
<div class="grid" id="live"></div>
<dialog id="clip"><div class="stagewrap"><video controls></video>
  <svg class="boxov" id="boxov" viewBox="0 0 100 100" preserveAspectRatio="none"></svg></div>
  <div class="vmeta"><span id="clipmeta"></span>
  <span><button class="boxbtn" id="boxbtn">show boxes</button> &middot; <a id="cliplink" target="_blank">open</a></span></div></dialog>
<script>
const $=s=>document.querySelector(s), params=()=>{const p=new URLSearchParams();
  p.set('since',$('#since').value); if($('#taxon').value)p.set('taxon',$('#taxon').value);
  if($('#camera').value)p.set('camera',$('#camera').value); return p.toString();};
function tick(){$('#clock').textContent=new Date().toLocaleString();}
async function loadHealth(){try{const h=await (await fetch('/api/health?since=24h')).json();
  const st=await (await fetch('/api/status')).json();
  const watching=st.filter(c=>c.watching_now).length, total=st.filter(c=>c.state!=='never seen by the monitor').length;
  let s;
  if(h.samples){const dr=h.segments_dropped; s=`<b>Analysis (24h):</b> ${h.frames_analyzed.toLocaleString()} frames &middot; `+
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
async function loadVisits(){const r=await (await fetch('/api/visits?'+params())).json();
  const v=(r.visits||[]).filter(x=>x.has_thumbnail);
  $('#empty').style.display=v.length?'none':'block';
  $('#grid').innerHTML=v.map(x=>`<a class="tile" href="#" data-id="${x.id}" data-clip="${x.clip?1:0}">`+
    `<img loading="lazy" src="/media/visit/${x.id}/thumb" alt="${x.label}">`+
    `<div class="meta"><div class="sp">${x.label}${x.max_at_once>1?' ×'+x.max_at_once:''}</div>`+
    `<div class="sub">${x.camera} &middot; ${(x.started||'').slice(11,16)}</div></div></a>`).join('');
  document.querySelectorAll('#grid .tile').forEach(t=>t.onclick=ev=>{ev.preventDefault();
    const id=t.dataset.id; const dlg=$('#clip'); const vid=dlg.querySelector('video');
    const hasClip=t.dataset.clip==='1';
    if(hasClip){vid.src='/media/visit/'+id+'/clip'; vid.style.display='';}else{vid.removeAttribute('src');vid.style.display='none';}
    curVisit=id; resetBoxes(); $('#boxbtn').disabled=!hasClip;
    $('#clipmeta').textContent=t.querySelector('.sp').textContent+' — '+t.querySelector('.sub').textContent;
    $('#cliplink').href='/media/visit/'+id+'/clip'; dlg.showModal();
    if(hasClip) toggleBoxes();});}  // show the boxed still by default
let curVisit=null, boxesOn=false, boxData=null;
function resetBoxes(){boxesOn=false;boxData=null;const s=$('#boxov');s.style.display='none';s.innerHTML='';
  const b=$('#boxbtn');b.classList.remove('on');b.textContent='show boxes';}
function drawBoxes(){const svg=$('#boxov'), vid=$('#clip').querySelector('video');
  svg.style.width=(vid.clientWidth||vid.offsetWidth)+'px';   // SVG is replaced: size it to the video
  svg.style.height=(vid.clientHeight||vid.offsetHeight)+'px';
  if(!boxData||!boxData.frames||!boxData.frames.length){svg.innerHTML='<text x="2" y="7">no boxes stored for this visit</text>';svg.style.display='block';return;}
  const f=boxData.frames[boxData.best]||boxData.frames[0]; let s='';
  (f.boxes||[]).forEach(b=>{const x=b.x*100,y=b.y*100,w=b.w*100,h=b.h*100;
    s+=`<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${w.toFixed(2)}" height="${h.toFixed(2)}"/>`;
    if(b.conf!=null)s+=`<text x="${x.toFixed(2)}" y="${Math.max(3.2,y).toFixed(2)}">${Math.round(b.conf*100)}%</text>`;});
  svg.innerHTML=s; svg.style.display='block';}
async function toggleBoxes(){const b=$('#boxbtn'); if(boxesOn){resetBoxes();return;}
  if(!boxData){try{boxData=await (await fetch('/api/visit/'+curVisit+'/boxes')).json();}catch(e){boxData={frames:[]};}}
  boxesOn=true;b.classList.add('on');b.textContent='hide boxes';
  const vid=$('#clip').querySelector('video'), f=boxData.frames&&boxData.frames[boxData.best];
  if(f&&vid.getAttribute('src')){try{vid.pause();if(f.offset)vid.currentTime=f.offset;}catch(e){}}
  drawBoxes();}
$('#boxbtn').onclick=toggleBoxes;
// once the clip's dimensions are known, size the overlay and seek to the best-box frame
$('#clip').querySelector('video').addEventListener('loadeddata',()=>{if(!boxesOn)return;
  const vid=$('#clip').querySelector('video'), f=boxData&&boxData.frames&&boxData.frames[boxData.best];
  if(f){try{vid.pause();if(f.offset)vid.currentTime=f.offset;}catch(e){}}
  drawBoxes();});
$('#clip').onclick=e=>{if(e.target.id==='clip')$('#clip').close();};
async function loadStats(){try{const cc=$('#chartcam').value;
  const d=await (await fetch('/api/stats?since='+$('#since').value+(cc?'&camera='+encodeURIComponent(cc):''))).json();
  const pts=d.points||[]; const el=$('#chart');
  if(!pts.length){el.innerHTML='<div class="empty" style="padding:1rem">'+(d.camera?'no per-camera stats for '+d.camera+' yet (recorded from this build on)':'no stats yet')+'</div>';return;}
  const W=1000,H=132,pad=4,n=pts.length,cap=1800;
  const maxLag=Math.max(cap, ...pts.map(p=>p.lag));
  const X=i=>pad+i*(W-2*pad)/(n-1||1), Y=v=>H-pad-(v/maxLag)*(H-2*pad);
  let area='M'+X(0).toFixed(1)+','+(H-pad);
  pts.forEach((p,i)=>area+=' L'+X(i).toFixed(1)+','+Y(p.lag).toFixed(1));
  area+=' L'+X(n-1).toFixed(1)+','+(H-pad)+' Z';
  const drops=pts.map((p,i)=>p.dropped>0?`<rect x="${X(i).toFixed(1)}" y="${H-7}" width="2.5" height="7" fill="var(--warn)"/>`:'').join('');
  const capY=Y(cap).toFixed(1);
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`+
    `<line x1="0" y1="${capY}" x2="${W}" y2="${capY}" stroke="var(--warn)" stroke-dasharray="5 5" stroke-width="1" opacity=".55"/>`+
    `<path d="${area}" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.5"/>${drops}</svg>`+
    `<div class="clabel"><span>${d.camera?'<b>'+d.camera+'</b> &middot; ':''}analysis lag (area) &middot; <span class="bad">drops</span> (ticks) &middot; dashed = 30-min cap</span>`+
    `<span>${(d.from||'').slice(5,16)} &rarr; ${(d.to||'').slice(11,16)}</span></div>`;
}catch(e){$('#chart').innerHTML='<div class="empty" style="padding:1rem">stats unavailable</div>';}}
['since','taxon','camera'].forEach(id=>$('#'+id).onchange=()=>{loadVisits();loadStats();});
$('#chartcam').onchange=loadStats;
tick();setInterval(tick,1000);loadCameras();loadHealth();loadStats();loadVisits();
setInterval(()=>{loadHealth();loadStats();},60000);
</script></div></body></html>"""


ZONES_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Zones &middot; Yard Sentry</title>
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
.brand a{color:inherit;text-decoration:none}
.who{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.row{display:flex;gap:.6rem;flex-wrap:wrap;align-items:center;margin-bottom:.8rem}
select,button{font-family:var(--mono);font-size:.78rem;background:var(--frame);color:var(--ink);
  border:1px solid var(--line);border-radius:7px;padding:.35rem .6rem;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button:disabled{opacity:.5;cursor:default}
.tools{display:flex;gap:.3rem}
.tools label{font-family:var(--mono);font-size:.72rem;border:1px solid var(--line);border-radius:999px;
  padding:.2rem .6rem;cursor:pointer;color:var(--muted);display:inline-flex;gap:.3rem;align-items:center}
.tools input{accent-color:var(--accent)}
.tools label.roi.on{color:var(--accent);border-color:var(--accent)}
.tools label.motion_roi.on{color:var(--good);border-color:var(--good)}
.tools label.motion_mask.on{color:var(--warn);border-color:var(--warn)}
.stage{position:relative;border:1px solid var(--line);border-radius:11px;overflow:hidden;background:var(--chrome);
  max-width:960px;aspect-ratio:16/9;min-height:240px;touch-action:none;user-select:none}
.stage img{width:100%;height:100%;object-fit:contain;display:block}
.stage .msg{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:.74rem;color:var(--muted);text-align:center;padding:1rem}
.stage.noshot .msg{display:flex}
.ov{position:absolute;inset:0;cursor:crosshair}
.zr{position:absolute;border:2px solid var(--accent);background:rgba(91,107,140,.14);pointer-events:none}
.zr.roi{border-color:var(--accent);background:rgba(91,107,140,.14)}
.zr.motion_roi{border-color:var(--good);background:rgba(79,122,82,.16)}
.zr.motion_mask{border-color:var(--warn);background:rgba(165,81,74,.16)}
.zr .tag{position:absolute;top:-1px;left:-1px;font-family:var(--mono);font-size:.6rem;padding:0 .25rem;
  color:#fff;background:var(--accent)}
.zr.motion_roi .tag{background:var(--good)} .zr.motion_mask .tag{background:var(--warn)}
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
  </div>
  <button id="snap">reload snapshot</button>
</div>
<div class="hint">Drag on the frame to draw. <b>Analysis ROI</b> and <b>motion ROI</b> are single rectangles (a new draw replaces); <b>motion mask</b> is a list (each draw adds a box). Zones are fractions of the frame; see what each does below.</div>
<div class="cols">
  <div class="stage" id="stage"><img id="shot" alt="snapshot">
  <div class="msg">snapshot unavailable (camera busy or offline) &mdash; you can still draw and save; zones are fractions of the frame</div>
  <div class="ov" id="ov"></div></div>
  <div class="side">
    <div><b>Analysis ROI</b> <span class="k">tier-2 crop</span>
      <div class="zdesc">Crops the frame before SpeciesNet looks, so it only analyses this region &mdash; cheaper, and sharper on what matters (e.g. a feeder). Doesn&rsquo;t change how <i>often</i> analysis runs. Changing it makes the camera relearn its scenery.</div>
      <ul class="list" id="l-roi"></ul></div>
    <div style="margin-top:.6rem"><b>Motion ROI</b> <span class="k">tier-1 gate</span>
      <div class="zdesc">Only motion <i>inside</i> this box wakes the detector, so fewer frames get analysed at all. The biggest lever on load &mdash; use it to point a camera at just the active area.</div>
      <ul class="list" id="l-motion_roi"></ul></div>
    <div style="margin-top:.6rem"><b>Motion mask</b> <span class="k">tier-1 ignore</span>
      <div class="zdesc">Motion <i>inside</i> these boxes is ignored. Mute known false motion &mdash; a swaying branch, a street, a neighbour&rsquo;s window &mdash; so it never wakes the detector. Add as many as you need.</div>
      <ul class="list" id="l-motion_mask"></ul></div>
    <div class="row" style="margin-top:1rem">
      <button id="apply" class="primary">apply &amp; reload</button>
      <button id="revert">revert</button>
    </div>
    <div class="status" id="status"></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
let all={}, editable=true, name=null, zones={roi:null,motion_roi:null,motion_mask:[]}, tool='roi';
const f3=v=>(+v).toFixed(3);
function tool_of(){return document.querySelector('input[name=tool]:checked').value;}
function setToolUI(){document.querySelectorAll('.tools label').forEach(l=>l.classList.remove('on'));
  document.querySelector('.tools label.'+tool).classList.add('on');}
function render(){
  const ov=$('#ov'); ov.querySelectorAll('.zr').forEach(e=>e.remove());
  const put=(r,cls,tag)=>{if(!r)return;const d=document.createElement('div');d.className='zr '+cls;
    d.style.left=(r[0]*100)+'%';d.style.top=(r[1]*100)+'%';d.style.width=(r[2]*100)+'%';d.style.height=(r[3]*100)+'%';
    d.innerHTML='<span class="tag">'+tag+'</span>';ov.appendChild(d);};
  put(zones.roi,'roi','ROI'); put(zones.motion_roi,'motion_roi','mROI');
  zones.motion_mask.forEach((r,i)=>put(r,'motion_mask','m'+(i+1)));
  const li=(r,cls,label,idx)=>`<li><span class="k">${label}</span><span>[${r.map(f3).join(', ')}] `+
    `<button data-cls="${cls}" data-i="${idx}">&times;</button></span></li>`;
  $('#l-roi').innerHTML=zones.roi?li(zones.roi,'roi','ROI',-1):'<li class="k">none</li>';
  $('#l-motion_roi').innerHTML=zones.motion_roi?li(zones.motion_roi,'motion_roi','mROI',-1):'<li class="k">none</li>';
  $('#l-motion_mask').innerHTML=zones.motion_mask.length?zones.motion_mask.map((r,i)=>li(r,'motion_mask','m'+(i+1),i)).join(''):'<li class="k">none</li>';
  document.querySelectorAll('.list button').forEach(b=>b.onclick=()=>{const c=b.dataset.cls,i=+b.dataset.i;
    if(c==='motion_mask')zones.motion_mask.splice(i,1);else zones[c]=null;render();});
}
function loadCam(n){name=n;const c=all[n];
  zones={roi:c.roi?c.roi.slice():null,motion_roi:c.motion_roi?c.motion_roi.slice():null,
    motion_mask:(c.motion_mask||[]).map(r=>r.slice())};
  $('#shot').src='/api/snapshot/'+n+'?w=960&t='+Date.now();render();status('');}
function status(msg,cls){const s=$('#status');s.textContent=msg;s.className='status'+(cls?' '+cls:'');}
// draw
let drag=null;
const ov=$('#ov');
function pt(ev){const r=ov.getBoundingClientRect();
  return [Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width)),Math.min(1,Math.max(0,(ev.clientY-r.top)/r.height))];}
ov.addEventListener('pointerdown',ev=>{if(!editable)return;ev.preventDefault();ov.setPointerCapture(ev.pointerId);
  drag={start:pt(ev),el:document.createElement('div')};drag.el.className='zr '+tool;ov.appendChild(drag.el);});
ov.addEventListener('pointermove',ev=>{if(!drag)return;const p=pt(ev),s=drag.start;
  const x=Math.min(s[0],p[0]),y=Math.min(s[1],p[1]),w=Math.abs(p[0]-s[0]),h=Math.abs(p[1]-s[1]);
  Object.assign(drag.el.style,{left:x*100+'%',top:y*100+'%',width:w*100+'%',height:h*100+'%'});});
ov.addEventListener('pointerup',ev=>{if(!drag)return;const p=pt(ev),s=drag.start;drag.el.remove();drag=null;
  const x=Math.min(s[0],p[0]),y=Math.min(s[1],p[1]),w=Math.abs(p[0]-s[0]),h=Math.abs(p[1]-s[1]);
  if(w<0.01||h<0.01)return;const r=[+x.toFixed(4),+y.toFixed(4),+w.toFixed(4),+h.toFixed(4)];
  if(tool==='motion_mask')zones.motion_mask.push(r);else zones[tool]=r;render();});
document.querySelectorAll('input[name=tool]').forEach(i=>i.onchange=()=>{tool=tool_of();setToolUI();});
$('#shot').onerror=()=>$('#stage').classList.add('noshot');
$('#shot').onload=()=>$('#stage').classList.remove('noshot');
$('#camera').onchange=()=>loadCam($('#camera').value);
$('#snap').onclick=()=>{$('#shot').src='/api/snapshot/'+name+'?w=960&t='+Date.now();};
$('#revert').onclick=()=>loadCam(name);
$('#apply').onclick=async()=>{if(!editable){status('editing is disabled','err');return;}
  $('#apply').disabled=true;status('saving & reloading the monitor…');
  try{const res=await fetch('/api/zones/'+encodeURIComponent(name),{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(zones)});
  const d=await res.json();
  if(!res.ok){status(d.error||'save failed','err');}
  else{all[name]=d.zones;let m='saved. '+(d.reloaded?'monitor restarting':'monitor NOT reloaded: '+d.reload_detail);
    if(d.scenery_reset)m+=' · scenery will relearn';status(m,d.reloaded?'ok':'err');}
  }catch(e){status('network error: '+e,'err');}finally{$('#apply').disabled=false;}};
async function init(){const d=await (await fetch('/api/zones')).json();editable=d.editable;
  d.cameras.filter(c=>c.stream).forEach(c=>all[c.name]=c);
  $('#camera').innerHTML=Object.keys(all).map(n=>`<option>${n}</option>`).join('');
  if(!editable){$('#apply').disabled=true;status('read-only (CAMERA_MCP_WEB_READONLY set)','err');}
  const first=Object.keys(all)[0]; if(first)loadCam(first);}
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
