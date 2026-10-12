# Copyright (c) 2026 Eric Cooper.
"""A read-only web dashboard for the Studio: browse visits and watch how the monitor is doing,
over the same History API the MCP tools use. Starlette + uvicorn, no new framework.

    python -m camera_mcp.web [--config cameras.yaml] [--data-dir data] [--host 0.0.0.0] [--port 8080]

v1 is read-only (visit browser + analytics + live snapshots). Editing settings and external
access (auth) are later phases — build it behind a gate before exposing it off the LAN.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from camera_mcp.capture import CaptureError, grab_any
from camera_mcp.config import (ConfigError, default_config_path, default_data_dir, find_camera,
                               load_cameras, load_site)
from camera_mcp.history import History, HistoryError


def build_app(cameras: dict, data_dir: Path, site=None) -> Starlette:
    history = History(data_dir / "monitor.db", cameras, site)

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
        Route("/api/cameras", api_cameras),
        Route("/api/status", api_status),
        Route("/api/health", api_health),
        Route("/api/visits", api_visits),
        Route("/api/summary", api_summary),
        Route("/api/snapshot/{camera}", api_snapshot),
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
dialog .vmeta{font-family:var(--mono);font-size:.74rem;padding:.6rem .8rem;display:flex;justify-content:space-between}
dialog .vmeta a{color:var(--accent)}
</style></head><body><div class="wrap">
<div class="bar"><div class="brand">&#129437; Yard Sentry</div><div class="who" id="clock"></div></div>
<div class="health" id="health">loading&hellip;</div>
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
<dialog id="clip"><video controls></video><div class="vmeta"><span id="clipmeta"></span>
  <a id="cliplink" target="_blank">open</a></div></dialog>
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
  $('#camera').insertAdjacentHTML('beforeend',cs.filter(c=>c.enabled).map(c=>`<option value="${c.name}">${c.name}</option>`).join(''));
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
    if(t.dataset.clip==='1'){vid.src='/media/visit/'+id+'/clip'; vid.style.display='';}else{vid.removeAttribute('src');vid.style.display='none';}
    $('#clipmeta').textContent=t.querySelector('.sp').textContent+' — '+t.querySelector('.sub').textContent;
    $('#cliplink').href='/media/visit/'+id+'/clip'; dlg.showModal();});}
$('#clip').onclick=e=>{if(e.target.id==='clip')$('#clip').close();};
['since','taxon','camera'].forEach(id=>$('#'+id).onchange=loadVisits);
tick();setInterval(tick,1000);loadCameras();loadHealth();loadVisits();setInterval(loadHealth,60000);
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
        cameras = load_cameras(args.config, require_login=False)
        site = load_site(args.config)
    except ConfigError as e:
        raise SystemExit(f"web: {e}")
    app = build_app(cameras, data_dir, site)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
