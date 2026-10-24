# camera-mcp

[![CI](https://github.com/ecoop/camera-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/ecoop/camera-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

_Last updated: 2026-10-24_

An MCP server and an always-on monitor for home IP cameras, built to watch for wildlife on one
machine without saturating it.

- **Motion first, models second.** A lightweight motion frontend watches every camera on small
  grayscale frames and opens a window only when something moves; only those windows reach the
  detector. Analysis regions, motion masks and exclusion zones cut the work further, so six cameras
  fit on one Apple-Silicon GPU.
- **Detect, then identify.** MegaDetector finds animals, people and vehicles; SpeciesNet names the
  animal, limited to species found in your area. Each arrival-to-departure is logged as one visit in
  SQLite, with a thumbnail, a full-resolution still and a clip.
- **Junk control.** A memory of fixed scenery, "ignore" zones that drop boxes before they are
  classified, and a reversible gate for unidentified visits keep leaves, fruit and feeder roofs out
  of the log.
- **Ask Claude.** The MCP tools below answer live and history questions (what came by overnight, is
  the yard clear) and say plainly when a camera wasn't being watched.
- **A web dashboard** for browsing visits and analytics, tagging detections down to single boxes,
  favourites, a labels map, drawing zones, and a plain-English ask box.
- **A morning brief** by email or iMessage.

The design is in [docs/phase2-design.md](docs/phase2-design.md), backed by the
[SpeciesNet eval](docs/speciesnet-eval-2026-10-08.md). SpeciesNet's full label list is in
[docs/speciesnet-labels-v4.0.3a.txt](docs/speciesnet-labels-v4.0.3a.txt).

## Tools

| Tool | What it returns |
|---|---|
| `list_cameras` | name, label, tags, aliases, and whether each camera answers right now (a TCP check, no decode) |
| `snapshot(camera, max_width=1024)` | one current frame as a JPEG image, downscaled to `max_width`; from the low-res stream if the camera's main stream is busy (Tapo cameras allow only a couple of main-stream clients) |
| `find_visits(since, until, cameras, tag, taxon, overlapping, min_score, limit)` | animal visits in a stretch of time, newest first; `still_there` marks one happening now |
| `activity_summary(since, until, group_by, cameras, tag, taxon)` | visit counts by `species`, `group`, `camera`, local `hour` or `day` |
| `visit_media(visit_id)` | the thumbnail of a visit's best sighting |
| `monitor_status()` | per camera: watched now?, last frame and motion, looks moved?, visits in progress, errors |

Resources: `cameras://list` (JSON) and `cameras://{name}/latest.jpg`.

All tools are read-only. Cameras are addressed by name (loosely: "west alley" finds `west_alley`);
a source URL or credential is never returned. The history tools read the monitor's
`data/monitor.db` read-only.

`since` and `until` take ISO times or durations ago (`15m`, `4h`, `3d`, `2w`), `today`,
`yesterday`. `taxon` takes a group (`songbirds`, `jays`, `crows`, `doves/pigeons`, `hawks`, `owls`,
`cat`, `dog`, `rat`, `squirrel`, `skunk`, `raccoon`, `opossum`, `fox`, `coyote`, `person`), the union
`night-critters`, a class (`bird`, `mammal`) or a common name. `overlapping` finds visits that
overlapped another kind on the same camera: `taxon=cat, overlapping=bird` is cats among the birds.

Every history answer carries `coverage`: the share of the time each camera was actually being
watched (`null` if the log predates that record). "Nothing seen" only means nothing was there when
coverage is near 1. "Is the dog yard clear?" is `find_visits(since="15m", tag="dog_yard",
taxon="night-critters")`.

## Setup

Needs Python 3.12+ and the `ffmpeg` binary (Homebrew: `brew install ffmpeg`). Species identification
(SpeciesNet, the `detect` extra) needs Python below 3.14 until SpeciesNet supports it; everything
else also runs on 3.14.

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
cp cameras.example.yaml cameras.yaml
```

`cameras.yaml` is git-ignored. A Tapo camera needs only a name and its IP:

```yaml
cameras:
  - name: east_fence
    label: Back fence, east
    host: 192.0.2.10
```

The RTSP URL is built from the host (`rtsp://<user>:<pass>@<host>/stream1`). The login is the
Camera Account set in the Tapo app, and it never goes in a file. On macOS, store it once in the
Keychain; `security` asks for the password, so it isn't typed on the command line:

```bash
security add-generic-password -U -s camera-mcp -a <camera user> -w
```

`$CAMERA_USER` and `$CAMERA_PASS`, if set, take precedence (for other systems, and the tests). On a
headless or remote host the login Keychain is locked in an SSH session, so put
`CAMERA_USER=` and `CAMERA_PASS=` lines in `~/.config/camera-mcp/credentials` (`chmod 600`,
or `$CAMERA_MCP_CREDENTIALS` to point elsewhere) and that is used instead. The same file also
holds the optional email and alert settings (`SMTP_*`, `SMS_ALERT_TO`); see
[`credentials.example`](credentials.example) for every key the code reads, with placeholder
values and the exact `KEY=value` format it expects.
Anything else goes in `source` instead of `host`: a video file (which is how the tests run, and
how to try the server away from home) or a full URL for a non-Tapo camera, where `${VAR}` is
expanded from the environment.

## Use from Claude Code

**As a plugin** (the easy way): this repository is also a Claude Code plugin marketplace. The plugin
brings the MCP server and a Skill, `camera-history`, that teaches Claude how to answer from it:
common questions with the exact tool calls, and how to read coverage, paused cameras, hidden
visits and your tags honestly. It also adds two shortcuts, `/camera-mcp:overnight` and
`/camera-mcp:yard-check [camera tag]`.

```
/plugin marketplace add ecoop/camera-mcp
/plugin install camera-mcp@camera-mcp
```

On install it asks for two settings: the **monitor host** (`user@host`, reached over key-based SSH;
leave it empty if Claude Code runs on the monitor's machine) and the **camera-mcp folder** there
(default `~/projects/camera-mcp`). `CAMERA_MCP_HOST` / `CAMERA_MCP_DIR` in the environment override
them. The plugin's tools are named `mcp__plugin_camera-mcp_cameras__<tool>`; if you added the server
by hand before (below), remove that entry (`claude mcp remove cameras`) so Claude doesn't see every
tool twice, and update any permission rules that named the old tools. To try it without installing:
`claude --plugin-dir plugin`. The plugin lives in `plugin/`; `tests/test_plugin.py` keeps its Skill
true to the server's tools and the taxon list (`plugin/skills/camera-history/reference.md` is
generated from `taxa.py`).

**By hand:**

```bash
claude mcp add cameras -- ~/projects/camera-mcp/.venv/bin/camera-mcp --config ~/projects/camera-mcp/cameras.yaml --data-dir ~/projects/camera-mcp/data
```

No password goes into Claude's settings: the server reads the login from the Keychain, and only
`snapshot` needs it. Without a login the server still starts, the history tools work, and
`snapshot` says how to add one.

## Probe the cameras

```bash
.venv/bin/python -m camera_mcp.probe
```

For every camera in `cameras.yaml`, this checks that it answers and accepts the login, and reports
what its streams deliver (codec, resolution, frame rate). Tapo cameras are checked on both
`/stream1` and `/stream2`. It's read-only: each stream is opened, its header read, and closed.
It exits non-zero if any stream fails.

## The monitor

To run the monitor as an always-on service (its real home is the Mac Studio, where the GPU keeps
SpeciesNet fast enough), see [deploy/README.md](deploy/README.md). Health checks run beside it
(`camera-watchdog`): a text when a camera's video stops inside its watch hours, and a midday report
on whether the detector kept up.


```bash
uv pip install --python .venv/bin/python -e '.[detect]'   # SpeciesNet, for animal identification
.venv/bin/camera-monitor                                    # all enabled cameras, until Ctrl-C
.venv/bin/camera-monitor --camera "west alley" --duration 300
```

For each camera, the monitor:
- **records** the main stream into a ring buffer of 2-second files, copied rather than re-encoded;
- **watches** the low-resolution stream for motion, at 5 frames per second;
- **identifies** what's there: while there's motion, or an animal is still in view, it runs
  [SpeciesNet](https://github.com/google/cameratrapai) on one frame per second from the buffer.
  Every box is classified, and on `wide_view` cameras the detector also looks at tiles of the
  frame, so small animals far away aren't lost;
- **logs visits**: one per kind of animal per camera, from arrival until it's gone 30 seconds
  (a flock is one visit with a head count; a cat among the birds is a second visit). Each visit
  gets a label by vote across its frames, a group (`songbirds`, `jays`, `cat`, `raccoon`...),
  a thumbnail of the best sighting and a clip with the seconds before and after.

Everything goes to `data/` (git-ignored; `--data-dir` or `$CAMERA_MCP_DATA` to move it):
`monitor.db` (visits, their frames, motion windows, camera status), `media/` (thumbnails) and
`clips/`. Dropped streams are reopened with backoff. The first detection loads the model, which
downloads its weights (about 500 MB) on first use.

Without SpeciesNet installed, or with `--detector none`, the monitor watches for motion only and
keeps a clip of every motion window. With it, clips are kept for visits, so leaves, shadows and
insects in the infrared light don't fill the disk; set `training_capture: true` on a camera to
keep motion clips too. On a slow machine analysis thins to one frame per segment when it falls
behind; SpeciesNet takes about 4 s per frame on an Intel MacBook Pro's CPU.

Where the cameras are goes under `site:` in `cameras.yaml`. SpeciesNet drops species not found
there, and an `allowlist` turns any other species into its class ("mammal", "bird"). `timezone`
sets the hours of the day that `activity_summary` reports (default: this machine's):

```yaml
site:
  country: USA
  admin1_region: CA
  timezone: America/Los_Angeles
  allowlist: [domestic cat, domestic dog, northern raccoon, striped skunk, virginia opossum, aves]
```

The web dashboard's species filter is built from what's actually been seen (most frequent first,
with `night-critters` once one has shown up). To pin it to a fixed menu instead, set
`menu_species: [night-critters, cat, raccoon, …]` under `site:` — then only those show, in that
order.

Settings, site-wide under `monitor:` or per camera: `pre_seconds`, `post_seconds`,
`quiet_seconds`, `motion_fps`, `pixel_threshold`, `min_changed`, `detect_every_seconds`,
`visit_close_seconds`, `min_box_conf`, `max_boxes`, `save_people_images` (default true),
`detect_vehicles` (default false; for a driveway camera) and `training_capture`.

**Learned scenery.** Fixed objects that the detector keeps mistaking for an animal or a person
(an empty feeder, a cloth on a rail, clothespins on a line) are learned after 5 sightings over 10
minutes at the same place and size, and ignored from then on (`data/scenery/`). Named animals are
never ignored, however still they sit.

**Unnamed-visit gate.** Swaying leaves and fruit don't sit still enough to be learned as scenery,
so they become visits of a bare "animal". Per camera, `unnamed_min_conf` hides such a visit when
its best detector box is below that confidence, and `unnamed_min_travel` hides it when its boxes
moved less than that many box sizes (a leaf sways in place; an animal crosses the frame) unless
its best box reached `unnamed_travel_conf` (default 0.5). Both are off (0) by default. Hidden visits
are kept, with the reason, and counted as `hidden` in history answers;
`find_visits(include_suppressed=true)` lists them. Visits with a name are never hidden. Try values
against the camera's own history first (read-only):

```bash
.venv/bin/camera-backtest --since 7d --camera east_fence --min-conf 0.3 --min-travel 0.5
```

To check the gate by eye, pick species "animal" and tick **+ hidden** on the dashboard: hidden
visits show a badge with the reason. Open one and mark it real or false; the viewer moves on to
the next unlabelled visit (or press **next unlabelled ›**). The backtest's `lbl_real` column then
counts real animals the rules would hide. A label is per visit, but a visit you mark real
can still have a junk box or two (a bird on the feeder plus a box on the clothesline), so single
boxes can be tagged too, and any tag can say what the animal actually was (a species: the right
name for a wrong-species visit, or a name for an unnamed one; never on a false one). A box with no
tag of its own takes its visit's where that must hold: every box of a false or wrong-species visit,
and of a real visit whose boxes all stayed in one place. A real visit whose boxes moved around
leaves them untagged.

In the viewer, click a box on the still (a click, not a drag) and tag it in the row below; boxes
are outlined by their tag (green real, grey false, amber wrong species, red none yet; dashed when
the tag comes from the visit's), with a ✓ / ✗ / ≠ mark. Tagging a box real makes an untagged visit
real. A real or wrong-species tag has a **what was it?** field, suggesting names you've used and
groups seen here; a wrong-species tag waits for it instead of moving on. **next box to tag ›**
walks the real visits whose still still has an untagged box (narrowed to the camera filter): the
ones to revisit before the stills are used as training data.

**Favourites.** Separate from tags, **☆ favourite** in the viewer stars a visit as a keeper. Starred
while its clip plays, it keeps that moment and reopens there (paused); starred on the image, it
opens on the still. Starred visits show ★ after their name, and **★ favourites** in the filter bar
lists them all, from any time (the window is ignored) and hidden or not. Nothing cleans up visit
media today; any future cleanup must leave favourites alone.

In the viewer, scroll to zoom and drag to pan. **full-res image** opens the still in its own tab,
filling the window, with the same scroll-zoom and drag-pan (double-click fits; **boxes** turns the
detector's boxes off; **jpeg ↗** is the file itself).

**Exclusion zones.** In the zones editor (`/zones`), the **exclude** tool marks spots that only
ever produce junk: a feeder roof's underside, a patch of hedge. Each zone has a mode (click it to
switch):

- **Ignore**: anything boxed here (box centre in the zone) is dropped as soon as MegaDetector finds
  it, before SpeciesNet classifies it, so it never opens or extends a visit and can't keep the camera
  analysing. For spots no animal can be. (A phantom "bird" on a feeder roof's underside once held a
  visit open all night: 72% of that camera's analysed frames in a day.) Dropped boxes leave no visit
  to show again, but the monitor log keeps a periodic tally ("exclusion zones dropped N on …"); to
  undo, switch the zone to **hide unrecognized** or delete it.
- **Hide unrecognized**: a visit SpeciesNet couldn't identify (a bare "animal"), with all its boxes
  in the zone, is hidden when it ends (reason `zone`): kept, and shown again with one click. An
  identified animal is never hidden. It can only act at the end: whether a visit is identified is
  only settled then. It doesn't reduce load.

An exclusion zone can also have a **box size** range (optional, per zone; `min_size` / `max_size`
in `zones.json`): only boxes whose larger side, as a fraction of the frame, is at least the "from"
and below the "below" count as in the zone. Size separates a fixed object from an animal at the
same spot, which position alone can't:

- **from** (big fixed object, small animals): on backyard, 99.9% of the ~70k boxes centred on a
  feeder roof were 0.052 or larger (median 0.075), while birds crossing there boxed at 0.015–0.03.
  From 0.04 lets a polygon be generous without dropping those birds.
- **below** (small fixed object, bigger animals): on west_alley, a bolt-loop swinging on a squirrel
  shield boxed at 0.024–0.032, while a rat sitting on it boxed at 0.05. A small zone over the
  bolt's swing with below 0.04 drops the bolt and keeps the rat.

To size a zone for something that moves (a swinging part), draw it over the whole spread of its
boxes, not one frame's: the labels map shows where past boxes landed. Zones without a size range
behave as before.

Any visit can be shown or hidden by hand from the viewer (**show** / **hide**), to undo a wrong call
by the gate or a zone. Zones apply to new visits. In `zones.json` the modes are `ignore` and
`hide_unrecognized` (the earlier `all` / `unnamed` are still read).

**Drawing zones.** The editor zooms (wheel, trackpad pinch, or **+ / − / fit**) and pans
(**Space**-drag, middle-drag or **✋**), on a full-resolution snapshot. Motion ROI, motion mask and
exclusion zones can be **polygons** as well as rectangles: pick **polygon**, click points, close
with **Enter**, a double-click or a click on the first point; drag a point to move it, **Alt**-click
to delete one. The analysis ROI is always a rectangle (the detector needs one). Polygons cost no
throughput: motion zones are rasterised into a pixel mask once at startup (per frame it's one
multiply, whatever the shape), and exclusion zones are checked once per visit. In `zones.json` a
polygon is `{"poly": [[x, y], ...]}` beside the `[x, y, w, h]` rectangles.

**Moved-camera check.** Every 5 minutes, when nothing is moving, the monitor compares each
camera's view with a stored reference (one for day, one for infrared night, in `data/views/`).
If three checks in a row don't match, it logs a warning and sets `view_moved_since` in the
camera's status: a raccoon may have tilted it. After re-aiming a camera on purpose, forget its
old reference:

```bash
.venv/bin/camera-monitor --reset-view east_fence
```

## Labels map

The dashboard's **map** view plots past visits for one camera over its picture: one dot per visit,
at its most confident box (the one its viewer shows), coloured by your label (false, real, wrong
species, and optionally unlabelled). Hover for the label, time and frame count; click a dot to open
the visit in the usual viewer and re-check or re-label it (the dot recolours). Scroll to zoom, drag
to pan. It uses the dashboard's camera, species and time filters. Clusters of "false" dots are where
an exclusion zone may belong; a "real" dot inside one is where a zone would cost you. After a camera
is re-aimed, older visits plot against the new view, so narrow the time window to after the change.

**day** / **night (IR)** show visits seen in colour or under the camera's infrared light apart
(junk behaves differently in each). With only one ticked, the picture is the newest still of that
kind in the window, so night dots sit on a night view. Each visit's light is read from its
thumbnail when it's saved: an infrared picture has no colour at all. Visits logged before this are
filled in when the monitor next starts.

## Ask the dashboard

The dashboard can answer plain-English questions about the visit log ("all skunks in the last 24
hours?", "when do raccoons come?", "was every camera watched last night?"). Claude (`claude-opus-5-5`,
low effort) answers with the MCP server's own read-only history tools (`find_visits`,
`activity_summary`, `monitor_status`, `analysis_health`), listed and called in process, so each tool
is defined once, in `server.py`, for the ask box, Claude Code and any agent alike. Visits it mentions
show as thumbnails you can open, label or hide. **Only metadata leaves the machine** (labels, times,
counts, camera names), never images or clips. It can't change settings or control cameras.

It's off until the host has an API key. To turn it on:

```bash
.venv/bin/pip install '.[ask]'
```

then add a line to the credentials file (`~/.config/camera-mcp/credentials`, chmod 600; the
environment overrides it), and restart the web agent:

```
ANTHROPIC_API_KEY=sk-ant-...
ASK_DAILY_USD=1.00
```

Spend is capped per day (`ASK_DAILY_USD`, default $1.00), tracked in `data/ask_usage.json` and
checked before every model call; the box shows each answer's cost and the day's total. A typical
question costs a few cents (the box shows the real figure). If Claude declines a question, it's retried server-side on
another model (`fallbacks: "default"`).

**Agent SDK backend.** With `ASK_BACKEND=agent` in the credentials file (and `pip install '.[agent]'`),
the box answers through the Claude Agent SDK instead of its own tool loop: a locked-down Claude Code
process per question whose only tools are those four, from `camera-mcp --metadata-only` over MCP.
Same answers, cap and ledger; a few seconds' startup. Delete the line to go back.

## Scheduled agents

`camera-agent <job>` runs one job with the same lock-down and writes `data/agent/<job>/<date>.md`
(`--email` also mails it). A job's prompt is the plugin skill of the same name, so
`/camera-mcp:weekly-digest` in Claude Code runs the same text. Jobs use `claude-sonnet-5-5`, at most
$0.25 a run and `AGENT_DAILY_USD` (default $0.50) a day, tracked in `data/agent_usage.json`.

| job | what it writes |
|---|---|
| `weekly-digest` | the last 7 days against the 7 before: what changed, firsts, coverage gaps |

`deploy/install-agent.sh weekly-digest` schedules it weekly (Sunday 08:00 by default, by email).

## Benchmark

```bash
.venv/bin/python -m camera_mcp.bench --clip some.mp4 --clip other.mp4    # or --camera backyard
```

Times SpeciesNet the way the monitor runs it (every box classified), on the whole frame and
tiled, and says how many busy cameras this machine can keep up with at one analysed frame per
second each. On a 2020 Intel MacBook Pro (CPU only: PyTorch can't use Iris Plus graphics) it
measured 7.2 s per frame whole, 20.4 s tiled, so about 0.14 busy cameras; Apple Silicon runs it
on the GPU (MPS).

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests generate a clip with ffmpeg and use it as a camera. No network or hardware is needed.

CI (`.github/workflows/ci.yml`) runs on every PR, on Python 3.12 and 3.13, the
versions the whole product (species identification included) runs on: it builds the wheel,
installs it, imports every module from the installed package, starts the entry point, and runs the
tests (without SpeciesNet, which the tests stand in for). A PR merges only once CI passes.

## Design notes

- **Capture goes through the `ffmpeg` binary, not OpenCV's video backends.** OpenCV's macOS wheels
  don't reliably bundle FFmpeg (4.14 and 5.0 on Intel Macs have none, so no RTSP). The binary can
  also use hardware decode on Apple Silicon.
- **The server holds no per-camera state.** Each snapshot opens the stream, takes one frame and
  closes it. Continuous watching (detection, buffering, recording) belongs to a separate monitor
  process, which will write events the server reads (phase 2).
