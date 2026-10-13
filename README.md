# camera-mcp

[![CI](https://github.com/ecoop/camera-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/ecoop/camera-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-MIT-blue)

_Last updated: 2026-10-13_

An MCP server over the home IP cameras. A model can list the cameras and look at a current frame
from any of them; detections, activity and recording come in later phases. Combines the camera
work in `camera-controller` and `wildlife-cv-pipeline`. The original design is in
`../camera-controller/MCP_SERVER_DESIGN.md`; phase 2 (monitor, visit log, history tools) is in
[docs/phase2-design.md](docs/phase2-design.md), backed by the
[SpeciesNet eval](docs/speciesnet-eval-2026-10-08.md). SpeciesNet's full label list is in
[docs/speciesnet-labels-v4.0.3a.txt](docs/speciesnet-labels-v4.0.3a.txt).

Household context and motivations stay out of the repo, in the git-ignored `notes/` folder.

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

Needs Python 3.11+ and the `ffmpeg` binary (Homebrew: `brew install ffmpeg`).

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
SpeciesNet fast enough), see [deploy/README.md](deploy/README.md).


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

Settings, site-wide under `monitor:` or per camera: `pre_seconds`, `post_seconds`,
`quiet_seconds`, `motion_fps`, `pixel_threshold`, `min_changed`, `detect_every_seconds`,
`visit_close_seconds`, `min_box_conf`, `max_boxes`, `save_people_images` (default true),
`detect_vehicles` (default false; for a driveway camera) and `training_capture`.

**Learned scenery.** Fixed objects that the detector keeps mistaking for an animal or a person
(an empty feeder, a cloth on a rail, clothespins on a line) are learned after 5 sightings over 10
minutes at the same place and size, and ignored from then on (`data/scenery/`). Named animals are
never ignored, however still they sit.

**Moved-camera check.** Every 5 minutes, when nothing is moving, the monitor compares each
camera's view with a stored reference (one for day, one for infrared night, in `data/views/`).
If three checks in a row don't match, it logs a warning and sets `view_moved_since` in the
camera's status: a raccoon may have tilted it. After re-aiming a camera on purpose, forget its
old reference:

```bash
.venv/bin/camera-monitor --reset-view east_fence
```

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

CI (`.github/workflows/ci.yml`) runs on every PR and on pushes to `main`, on Python 3.11–3.14:
it builds the wheel, installs it, imports every module from the installed package, starts the
entry point, and runs the tests. A PR merges only once CI passes.

## Design notes

- **Capture goes through the `ffmpeg` binary, not OpenCV's video backends.** OpenCV's macOS wheels
  don't reliably bundle FFmpeg (4.14 and 5.0 on Intel Macs have none, so no RTSP). The binary can
  also use hardware decode on Apple Silicon.
- **The server holds no per-camera state.** Each snapshot opens the stream, takes one frame and
  closes it. Continuous watching (detection, buffering, recording) belongs to a separate monitor
  process, which will write events the server reads (phase 2).
