# camera-mcp

_Last updated: 2026-10-08_

An MCP server over the home IP cameras. A model can list the cameras and look at a current frame
from any of them; detections, activity and recording come in later phases. Combines the camera
work in `camera-controller` and `wildlife-cv-pipeline`. The original design is in
`../camera-controller/MCP_SERVER_DESIGN.md`; phase 2 (monitor, visit log, history tools) is in
[docs/phase2-design.md](docs/phase2-design.md), backed by the
[SpeciesNet eval](docs/speciesnet-eval-2026-10-08.md). SpeciesNet's full label list is in
[docs/speciesnet-labels-v4.0.3a.txt](docs/speciesnet-labels-v4.0.3a.txt).

Household context and motivations stay out of the repo, in the git-ignored `notes/` folder.

## Tools (phase 1)

| Tool | What it returns |
|---|---|
| `list_cameras` | name, label, enabled, stream or file, and whether it answers right now (a TCP check, no decode) |
| `snapshot(camera, max_width=1024)` | one current frame as a JPEG image, downscaled to `max_width` |

Resources: `cameras://list` (JSON) and `cameras://{name}/latest.jpg`.

Both tools are marked read-only. Cameras are addressed by name; a source URL or credential is never
returned.

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

The RTSP URL is built from the host (`rtsp://<user>:<pass>@<host>/stream1`), with the credentials
taken from `$CAMERA_USER` and `$CAMERA_PASS`, the Camera Account set in the Tapo app. Credentials
never go in the file. Anything else goes in `source` instead of `host`: a video file (which is how
the tests run, and how to try the server away from home) or a full URL for a non-Tapo camera,
where `${VAR}` is expanded from the environment.

## Use from Claude Code

```bash
claude mcp add cameras -e CAMERA_USER=... -e CAMERA_PASS=... -- ~/projects/camera-mcp/.venv/bin/camera-mcp --config ~/projects/camera-mcp/cameras.yaml
```

## Probe the cameras

```bash
.venv/bin/python -m camera_mcp.probe
```

For every camera in `cameras.yaml`, this checks that it answers and accepts the login, and reports
what its streams deliver (codec, resolution, frame rate). Tapo cameras are checked on both
`/stream1` and `/stream2`. It's read-only: each stream is opened, its header read, and closed.
It exits non-zero if any stream fails.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests generate a clip with ffmpeg and use it as a camera. No network or hardware is needed.

CI (`.github/workflows/ci.yml`) runs on every PR and on pushes to `main`, on Python 3.11 and 3.13:
it builds the wheel, installs it, imports every module from the installed package, starts the
entry point, and runs the tests. A PR merges only once CI passes.

## Design notes

- **Capture goes through the `ffmpeg` binary, not OpenCV's video backends.** OpenCV's macOS wheels
  don't reliably bundle FFmpeg (4.14 and 5.0 on Intel Macs have none, so no RTSP). The binary can
  also use hardware decode on Apple Silicon.
- **The server holds no per-camera state.** Each snapshot opens the stream, takes one frame and
  closes it. Continuous watching (detection, buffering, recording) belongs to a separate monitor
  process, which will write events the server reads (phase 2).
