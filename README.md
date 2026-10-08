# camera-mcp

_Last updated: 2026-10-08_

An MCP server over the home IP cameras. A model can list the cameras and look at a current frame
from any of them; detections, activity and recording come in later phases. Combines the camera
work in `camera-controller` and `wildlife-cv-pipeline`. The design is in
`../camera-controller/MCP_SERVER_DESIGN.md`.

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

`cameras.yaml` is git-ignored. Credentials stay out of it: write `${CAMERA_USER}` and
`${CAMERA_PASS}` in the source URL and set them in the environment. A video file works as a
camera, which is how the tests run and how to try the server away from home.

## Use from Claude Code

```bash
claude mcp add cameras -e CAMERA_USER=... -e CAMERA_PASS=... -- ~/projects/camera-mcp/.venv/bin/camera-mcp --config ~/projects/camera-mcp/cameras.yaml
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests generate a clip with ffmpeg and use it as a camera. No network or hardware is needed.

## Design notes

- **Capture goes through the `ffmpeg` binary, not OpenCV's video backends.** OpenCV's macOS wheels
  don't reliably bundle FFmpeg (4.14 and 5.0 on Intel Macs have none, so no RTSP). The binary can
  also use hardware decode on Apple Silicon.
- **The server holds no per-camera state.** Each snapshot opens the stream, takes one frame and
  closes it. Continuous watching (detection, buffering, recording) belongs to a separate monitor
  process, which will write events the server reads (phase 2).
