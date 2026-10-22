# Copyright (c) 2026 Eric Cooper.
"""The camera MCP server: live snapshots, and questions about what the cameras have seen.

Live tools (list_cameras, snapshot) open the cameras directly. History tools (find_visits,
activity_summary, visit_media, monitor_status, analysis_health) read the monitor's visit log,
read-only. These are the one definition of the tools: the dashboard's ask box (ask.py) offers
Claude the metadata-only ones by listing and calling this server in process, and an agent can
connect to it over MCP.

Run over stdio:  camera-mcp --config ~/projects/camera-mcp/cameras.yaml [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel

from camera_mcp.capture import CaptureError, grab_any, reachable, to_jpeg
from camera_mcp.config import (
    Camera, ConfigError, default_config_path, default_data_dir, find_camera, load_cameras, load_site,
)
from camera_mcp.history import GROUP_BY, History, HistoryError

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


class CameraInfo(BaseModel):
    name: str
    label: str
    enabled: bool
    kind: str  # "stream" or "file"
    reachable: bool
    tags: list[str]
    aliases: list[str]


INSTRUCTIONS = """\
Home cameras watched by an always-on monitor that logs animal visits.

Live: list_cameras, then snapshot for a current frame (downscaled to max_width).

History: a visit is one kind of animal on one camera, from arrival until it's gone; a flock
is one visit with max_at_once. find_visits lists them, activity_summary counts them (by
species, group, camera, local hour or day), visit_media shows a visit's thumbnail, and
monitor_status says which cameras are being watched now. analysis_health says how the detector
kept up (throughput, backlog/lag, and anything dropped on overload).

Filters: `since`/`until` take ISO times (local if no zone) or durations ago (15m, 4h, 3d, 2w),
today, yesterday. `taxon` takes a group (songbirds, jays, crows, doves/pigeons, hawks, owls,
cat, dog, rat, squirrel, rodent, skunk, raccoon, opossum, fox, coyote, person), the union
night-critters (raccoon, skunk, opossum), a class (bird, mammal), or a common name. `tag`
picks cameras by role (list_cameras shows tags). Camera names match loosely.

Be honest about absence: every history answer has `coverage`, the share of the time each
camera was actually watched, and `unanalyzed_motion_seconds`, motion the detector skipped
because it was too far behind. "Nothing seen" only means nothing was there if coverage is near
1, nothing went unanalyzed and the camera doesn't look moved; otherwise say which camera wasn't
fully watched. Unnamed "animal" visits a camera's gate judged to be leaves or similar are hidden
and counted in `hidden`; find_visits(include_suppressed=true) lists them with the reason. A visit with
still_there=true is happening now. Plain "bird" visits couldn't be sorted into a group: when
asked about songbirds, mention them as possibly songbirds too.
"""


def build_server(cameras: dict[str, Camera], history: History | None = None, data_dir: Path | None = None) -> MCPServer:
    server = MCPServer(name="camera-mcp", instructions=INSTRUCTIONS)

    def ask(fn, *args, **kwargs):
        if history is None:
            raise ToolError("the visit log isn't configured for this server")
        try:
            return fn(*args, **kwargs)
        except HistoryError as e:
            raise ToolError(str(e)) from None

    def lookup(name: str) -> Camera:
        cam = find_camera(cameras, name)
        if cam is None:
            raise ToolError(f"unknown camera {name!r}; known: {', '.join(sorted(cameras))}")
        if not cam.enabled:
            raise ToolError(f"camera {cam.name!r} is disabled in the config")
        return cam

    def info() -> list[CameraInfo]:
        return [
            CameraInfo(
                name=c.name,
                label=c.label,
                enabled=c.enabled,
                kind="stream" if c.is_stream else "file",
                reachable=c.enabled and reachable(c),
                tags=list(c.tags),
                aliases=list(c.aliases),
            )
            for c in cameras.values()
        ]

    @server.tool(annotations=READ_ONLY)
    async def list_cameras() -> list[CameraInfo]:
        """List the configured cameras and whether each one answers right now (a quick network
        check, no video decode)."""
        return await anyio.to_thread.run_sync(info)

    @server.tool(annotations=READ_ONLY)
    async def snapshot(camera: str, max_width: int = 1024) -> list:
        """Take one current frame from a camera and return it as a JPEG image.

        Args:
            camera: camera name or alias, as returned by list_cameras (case, spaces and
                underscores are ignored: "west alley" finds west_alley)
            max_width: downscale to at most this many pixels wide (160-1920)
        """
        cam = lookup(camera)
        max_width = max(160, min(max_width, 1920))
        try:
            frame, note = await anyio.to_thread.run_sync(grab_any, cam)
        except CaptureError as e:
            raise ToolError(str(e)) from None
        data, w, h = to_jpeg(frame.image, max_width=max_width)
        taken = datetime.fromtimestamp(frame.taken_at).astimezone().isoformat(timespec="seconds")
        return [Image(data=data, format="jpeg"), f"{cam.name} ({cam.label or 'no label'}) at {taken}, {w}x{h}{note}"]

    @server.resource("cameras://list", mime_type="application/json")
    def camera_list() -> str:
        """The configured cameras, as JSON."""
        return json.dumps([c.model_dump() for c in info()], indent=2)

    @server.resource("cameras://{name}/latest.jpg", mime_type="image/jpeg")
    def latest(name: str) -> bytes:
        """A current frame from the named camera, as JPEG (1024 px wide at most)."""
        try:
            return to_jpeg(grab_any(lookup(name))[0].image)[0]
        except CaptureError as e:
            raise ToolError(str(e)) from None

    @server.tool(annotations=READ_ONLY)
    async def find_visits(since: str = "24h", until: str | None = None, cameras: list[str] | None = None,
                          tag: str | None = None, taxon: str | None = None, overlapping: str | None = None,
                          min_score: float = 0.0, limit: int = 50,
                          include_suppressed: bool = False) -> dict[str, Any]:
        """Animal visits in a stretch of time, newest first, with how much of it each camera was watched.
        Use for which / when / show-me questions.

        Args:
            since: start: ISO time or a duration ago (15m, 4h, 3d, 2w), today, yesterday
            until: end (default now)
            cameras: camera names (default all)
            tag: only cameras with this tag, e.g. feeder or dog_yard
            taxon: group, union, class or common name, e.g. songbirds, night-critters, bird, striped skunk
            overlapping: only visits that overlapped a visit of this taxon on the same camera
                (e.g. taxon=cat, overlapping=bird: cats at the feeder while birds were there)
            min_score: only visits whose label won at least this share of the vote (0-1)
            limit: at most this many visits (1-500); `total` says how many matched
            include_suppressed: also list unnamed visits the gate hid as likely leaves (each has a
                `suppressed` reason); otherwise they're only counted in `hidden`
        """
        return await anyio.to_thread.run_sync(lambda: ask(
            history.find_visits, since, until, cameras, tag, taxon, overlapping, min_score, limit,
            include_suppressed))

    @server.tool(annotations=READ_ONLY)
    async def activity_summary(since: str = "7d", until: str | None = None,
                               group_by: Literal["species", "group", "camera", "hour", "day"] = "species",
                               cameras: list[str] | None = None, tag: str | None = None,
                               taxon: str | None = None) -> dict[str, Any]:
        """Count visits in a stretch of time, grouped, with how much of it each camera was watched.
        Use for how-many / when-most / busiest questions.

        Args:
            since: start: ISO time or a duration ago (15m, 4h, 3d, 2w), today, yesterday
            until: end (default now)
            group_by: species, group, camera, hour (local hour of day) or day (local date)
            cameras: camera names (default all)
            tag: only cameras with this tag
            taxon: only visits of this group, union, class or common name
        """
        if group_by not in GROUP_BY:
            raise ToolError(f"group_by must be one of {', '.join(GROUP_BY)}")
        return await anyio.to_thread.run_sync(lambda: ask(
            history.activity_summary, since, until, group_by, cameras, tag, taxon))

    @server.tool(annotations=READ_ONLY)
    async def visit_media(visit_id: int) -> list:
        """The thumbnail of a visit's best sighting, as a JPEG image, with a caption.

        Args:
            visit_id: the id from find_visits
        """
        if data_dir is None:
            raise ToolError("the data directory isn't configured for this server")
        data, caption = await anyio.to_thread.run_sync(lambda: ask(history.visit_thumbnail, visit_id, data_dir))
        return [Image(data=data, format="jpeg"), caption]

    @server.tool(annotations=READ_ONLY)
    async def monitor_status() -> list[dict[str, Any]]:
        """Per camera: is the monitor watching it now, when it last saw a frame and motion, whether
        it looks moved (tilted, blocked), how many visits are happening now, and any error."""
        return await anyio.to_thread.run_sync(lambda: ask(history.monitor_status))

    @server.tool(annotations=READ_ONLY)
    async def analysis_health(since: str = "24h", until: str | None = None) -> dict[str, Any]:
        """How the detector kept up over a window (one shared model on this machine): frames
        analysed and mean seconds per frame (throughput), peak queue depth and how far behind it
        fell (backlog/lag), and segments dropped on sustained overload (loss). `since`/`until` take
        durations ago (2h, 24h) or ISO times. 0 dropped with low peak lag means it kept up."""
        return await anyio.to_thread.run_sync(lambda: ask(history.analysis_health, since, until))

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="camera-mcp", description=__doc__)
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="camera YAML (default: $CAMERA_MCP_CONFIG, else cameras.yaml at the repo root)",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="the monitor's data directory, with monitor.db (default: $CAMERA_MCP_DATA, else data/)")
    args = parser.parse_args(argv)
    try:
        cameras = load_cameras(args.config, require_login=False)  # history needs no login; snapshot says so
        site = load_site(args.config)
    except ConfigError as e:
        print(f"camera-mcp: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    history = History(args.data_dir / "monitor.db", cameras, site)
    build_server(cameras, history, args.data_dir).run()  # stdio


if __name__ == "__main__":
    main()
