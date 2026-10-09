# Copyright (c) 2026 Eric Cooper.
"""The camera MCP server (phase 1: list and snapshot).

Run over stdio:  camera-mcp --config ~/projects/camera-mcp/cameras.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import anyio
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel

from camera_mcp.capture import CaptureError, grab_frame, reachable, to_jpeg
from camera_mcp.config import Camera, default_config_path, find_camera, load_cameras

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


class CameraInfo(BaseModel):
    name: str
    label: str
    enabled: bool
    kind: str  # "stream" or "file"
    reachable: bool
    tags: list[str]
    aliases: list[str]


def build_server(cameras: dict[str, Camera]) -> MCPServer:
    server = MCPServer(
        name="camera-mcp",
        instructions=(
            "Home IP cameras, addressed by name. Call list_cameras first. snapshot returns one "
            "current frame as an image; it is downscaled to max_width to keep tokens down."
        ),
    )

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
            frame = await anyio.to_thread.run_sync(grab_frame, cam)
        except CaptureError as e:
            raise ToolError(str(e)) from None
        data, w, h = to_jpeg(frame.image, max_width=max_width)
        taken = datetime.fromtimestamp(frame.taken_at).astimezone().isoformat(timespec="seconds")
        return [Image(data=data, format="jpeg"), f"{cam.name} ({cam.label or 'no label'}) at {taken}, {w}x{h}"]

    @server.resource("cameras://list", mime_type="application/json")
    def camera_list() -> str:
        """The configured cameras, as JSON."""
        return json.dumps([c.model_dump() for c in info()], indent=2)

    @server.resource("cameras://{name}/latest.jpg", mime_type="image/jpeg")
    def latest(name: str) -> bytes:
        """A current frame from the named camera, as JPEG (1024 px wide at most)."""
        try:
            return to_jpeg(grab_frame(lookup(name)).image)[0]
        except CaptureError as e:
            raise ToolError(str(e)) from None

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="camera-mcp", description=__doc__)
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="camera YAML (default: $CAMERA_MCP_CONFIG, else cameras.yaml at the repo root)",
    )
    args = parser.parse_args(argv)
    try:
        cameras = load_cameras(args.config)
    except ValueError as e:
        print(f"camera-mcp: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    build_server(cameras).run()  # stdio


if __name__ == "__main__":
    main()
