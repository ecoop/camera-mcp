# Copyright (c) 2026 Eric Cooper.
"""Assemble the dashboard from its route modules, and run it (see the package docstring)."""

from __future__ import annotations

import argparse
from pathlib import Path

from starlette.applications import Starlette

from camera_mcp.config import ConfigError, default_config_path, default_data_dir, load_cameras, load_site
from camera_mcp.web import askbox, media, pages, tags, visits, zones
from camera_mcp.web.context import Context


def build_app(cameras: dict, data_dir: Path, site=None, config: Path | None = None) -> Starlette:
    ctx = Context(cameras, data_dir, site, config)
    return Starlette(routes=[*pages.routes(ctx), *visits.routes(ctx), *askbox.routes(ctx), *zones.routes(ctx),
                             *media.routes(ctx), *tags.routes(ctx)])


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from camera_mcp import web

    parser = argparse.ArgumentParser(prog="python -m camera_mcp.web", description=web.__doc__,
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
    app = build_app(cameras, data_dir, site, Path(args.config))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
