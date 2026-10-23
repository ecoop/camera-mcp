# Copyright (c) 2026 Eric Cooper.
"""What every route module shares: the cameras, the visit log, the data folder and whether editing
is on, plus two small helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from starlette.responses import JSONResponse

from camera_mcp.config import default_config_path
from camera_mcp.history import History, HistoryError


@dataclass
class Context:
    cameras: dict  # name -> Camera; the zones route replaces entries in place after a save
    data_dir: Path
    site: object | None = None
    config: Path | None = None  # the camera YAML, for anything that starts its own camera-mcp server
    history: History = field(init=False)
    editable: bool = field(init=False)  # CAMERA_MCP_WEB_READONLY=1 turns every write off

    def __post_init__(self):
        self.history = History(self.db_path, self.cameras, self.site)
        self.config = Path(self.config) if self.config else Path(default_config_path())
        self.editable = os.environ.get("CAMERA_MCP_WEB_READONLY", "").lower() not in ("1", "true", "yes")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "monitor.db"

    @property
    def zones_path(self) -> Path:
        return self.data_dir / "zones.json"

    def query(self, fn, *a, **k):
        """fn's result, or a 503 response when the visit log can't answer (e.g. none yet)."""
        try:
            return fn(*a, **k)
        except HistoryError as e:
            return JSONResponse({"error": str(e)}, status_code=503)


async def to_thread(fn, *a):
    import anyio
    return await anyio.to_thread.run_sync(lambda: fn(*a))
