# Copyright (c) 2026 Eric Cooper. All rights reserved.
"""Camera configuration: a YAML list of named cameras.

A camera is usually just a `host` (its IP address): the RTSP URL is built here, with the
credentials from $CAMERA_USER and $CAMERA_PASS and the Tapo main-stream path. Anything
else (a video file, another brand's URL) goes in `source`, where `${VAR}` is expanded
from the environment. Either way the file holds no credentials, and sources are never
returned to an MCP client; cameras are addressed by name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_USERINFO = re.compile(r"(?<=://)[^/@\s]*@")

# Tapo cameras: /stream1 is full resolution, /stream2 low resolution; both always on.
MAIN_STREAM = "/stream1"
USER_ENV, PASS_ENV = "CAMERA_USER", "CAMERA_PASS"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    label: str = ""
    enabled: bool = True

    @property
    def is_stream(self) -> bool:
        return "://" in self.source

    def endpoint(self) -> tuple[str, int] | None:
        """(host, port) for a network source, else None."""
        if not self.is_stream:
            return None
        parts = urlsplit(self.source)
        default = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}.get(parts.scheme, 554)
        return (parts.hostname or "", parts.port or default)


def redact(text: str) -> str:
    """Strip user:password@ from any URL in `text` (OpenCV errors can echo the URL)."""
    return _USERINFO.sub("***@", text)


def _expand(value: str, where: str) -> str:
    def sub(m: re.Match[str]) -> str:
        var = m.group(1)
        if var not in os.environ:
            raise ConfigError(f"{where}: environment variable {var} is not set")
        return os.environ[var]

    return os.path.expanduser(_VAR.sub(sub, value))


def _rtsp_url(host: str, where: str) -> str:
    missing = [v for v in (USER_ENV, PASS_ENV) if v not in os.environ]
    if missing:
        raise ConfigError(f"{where}: environment variable {' and '.join(missing)} is not set")
    user, password = (quote(os.environ[v], safe="") for v in (USER_ENV, PASS_ENV))
    return f"rtsp://{user}:{password}@{host}{MAIN_STREAM}"


def load_cameras(path: str | Path) -> dict[str, Camera]:
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"camera config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    cameras: dict[str, Camera] = {}
    for i, raw in enumerate(data.get("cameras") or []):
        name = raw.get("name")
        host = raw.get("host")
        source = raw.get("source") or raw.get("rtsp_url")  # rtsp_url: wildlife-cv-pipeline's key
        if name is None or not (host or source):
            raise ConfigError(f"cameras[{i}]: needs a name and a host (or a source)")
        if host and source:
            raise ConfigError(f"cameras[{i}]: give a host or a source, not both")
        if not isinstance(name, str):  # e.g. `name: off` or `name: yes`, which YAML reads as booleans
            raise ConfigError(f"cameras[{i}]: name {name!r} is not a string; quote it in the YAML")
        if name in cameras:
            raise ConfigError(f"cameras[{i}]: duplicate name {name!r}")
        where = f"camera {name!r}"
        cameras[name] = Camera(
            name=name,
            source=_rtsp_url(str(host), where) if host else _expand(str(source), where),
            label=str(raw.get("label", "")),
            enabled=bool(raw.get("enabled", True)),
        )
    if not cameras:
        raise ConfigError(f"{path}: no cameras defined")
    return cameras
