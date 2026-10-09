# Copyright (c) 2026 Eric Cooper.
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
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from urllib.parse import quote, urlsplit

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_USERINFO = re.compile(r"(?<=://)[^/@\s]*@")

# Tapo cameras: /stream1 is full resolution, /stream2 low resolution; both always on.
MAIN_STREAM, SUB_STREAM = "/stream1", "/stream2"
USER_ENV, PASS_ENV = "CAMERA_USER", "CAMERA_PASS"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class MonitorSettings:
    """How the monitor watches a camera. Set site-wide under `monitor:` and per camera."""

    pre_seconds: float = 5.0  # clip footage kept from before motion started
    post_seconds: float = 10.0  # ...and after it ended
    quiet_seconds: float = 10.0  # a motion window closes after this long without motion
    segment_seconds: float = 2.0  # ring-buffer segment length (cut at keyframes, so a minimum)
    motion_fps: float = 5.0  # frames per second decoded from the low-resolution stream
    motion_width: int = 320  # those frames are scaled to this width, in grayscale
    pixel_threshold: int = 25  # a pixel counts as changed if it differs this much (0-255) from the background
    min_changed: float = 0.002  # motion = at least this fraction of pixels changed (low, for recall)
    view_check_seconds: float = 300.0  # how often to check whether the camera has been moved (0: never)
    view_threshold: float = 0.5  # below this edge match (0-1), a check counts as "looks moved"


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    label: str = ""
    enabled: bool = True
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    monitor: MonitorSettings = field(default_factory=MonitorSettings)

    @property
    def is_stream(self) -> bool:
        return "://" in self.source

    def substream(self) -> Camera | None:
        """The same camera on its low-resolution stream, for a Tapo RTSP source; else None."""
        if not (self.source.startswith("rtsp://") and self.source.endswith(MAIN_STREAM)):
            return None
        return replace(self, source=self.source[: -len(MAIN_STREAM)] + SUB_STREAM)

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


def normalize_name(name: str) -> str:
    """Names match loosely: case, spaces, underscores and hyphens are ignored."""
    return re.sub(r"[\s_-]+", "", name).lower()


def find_camera(cameras: dict[str, Camera], query: str) -> Camera | None:
    """The camera whose name or alias matches `query` loosely ("West Alley" finds west_alley)."""
    want = normalize_name(query)
    for cam in cameras.values():
        if want in {normalize_name(n) for n in (cam.name, *cam.aliases)}:
            return cam
    return None


def _settings(base: MonitorSettings, raw: dict, where: str, *, strict: bool) -> MonitorSettings:
    """`base` with any MonitorSettings keys in `raw` applied. `strict`: other keys are errors."""
    known = {f.name: f for f in fields(MonitorSettings)}
    if strict and (unknown := sorted(set(raw) - set(known))):
        raise ConfigError(f"{where}: unknown setting {', '.join(unknown)}")
    changes = {}
    for key in set(raw) & set(known):
        kind = int if known[key].type == "int" else float
        try:
            changes[key] = kind(raw[key])
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: {key} must be a number, not {raw[key]!r}") from None
        if changes[key] < 0:
            raise ConfigError(f"{where}: {key} must not be negative")
    settings = replace(base, **changes)
    if settings.segment_seconds < 2:  # segment files are named by the second they start
        raise ConfigError(f"{where}: segment_seconds must be at least 2")
    return settings


def _strings(value, where: str, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
        raise ConfigError(f"{where}: {key} must be a list of strings")
    return tuple(items)


def default_config_path() -> str:
    """$CAMERA_MCP_CONFIG, else cameras.yaml at the repo root."""
    return os.environ.get("CAMERA_MCP_CONFIG", str(Path(__file__).resolve().parents[2] / "cameras.yaml"))


def load_cameras(path: str | Path) -> dict[str, Camera]:
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"camera config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    site = _settings(MonitorSettings(), data.get("monitor") or {}, "monitor", strict=True)
    cameras: dict[str, Camera] = {}
    seen: dict[str, str] = {}  # normalized name or alias -> camera
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
        where = f"camera {name!r}"
        aliases = _strings(raw.get("aliases"), where, "aliases")
        for n in (name, *aliases):
            if (other := seen.setdefault(normalize_name(n), name)) != name or (n == name and name in cameras):
                raise ConfigError(f"cameras[{i}]: {n!r} clashes with camera {other!r} (names match loosely)")
        cameras[name] = Camera(
            name=name,
            source=_rtsp_url(str(host), where) if host else _expand(str(source), where),
            label=str(raw.get("label", "")),
            enabled=bool(raw.get("enabled", True)),
            tags=_strings(raw.get("tags"), where, "tags"),
            aliases=aliases,
            monitor=_settings(site, raw, where, strict=False),
        )
    if not cameras:
        raise ConfigError(f"{path}: no cameras defined")
    return cameras
