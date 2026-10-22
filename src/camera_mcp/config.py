# Copyright (c) 2026 Eric Cooper.
"""Camera configuration: a YAML list of named cameras.

A camera is usually just a `host` (its IP address): the RTSP URL is built here, with the
Tapo main-stream path and the cameras' login: $CAMERA_USER and $CAMERA_PASS if set, else the
macOS Keychain item "camera-mcp" (so no password has to sit in a config file). Anything
else (a video file, another brand's URL) goes in `source`, where `${VAR}` is expanded
from the environment. Either way the file holds no credentials, and sources are never
returned to an MCP client; cameras are addressed by name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, tzinfo
from pathlib import Path
from urllib.parse import quote, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from camera_mcp.credentials import NO_LOGIN, _expand, camera_login
from camera_mcp.errors import (
    ConfigError,  # also re-exported: callers import it from config
)
from camera_mcp.zones import (
    Exclude,
    Rect,
    Shape,
    apply_zone_overlay,
    load_zone_overlay,
    parse_excludes,
    parse_rect,
    parse_shape,
    parse_shapes,
)

# Tapo cameras: /stream1 is full resolution, /stream2 low resolution; both always on.
MAIN_STREAM, SUB_STREAM = "/stream1", "/stream2"


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
    min_changed: float = 0.002  # motion = at least this fraction of pixels changed (low, for recall)...
    min_blob_fraction: float = 0.0002  # ...or one compact patch this big (a small bird far from the camera)
    view_check_seconds: float = 300.0  # how often to check whether the camera has been moved (0: never)
    view_threshold: float = 0.5  # below this edge match (0-1), a check counts as "looks moved"
    detect_every_seconds: float = 1.0  # while there's motion or an open visit, analyse a frame this often
    tile_rows: int = 0  # analysis tiling grid rows (0 = the wide_view default: 2 if tagged, else 1)
    tile_cols: int = 0  # analysis tiling grid cols; rows*cols <= 1 means the whole frame, no tiling
    analysis_width: int = 0  # scale the analysis frame to this width before inference (0 = native)
    priority: int = 0  # analyzer service order under backlog: a higher-priority camera's queued segments
    # are analysed before a lower one's (ties: least-recently-served). Each camera stays in its own order.
    max_defer_seconds: float = 1800.0  # a motion segment is deferred (queued) up to this long; only past
    # it is dropped and logged as unanalyzed. Bursts are absorbed and caught up in idle time.
    max_clip_seconds: float = 120.0  # clips are cut off this long after the visit starts (a dove can sit for an hour)
    visit_close_seconds: float = 30.0  # a visit ends after this long without its animal being detected
    min_box_conf: float = 0.2  # detector boxes below this confidence are ignored
    max_boxes: int = 8  # boxes classified per frame (the most confident); the rest share neighbours' labels
    # The unnamed-visit gate (gate.py): a closed visit of a bare "animal" (unnamed by the classifier) is
    # stored but hidden when it fails a rule. 0 turns a rule off; try values with `camera-backtest` first.
    unnamed_min_conf: float = 0.0  # hide it if its best box is below this detector confidence
    unnamed_min_travel: float = 0.0  # hide it if its boxes moved less than this many box sizes (swaying leaves)...
    unnamed_travel_conf: float = 0.5  # ...unless its best box reached this confidence
    save_people_images: bool = True  # keep a thumbnail and clip for person visits
    detect_vehicles: bool = False  # log vehicles (for a driveway camera); elsewhere they're false alarms
    training_capture: bool = False  # keep a clip of every motion window, animal or not (training data)


@dataclass(frozen=True)
class Site:
    """Where the cameras are and what lives there. The `site:` section of cameras.yaml."""

    country: str = "USA"  # ISO 3166-1 alpha-3; SpeciesNet drops species not found there
    admin1_region: str = ""  # e.g. "CA"; narrows that to a state
    allowlist: tuple[str, ...] = ()  # common names or taxonomy prefixes; empty: everything allowed
    timezone: str = ""  # IANA name, e.g. "America/Los_Angeles", for hours of the day; "": this machine's
    alert_species: tuple[str, ...] = ()  # species/groups that trigger a real-time alert (e.g. coyote, owl)
    menu_species: tuple[str, ...] = ()  # if set, the web species filter shows only these, in this order;
    # empty (the default) derives the menu from what's actually been seen

    def tz(self) -> tzinfo:
        return ZoneInfo(self.timezone) if self.timezone else datetime.now().astimezone().tzinfo


def parse_active_window(spec: str) -> tuple[int, int] | None:
    """Parse a local-time "HH:MM-HH:MM" watch window into (start_minute, end_minute).

    Blank means always watched (None). A window whose start is after its end wraps past
    midnight, so "19:00-07:00" is the overnight hours. Raises ConfigError on a bad spec.
    """
    spec = spec.strip()
    if not spec:
        return None

    def _minute(t: str) -> int:
        h, m = t.strip().split(":")
        h, m = int(h), int(m)
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
        return h * 60 + m

    try:
        start_s, end_s = spec.split("-")
        start, end = _minute(start_s), _minute(end_s)
    except ValueError:
        raise ConfigError(f"active window must be HH:MM-HH:MM, got {spec!r}") from None
    if start == end:
        raise ConfigError(f"active window is empty (start equals end): {spec!r}")
    return (start, end)


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    label: str = ""
    enabled: bool = True
    tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    watermark: str = ""  # tiny label stamped on this camera's thumbnails; defaults to the name
    monitor: MonitorSettings = field(default_factory=MonitorSettings)
    needs_login: bool = False  # a host camera loaded without a login: it can't be opened
    active: str = ""  # local-time "HH:MM-HH:MM" window this camera is watched; blank: always
    motion_mask: tuple[Shape, ...] = ()  # tier-1 ignore zones: motion here never wakes analysis
    motion_roi: Shape | None = None  # if set, only motion inside this zone counts
    roi: Rect | None = None  # tier-2 analysis crop: inference runs on this region of the frame only
    exclude: tuple[Exclude, ...] = ()  # exclusion zones: a visit whose boxes all sit in them is hidden (gate.py)
    schedule: tuple[tuple[int, int, dict], ...] = ()  # time-of-day monitor overrides (from `when:`)

    @property
    def is_stream(self) -> bool:
        return "://" in self.source

    def is_active(self, now: datetime) -> bool:
        """Whether this camera should be watched at local time `now` (always, with no window)."""
        window = parse_active_window(self.active)
        if window is None:
            return True
        start, end = window
        minute = now.hour * 60 + now.minute
        return start <= minute < end if start < end else (minute >= start or minute < end)

    def settings_at(self, now: datetime, base: MonitorSettings | None = None) -> MonitorSettings:
        """The monitor settings in force at local time `now`: the base, with the first matching
        `when:` period's overrides applied (periods wrap past midnight, like the active window)."""
        base = base or self.monitor
        minute = now.hour * 60 + now.minute
        for start, end, overrides in self.schedule:
            inside = start <= minute < end if start < end else (minute >= start or minute < end)
            if inside:
                return replace(base, **overrides)
        return base

    @property
    def short(self) -> str:
        """The thumbnail watermark label: the `watermark` field if set, else the name."""
        return self.watermark or self.name

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


def _rtsp_url(host: str, login: tuple[str, str] | None) -> str:
    userinfo = ":".join(quote(part, safe="") for part in login) + "@" if login else ""
    return f"rtsp://{userinfo}{host}{MAIN_STREAM}"


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
        if known[key].type == "bool":
            if not isinstance(raw[key], bool):
                raise ConfigError(f"{where}: {key} must be true or false, not {raw[key]!r}")
            changes[key] = raw[key]
            continue
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


def _active(raw: dict, where: str) -> str:
    """The camera's `active:` window spec, validated now so a bad one fails at load."""
    spec = str(raw.get("active", ""))
    try:
        parse_active_window(spec)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from None
    return spec


SCHEDULE_KEYS = {"tile_rows", "tile_cols", "detect_every_seconds", "analysis_width", "priority"}  # a `when:` may override these


def _schedule(raw: dict, where: str) -> tuple[tuple[int, int, dict], ...]:
    """Parse a camera's `when:` list into (start_minute, end_minute, overrides) periods."""
    entries = raw.get("when")
    if entries is None:
        return ()
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ConfigError(f"{where}: when must be a list of {{hours: ..., <overrides>}} entries")
    periods = []
    for i, entry in enumerate(entries):
        at = f"{where}: when[{i}]"
        if "hours" not in entry:
            raise ConfigError(f"{at} needs an 'hours' window")
        window = parse_active_window(str(entry["hours"]))
        if window is None:
            raise ConfigError(f"{at}: hours must be HH:MM-HH:MM")
        overrides = {k: v for k, v in entry.items() if k != "hours"}
        if bad := sorted(set(overrides) - SCHEDULE_KEYS):
            raise ConfigError(f"{at} cannot override {', '.join(bad)} (only {', '.join(sorted(SCHEDULE_KEYS))})")
        if not overrides:
            raise ConfigError(f"{at} has no overrides")
        # Reuse _settings to type-check and convert, then keep only the overridden fields.
        converted = _settings(MonitorSettings(), overrides, at, strict=True)
        periods.append((window[0], window[1], {k: getattr(converted, k) for k in overrides}))
    return tuple(periods)


def load_site(path: str | Path) -> Site:
    """The `site:` section of the camera config (defaults if there is none)."""
    data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    raw = data.get("site") or {}
    known = {"country", "admin1_region", "allowlist", "timezone", "alert_species", "menu_species"}
    if unknown := sorted(set(raw) - known):
        raise ConfigError(f"site: unknown setting {', '.join(unknown)}")
    site = Site(
        country=str(raw.get("country", Site.country)),
        admin1_region=str(raw.get("admin1_region", "")),
        allowlist=_strings(raw.get("allowlist"), "site", "allowlist"),
        timezone=str(raw.get("timezone", "")),
        alert_species=_strings(raw.get("alert_species"), "site", "alert_species"),
        menu_species=_strings(raw.get("menu_species"), "site", "menu_species"),
    )
    try:
        site.tz()
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigError(f"site: unknown timezone {site.timezone!r}") from None
    return site


def default_data_dir() -> Path:
    """$CAMERA_MCP_DATA, else data/ at the repo root: the monitor's buffer, clips and monitor.db."""
    return Path(os.environ.get("CAMERA_MCP_DATA") or Path(__file__).resolve().parents[2] / "data")


def default_config_path() -> str:
    """$CAMERA_MCP_CONFIG, else cameras.yaml at the repo root."""
    return os.environ.get("CAMERA_MCP_CONFIG", str(Path(__file__).resolve().parents[2] / "cameras.yaml"))


def load_cameras(
    path: str | Path, *, require_login: bool = True, zones: str | Path | None = None
) -> dict[str, Camera]:
    """The cameras in a config file. Host cameras need the login (see camera_login); with
    `require_login=False` (the MCP server, whose history tools never open a camera) they load
    without it and are marked `needs_login`. With `zones`, the app-owned overlay at that path is
    merged on top of each camera's roi/motion_roi/motion_mask (see load_zone_overlay)."""
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
        source = raw.get("source") or raw.get("rtsp_url")  # rtsp_url: the key name an earlier pipeline used
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
        login = camera_login() if host else None
        if host and login is None and require_login:
            raise ConfigError(f"{where}: {NO_LOGIN}")
        cameras[name] = Camera(
            name=name,
            source=_rtsp_url(str(host), login) if host else _expand(str(source), where),
            needs_login=bool(host) and login is None,
            label=str(raw.get("label", "")),
            enabled=bool(raw.get("enabled", True)),
            tags=_strings(raw.get("tags"), where, "tags"),
            aliases=aliases,
            watermark=str(raw.get("watermark", "")),
            active=_active(raw, where),
            motion_mask=parse_shapes(raw.get("motion_mask"), where, "motion_mask"),
            motion_roi=parse_shape(raw["motion_roi"], where, "motion_roi") if raw.get("motion_roi") else None,
            roi=parse_rect(raw["roi"], where, "roi") if raw.get("roi") else None,
            exclude=parse_excludes(raw.get("exclude"), where),
            schedule=_schedule(raw, where),
            monitor=_settings(site, raw, where, strict=False),
        )
    if not cameras:
        raise ConfigError(f"{path}: no cameras defined")
    if zones is not None:
        apply_zone_overlay(cameras, load_zone_overlay(zones))
    return cameras
