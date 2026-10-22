# Copyright (c) 2026 Eric Cooper.
"""Zones on a camera's picture: shapes (rectangles and polygons, as fractions of the frame), exclusion
zones (a shape, a mode and an optional size range), the geometry the monitor and gate test boxes
against, and the zone overlay file the dashboard's editor writes (data/zones.json)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from camera_mcp.errors import ConfigError

if TYPE_CHECKING:
    from camera_mcp.config import Camera


Rect = tuple[float, float, float, float]  # (x, y, w, h), each a fraction of the frame (0..1)


def parse_rect(value, where: str, key: str) -> Rect:
    """Validate an [x, y, w, h] rectangle of frame fractions (within the frame)."""
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: {key} must be [x, y, w, h] fractions, got {value!r}") from None
    if not (0 <= x and 0 <= y and w > 0 and h > 0 and x + w <= 1 and y + h <= 1):
        raise ConfigError(f"{where}: {key} {list(value)} is out of bounds (0..1, x+w<=1, y+h<=1)")
    return (x, y, w, h)


Poly = tuple[tuple[float, float], ...]  # a polygon: 3+ (x, y) vertices, each a fraction of the frame


Shape = Rect | Poly  # a zone's shape (the analysis roi is always a Rect: the detector needs one)


def is_poly(shape: Shape) -> bool:
    return bool(shape) and isinstance(shape[0], tuple)


def parse_poly(value, where: str, key: str) -> Poly:
    """Validate a polygon: a list of 3+ [x, y] vertices, each within the frame."""
    try:
        pts = tuple((float(x), float(y)) for x, y in value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: {key} polygon must be a list of [x, y] fractions, got {value!r}") from None
    if len(pts) < 3:
        raise ConfigError(f"{where}: {key} polygon needs at least 3 points, got {len(pts)}")
    if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in pts):
        raise ConfigError(f"{where}: {key} polygon has a point outside the frame (0..1)")
    return pts


def parse_shape(value, where: str, key: str) -> Shape:
    """A zone shape: an [x, y, w, h] rectangle, or {"poly": [[x, y], ...]}."""
    if isinstance(value, dict):
        if "poly" not in value:
            raise ConfigError(f"{where}: {key} must be [x, y, w, h] or {{poly: [[x, y], ...]}}")
        return parse_poly(value["poly"], where, key)
    return parse_rect(value, where, key)


def shape_json(shape: Shape):
    """A shape as it's written to zones.json and served to the editor."""
    return {"poly": [list(p) for p in shape]} if is_poly(shape) else list(shape)


def shape_contains(shape: Shape, x: float, y: float) -> bool:
    """Whether the point (x, y) lies inside the shape (frame fractions); edges count as inside."""
    if not is_poly(shape):
        sx, sy, w, h = shape
        return sx <= x <= sx + w and sy <= y <= sy + h
    inside, n = False, len(shape)
    for i in range(n):  # even-odd ray casting
        (x0, y0), (x1, y1) = shape[i], shape[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


# What an exclusion zone does. "ignore": any box centred in it is dropped as soon as it's found, before
# it's classified (never a visit). "hide_unrecognized": a visit that SpeciesNet couldn't identify, with all
# its boxes in it, is hidden when it ends (reversibly). Earlier names, still accepted: "all", "unnamed".
EXCLUDE_MODES = ("hide_unrecognized", "ignore")


LEGACY_MODES = {"all": "ignore", "unnamed": "hide_unrecognized"}


Exclude = tuple  # an exclusion zone: (shape, mode) or (shape, mode, min_size[, max_size]); see zone_parts


def zone_parts(zone: Exclude) -> tuple[Shape, str, float, float]:
    """(shape, mode, min_size, max_size) of an exclusion zone; each size is 0 (no limit) when unset."""
    shape, mode, *rest = zone
    rest = [float(r) for r in rest] + [0.0, 0.0]
    return shape, mode, rest[0], rest[1]


def zone_holds(zone: Exclude, cx: float, cy: float, w: float, h: float) -> bool:
    """Whether a box (centre and size, fractions of the full frame) is in the zone: its centre is
    inside the shape and its larger side is within the zone's size range, if it has one: at least
    min_size, below max_size. A min size tells a fixed object (a feeder roof, boxed ~0.07 wide) from
    a small bird passing the same spot; a max size does the reverse, telling a small fixed thing (a
    bolt on a squirrel shield, boxed ~0.03) from a rat sitting on it (~0.05). Position alone can't."""
    shape, _, min_size, max_size = zone_parts(zone)
    size = max(w, h)
    return size >= min_size and (not max_size or size < max_size) and shape_contains(shape, cx, cy)


def _size(z: dict, key: str, where: str) -> float:
    value = z.get(key) or 0
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}: exclude {key} must be a number, not {z.get(key)!r}") from None
    if not 0 <= value < 1:
        raise ConfigError(f"{where}: exclude {key} must be at least 0 and below 1 (a fraction of the frame)")
    return value


def parse_excludes(value, where: str) -> tuple[Exclude, ...]:
    """A list of exclusion zones, each {"rect": [x, y, w, h]} or {"poly": [[x, y], ...]}, with
    "mode": "ignore" | "hide_unrecognized" (empty if omitted), and optionally a size range,
    "min_size" and/or "max_size" (0..1: only boxes whose larger side is at least min_size and below
    max_size, as fractions of the frame, count). Mode defaults to "hide_unrecognized", the cautious
    one; the earlier names "all" / "unnamed" are read as those."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(z, dict) for z in value):
        raise ConfigError(f"{where}: exclude must be a list of {{rect|poly, mode: ignore|hide_unrecognized}}")
    out = []
    for z in value:
        mode = z.get("mode", "hide_unrecognized")
        mode = LEGACY_MODES.get(mode, mode)
        if mode not in EXCLUDE_MODES:
            raise ConfigError(f"{where}: exclude mode must be one of {', '.join(EXCLUDE_MODES)}, not {mode!r}")
        shape = parse_poly(z["poly"], where, "exclude") if "poly" in z else parse_rect(z.get("rect"), where, "exclude")
        min_size, max_size = _size(z, "min_size", where), _size(z, "max_size", where)
        if max_size and max_size <= min_size:
            raise ConfigError(f"{where}: exclude max_size must be above min_size, or no box would count")
        out.append((shape, mode, min_size, max_size) if max_size else (shape, mode, min_size) if min_size else (shape, mode))
    return tuple(out)


def exclude_json(zone: Exclude) -> dict:
    shape, mode, min_size, max_size = zone_parts(zone)
    out = {"poly": [list(p) for p in shape], "mode": mode} if is_poly(shape) else {"rect": list(shape), "mode": mode}
    if min_size:
        out["min_size"] = min_size
    if max_size:
        out["max_size"] = max_size
    return out


def parse_shapes(value, where: str, key: str) -> tuple[Shape, ...]:
    """A list of zone shapes, rectangles and/or polygons (empty if omitted)."""
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(r, (list, tuple, dict)) for r in value):
        raise ConfigError(f"{where}: {key} must be a list of [x, y, w, h] rectangles or {{poly: ...}} polygons")
    return tuple(parse_shape(r, where, key) for r in value)


parse_rects = parse_shapes  # the old name: motion masks were rectangles only


ZONE_FIELDS = ("roi", "motion_roi", "motion_mask", "exclude")  # per-camera zones the web editor owns


def load_zone_overlay(path: str | Path) -> dict[str, dict]:
    """The app-owned zone overrides (data/zones.json), parsed and validated. Maps a camera name to
    the zone fields it overrides: `roi`/`motion_roi` as a Rect (or None to clear), `motion_mask`
    as a tuple of Rects and `exclude` as a tuple of (Rect, mode). A missing file is no overrides ({}); a malformed one fails loudly, the way
    a bad cameras.yaml does. The web editor is the sole writer."""
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        raise ConfigError(f"{path}: unreadable zone overlay ({e})") from None
    cams = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cams, dict):
        raise ConfigError(f"{path}: zone overlay needs a 'cameras' object")
    overlay: dict[str, dict] = {}
    for name, raw in cams.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"zones[{name!r}]: must be an object of zone fields")
        where = f"zones for {name!r}"
        over: dict = {}
        if "roi" in raw:
            over["roi"] = parse_rect(raw["roi"], where, "roi") if raw["roi"] else None
        if "motion_roi" in raw:
            over["motion_roi"] = parse_shape(raw["motion_roi"], where, "motion_roi") if raw["motion_roi"] else None
        if "motion_mask" in raw:
            over["motion_mask"] = parse_shapes(raw["motion_mask"], where, "motion_mask")
        if "exclude" in raw:
            over["exclude"] = parse_excludes(raw["exclude"], where)
        if over:
            overlay[name] = over
    return overlay


def apply_zone_overlay(cameras: dict[str, Camera], overlay: dict[str, dict]) -> dict[str, Camera]:
    """Merge a zone overlay into `cameras` in place, overriding roi/motion_roi/motion_mask/exclude where the
    overlay sets them. Overlay names matching no camera are ignored. Returns the same dict."""
    for name, over in overlay.items():
        if name in cameras:
            cameras[name] = replace(cameras[name], **over)
    return cameras


def save_zone_overlay(path: str | Path, overlay_json: dict) -> None:
    """Atomically write the app-owned zone overlay (plain, already-validated JSON)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(overlay_json, indent=2))
    tmp.replace(path)


def zone_drop(camera: Camera):
    """A test for the detector: True for a box (fractions of the analysis frame, i.e. the camera's
    roi crop) whose centre lies in one of the camera's "ignore" exclusion zones (fractions of the
    full frame), and whose size is in the zone's size range if it has one (zone_holds). None
    when the camera has none. Such a box is dropped before it's classified, so it can't open or extend
    a visit: a phantom on a feeder roof can't keep the GPU busy all night, while a small bird passing
    the same spot (below the zone's min_size) is kept."""
    zones = [z for z in camera.exclude if zone_parts(z)[1] == "ignore"]
    if not zones:
        return None
    rx, ry, rw, rh = camera.roi or (0.0, 0.0, 1.0, 1.0)

    def drop(b) -> bool:
        cx, cy = rx + (b.x + b.w / 2) * rw, ry + (b.y + b.h / 2) * rh
        return any(zone_holds(z, cx, cy, b.w * rw, b.h * rh) for z in zones)
    return drop
