# Copyright (c) 2026 Eric Cooper.
"""The unnamed-visit gate: hide "animal" visits that look like swaying leaves, not animals.

MegaDetector boxes leaves, fruit and fence knots as "animal", and the classifier can't name
them, so they become visits of a bare "animal". Two things tell them from real animals, measured
on the live log (see the backtest below):

- **Confidence.** A visit the classifier *could* name almost never peaks below 0.3 detector
  confidence; a third or more of the unnamed ones do.
- **Travel.** A leaf sways about a fixed point; an animal crosses the frame. Travel is how far the
  box centres spread over the visit, in units of the box's own size.

A closed visit keyed "animal" that fails a rule is *suppressed*, not deleted: it is stored with
the reason, hidden from history answers by default, and still there to review and label. Visits
with a name are never touched. Every rule is off (0) until a camera sets it.

The backtest (`camera-backtest`) replays the rules over the stored log, read-only, so a
camera's thresholds can be chosen from its own history before they're switched on.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from camera_mcp.config import default_data_dir
from camera_mcp.zones import zone_holds, zone_parts
from camera_mcp.times import iso, parse_when
from camera_mcp.visits import Box

GATED = "animal"  # the visit key the gate judges: a detection the classifier couldn't name
NAMED_EXCLUDE = ("person", "mammal", "bird", "vehicle", "animal")  # labels that aren't a real name


@dataclass(frozen=True)
class GateRules:
    min_conf: float = 0.0  # suppress if the visit's best box is below this confidence (0: off)
    min_travel: float = 0.0  # suppress if its boxes travel less than this many box sizes (0: off)...
    travel_conf: float = 0.5  # ...and its best box is below this confidence (a confident box is kept)

    @classmethod
    def of(cls, settings) -> GateRules:
        """The rules in a camera's MonitorSettings."""
        return cls(settings.unnamed_min_conf, settings.unnamed_min_travel, settings.unnamed_travel_conf)

    @property
    def active(self) -> bool:
        return bool(self.min_conf or self.min_travel)


def travel(boxes: list[Box]) -> float | None:
    """How far the box centres spread, as a multiple of the boxes' median size (the larger side).
    None with fewer than two boxes: a single sighting can't show whether it moved."""
    if len(boxes) < 2:
        return None
    xs = [b.x + b.w / 2 for b in boxes]
    ys = [b.y + b.h / 2 for b in boxes]
    size = sorted(max(b.w, b.h) for b in boxes)[len(boxes) // 2]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / max(size, 1e-3)


def judge(key: str, boxes: list[Box], rules: GateRules) -> str | None:
    """Why a closed visit should be suppressed ("low_conf" or "stationary"), or None to keep it.
    `boxes` are all of the visit's detector boxes, every frame."""
    if key != GATED or not rules.active or not boxes:
        return None
    peak = max(b.conf for b in boxes)
    if rules.min_conf and peak < rules.min_conf:
        return "low_conf"
    if rules.min_travel and peak < rules.travel_conf:
        moved = travel(boxes)
        if moved is not None and moved < rules.min_travel:
            return "stationary"
    return None


def in_zones(key: str, boxes: list[Box], zones, roi=None) -> str | None:
    """"zone" when every one of a closed visit's boxes sits in an exclusion zone that applies to it,
    else None. Boxes are fractions of the analysis frame (the camera's `roi` crop, if any); zones are
    fractions of the full frame. A box is in a zone when its centre is (and its size is in the zone's
    size range, if it has one: zones.zone_holds). A "hide_unrecognized" zone only
    applies to visits keyed "animal" (SpeciesNet couldn't identify them); an "ignore" zone applies to any
    (normally its boxes never get this far: see zones.zone_drop). One box outside keeps the visit."""
    if not zones or not boxes:
        return None
    rx, ry, rw, rh = roi or (0.0, 0.0, 1.0, 1.0)
    applies = [z for z in zones if zone_parts(z)[1] == "ignore" or key == GATED]
    if not applies:
        return None
    for b in boxes:
        cx, cy = rx + (b.x + b.w / 2) * rw, ry + (b.y + b.h / 2) * rh
        if not any(zone_holds(z, cx, cy, b.w * rw, b.h * rh) for z in applies):
            return None
    return "zone"


# ---- backtest ---------------------------------------------------------------------------

def _visit_boxes(db: sqlite3.Connection, since: str, cameras: list[str]) -> list[tuple]:
    """(id, camera, key, is_named, verdict, boxes) for every closed visit since `since` (UTC ISO)."""
    where, params = "v.started_at >= ? AND v.open = 0", [since]
    if cameras:
        where += f" AND v.camera IN ({', '.join('?' * len(cameras))})"
        params += cameras
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    verdict = ("(SELECT verdict FROM visit_labels l WHERE l.visit_id = v.id)" if "visit_labels" in tables
               else "NULL")
    visits = db.execute(f"SELECT v.id, v.camera, v.grp, v.label, v.taxonomy, {verdict} FROM visits v"
                        f" WHERE {where}", params).fetchall()
    boxes: dict[int, list[Box]] = defaultdict(list)
    for vid, raw in db.execute(f"SELECT f.visit_id, f.boxes FROM visit_frames f JOIN visits v ON v.id = f.visit_id"
                               f" WHERE {where}", params):
        for b in json.loads(raw):
            boxes[vid].append(Box(*b["bbox"], conf=b["conf"]))
    return [(vid, cam, grp, bool(tax) and label not in NAMED_EXCLUDE, ver, boxes[vid])
            for vid, cam, grp, label, tax, ver in visits]


def backtest(db_path: Path, rules: GateRules, since: str, cameras: list[str] | None = None) -> dict[str, dict]:
    """Per camera: unnamed visits and how many the rules would suppress (by reason); named visits
    and how many the same rules *would* hit if applied to them (a stand-in for real animals the
    gate would lose); and, where visits are labelled, how the suppressed ones were labelled."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        rows = _visit_boxes(db, since, cameras or [])
    out: dict[str, dict] = {}
    for _vid, cam, key, named, verdict, boxes in rows:
        c = out.setdefault(cam, {"unnamed": 0, "suppressed": 0, "low_conf": 0, "stationary": 0,
                                 "named": 0, "named_hit": 0, "labelled_false": 0, "labelled_real": 0})
        if key == GATED:
            c["unnamed"] += 1
            if reason := judge(key, boxes, rules):
                c["suppressed"] += 1
                c[reason] += 1
                if verdict == "false":
                    c["labelled_false"] += 1
                elif verdict == "real":
                    c["labelled_real"] += 1  # a real animal the gate would hide
        elif named:
            c["named"] += 1
            c["named_hit"] += judge(GATED, boxes, rules) is not None
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="camera-backtest", description="Replay the unnamed-visit gate over the stored visit log (read-only):"
        " how many unnamed visits each camera's rules would hide, and how many named ones they'd have hit.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="where monitor.db is")
    parser.add_argument("--since", default="7d", help="how far back: a duration (3d, 2w) or an ISO time")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME", help="only this camera (repeatable)")
    parser.add_argument("--min-conf", type=float, default=0.0, help="unnamed_min_conf to try (0: off)")
    parser.add_argument("--min-travel", type=float, default=0.0, help="unnamed_min_travel to try (0: off)")
    parser.add_argument("--travel-conf", type=float, default=0.5, help="unnamed_travel_conf to try")
    args = parser.parse_args(argv)
    rules = GateRules(args.min_conf, args.min_travel, args.travel_conf)
    if not rules.active:
        parser.error("set --min-conf and/or --min-travel: with neither, the gate suppresses nothing")
    db = args.data_dir / "monitor.db"
    if not db.exists():
        parser.error(f"no visit log at {db}")
    since = iso(parse_when(args.since, time.time(), datetime.now().astimezone().tzinfo))
    result = backtest(db, rules, since, args.camera)
    head = (f"{'camera':12} {'unnamed':>7} {'hidden':>7} {'%':>4} {'low_conf':>8} {'still':>6}"
            f" | {'named':>5} {'hit':>4} | {'lbl_false':>9} {'lbl_real':>8}")
    print(f"rules: min_conf={rules.min_conf} min_travel={rules.min_travel} travel_conf={rules.travel_conf}"
          f"  since {args.since}")
    print(head)
    print("-" * len(head))
    for cam, c in sorted(result.items()):
        pct = round(100 * c["suppressed"] / c["unnamed"]) if c["unnamed"] else 0
        print(f"{cam:12} {c['unnamed']:7} {c['suppressed']:7} {pct:4} {c['low_conf']:8} {c['stationary']:6}"
              f" | {c['named']:5} {c['named_hit']:4} | {c['labelled_false']:9} {c['labelled_real']:8}")
    print("\nhit: named visits the same rules would have hidden (stand-in for real animals lost).\n"
          "lbl_real: hidden visits a human labelled real — each one is a miss; label more to trust the rules.")


if __name__ == "__main__":
    main()
