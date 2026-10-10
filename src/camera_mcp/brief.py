# Copyright (c) 2026 Eric Cooper.
"""A short wildlife brief from the monitor's visit log, for a scheduled morning message.

Plain text, sized for a phone notification: what came by in the window, with the two things
worth knowing first — was the dog yard clear of night-critters, and were cats at the feeders —
then the overall counts, and any gaps (a camera down, motion the detector couldn't keep up with).

    python -m camera_mcp.brief [--since 14h] [--config cameras.yaml] [--data-dir data]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from camera_mcp.config import ConfigError, default_config_path, default_data_dir, load_cameras, load_site
from camera_mcp.history import History, HistoryError


def _count(visits: list[dict]) -> str:
    return "no visits" if not visits else f"{len(visits)} visit{'s' if len(visits) != 1 else ''}"


def _times(visits: list[dict], n: int = 3) -> str:
    """A few local clock times from visits (newest first)."""
    stamps = [v["started"][11:16] for v in visits[:n]]
    more = "" if len(visits) <= n else f" +{len(visits) - n} more"
    return ", ".join(stamps) + more


GENERIC_GROUPS = {"animal", "mammal", "rodent", "bird", "person", "vehicle"}  # not a species worth naming


def build_subject(history: History, since: str = "14h") -> str:
    """An email subject that leads with the content: the identified species seen overnight."""
    from datetime import datetime

    summary = history.activity_summary(since=since, group_by="group")
    try:
        day = datetime.fromisoformat(summary["to"]).strftime("%b %-d")
    except (ValueError, TypeError):
        day = summary["to"][:10]
    named = [r["key"] for r in summary["rows"] if r["key"] not in GENERIC_GROUPS]
    tail = f" ({', '.join(named[:4])})" if named else ""
    return f"Overnight visitors — {day}{tail}"


def build_brief(history: History, since: str = "14h") -> str:
    lines: list[str] = []
    summary = history.activity_summary(since=since, group_by="group")
    window = f"{summary['from'][5:16].replace('T', ' ')} → {summary['to'][11:16]}"
    lines.append(f"\U0001f426 Wildlife brief — last night ({window})")

    def by_tag(tag: str, taxon: str) -> list[dict]:
        """find_visits for a tag, or [] if no camera has that tag (not every site has every role)."""
        try:
            return history.find_visits(since=since, tag=tag, taxon=taxon, limit=50)["visits"]
        except HistoryError:
            return []

    # The two questions worth leading with.
    if any("dog_yard" in c.tags for c in history.cameras.values()):
        critters = by_tag("dog_yard", "night-critters")
        if critters:
            where = sorted({v["camera"] for v in critters})
            lines.append(f"⚠ Backyard: {_count(critters)} (raccoon/skunk/opossum) on "
                         f"{', '.join(where)} — {_times(critters)}")
        else:
            lines.append("✓ Backyard: clear of raccoons/skunks/opossums overnight")

    cats = by_tag("feeder", "cat")
    if cats:
        lines.append(f"\U0001f408 Cats at feeders: {_count(cats)} — {_times(cats)}")

    # Overall counts by group.
    rows = [r for r in summary["rows"] if r["key"] not in ("person",)]
    if rows:
        tally = ", ".join(f"{r['key']} {r['visits']}" for r in rows[:8])
        lines.append(f"Overnight: {summary['total_visits']} visits — {tally}")
    else:
        lines.append("Overnight: nothing detected")

    # Gaps worth flagging: a camera down or barely watched, a big unanalysed backlog, or a
    # camera that looks moved. (A uniformly < 100% watched just means the monitor started
    # mid-window; that's not an ongoing problem, so it isn't flagged.) Streams only.
    streams = {c.name for c in history.cameras.values() if c.is_stream}
    gaps = []
    for c in summary["coverage"]:
        if c["camera"] not in streams:
            continue
        watched, unseen = c.get("watched"), c.get("unanalysed_motion_seconds") or 0
        if watched is not None and watched < 0.5:
            gaps.append(f"{c['camera']} down/partial ({watched * 100:.0f}%)")
        elif unseen > 600:
            gaps.append(f"{c['camera']} {unseen // 60}m motion unanalysed")
        if c.get("looks_moved_since"):
            gaps.append(f"{c['camera']} looks moved")
    if gaps:
        lines.append("⚠ Note: " + "; ".join(sorted(set(gaps))))
    return "\n".join(lines)


NOTABLE_TAXA = ("night-critters", "cat", "rat", "fox", "coyote")  # worth a picture in the brief


def _notable(history: History, since: str, data_dir, limit: int = 6):
    """(jpeg bytes, caption) for the night's notable, identified visits (newest first)."""
    picked: dict[int, dict] = {}
    for taxon in NOTABLE_TAXA:
        for v in history.find_visits(since=since, taxon=taxon, limit=10)["visits"]:
            if v["has_thumbnail"]:
                picked.setdefault(v["id"], v)
    out = []
    for v in sorted(picked.values(), key=lambda v: v["started"], reverse=True)[:limit]:
        try:
            data, _ = history.visit_thumbnail(v["id"], data_dir)
            out.append((data, f"{v['label']} {v['camera']} {v['started'][11:16]}"))
        except HistoryError:
            continue
    return out


def write_montage(history: History, since: str, data_dir, out_path, limit: int = 6) -> bool:
    """A labelled grid of the notable visitors' thumbnails. False if there were none."""
    import cv2
    import numpy as np

    items = _notable(history, since, data_dir, limit)
    if not items:
        return False
    tiles = []
    for data, caption in items:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img if img is not None else np.zeros((180, 320, 3), np.uint8), (320, 180))
        cv2.rectangle(img, (0, 158), (320, 180), (0, 0, 0), -1)
        cv2.putText(img, caption[:40], (4, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    cols = 3 if len(tiles) > 1 else 1
    while len(tiles) % cols:
        tiles.append(np.zeros((180, 320, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    # imencode (not imwrite): robust regardless of what else has patched cv2.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m camera_mcp.brief", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="14h", help="how far back to look (default 14h)")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--data-dir", default=None, help="monitor data dir (default: $CAMERA_MCP_DATA, else data/)")
    parser.add_argument("--montage", default=None, help="also write a labelled image of notable visitors here")
    parser.add_argument("--email", action="store_true", help="send the brief by email (SMTP_* settings) instead "
                        "of printing; also writes the dated file and montage under the data dir")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    try:
        cameras = load_cameras(args.config, require_login=False)
        site = load_site(args.config)
        history = History(data_dir / "monitor.db", cameras, site)
        text = build_brief(history, args.since)
        if not args.email:
            print(text)
            if args.montage:
                write_montage(history, args.since, data_dir, Path(args.montage))
            return
        # --email: write the dated file + montage, then send.
        from datetime import datetime

        from camera_mcp.mailer import MailError, email_settings, send_email

        settings = email_settings()
        if not settings:
            print("brief: no mail settings (set SMTP_HOST, SMTP_FROM, SMTP_TO)", file=sys.stderr)
            raise SystemExit(1)
        stamp = datetime.now(site.tz()).strftime("%Y-%m-%d")
        briefs = data_dir / "briefs"
        briefs.mkdir(parents=True, exist_ok=True)
        (briefs / f"{stamp}.txt").write_text(text)
        montage = briefs / f"{stamp}.jpg"
        image = montage if write_montage(history, args.since, data_dir, montage) else None
        try:
            send_email(build_subject(history, args.since), text, settings, image)
        except MailError as e:
            print(f"brief: {e}", file=sys.stderr)
            raise SystemExit(1) from None
    except (ConfigError, HistoryError) as e:
        print(f"brief: {e}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
