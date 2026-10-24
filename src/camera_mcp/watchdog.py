# Copyright (c) 2026 Eric Cooper.
"""Health checks that run beside the monitor (from launchd), not inside it, so they still report
when the monitor itself has stopped.

- offline (every few minutes): a short alert when a camera has sent no video for OFFLINE_AFTER
  inside its watch hours, and one more when its video is back. Paused hours don't count, nor the
  first OFFLINE_AFTER after a watch window opens. Each outage alerts once: what's been sent is kept
  in data/watchdog.json. Alerts go to SMS_ALERT_TO (an email->SMS gateway) if set, else SMTP_TO.
- midday (daily): an email on whether the detector kept up over the last 4 hours.

    camera-watchdog offline [--dry-run] | midday   [--config cameras.yaml] [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

from camera_mcp.config import ConfigError, default_config_path, default_data_dir, load_cameras, load_site
from camera_mcp.history import History
from camera_mcp.times import epoch

OFFLINE_AFTER = 20 * 60  # seconds without video, inside watch hours, before a camera counts as offline
STATE = "watchdog.json"


def read_status(db_path: Path) -> dict[str, dict]:
    """Each camera's row from the monitor's camera_status, with last_frame as epoch seconds."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT camera, state, last_frame_at, error FROM camera_status").fetchall()
    return {r["camera"]: {"state": r["state"], "error": r["error"],
                          "last_frame": epoch(r["last_frame_at"]) if r["last_frame_at"] else None} for r in rows}


def offline_changes(cameras: dict, status: dict[str, dict], now: float, tz,
                    alerted: dict[str, str]) -> tuple[dict[str, str], list[str], set[str]]:
    """Which cameras just went offline (name -> why), which alerted ones have video again, and which
    should have been watched all along (inside their watch hours for the whole OFFLINE_AFTER)."""
    local = lambda t: datetime.fromtimestamp(t, tz)  # noqa: E731
    down, back, due = {}, [], set()
    for name, cam in cameras.items():
        if not cam.enabled:
            continue
        st = status.get(name) or {"state": "never seen by the monitor", "error": None, "last_frame": None}
        last = st["last_frame"]
        silent = last is None or now - last > OFFLINE_AFTER
        if cam.is_active(local(now)) and cam.is_active(local(now - OFFLINE_AFTER)):
            due.add(name)
        if silent and name in due and name not in alerted:
            since = f"since {local(last):%-I:%M%p}".lower() if last else "yet"
            why = f"{st['state']}: {st['error']}" if st.get("error") else st["state"]
            down[name] = f"No video from {name} {since} ({why})"
        elif name in alerted and not silent:
            back.append(name)
    return down, back, due


def offline_message(down: dict[str, str], back: list[str], due: set[str], alerted: dict[str, str]) -> str:
    lines = []
    if down and len(due) > 1 and due <= set(down) | set(alerted):  # every camera that should be watched
        lines.append(f"No video from any camera for {OFFLINE_AFTER // 60}+ min: is the monitor running?")
    else:
        lines += down.values()
    lines += [f"{n} is back" for n in back]
    return "\n".join(lines)


def check_offline(config: Path, data_dir: Path, dry_run: bool = False, now: float | None = None) -> str:
    """Alert on cameras that just went offline or came back; returns what was (or would be) sent."""
    from camera_mcp.mailer import alert_recipient, email_settings, send_email

    now = time.time() if now is None else now
    cameras = load_cameras(str(config), require_login=False)
    tz = load_site(str(config)).tz()
    state_path = data_dir / STATE
    try:
        alerted = json.loads(state_path.read_text()).get("offline", {})
    except (OSError, ValueError):
        alerted = {}
    down, back, due = offline_changes(cameras, read_status(data_dir / "monitor.db"), now, tz, alerted)
    if not down and not back:
        return ""
    body = offline_message(down, back, due, alerted)
    if dry_run:
        return body
    settings, sms = email_settings(), alert_recipient()
    if not settings:
        raise ConfigError("no mail settings (SMTP_HOST, SMTP_FROM, SMTP_TO) to send alerts with")
    send_email("" if sms else "Cameras: " + ("offline" if down else "back"), body,
               {**settings, "SMTP_TO": sms} if sms else settings)  # only marked sent once it went
    alerted = {n: t for n, t in alerted.items() if n not in back} | {n: datetime.fromtimestamp(now, tz).isoformat()
                                                                    for n in down}
    state_path.write_text(json.dumps({"offline": alerted}, indent=1))
    return body


def midday_report(history: History) -> tuple[str, str]:
    """The midday email: subject and body, on how the detector kept up over the last 4 hours."""
    hc = history.analysis_health(since="4h")
    if not hc.get("samples"):
        return "Yard Sentry: midday health — no data", "No analysis stats in the window."
    drops, lag = hc["segments_dropped"], hc["peak_lag_seconds"]
    verdict = "keeping up" if (drops == 0 and lag < 120) else ("behind, not dropping" if drops == 0 else "DROPPING")
    body = (f"Midday analysis health: {verdict}.\n\n"
            f"  frames analysed: {hc['frames_analyzed']:,}\n"
            f"  dropped: {drops}\n"
            f"  peak lag: {lag:.0f}s   peak queue depth: {hc['peak_queue_depth']}\n"
            f"  mean per frame: {hc.get('mean_seconds_per_frame')}s\n"
            f"  window: {hc['from']} -> {hc['to']}\n")
    return "Yard Sentry: midday health — " + verdict, body


def main(argv: list[str] | None = None) -> None:
    from camera_mcp.mailer import MailError, email_settings, send_email

    parser = argparse.ArgumentParser(prog="camera-watchdog", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["offline", "midday"])
    parser.add_argument("--dry-run", action="store_true", help="offline: print what would be sent; send nothing")
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    try:
        if args.check == "offline":
            sent = check_offline(Path(args.config), data_dir, args.dry_run)
            if sent:
                print(f"{stamp} {'would send' if args.dry_run else 'sent'}: {sent!r}")
            return
        settings = email_settings()
        if not settings:
            raise ConfigError("no mail settings (SMTP_HOST, SMTP_FROM, SMTP_TO)")
        cameras = load_cameras(str(args.config), require_login=False)
        subject, body = midday_report(History(data_dir / "monitor.db", cameras, load_site(str(args.config))))
        send_email(subject, body, settings)
        print(f"{stamp} sent: {subject}")
    except (ConfigError, MailError, sqlite3.Error) as e:
        print(f"{stamp} camera-watchdog {args.check}: {e}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

