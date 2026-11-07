# Copyright (c) 2026 Eric Cooper.
"""Real-time alerts: a short email (or SMS, through an email-to-SMS gateway) the first time a
high-priority species is seen, or anything turns up in a camera's alert area, then quiet for a while.
An email carries a close-up of what set it off, the full-resolution frame, and a link to the visit."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# Called only when an alert is actually sent (encoding a full frame isn't free): returns
# (close-up JPEG, full-resolution JPEG), either of which may be None.
Photo = Callable[[], tuple["bytes | None", "bytes | None"]]


class Alerter:
    """Sends a short alert (an email, or an SMS via an email->SMS gateway) the first time a
    high-priority species is seen, then stays quiet for COOLDOWN per species so a lingering or
    returning animal does not spam. Alert areas (zones.AlertArea) work the same way, quiet for
    COOLDOWN per camera and area, whatever the species. Inactive (a no-op) unless mail settings and a
    recipient are set; then it alerts for the alert species and any alert areas, whichever are
    configured. The message is its own subject; given a photo it shows the close-up inline and
    attaches the full frame, and given the dashboard's address (web_url) and the visit, it links to it.

    The alert fires when the sighting is *analysed*, which under a backlog can trail wall-clock;
    the message carries the actual sighting time. Give the cameras that matter for alerts a higher
    `priority` so their backlog is analysed before quieter cameras' — within one camera, segments
    are still analysed in time order, so a fresh sighting waits behind that camera's own backlog."""

    COOLDOWN = 300.0  # seconds between alerts for the same species, or the same alert area

    def __init__(self, species, settings, recipient, tz, web_url: str | None = None):
        self.species = set(species)
        self.settings = settings
        self.recipient = recipient
        self.tz = tz
        self.web_url = web_url  # the dashboard's address, for a link to the visit (None: no link)
        self._last: dict = {}  # species, or ("area", camera, area name) -> when it last alerted

    @property
    def active(self) -> bool:
        return bool(self.settings and self.recipient)

    def maybe_alert(self, camera: str, species: str, t: float, photo: Photo | None = None,
                    visit: int | None = None) -> None:
        if species in self.species:
            self._send(species, f"{species} at {camera}, {self._when(t)}", camera, t, photo, visit)

    def maybe_area_alert(self, camera: str, area: str, label: str, t: float, photo: Photo | None = None,
                         visit: int | None = None) -> None:
        """Something labelled `label` in the camera's alert area `area` (the label may well be wrong;
        it's in the text so the reader can judge)."""
        self._send(("area", camera, area), f"{area}: {label} at {camera}, {self._when(t)}", camera, t, photo, visit)

    def _local(self, t: float) -> datetime:
        return datetime.fromtimestamp(t, timezone.utc).astimezone(self.tz)

    def _when(self, t: float) -> str:
        return self._local(t).strftime("%-I:%M%p").lower()

    def _send(self, key, body: str, camera: str, t: float, photo: Photo | None, visit: int | None) -> None:
        if not self.active:
            return
        now = time.time()
        if now - self._last.get(key, 0.0) < self.COOLDOWN:
            return
        self._last[key] = now
        try:
            from camera_mcp.mailer import MailError, send_email
            close = full = None
            if photo is not None:
                try:
                    close, full = photo()
                except Exception:  # an alert without a picture beats no alert
                    log.exception("alert photo failed for %s", body)
            link = f"{self.web_url}/?visit={visit}" if self.web_url and visit is not None else None  # opens it to tag
            attachments = ((f"{camera}-{self._local(t):%Y-%m-%d-%H%M%S}.jpg", full),) if full else ()
            text = body + (f" (visit {visit})" if visit is not None else "")
            try:
                send_email(body, text, {**self.settings, "SMTP_TO": self.recipient}, close,
                           link=link, attachments=attachments)
                log.info("alert sent: %s", body)
            except MailError as e:
                log.warning("alert send failed (%s): %s", body, e)
        except Exception:  # never let an alert crash analysis
            log.exception("alert error for %s", body)
