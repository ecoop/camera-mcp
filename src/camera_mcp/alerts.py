# Copyright (c) 2026 Eric Cooper.
"""Real-time alerts: a short SMS (through an email-to-SMS gateway) the first time a high-priority
species is seen, or anything turns up in a camera's alert area, then quiet for a while."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class Alerter:
    """Sends a short SMS (via an email->SMS gateway) the first time a high-priority species is
    seen, then stays quiet for COOLDOWN per species so a lingering or returning animal does not
    spam. Alert areas (zones.AlertArea) work the same way, quiet for COOLDOWN per camera and area,
    whatever the species. Inactive (a no-op) unless mail settings and a recipient are set; then it texts
    for the alert species and any alert areas, whichever are configured.

    The alert fires when the sighting is *analysed*, which under a backlog can trail wall-clock;
    the message carries the actual sighting time. Give the cameras that matter for alerts a higher
    `priority` so their backlog is analysed before quieter cameras' — within one camera, segments
    are still analysed in time order, so a fresh sighting waits behind that camera's own backlog."""

    COOLDOWN = 300.0  # seconds between alerts for the same species, or the same alert area

    def __init__(self, species, settings, recipient, tz):
        self.species = set(species)
        self.settings = settings
        self.recipient = recipient
        self.tz = tz
        self._last: dict = {}  # species, or ("area", camera, area name) -> when it last texted

    @property
    def active(self) -> bool:
        return bool(self.settings and self.recipient)

    def maybe_alert(self, camera: str, species: str, t: float) -> None:
        if species in self.species:
            self._send(species, f"{species} at {camera}, {self._when(t)}")

    def maybe_area_alert(self, camera: str, area: str, label: str, t: float) -> None:
        """Something labelled `label` in the camera's alert area `area` (the label may well be wrong;
        it's in the text so the reader can judge)."""
        self._send(("area", camera, area), f"{area}: {label} at {camera}, {self._when(t)}")

    def _when(self, t: float) -> str:
        return datetime.fromtimestamp(t, timezone.utc).astimezone(self.tz).strftime("%-I:%M%p").lower()

    def _send(self, key, body: str) -> None:
        if not self.active:
            return
        now = time.time()
        if now - self._last.get(key, 0.0) < self.COOLDOWN:
            return
        self._last[key] = now
        try:
            from camera_mcp.mailer import MailError, send_email
            try:
                send_email("", body, {**self.settings, "SMTP_TO": self.recipient})
                log.info("alert sent: %s", body)
            except MailError as e:
                log.warning("alert send failed (%s): %s", body, e)
        except Exception:  # never let an alert crash analysis
            log.exception("alert error for %s", body)
