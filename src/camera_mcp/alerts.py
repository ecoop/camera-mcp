# Copyright (c) 2026 Eric Cooper.
"""Real-time alerts: a short SMS (through an email-to-SMS gateway) the first time a high-priority
species is seen, then quiet for a while per species."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class Alerter:
    """Sends a short SMS (via an email->SMS gateway) the first time a high-priority species is
    seen, then stays quiet for COOLDOWN per species so a lingering or returning animal does not
    spam. Inactive (a no-op) unless alert species, mail settings, and a recipient are all set.

    The alert fires when the sighting is *analysed*, which under a backlog can trail wall-clock;
    the message carries the actual sighting time. Give the cameras that matter for alerts a higher
    `priority` so their backlog is analysed before quieter cameras' — within one camera, segments
    are still analysed in time order, so a fresh sighting waits behind that camera's own backlog."""

    COOLDOWN = 300.0  # seconds between alerts for the same species

    def __init__(self, species, settings, recipient, tz):
        self.species = set(species)
        self.settings = settings
        self.recipient = recipient
        self.tz = tz
        self._last: dict[str, float] = {}

    @property
    def active(self) -> bool:
        return bool(self.species and self.settings and self.recipient)

    def maybe_alert(self, camera: str, species: str, t: float) -> None:
        if not self.active or species not in self.species:
            return
        now = time.time()
        if now - self._last.get(species, 0.0) < self.COOLDOWN:
            return
        self._last[species] = now
        when = datetime.fromtimestamp(t, timezone.utc).astimezone(self.tz).strftime("%-I:%M%p").lower()
        body = f"{species} at {camera}, {when}"
        try:
            from camera_mcp.mailer import MailError, send_email
            try:
                send_email("", body, {**self.settings, "SMTP_TO": self.recipient})
                log.info("alert sent: %s", body)
            except MailError as e:
                log.warning("alert send failed (%s): %s", body, e)
        except Exception:  # never let an alert crash analysis
            log.exception("alert error for %s", body)
