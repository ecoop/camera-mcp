# Copyright (c) 2026 Eric Cooper.
"""Send the brief by email, with the montage shown inline in the message body.

Settings come from the environment or the credentials file (read_credentials_file):

    SMTP_HOST   mail server, e.g. smtp.example.com      (required)
    SMTP_PORT   587 (STARTTLS, default) or 465 (SSL)
    SMTP_FROM   sender, e.g. camera@example.com          (required)
    SMTP_TO     recipient; comma-separated for several   (required)
    SMTP_USER   login user (default: SMTP_FROM)
    SMTP_PASS   login password (omit for an unauthenticated relay)

For real-time alerts, SMS_ALERT_TO is where they go: an email address, or an email->SMS gateway
(e.g. 15551234567@vtext.com; AT&T's txt.att.net no longer delivers). WEB_URL, if set, is the web
dashboard's address as seen from the reader's phone (e.g. http://studio.local:8080), so an alert can
link to its visit.

No address or password is ever hard-coded or committed; the real values live only in the
host's git-ignored credentials file.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from html import escape
from pathlib import Path

from camera_mcp.credentials import read_credentials_file

SETTINGS = ("SMTP_HOST", "SMTP_PORT", "SMTP_FROM", "SMTP_TO", "SMTP_USER", "SMTP_PASS",
            "SMTP_CA_FILE", "SMTP_INSECURE")


class MailError(RuntimeError):
    pass


def email_settings() -> dict[str, str] | None:
    """Mail settings from the credentials file, overridden by the environment. None if the
    required ones (host, from, to) aren't all set."""
    merged = {**read_credentials_file(), **{k: os.environ[k] for k in SETTINGS if os.environ.get(k)}}
    settings = {k: merged[k] for k in SETTINGS if merged.get(k)}
    if not all(settings.get(k) for k in ("SMTP_HOST", "SMTP_FROM", "SMTP_TO")):
        return None
    return settings


def alert_recipient() -> str | None:
    """The address real-time alerts go to (SMS_ALERT_TO) — an email->SMS gateway like
    15551234567@txt.att.net — from the credentials file or the environment. None if unset."""
    merged = {**read_credentials_file(), **os.environ}
    return merged.get("SMS_ALERT_TO") or None


def web_url() -> str | None:
    """The web dashboard's address for links in alerts (WEB_URL), from the credentials file or the
    environment, without a trailing slash. None if unset: alerts then carry no link."""
    merged = {**read_credentials_file(), **os.environ}
    return (merged.get("WEB_URL") or "").rstrip("/") or None


def _tls_context(settings: dict[str, str]) -> ssl.SSLContext:
    """A TLS context that actually verifies the mail server. Python's default context can miss a
    valid chain (its CA paths differ from the system's); certifi's current bundle verifies them.
    SMTP_CA_FILE overrides the bundle; SMTP_INSECURE=1 disables verification (for a mail server
    with a private/self-signed cert you trust)."""
    if str(settings.get("SMTP_INSECURE", "")).lower() in ("1", "true", "yes"):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    ca_file = settings.get("SMTP_CA_FILE")
    if not ca_file:
        try:
            import certifi
            ca_file = certifi.where()
        except ImportError:
            ca_file = None
    return ssl.create_default_context(cafile=ca_file)


def _message(subject: str, body: str, settings: dict[str, str], image: Path | bytes | None,
             link: str | None = None, attachments: tuple[tuple[str, bytes], ...] = ()) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["SMTP_FROM"]
    msg["To"] = settings["SMTP_TO"]
    msg.set_content(body + (f"\n\n{link}" if link else ""))  # plain-text part, for clients without HTML
    data = image.read_bytes() if isinstance(image, Path) and image.is_file() else image if isinstance(image, bytes) else None
    if data is not None or link:
        html = (f"<div style='font:14px/1.4 -apple-system,system-ui,sans-serif'>"
                f"<pre style='font:13px/1.4 ui-monospace,monospace;white-space:pre-wrap'>{escape(body)}</pre>"
                + (f"<p><a href='{escape(link, quote=True)}'>Open the visit</a></p>" if link else "")
                + ("<img src='cid:montage' style='max-width:100%;border-radius:6px'>" if data is not None else "")
                + "</div>")
        msg.add_alternative(html, subtype="html")
        if data is not None:
            msg.get_payload()[1].add_related(data, maintype="image", subtype="jpeg", cid="<montage>")
    for name, blob in attachments:  # e.g. the full-resolution frame, to save or zoom
        msg.add_attachment(blob, maintype="image", subtype="jpeg", filename=name)
    return msg


def send_email(subject: str, body: str, settings: dict[str, str], image: Path | bytes | None = None, *,
               link: str | None = None, attachments: tuple[tuple[str, bytes], ...] = ()) -> None:
    """Send one message: `image` shown inline, `link` as a button under the text, `attachments`
    as (filename, JPEG bytes). Raises MailError on any SMTP or network failure."""
    host = settings["SMTP_HOST"]
    port = int(settings.get("SMTP_PORT") or 587)
    recipients = [a.strip() for a in settings["SMTP_TO"].split(",") if a.strip()]
    msg = _message(subject, body, settings, image, link, attachments)
    context = _tls_context(settings)
    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, context=context, timeout=30)
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        with server:
            if port != 465:
                server.starttls(context=context)
            if settings.get("SMTP_USER") and settings.get("SMTP_PASS"):
                server.login(settings["SMTP_USER"], settings["SMTP_PASS"])
            server.send_message(msg, to_addrs=recipients)
    except (smtplib.SMTPException, OSError, ssl.SSLError) as e:
        raise MailError(f"sending mail via {host}:{port} failed: {e}") from None
