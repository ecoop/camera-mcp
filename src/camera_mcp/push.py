# Copyright (c) 2026 Eric Cooper.
"""Push notifications for real-time alerts, through ntfy (https://ntfy.sh, or a server you run): a
phone with the ntfy app subscribed to the topic gets the alert in seconds, with the close-up and a
tap that opens the visit. No account or carrier registration is needed.

Settings come from the credentials file, overridden by the environment:

    NTFY_TOPIC     the topic to publish to                              (required)
    NTFY_SERVER    https://ntfy.sh (default), or your own server
    NTFY_TOKEN     an access token, for a server or topic that needs one
    NTFY_PRIORITY  1-5 or min/low/default/high/max (default high; max can break through Do Not Disturb)

On a public server anyone who knows the topic name can read it, so pick a long, unguessable one and
keep it, like the other secrets, only in the host's git-ignored credentials file.
"""

from __future__ import annotations

import base64
import os
import ssl
import urllib.error
import urllib.request
from urllib.parse import quote

from camera_mcp.credentials import read_credentials_file

SETTINGS = ("NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN", "NTFY_PRIORITY")
DEFAULT_SERVER = "https://ntfy.sh"
TIMEOUT = 15.0


class PushError(RuntimeError):
    pass


def push_settings() -> dict[str, str] | None:
    """ntfy settings from the credentials file, overridden by the environment. None without a topic."""
    merged = {**read_credentials_file(), **{k: os.environ[k] for k in SETTINGS if os.environ.get(k)}}
    settings = {k: merged[k].strip() for k in SETTINGS if merged.get(k, "").strip()}
    return settings if settings.get("NTFY_TOPIC") else None


def _header(text: str) -> str:
    """A header value ntfy reads back as text: as is if it's ASCII, else an RFC 2047 encoded word."""
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return f"=?UTF-8?B?{base64.b64encode(text.encode()).decode()}?="


def push_request(title: str, message: str, settings: dict[str, str], image: bytes | None = None,
                 filename: str = "alert.jpg", link: str | None = None) -> urllib.request.Request:
    """The ntfy publish request: with an image, the image is the body (an attachment the app shows)
    and the message rides in a header; without one, the message is the body."""
    url = f"{settings.get('NTFY_SERVER', DEFAULT_SERVER).rstrip('/')}/{quote(settings['NTFY_TOPIC'], safe='')}"
    headers = {"Title": _header(title), "Priority": settings.get("NTFY_PRIORITY", "high"), "Tags": "camera"}
    if link:
        headers["Click"] = link
    if settings.get("NTFY_TOKEN"):
        headers["Authorization"] = f"Bearer {settings['NTFY_TOKEN']}"
    if image:
        headers |= {"Message": _header(message), "Filename": filename}
        return urllib.request.Request(url, data=image, headers=headers, method="PUT")
    return urllib.request.Request(url, data=message.encode(), headers=headers, method="POST")


def send_push(title: str, message: str, settings: dict[str, str], image: bytes | None = None,
              filename: str = "alert.jpg", link: str | None = None) -> None:
    """Publish one notification; raises PushError if the server can't be reached or refuses it."""
    req = push_request(title, message, settings, image, filename, link)
    try:
        import certifi  # a current CA bundle (Python's default paths can miss a valid chain)
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=ctx) as r:
            r.read()
    except urllib.error.HTTPError as e:
        raise PushError(f"ntfy refused the alert: HTTP {e.code} {e.read()[:200].decode(errors='replace')}") from None
    except (urllib.error.URLError, OSError) as e:
        raise PushError(f"couldn't reach ntfy: {e}") from None
