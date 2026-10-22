# Copyright (c) 2026 Eric Cooper.
"""The cameras' login and other secrets: $CAMERA_USER / $CAMERA_PASS, else the macOS Keychain item
"camera-mcp", else the git-ignored credentials file a headless host uses; `${VAR}` expansion in
sources; and redacting logins out of anything that might be shown or logged."""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path

from camera_mcp.errors import ConfigError


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


_USERINFO = re.compile(r"(?<=://)[^/@\s]*@")


USER_ENV, PASS_ENV = "CAMERA_USER", "CAMERA_PASS"


KEYCHAIN_SERVICE = "camera-mcp"


CREDENTIALS_ENV = "CAMERA_MCP_CREDENTIALS"  # path override for the credentials file


DEFAULT_CREDENTIALS = "~/.config/camera-mcp/credentials"  # CAMERA_USER=.. / CAMERA_PASS=.. lines, chmod 600


NO_LOGIN = (f"no camera login: set {USER_ENV} and {PASS_ENV}, store it in the macOS Keychain with "
            f"`security add-generic-password -U -s {KEYCHAIN_SERVICE} -a <camera user> -w`, or put "
            f"{USER_ENV}=.. and {PASS_ENV}=.. lines in {DEFAULT_CREDENTIALS} (chmod 600). The Keychain is "
            "unreadable in an SSH session, so a headless/remote host uses the file.")


def redact(text: str) -> str:
    """Strip user:password@ from any URL in `text` (OpenCV errors can echo the URL)."""
    return _USERINFO.sub("***@", text)


def _expand(value: str, where: str) -> str:
    def sub(m: re.Match[str]) -> str:
        var = m.group(1)
        if var not in os.environ:
            raise ConfigError(f"{where}: environment variable {var} is not set")
        return os.environ[var]

    return os.path.expanduser(_VAR.sub(sub, value))


@functools.cache
def _keychain_login() -> tuple[str, str] | None:
    """The login stored in the macOS Keychain as a generic password, service "camera-mcp",
    account = the camera user. None if there's no Keychain (not macOS) or no such item."""
    security = shutil.which("security")
    if security is None:
        return None
    try:
        item = subprocess.run([security, "find-generic-password", "-s", KEYCHAIN_SERVICE],
                              capture_output=True, text=True, timeout=10)
        secret = subprocess.run([security, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    account = re.search(r'"acct"<blob>="([^"]*)"', item.stdout)
    if item.returncode or secret.returncode or not account:
        return None
    return account.group(1), secret.stdout.rstrip("\n")


def read_credentials_file() -> dict[str, str]:
    """`KEY=VALUE` lines from the credentials file ($CAMERA_MCP_CREDENTIALS, else
    ~/.config/camera-mcp/credentials), the git-ignored secrets store on a host. Empty if absent.
    Holds the camera login (CAMERA_USER/PASS) and, optionally, the mail settings (SMTP_*)."""
    for candidate in (os.environ.get(CREDENTIALS_ENV), DEFAULT_CREDENTIALS, ".env"):
        if not candidate:
            continue
        try:
            text = Path(candidate).expanduser().read_text()
        except OSError:
            continue
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip('"').strip("'")
        if out:
            return out
    return {}


def _file_login() -> tuple[str, str] | None:
    """The camera login from the credentials file, for headless/remote hosts where the login
    Keychain is locked in an SSH session. None if absent or incomplete."""
    creds = read_credentials_file()
    user, password = creds.get(USER_ENV), creds.get(PASS_ENV)
    return (user, password) if user and password else None


def camera_login() -> tuple[str, str] | None:
    """(user, password) for the cameras: the environment, then the Keychain (desktop session),
    then a credentials file (headless/SSH, where the Keychain is locked)."""
    if os.environ.get(USER_ENV) and os.environ.get(PASS_ENV):
        return os.environ[USER_ENV], os.environ[PASS_ENV]
    return _keychain_login() or _file_login()
