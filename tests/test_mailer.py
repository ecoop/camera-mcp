# Copyright (c) 2026 Eric Cooper.
"""Email delivery: settings resolution and sending, with a fake SMTP server (no network)."""

from __future__ import annotations

import pytest

from camera_mcp import mailer


@pytest.fixture
def creds_file(tmp_path, monkeypatch):
    f = tmp_path / "credentials"
    f.write_text("CAMERA_USER=cam\nCAMERA_PASS=p\nSMTP_HOST=smtp.example.com\nSMTP_FROM=camera@example.com\n"
                 "SMTP_TO=me@example.com\nSMTP_USER=camera@example.com\nSMTP_PASS=mailpass\n")
    monkeypatch.setenv("CAMERA_MCP_CREDENTIALS", str(f))
    for k in mailer.SETTINGS:
        monkeypatch.delenv(k, raising=False)
    return f


def test_email_settings_from_file_and_env_override(creds_file, monkeypatch):
    s = mailer.email_settings()
    assert s["SMTP_HOST"] == "smtp.example.com" and s["SMTP_TO"] == "me@example.com"
    monkeypatch.setenv("SMTP_TO", "other@example.com")  # env wins
    assert mailer.email_settings()["SMTP_TO"] == "other@example.com"


def test_email_settings_none_when_incomplete(tmp_path, monkeypatch):
    f = tmp_path / "c"; f.write_text("SMTP_HOST=smtp.example.com\n")  # no FROM/TO
    monkeypatch.setenv("CAMERA_MCP_CREDENTIALS", str(f))
    for k in mailer.SETTINGS:
        monkeypatch.delenv(k, raising=False)
    assert mailer.email_settings() is None


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port
        self.started_tls = False
        self.login_args = None
        self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg, to_addrs=None):
        self.sent = (msg, to_addrs)


def test_send_email_starttls_login_and_inline_image(creds_file, tmp_path, monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    img = tmp_path / "m.jpg"; img.write_bytes(b"\xff\xd8 jpeg bytes")
    mailer.send_email("Subject line", "line1\nline2", mailer.email_settings(), img)
    (srv,) = FakeSMTP.instances
    assert srv.host == "smtp.example.com" and srv.port == 587 and srv.started_tls
    assert srv.login_args == ("camera@example.com", "mailpass")
    msg, to = srv.sent
    assert msg["From"] == "camera@example.com" and to == ["me@example.com"]
    assert msg["Subject"] == "Subject line"
    assert any(part.get_content_type() == "image/jpeg" for part in msg.walk())  # montage attached inline


def test_send_email_uses_ssl_on_465(creds_file, monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", FakeSMTP)
    monkeypatch.setenv("SMTP_PORT", "465")
    mailer.send_email("s", "body", mailer.email_settings())
    (srv,) = FakeSMTP.instances
    assert srv.port == 465 and not srv.started_tls  # SSL from the start, no STARTTLS


def test_send_email_wraps_failures(creds_file, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(mailer.smtplib, "SMTP", boom)
    with pytest.raises(mailer.MailError, match="connection refused"):
        mailer.send_email("s", "b", mailer.email_settings())


def test_tls_context_uses_a_ca_bundle_and_honours_insecure():
    import ssl
    from camera_mcp.mailer import _tls_context
    secure = _tls_context({})
    assert secure.verify_mode == ssl.CERT_VERIFY if hasattr(ssl, "CERT_VERIFY") else secure.verify_mode == ssl.CERT_REQUIRED
    assert secure.check_hostname is True
    insecure = _tls_context({"SMTP_INSECURE": "1"})
    assert insecure.verify_mode == ssl.CERT_NONE and insecure.check_hostname is False
