# Copyright (c) 2026 Eric Cooper.
"""The error a bad configuration raises (its own module so zones and credentials can raise it
without importing config)."""

from __future__ import annotations


class ConfigError(ValueError):
    pass
