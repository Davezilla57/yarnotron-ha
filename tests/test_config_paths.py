"""Where Yarnotron keeps its data, per platform."""

from __future__ import annotations

from pathlib import Path

import pytest

from yarnotron.core import config
from yarnotron.core.config import application_support_dir


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("YARNOTRON_DATA_DIR", "XDG_DATA_HOME", "APPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))


def test_explicit_override_wins_everywhere(monkeypatch):
    monkeypatch.setenv("YARNOTRON_DATA_DIR", "/data")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")
    monkeypatch.setattr(config.sys, "platform", "linux")

    assert application_support_dir() == Path("/data")


def test_xdg_data_home_is_honoured(monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")
    monkeypatch.setattr(config.sys, "platform", "linux")

    assert application_support_dir() == Path("/xdg/Yarnotron")


def test_macos_keeps_the_library_default(monkeypatch):
    # Existing installs must not move: this is where a Mac's stories already are.
    monkeypatch.setattr(config.sys, "platform", "darwin")

    assert application_support_dir() == Path(
        "/home/someone/Library/Application Support/Yarnotron"
    )


def test_linux_defaults_to_the_xdg_location_not_library(monkeypatch):
    # A container or Linux server has no ~/Library; writing there was a bug.
    monkeypatch.setattr(config.sys, "platform", "linux")

    assert application_support_dir() == Path("/home/someone/.local/share/Yarnotron")


def test_windows_uses_appdata(monkeypatch):
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "/appdata")

    assert application_support_dir() == Path("/appdata/Yarnotron")
