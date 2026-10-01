"""Starting the server on a LAN address without a token should not be silent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_yarnotron.py"


@pytest.fixture(scope="module")
def start_script():
    spec = importlib.util.spec_from_file_location("start_yarnotron", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_never_warns(start_script, host, tmp_path):
    assert start_script.exposure_warning(host, env={}, dotenv=tmp_path / ".env") is None


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_a_network_address_without_a_token_warns(start_script, host, tmp_path):
    message = start_script.exposure_warning(host, env={}, dotenv=tmp_path / ".env")

    assert message is not None
    assert "YARNOTRON_API_TOKEN" in message


def test_a_token_in_the_environment_silences_it(start_script, tmp_path):
    env = {"YARNOTRON_API_TOKEN": "abc"}

    assert start_script.exposure_warning("0.0.0.0", env=env, dotenv=tmp_path / ".env") is None


def test_a_token_in_dotenv_silences_it(start_script, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("YARNOTRON_DEFAULT_MODEL=x\nYARNOTRON_API_TOKEN=abc\n")

    assert start_script.exposure_warning("0.0.0.0", env={}, dotenv=dotenv) is None


@pytest.mark.parametrize("line", ["YARNOTRON_API_TOKEN=", "# YARNOTRON_API_TOKEN=abc", "YARNOTRON_API_TOKEN=  "])
def test_a_blank_or_commented_token_does_not_count(start_script, tmp_path, line):
    dotenv = tmp_path / ".env"
    dotenv.write_text(line + "\n")

    assert start_script.exposure_warning("0.0.0.0", env={}, dotenv=dotenv) is not None


def test_quoted_dotenv_values_count(start_script, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text('YARNOTRON_API_TOKEN="abc"\n')

    assert start_script.exposure_warning("0.0.0.0", env={}, dotenv=dotenv) is None
