"""The backend must be usable on a machine with no audio stack.

A Home Assistant deployment runs Yarnotron in a container (or on a server) with
no microphone, speaker or PortAudio. `voice.tts` used to import numpy and
sounddevice at module scope, so the parent dashboard's voice picker
(`GET /v1/preferences/voices`) returned a 500 there.
"""

from __future__ import annotations

import importlib
import sys

import httpx
import pytest
from fastapi import FastAPI

from yarnotron.api.routes import preferences


@pytest.fixture
def no_audio_stack(monkeypatch):
    """Make `import numpy` and `import sounddevice` fail, as on a bare server."""
    for name in ("numpy", "sounddevice", "kokoro_onnx"):
        monkeypatch.setitem(sys.modules, name, None)  # None -> ImportError on import
    monkeypatch.delitem(sys.modules, "yarnotron.voice.tts", raising=False)
    yield
    # Do not leave a tts module bound to the broken imports for later tests.
    sys.modules.pop("yarnotron.voice.tts", None)


def test_tts_module_imports_without_audio_libraries(no_audio_stack):
    tts = importlib.import_module("yarnotron.voice.tts")

    assert tts.kokoro_is_available is not None
    assert tts.create_speaker("none").__class__.__name__ == "NullSpeaker"


async def test_voices_endpoint_survives_without_audio_libraries(no_audio_stack):
    app = FastAPI()
    app.include_router(preferences.router, prefix="/v1")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/preferences/voices")

    assert response.status_code == 200
    voices = response.json()
    assert voices[0]["provider"] == "auto"  # the "System default" entry is always there


async def test_voice_sample_reports_a_clean_error_without_audio_libraries(no_audio_stack):
    app = FastAPI()
    app.include_router(preferences.router, prefix="/v1")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/preferences/voices/sample", json={"tts_provider": "kokoro"})

    # A parent clicking "play sample" on a headless server gets a message, not a crash.
    assert response.status_code == 400
