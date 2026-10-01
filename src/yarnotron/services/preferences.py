from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import Depends

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.preferences import (
    ParentPreferences,
    UpdateParentPreferencesRequest,
    VoiceOption,
)


class ParentPreferencesStore:
    def __init__(self, settings: Settings) -> None:
        self._path = Path(settings.preferences_path)

    def get(self) -> ParentPreferences:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return ParentPreferences()
        return ParentPreferences.model_validate(data)

    def update(self, request: UpdateParentPreferencesRequest) -> ParentPreferences:
        preferences = ParentPreferences(
            tts_provider=request.tts_provider,
            tts_voice=request.tts_voice or None,
            space_friend_howl_enabled=request.space_friend_howl_enabled,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(preferences.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        return preferences


def list_voice_options() -> list[VoiceOption]:
    voices = [VoiceOption(id="", name="System default", provider="auto")]
    voices.extend(_kokoro_voice_options())
    if shutil.which("say") is None:
        return voices

    result = subprocess.run(["say", "-v", "?"], check=False, capture_output=True, text=True)
    voices.extend(_parse_macos_voices(result.stdout))
    return voices


# Kokoro voice ids encode language and gender: "af_heart" is American English,
# female. Expanded here so the picker reads as names rather than codes.
_KOKORO_PREFIXES = {
    "af": ("American English", "female"),
    "am": ("American English", "male"),
    "bf": ("British English", "female"),
    "bm": ("British English", "male"),
}


def _kokoro_voice_options() -> list[VoiceOption]:
    # Imported here, not at module scope: voice/tts pulls in numpy and
    # sounddevice, and the backend must start without the voice extra.
    from yarnotron.voice.tts import kokoro_is_available, kokoro_model_paths

    if not kokoro_is_available():
        return []
    try:
        from kokoro_onnx import Kokoro

        model, voices_path = kokoro_model_paths()
        names = sorted(Kokoro(str(model), str(voices_path)).get_voices())
    except Exception:
        # A missing or broken model must not take the dashboard down.
        return []

    options = []
    for name in names:
        prefix, _, label = name.partition("_")
        locale, gender = _KOKORO_PREFIXES.get(prefix, (None, None))
        pretty = label.replace("_", " ").title() or name
        options.append(
            VoiceOption(
                id=name,
                name=f"{pretty} ({gender})" if gender else pretty,
                provider="kokoro",
                locale=locale,
            )
        )
    return options


def play_voice_sample(
    *,
    provider: str,
    voice: str | None,
    text: str,
) -> None:
    from yarnotron.voice.tts import (
        DEFAULT_KOKORO_VOICE,
        KokoroSpeaker,
        kokoro_is_available,
        kokoro_model_paths,
    )

    resolved_provider = provider
    if provider == "auto":
        # Match what the voice loop would actually pick, so the sample the
        # parent hears is the voice the child will hear.
        resolved_provider = "kokoro" if kokoro_is_available() else "macos-say"

    if resolved_provider == "kokoro":
        if not kokoro_is_available():
            raise RuntimeError("Kokoro voice model is not installed.")
        speaker = KokoroSpeaker(
            *kokoro_model_paths(),
            voice=voice or DEFAULT_KOKORO_VOICE,
        )
        speaker.speak(text)
        return

    if resolved_provider != "macos-say":
        raise RuntimeError(f"Voice samples are not supported for provider: {provider}")
    if shutil.which("say") is None:
        raise RuntimeError("macOS say is not available on this system.")

    command = ["say"]
    if voice:
        command.extend(["-v", voice])
    command.append(text)
    subprocess.run(command, check=False)


def _parse_macos_voices(output: str) -> list[VoiceOption]:
    voices = []
    for line in output.splitlines():
        match = re.match(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2}_[A-Z]{2})\s+#", line)
        if match is None:
            continue
        name = match.group("name").strip()
        voices.append(
            VoiceOption(
                id=name,
                name=name,
                provider="macos-say",
                locale=match.group("locale"),
            )
        )
    return voices


def get_parent_preferences_store(
    settings: Settings = Depends(get_settings),
) -> ParentPreferencesStore:
    return ParentPreferencesStore(settings)
