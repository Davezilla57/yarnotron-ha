from yarnotron.core.config import Settings
from yarnotron.models.preferences import UpdateParentPreferencesRequest
from yarnotron.services.preferences import ParentPreferencesStore, _parse_macos_voices


def test_parent_preferences_store_round_trips_voice_choice(tmp_path) -> None:
    store = ParentPreferencesStore(Settings(preferences_path=str(tmp_path / "preferences.json")))

    saved = store.update(
        UpdateParentPreferencesRequest(tts_provider="macos-say", tts_voice="Samantha")
    )
    loaded = store.get()

    assert saved.tts_provider == "macos-say"
    assert saved.tts_voice == "Samantha"
    assert saved.space_friend_howl_enabled is True
    assert loaded == saved


def test_parent_preferences_store_round_trips_space_friend_howl_choice(tmp_path) -> None:
    store = ParentPreferencesStore(Settings(preferences_path=str(tmp_path / "preferences.json")))

    saved = store.update(
        UpdateParentPreferencesRequest(
            tts_provider="macos-say",
            tts_voice="Samantha",
            space_friend_howl_enabled=False,
        )
    )
    loaded = store.get()

    assert saved.space_friend_howl_enabled is False
    assert loaded.space_friend_howl_enabled is False


def test_parse_macos_voices_handles_names_with_spaces() -> None:
    voices = _parse_macos_voices(
        """
Samantha            en_US    # Hello, my name is Samantha.
Good News           en_US    # Hello from Good News.
Thomas              fr_FR    # Bonjour, je m'appelle Thomas.
"""
    )

    assert [voice.id for voice in voices] == ["Samantha", "Good News", "Thomas"]
    assert voices[1].provider == "macos-say"
    assert voices[2].locale == "fr_FR"
