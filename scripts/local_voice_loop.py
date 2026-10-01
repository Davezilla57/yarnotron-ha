#!/usr/bin/env python3
"""Microphone-first local voice loop for Yarnotron."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from yarnotron.core.config import get_settings
from yarnotron.services.preferences import ParentPreferencesStore
from yarnotron.voice.audio import (
    RecordingSettings,
    VoiceActivityRecorder,
    list_audio_devices,
    parse_audio_device,
    wait_for_mic_permission_hint,
)
from yarnotron.voice.backend import BackendClient
from yarnotron.voice.loop import LocalVoiceLoop, VoiceLoopSettings
from yarnotron.voice.stt import FasterWhisperTranscriber, TranscriptionSettings
from yarnotron.voice.tts import (
    DEFAULT_KOKORO_VOICE,
    create_sound_effect_player,
    create_speaker,
    list_macos_voices,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8989")
    parser.add_argument("--story-id")
    parser.add_argument("--child-id", default="local_child")
    parser.add_argument("--device-id", default="macbook-local-voice")
    parser.add_argument("--list-audio-devices", action="store_true")
    parser.add_argument("--list-macos-voices", action="store_true")
    parser.add_argument("--input-device", help="Input device index or name for sounddevice.")
    parser.add_argument("--output-device", help="Output device index or name for Piper playback.")
    parser.add_argument(
        "--tts-provider",
        choices=["auto", "none", "kokoro", "macos-say", "piper"],
        default="auto",
    )
    parser.add_argument("--tts-voice", help="Provider-specific voice name, such as a macOS say voice.")
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Never stop speaking to listen. Use if replies are being cut short.",
    )
    parser.add_argument(
        "--barge-in-margin",
        type=float,
        help=(
            "How much louder than Yarnotron's own echo a voice must be to interrupt "
            "(default 1.4). Lower interrupts more eagerly; raise it in a noisy room."
        ),
    )
    parser.add_argument(
        "--barge-in-frames",
        type=int,
        help=(
            "Speech frames out of 12 needed to interrupt (default 8). Higher means "
            "an interruption must be more sustained."
        ),
    )
    parser.add_argument(
        "--kokoro-voice",
        default=DEFAULT_KOKORO_VOICE,
        help="Kokoro voice id, such as af_heart or bm_george.",
    )
    parser.add_argument("--piper-binary", default="piper")
    parser.add_argument("--piper-model", help="Path to a Piper .onnx voice model.")
    parser.add_argument("--piper-config", help="Optional path to a Piper voice JSON config.")
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Alias for --tts-provider none.",
    )
    parser.add_argument("--stt-model", default="base.en")
    parser.add_argument("--stt-device", default="cpu")
    parser.add_argument("--stt-compute-type", default="int8")
    parser.add_argument("--start-threshold", type=float, default=0.018)
    parser.add_argument("--stop-threshold", type=float, default=0.012)
    parser.add_argument("--silence-seconds", type=float, default=0.9)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    parser.add_argument("--wake-word", default="interpreter")
    parser.add_argument("--wake-response", default="I'm listening. What's your story?")
    parser.add_argument("--idle-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--no-space-friend-howl",
        action="store_true",
        help="Do not play the space-friend howl for unclear story requests.",
    )
    args = parser.parse_args()

    if args.list_audio_devices:
        print(list_audio_devices())
        return 0
    if args.list_macos_voices:
        print(list_macos_voices())
        return 0

    input_device = parse_audio_device(args.input_device)
    output_device = parse_audio_device(args.output_device)
    tts_provider = "none" if args.no_speak else args.tts_provider
    preferences = ParentPreferencesStore(get_settings()).get()
    tts_voice = args.tts_voice or preferences.tts_voice
    if not args.no_speak and args.tts_provider == "auto" and preferences.tts_provider != "auto":
        tts_provider = preferences.tts_provider

    backend = BackendClient(args.api)
    barge_in_settings = None
    if args.barge_in_margin is not None or args.barge_in_frames is not None:
        from yarnotron.voice.bargein import BargeInSettings

        defaults = BargeInSettings()
        barge_in_settings = replace(
            defaults,
            margin=args.barge_in_margin or defaults.margin,
            speech_frames_required=args.barge_in_frames or defaults.speech_frames_required,
        )
    speaker = create_speaker(
        provider=tts_provider,
        voice=tts_voice,
        piper_binary=args.piper_binary,
        piper_model=args.piper_model,
        piper_config=args.piper_config,
        output_device=output_device,
        kokoro_voice=args.kokoro_voice,
        barge_in=not args.no_barge_in,
        barge_in_settings=barge_in_settings,
    )

    recorder = VoiceActivityRecorder(
        RecordingSettings(
            input_device=input_device,
            start_threshold=args.start_threshold,
            stop_threshold=args.stop_threshold,
            silence_seconds=args.silence_seconds,
            max_seconds=args.max_seconds,
        )
    )
    transcriber = FasterWhisperTranscriber(
        TranscriptionSettings(
            model=args.stt_model,
            device=args.stt_device,
            compute_type=args.stt_compute_type,
        )
    )
    loop = LocalVoiceLoop(
        backend=backend,
        recorder=recorder,
        transcriber=transcriber,
        speaker=speaker,
        sound_effect_player=create_sound_effect_player(
            enabled=preferences.space_friend_howl_enabled and not args.no_space_friend_howl
        ),
        settings=VoiceLoopSettings(
            child_id=args.child_id,
            story_id=args.story_id,
            device_id=args.device_id,
            wake_word=args.wake_word,
            wake_response=args.wake_response,
            idle_timeout_seconds=args.idle_timeout_seconds,
        ),
        status=lambda message: print(message, flush=True),
    )

    try:
        wait_for_mic_permission_hint()
        loop.run_forever()
    except KeyboardInterrupt:
        print()
    except RuntimeError as exc:
        print(f"Yarnotron voice loop failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
