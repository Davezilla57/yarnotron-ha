from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol
import shutil
import subprocess
import time
import tempfile
import wave

# numpy and sounddevice are imported where they are used, not here. This module
# is reached from the API (voice picker, voice samples), and the API has to run
# on a headless host -- a container, a NAS, a server behind Home Assistant --
# where there is no PortAudio and the `voice` extra is not installed.

if TYPE_CHECKING:
    from yarnotron.voice.bargein import BargeInDetector, BargeInSettings


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


class SoundEffectPlayer(Protocol):
    def play(self, path: Path) -> None: ...


class NullSpeaker:
    def speak(self, text: str) -> None:
        return


class NullSoundEffectPlayer:
    def play(self, path: Path) -> None:
        return


class AfplaySoundEffectPlayer:
    def play(self, path: Path) -> None:
        if not path.exists() or shutil.which("afplay") is None:
            return
        subprocess.run(["afplay", str(path)], check=False)


class MacSaySpeaker:
    def __init__(self, voice: str | None = None) -> None:
        self.voice = voice

    def speak(self, text: str) -> None:
        if shutil.which("say") is None:
            return
        command = ["say"]
        if self.voice:
            command.extend(["-v", self.voice])
        command.append(text)
        subprocess.run(command, check=False)


class PiperSpeaker:
    def __init__(
        self,
        model: Path,
        binary: str = "piper",
        config: Path | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.model = model
        self.binary = binary
        self.config = config
        self.output_device = output_device

    def speak(self, text: str) -> None:
        if shutil.which(self.binary) is None:
            raise RuntimeError(f"Piper binary not found: {self.binary}")
        if not self.model.exists():
            raise RuntimeError(f"Piper voice model not found: {self.model}")

        output_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        command = [
            self.binary,
            "--model",
            str(self.model),
            "--output_file",
            str(output_path),
        ]
        if self.config is not None:
            command.extend(["--config", str(self.config)])

        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        try:
            _play_wav(output_path, self.output_device)
        finally:
            output_path.unlink(missing_ok=True)


DEFAULT_KOKORO_DIR = Path.home() / ".cache" / "kokoro"
KOKORO_MODEL_NAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_NAME = "voices-v1.0.bin"
# Chosen by listening to eight candidates on five real corpus replies rendered
# by scripts/render_voice_samples.py. Also a good choice on pace: voices differ
# by up to 64% in how long they take to speak identical text, and af_heart sits
# near the fast end (see docs/phase-2-voice.md).
DEFAULT_KOKORO_VOICE = "af_heart"


def kokoro_model_paths(directory: Path | None = None) -> tuple[Path, Path]:
    base = directory or DEFAULT_KOKORO_DIR
    return base / KOKORO_MODEL_NAME, base / KOKORO_VOICES_NAME


def kokoro_is_available(directory: Path | None = None) -> bool:
    model, voices = kokoro_model_paths(directory)
    return model.exists() and voices.exists()


class KokoroSpeaker:
    """Neural TTS via kokoro-onnx.

    Synthesis happens per sentence, which is the granularity the voice loop
    already speaks at, so the reply starts as soon as the first sentence is
    ready rather than when the whole reply is.

    The model is loaded eagerly so the cost lands at start-up rather than on
    the child's first turn.
    """

    def __init__(
        self,
        model: Path,
        voices: Path,
        voice: str = DEFAULT_KOKORO_VOICE,
        speed: float = 1.0,
        output_device: int | str | None = None,
        barge_in: "BargeInDetector | None" = None,
        input_device: int | str | None = None,
    ) -> None:
        from kokoro_onnx import Kokoro

        if not model.exists() or not voices.exists():
            raise RuntimeError(f"Kokoro model files not found in {model.parent}")
        self.voice = voice
        self.speed = speed
        self.output_device = output_device
        self.input_device = input_device
        self.barge_in = barge_in
        self._kokoro = Kokoro(str(model), str(voices))

    def speak(self, text: str) -> None:
        self.speak_until_interrupted(text)

    def speak_until_interrupted(self, text: str) -> bool:
        """Speak, returning True if the child talked over it and it was cut short.

        Without a detector this just plays to the end. With one, playback is
        non-blocking and the microphone is monitored throughout, which is what
        lets a child interrupt mid-sentence instead of waiting their turn.
        """
        if not text.strip():
            return False
        import sounddevice as sd

        samples, sample_rate = self._kokoro.create(
            text, voice=self.voice, speed=self.speed, lang="en-us"
        )

        if self.barge_in is None:
            sd.play(samples, samplerate=sample_rate, device=self.output_device)
            sd.wait()
            return False

        settings = self.barge_in.settings
        self.barge_in.reset()
        interrupted = False
        duration = len(samples) / sample_rate

        sd.play(samples, samplerate=sample_rate, device=self.output_device)
        started = time.monotonic()
        try:
            with sd.InputStream(
                samplerate=settings.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=settings.frame_samples,
                device=self.input_device,
            ) as stream:
                while time.monotonic() - started < duration:
                    frame, _overflowed = stream.read(settings.frame_samples)
                    if self.barge_in.feed(frame[:, 0]):
                        interrupted = True
                        break
        finally:
            if interrupted:
                sd.stop()
            else:
                sd.wait()
        return interrupted


class FallbackSpeaker:
    """Speak through the first engine that works, in preference order.

    A neural voice that fails at runtime should degrade to a plainer one rather
    than leaving the child in silence. Once an engine fails it is skipped for
    the rest of the session instead of being retried on every sentence.
    """

    def __init__(self, speakers: list[Speaker]) -> None:
        if not speakers:
            raise ValueError("FallbackSpeaker needs at least one speaker")
        self._speakers = speakers
        self._failed: set[int] = set()

    def speak(self, text: str) -> None:
        self.speak_until_interrupted(text)

    def speak_until_interrupted(self, text: str) -> bool:
        for index, speaker in enumerate(self._speakers):
            if index in self._failed:
                continue
            try:
                interruptible = getattr(speaker, "speak_until_interrupted", None)
                if interruptible is not None:
                    return bool(interruptible(text))
                speaker.speak(text)
                return False
            except Exception:
                self._failed.add(index)
        # Every engine failed; stay silent rather than raise into the turn loop.
        return False

    @property
    def active(self) -> Speaker | None:
        for index, speaker in enumerate(self._speakers):
            if index not in self._failed:
                return speaker
        return None


def create_speaker(
    provider: str,
    voice: str | None = None,
    piper_binary: str = "piper",
    piper_model: str | None = None,
    piper_config: str | None = None,
    output_device: int | str | None = None,
    kokoro_dir: Path | None = None,
    kokoro_voice: str = DEFAULT_KOKORO_VOICE,
    barge_in: bool = True,
    barge_in_settings: "BargeInSettings | None" = None,
    input_device: int | str | None = None,
) -> Speaker:
    resolved_provider = provider
    if provider == "auto":
        # Kokoro → Piper → system, the tiering phase 2 calls for. Only Kokoro
        # and the system voice can be chosen without configuration; Piper needs
        # a model path, so it is opt-in.
        if kokoro_is_available(kokoro_dir):
            resolved_provider = "kokoro"
        elif shutil.which("say"):
            resolved_provider = "macos-say"
        else:
            resolved_provider = "none"

    match resolved_provider:
        case "none":
            return NullSpeaker()
        case "macos-say":
            return MacSaySpeaker(voice=voice)
        case "kokoro":
            model, voices = kokoro_model_paths(kokoro_dir)
            detector = None
            if barge_in:
                from yarnotron.voice.bargein import BargeInDetector, load_silero

                detector = BargeInDetector(settings=barge_in_settings, vad=load_silero())
            kokoro = KokoroSpeaker(
                model=model,
                voices=voices,
                voice=kokoro_voice,
                output_device=output_device,
                barge_in=detector,
                input_device=input_device,
            )
            # Degrade rather than go silent if synthesis fails mid-session.
            tiers: list[Speaker] = [kokoro]
            if shutil.which("say"):
                tiers.append(MacSaySpeaker(voice=voice))
            return FallbackSpeaker(tiers) if len(tiers) > 1 else kokoro
        case "piper":
            if piper_model is None:
                raise ValueError("--piper-model is required when --tts-provider piper")
            return PiperSpeaker(
                model=Path(piper_model),
                binary=piper_binary,
                config=Path(piper_config) if piper_config else None,
                output_device=output_device,
            )
        case _:
            raise ValueError(f"Unsupported TTS provider: {provider}")


def create_sound_effect_player(enabled: bool = True) -> SoundEffectPlayer:
    if not enabled:
        return NullSoundEffectPlayer()
    return AfplaySoundEffectPlayer()


def list_macos_voices() -> str:
    if shutil.which("say") is None:
        return "macOS say is not available on this system."
    result = subprocess.run(["say", "-v", "?"], check=False, capture_output=True, text=True)
    return result.stdout


def _play_wav(path: Path, output_device: int | str | None = None) -> None:
    import numpy as np
    import sounddevice as sd

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        else:
            audio = audio.reshape(-1, 1)
        sd.play(audio, samplerate=sample_rate, device=output_device)
        sd.wait()
