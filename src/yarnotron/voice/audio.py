from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd


@dataclass
class RecordingSettings:
    sample_rate: int = 16_000
    input_device: int | str | None = None
    block_ms: int = 30
    start_threshold: float = 0.018
    stop_threshold: float = 0.012
    silence_seconds: float = 0.9
    max_seconds: float = 18.0
    preroll_seconds: float = 0.3


class VoiceActivityRecorder:
    def __init__(self, settings: RecordingSettings | None = None) -> None:
        self.settings = settings or RecordingSettings()

    def record_utterance(self, listen_timeout_seconds: float | None = None) -> Path | None:
        settings = self.settings
        started_at = time.monotonic()
        block_size = int(settings.sample_rate * settings.block_ms / 1000)
        preroll_blocks = max(1, int(settings.preroll_seconds * 1000 / settings.block_ms))
        silence_blocks_to_stop = max(1, int(settings.silence_seconds * 1000 / settings.block_ms))
        max_blocks = max(1, int(settings.max_seconds * 1000 / settings.block_ms))

        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        captured: list[np.ndarray] = []
        recording = False
        silent_blocks = 0

        with sd.InputStream(
            samplerate=settings.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            device=settings.input_device,
        ) as stream:
            while True:
                block, _overflowed = stream.read(block_size)
                samples = np.asarray(block[:, 0], dtype=np.float32)
                rms = float(np.sqrt(np.mean(np.square(samples))))

                if not recording:
                    if (
                        listen_timeout_seconds is not None
                        and time.monotonic() - started_at >= listen_timeout_seconds
                    ):
                        return None
                    preroll.append(samples.copy())
                    if rms >= settings.start_threshold:
                        recording = True
                        captured.extend(preroll)
                        captured.append(samples.copy())
                    continue

                captured.append(samples.copy())
                silent_blocks = silent_blocks + 1 if rms < settings.stop_threshold else 0

                if silent_blocks >= silence_blocks_to_stop or len(captured) >= max_blocks:
                    return _write_wav(captured, settings.sample_rate)


def _write_wav(blocks: list[np.ndarray], sample_rate: int) -> Path:
    audio = np.concatenate(blocks)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return path


def wait_for_mic_permission_hint() -> None:
    # macOS prompts for microphone permission on first capture. A small delay
    # keeps CLI output readable if the permission dialog appears.
    time.sleep(0.2)


def list_audio_devices() -> str:
    return str(sd.query_devices())


def parse_audio_device(value: str | None) -> int | str | None:
    if value is None or not value.strip():
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped
