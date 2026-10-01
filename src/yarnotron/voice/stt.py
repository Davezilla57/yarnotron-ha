from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class TranscriptionSettings:
    model: str = "base.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


class FasterWhisperTranscriber:
    def __init__(self, settings: TranscriptionSettings | None = None) -> None:
        self.settings = settings or TranscriptionSettings()
        self._model: WhisperModel | None = None

    def transcribe(self, audio_path: Path) -> str:
        model = self._load_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language=self.settings.language,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.settings.model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        return self._model

