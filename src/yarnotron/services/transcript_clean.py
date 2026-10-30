"""Tidy what speech recognition returns before anything reads it as the child's words."""

from __future__ import annotations

import re

# Whisper writes background sound as *thud*, [BLANK_AUDIO] or (music). They are not words, so they are not story.
_SOUND_TAG = re.compile(r"\*[^*\n]{1,40}\*|\[[^\]\n]{1,40}\]|\([^)\n]{1,40}\)")


def strip_sound_tags(text: str) -> str:
    return re.sub(r"\s+", " ", _SOUND_TAG.sub(" ", text)).strip()
