"""Sentence boundaries that do not break on "Mrs." or "Dr.": one rule, used wherever speech text is split."""

from __future__ import annotations

import re

ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "jr", "sr", "mt", "vs", "prof", "capt", "gen", "col", "lt", "sgt"})
_WORD_BEFORE = re.compile(r"([A-Za-z]+)$")


def is_abbreviation_dot(text: str, index: int) -> bool:
    """True when text[index] is the dot of an abbreviation such as "Mrs." (not the end of a sentence)."""
    if text[index] != ".":
        return False
    match = _WORD_BEFORE.search(text[:index])
    return bool(match) and match.group(1).casefold() in ABBREVIATIONS


def sentence_ends(text: str) -> list[int]:
    """Indices just after each real sentence end in `text`."""
    ends = []
    for index, char in enumerate(text):
        if char not in ".!?" or is_abbreviation_dot(text, index):
            continue
        if index + 1 < len(text) and text[index + 1] in ".!?":
            continue
        if index + 1 == len(text) or text[index + 1].isspace() or text[index + 1] in "\"'”’)":
            ends.append(index + 1)
    return ends


def split_sentences(text: str) -> list[str]:
    parts, start = [], 0
    for end in sentence_ends(text):
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def drop_trailing_question(text: str) -> str:
    """Remove the last sentence if it is a question; keep everything before it."""
    text = text.strip()
    if not text.endswith("?"):
        return text
    ends = [e for e in sentence_ends(text) if e < len(text)]
    return text[: ends[-1]].strip() if ends else ""
