"""Match a name that speech recognition garbled to the character or scene it was meant to be.

"Mrs. Drumston", "Mrs. Shumson" and "Mrs. Stratson" are all Mrs. Strumson. The model only sees the words it is
given, so it cannot tell, and used to answer "I'm not sure which one you mean" turn after turn. Code does the match
and the model sees the real name. The child's own words are still what gets stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from yarnotron.models.story import Story

HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "dr", "sir", "madam", "captain", "professor"})
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*")
NEAR = 0.74       # similarity needed on a normal turn
LOOSE = 0.45      # with a matching "Mrs." and a single candidate, on a turn that asks to change or remove something


@dataclass(frozen=True)
class NameMatch:
    kind: str          # "character" or "scene"
    item_id: str
    name: str


def _skeleton(word: str) -> str:
    """Consonant skeleton with look-alike sounds folded together, so Drumston and Strumson meet."""
    word = word.casefold()
    for a, b in (("sh", "s"), ("ch", "s"), ("ph", "f"), ("ck", "k"), ("th", "t"), ("str", "s"), ("dr", "s"), ("tr", "s")):
        word = word.replace(a, b)
    word = re.sub(r"[aeiouy]", "", word)
    word = word.translate(str.maketrans("dzcqg", "tsksk"))
    return re.sub(r"(.)\1+", r"\1", word)


def similarity(a: str, b: str) -> float:
    a, b = a.casefold(), b.casefold()
    if a == b:
        return 1.0
    raw = SequenceMatcher(None, a, b).ratio()
    sk_a, sk_b = _skeleton(a), _skeleton(b)
    skeleton = SequenceMatcher(None, sk_a, sk_b).ratio() if sk_a and sk_b else 0.0
    return max(raw, 0.5 * raw + 0.5 * skeleton)


def _entities(story: Story | None) -> list[tuple[str, str, str]]:
    if story is None:
        return []
    return [("character", c.id, c.name) for c in story.characters if c.name] + [
        ("scene", sc.id, sc.title) for sc in story.scenes if sc.title
    ]


def _split_name(name: str) -> tuple[str | None, list[str]]:
    words = _TOKEN.findall(name)
    if words and words[0].casefold() in HONORIFICS:
        return words[0].casefold(), words[1:]
    return None, words


def resolve_names(transcript: str, story: Story | None, *, loose: bool = False) -> tuple[str, list[NameMatch]]:
    """(transcript with near-miss names replaced by the real ones, what was matched)."""
    entities = _entities(story)
    if not entities:
        return transcript, []
    tokens = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(transcript)]
    words_lower = [t[2].casefold().strip("'’-") for t in tokens]
    spans: list[tuple[int, int, str, NameMatch]] = []
    exact: list[NameMatch] = []
    honorific_counts: dict[str, int] = {}
    for _, _, name in entities:
        honorific, _ = _split_name(name)
        if honorific:
            honorific_counts[honorific] = honorific_counts.get(honorific, 0) + 1

    for kind, item_id, name in entities:
        honorific, core = _split_name(name)
        if not core:
            continue
        size = len(core)
        for i in range(0, len(tokens) - size + 1):
            window = " ".join(words_lower[i : i + size])
            if window == " ".join(w.casefold() for w in core):
                exact.append(NameMatch(kind, item_id, name))
                break  # said correctly already
            if size == 1 and (len(window) < 4 or window in HONORIFICS):
                continue
            score = similarity(window, " ".join(core))
            after_honorific = i > 0 and words_lower[i - 1] in HONORIFICS
            same_honorific = after_honorific and honorific is not None and words_lower[i - 1] == honorific
            needed = LOOSE if (loose and same_honorific and honorific_counts.get(honorific, 0) == 1) else NEAR
            if score < needed:
                continue
            start = tokens[i - 1][0] if same_honorific else tokens[i][0]
            end = tokens[i + size - 1][1]
            spans.append((start, end, name, NameMatch(kind, item_id, name)))
            break
    if not spans:
        return transcript, exact
    spans.sort()
    out, last, matches, used = [], 0, [], set()
    for start, end, name, match in spans:
        if start < last or match.item_id in used:
            continue
        out.append(transcript[last:start])
        out.append(name)
        last = end
        matches.append(match)
        used.add(match.item_id)
    out.append(transcript[last:])
    return "".join(out), matches + [m for m in exact if m.item_id not in used]
