"""Read one string field out of a JSON object while it is still being written.

The story prompt puts `response_text` first in the object precisely so the
spoken reply can start before the patches are generated. That only helps if
something can pull the value out of a half-written document, which is what this
does.

Three things make it fiddly:

- gemma4 wraps its output in a ```json fence even with `format: "json"` set, so
  the document does not begin at the first character.
- Escapes are JSON escapes, and a `\\uXXXX` can be split across chunk
  boundaries, so a partial escape has to be held back rather than emitted.
- The closing quote has to be told apart from an escaped one.
"""

from __future__ import annotations

from yarnotron.services.sentences import is_abbreviation_dot

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class JsonStringFieldReader:
    """Emits the decoded value of one top-level string field as it arrives.

    Feed it raw model output in whatever chunks arrive. Each `feed` returns the
    text that became available in that chunk, which may be empty. Once `done`
    is true the field has been fully read and later chunks return nothing.
    """

    def __init__(self, field: str = "response_text") -> None:
        self._needle = f'"{field}"'
        self._buffer = ""
        self._in_string = False
        self._found_key = False
        self._done = False
        self._pending = ""

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, chunk: str) -> str:
        if self._done or not chunk:
            return ""
        self._buffer += chunk

        if not self._found_key:
            index = self._buffer.find(self._needle)
            if index == -1:
                # Keep only enough of the tail to catch a needle split across
                # chunks; the rest is fence, whitespace or braces we do not need.
                self._buffer = self._buffer[-len(self._needle) :]
                return ""
            self._found_key = True
            self._buffer = self._buffer[index + len(self._needle) :]

        if not self._in_string:
            # Between the key and its value there is a colon, maybe whitespace,
            # then the opening quote.
            quote = self._buffer.find('"')
            if quote == -1:
                return ""
            self._in_string = True
            self._buffer = self._buffer[quote + 1 :]

        return self._consume_string()

    def _consume_string(self) -> str:
        out: list[str] = []
        source = self._pending + self._buffer
        self._pending = ""
        self._buffer = ""
        index = 0

        while index < len(source):
            char = source[index]

            if char == "\\":
                remainder = source[index:]
                if len(remainder) < 2:
                    self._pending = remainder  # escape split across chunks
                    break
                marker = remainder[1]
                if marker == "u":
                    if len(remainder) < 6:
                        self._pending = remainder
                        break
                    try:
                        out.append(chr(int(remainder[2:6], 16)))
                    except ValueError:
                        out.append(remainder[:6])
                    index += 6
                    continue
                out.append(_ESCAPES.get(marker, marker))
                index += 2
                continue

            if char == '"':
                self._done = True
                return "".join(out)

            out.append(char)
            index += 1

        return "".join(out)


class SentenceAccumulator:
    """Groups streamed characters into speakable sentences.

    A sentence is released as soon as its terminator is followed by a space or
    another sentence, so text can go to the synthesiser while the model is
    still writing the next one. `flush` releases whatever is left at the end.
    """

    _TERMINATORS = ".!?"

    def __init__(self, minimum_chars: int = 12) -> None:
        self._buffer = ""
        self._minimum_chars = minimum_chars

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        sentences: list[str] = []

        while True:
            split_at = self._find_split()
            if split_at is None:
                break
            sentence = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:].lstrip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def _find_split(self) -> int | None:
        for index, char in enumerate(self._buffer):
            if char not in self._TERMINATORS:
                continue
            if index + 1 < self._minimum_chars:
                continue
            if is_abbreviation_dot(self._buffer, index):
                continue  # "Mrs. Strumson" is one name, not the end of a sentence
            # Only split when we can see what follows. A terminator alone is not
            # a sentence end: "3.5" has no space after it, "..." is a run, and
            # "it... here" continues in lower case.
            if index + 1 >= len(self._buffer):
                return None
            if self._buffer[index + 1] in self._TERMINATORS:
                continue
            if not self._buffer[index + 1].isspace():
                continue

            following = self._next_visible(index + 1)
            if following is None:
                return None  # wait for the next chunk to decide
            if following.isupper() or following in "\"'“‘([":
                return index + 1
        return None

    def _next_visible(self, start: int) -> str | None:
        for char in self._buffer[start:]:
            if not char.isspace():
                return char
        return None

    def flush(self) -> str:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
