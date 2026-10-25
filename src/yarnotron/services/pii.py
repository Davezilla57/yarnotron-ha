"""Remove contact details the child says out loud, before anything else sees them.

Street addresses, email addresses and phone numbers are removed deterministically, ahead of any model call,
any storage and any parent note. The child is told plainly that they cannot be taken or kept (see
`PII_NOT_KEPT_LINE`) and the story carries on. This is code, not prompt text: the model never receives the
original span.

Deliberately narrow. A false redaction teaches a child to stop talking, so a street name on its own
("Harbor Road") passes; a house number in front of one does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REDACTED = "[removed]"

PII_NOT_KEPT_LINE = (
    "I can't take or keep addresses or email addresses, but I can keep going with the story."
)

_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|oh"
)
_STREET_WORDS = (
    "street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd|court|ct|way|place|pl|terrace|"
    "circle|cir|highway|hwy|parkway|pkwy"
)
# Words that make "3 big dogs on the road" read as a sentence rather than an address.
_NOT_A_STREET_NAME = (
    "the|a|an|on|in|at|of|to|and|or|my|our|your|his|her|their|its|down|up|by|along|across|is|are|was|were|"
    "with|for|from|into|onto|over|under|that|this|these|those|it|we|they|he|she|i|you|then|there|here"
)

_TLDS = "com|net|org|edu|gov|io|co|uk|us|ca|au|info|me"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# Speech-to-text writes "sam dot jones at gmail dot com".
_SPOKEN_EMAIL = re.compile(
    rf"\b[a-z0-9]+(?:\s+(?:dot|underscore|dash)\s+[a-z0-9]+){{0,3}}\s+at\s+[a-z0-9]+(?:\s+dot\s+[a-z0-9]+)*"
    rf"\s+dot\s+(?:{_TLDS})\b",
    re.IGNORECASE,
)
_STREET = re.compile(
    rf"\b(?:\d{{1,6}}|(?:(?:{_NUMBER_WORDS})[\s-]+){{1,4}}(?:{_NUMBER_WORDS}))\s+"
    rf"(?:(?!(?:{_NOT_A_STREET_NAME})\b)[A-Za-z0-9'.]+\s+){{1,3}}(?:{_STREET_WORDS})\b\.?",
    re.IGNORECASE,
)
_PHONE_DIGITS = re.compile(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)")
_PHONE_WORDS = re.compile(rf"\b(?:(?:{_NUMBER_WORDS})[\s,-]+){{6,}}(?:{_NUMBER_WORDS})\b", re.IGNORECASE)


@dataclass(frozen=True)
class PiiResult:
    text: str
    kinds: tuple[str, ...]

    @property
    def found(self) -> bool:
        return bool(self.kinds)


def redact_contact_details(text: str) -> PiiResult:
    kinds: list[str] = []
    out = text

    def sub(pattern: re.Pattern[str], kind: str, value: str, min_digits: int = 0) -> str:
        def repl(match: re.Match[str]) -> str:
            if min_digits and sum(ch.isdigit() for ch in match.group(0)) < min_digits:
                return match.group(0)
            if kind not in kinds:
                kinds.append(kind)
            return REDACTED

        return pattern.sub(repl, value)

    out = sub(_EMAIL, "email_address", out)
    out = sub(_SPOKEN_EMAIL, "email_address", out)
    out = sub(_STREET, "street_address", out)
    out = sub(_PHONE_DIGITS, "phone_number", out, min_digits=7)
    out = sub(_PHONE_WORDS, "phone_number", out)
    if kinds:
        out = re.sub(rf"{re.escape(REDACTED)}(?:\s+{re.escape(REDACTED)})+", REDACTED, out).strip()
    return PiiResult(text=out, kinds=tuple(kinds))
