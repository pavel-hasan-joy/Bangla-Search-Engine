"""Orthographic grapheme cluster segmenter for Bengali and mixed script text.

A Bengali orthographic grapheme (akshara) consists of:
- Base: Consonant conjunct chain (C + hasanta + C + ...), or independent vowel, or Khanda-Ta.
- Attachments: Any attached vowel signs (matras), nukta, and modifiers (ং Anusvara, ঃ Visarga, ঁ Candrabindu).
- Conjunct chains (e.g. ক্ষ, স্ক্র, র্ক, ক্য, ত্ব) and modifying diacritics never get split apart.
"""

from __future__ import annotations

import re
from typing import Final

from bsearch.unicode.constants import (
    CONSONANTS,
    HASANTA,
    INDEPENDENT_VOWELS,
    MODIFIERS,
    NUKTA,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
)

_C_CHARS = "".join(sorted(CONSONANTS))
_C_REGEX = rf"[{re.escape(_C_CHARS)}]"

_V_CHARS = "".join(sorted(INDEPENDENT_VOWELS))
_MATRA_AND_MODIFIERS = "".join(sorted(VOWEL_SIGNS | MODIFIERS | {NUKTA}))

# Regex for a single orthographic Bengali grapheme or individual external character:
_RE_GRAPHEME: Final = re.compile(
    rf"(?:"
    # 1. Consonant base with optional hasanta conjunct chains:
    rf"{_C_REGEX}(?:{NUKTA})?"
    rf"(?:(?:[{re.escape(ZWNJ)}{re.escape(ZWJ)}])?{HASANTA}(?:[{re.escape(ZWNJ)}{re.escape(ZWJ)}])?{_C_REGEX}(?:{NUKTA})?)*"
    rf"[{re.escape(_MATRA_AND_MODIFIERS)}]*"
    rf"(?:{HASANTA})?"
    rf"|"
    # 2. Independent vowels or Khanda-Ta (ৎ) with optional modifiers:
    rf"[{re.escape(_V_CHARS)}\u09CE]"
    rf"[{re.escape(_MATRA_AND_MODIFIERS)}]*"
    rf"|"
    # 3. Newline pair \r\n or any other single character:
    rf"\r\n|."
    rf")",
    re.DOTALL,
)


def segment_graphemes(text: str) -> list[str]:
    """Segment input string into orthographic grapheme clusters.

    Preserves full conjunct chains (e.g. ক্ষ, স্ক্র, র্ক, ক্য, ত্ব)
    and ensures vowel signs (matras) and modifiers (ং, ঃ, ঁ) stay attached to their base.

    Args:
        text: Input string.

    Returns:
        List of grapheme cluster strings.
    """
    if not text:
        return []
    return _RE_GRAPHEME.findall(text)
