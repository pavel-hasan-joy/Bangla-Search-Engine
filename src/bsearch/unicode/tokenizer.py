"""Script-aware tokenizer for Bengali, Latin, and mixed text.

Guarantees:
- Never splits Bengali consonant conjuncts (e.g. ক্ষ, স্ক্র, র্ক, ক্য, ত্ব).
- Keeps Latin and alphanumeric tokens lowercased.
- Converts Bengali digits to ASCII for consistent indexing.
- Preserves intra-token hyphens (e.g. ঢাকা-সিলেট, covid-19) and apostrophes (e.g. user's, o'clock).
- Strips surrounding quotes and punctuation.
- Integrates optional inflectional stemming behind the stem=True flag.
"""

from __future__ import annotations

import re
from typing import Final

from bsearch.unicode.normalizer import normalize
from bsearch.unicode.stemmer import BanglaStemmer, get_default_stemmer

# Bengali Unicode block (0x0980 - 0x09FF)
_BENGALI_WORD_CHARS = r"[\u0980-\u09FF]"

# Base character for word tokens: alphanumeric ASCII or Bengali script
_BASE_WORD_CHAR = rf"(?:[a-zA-Z0-9]|{_BENGALI_WORD_CHARS})"

# Token pattern: word characters, optionally connected by intra-word hyphen, apostrophe,
# or decimal point in numbers (e.g. 'ঢাকা', 'শিক্ষা', 'ঢাকা-সিলেট', "user's", 'covid-19', '3.11')
_RE_TOKEN: Final = re.compile(
    rf"{_BASE_WORD_CHAR}+(?:['’\-]{_BASE_WORD_CHAR}+|\.\d+)*"
)


def tokenize(
    text: str,
    normalize_text: bool = True,
    stem: bool = False,
    stemmer: BanglaStemmer | None = None,
    lowercase: bool = True,
    digits_to_ascii: bool = True,
) -> list[str]:
    """Tokenize text into search index / query tokens.

    Args:
        text: Input string (Bengali, Latin, or mixed).
        normalize_text: Whether to apply Unicode normalizer first (default True).
        stem: Whether to apply rule-based suffix stemming to Bengali tokens (default False).
        stemmer: Optional custom BanglaStemmer instance (uses default if None).
        lowercase: Whether to lowercase Latin tokens (default True).
        digits_to_ascii: Whether to normalize Bengali numerals to ASCII numerals (default True).

    Returns:
        List of cleaned, script-aware tokens.
    """
    if not text:
        return []

    # 1. Normalization
    if normalize_text:
        text = normalize(text, digits_to_ascii=digits_to_ascii)

    # 2. Extract tokens matching the regex
    raw_tokens = _RE_TOKEN.findall(text)

    # 3. Process tokens
    stem_engine = stemmer or get_default_stemmer()
    tokens: list[str] = []

    for tok in raw_tokens:
        # Standardize curly apostrophe
        if "’" in tok:
            tok = tok.replace("’", "'")

        # Lowercase Latin characters
        if lowercase:
            tok = tok.lower()

        # Optional suffix stemming for Bengali tokens
        if stem:
            tok = stem_engine.stem(tok, enabled=True)

        if tok:
            tokens.append(tok)

    return tokens
