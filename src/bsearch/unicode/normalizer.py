"""Unicode normalization, character canonicalization, and orthographic repair for Bengali text.

Policies:
- Unicode Normalization: Canonical Decomposition followed by Canonical Composition (NFC).
- Canonical Forms: Standardizes decomposed য+়, ড+়, ঢ+় to atomic য় (\u09DF), ড় (\u09DC), ঢ় (\u09DD).
- Two-Part Vowel Signs: Composes split ে+া to ো (\u09CB) and ে+ৗ to ৌ (\u09CC).
- Repeated Hasanta: Collapses duplicate virama/hasanta sequences (্+) to a single hasanta (্).
- Pre-Base Vowel Mis-Ordering Repair: Detects pre-base vowel signs (ি, ে, ৈ) typed before
  a consonant or conjunct at word-initial or post-vowel positions and repositions them
  orthographically after the consonant base (e.g. 'িশক্ষা' -> 'শিক্ষা', 'েপ্রম' -> 'প্রেম').
- Bangla Digits: Maps Bengali numerals (০-৯) to ASCII digits (0-9) for consistent indexing.
- Danda Standardization: Standardizes double danda (॥) to danda (।) as sentence boundary.
- Zero Width Controls (ZWJ / ZWNJ Policy):
  - ZWJ (\u200D) and ZWNJ (\u200C) are rendering controls used to alter ligature display
    (e.g. ক্‍ক vs ক্ক, or ক্‌ক vs ক্ক).
  - Across different Bengali keyboards (Bijoy, Avro, Gboard), users produce inconsistent
    ZWJ/ZWNJ usage for the exact same semantic terms.
  - Policy: By default (preserve_formatting=False), ZWJ and ZWNJ are stripped during
    indexing and query normalization to prevent retrieval mismatches. If visual
    reconstruction is explicitly required, set preserve_formatting=True.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from bsearch.unicode.constants import (
    BANGLA_TO_ASCII_DIGITS,
    CONSONANTS,
    DANDA,
    DOUBLE_DANDA,
    HASANTA,
    NUKTA,
    NUKTA_CANONICAL_MAP,
    PRE_BASE_VOWEL_SIGNS,
    TWO_PART_VOWEL_MAP,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
)

# Regex to collapse multiple consecutive hasantas
_RE_REPEATED_HASANTA: Final = re.compile(rf"{HASANTA}{{2,}}")

# Regex to strip ZWJ and ZWNJ
_RE_ZERO_WIDTH: Final = re.compile(rf"[{re.escape(ZWNJ)}{re.escape(ZWJ)}]")

# Consonant character set for regex matching
_CONSONANT_CHARS = "".join(sorted(CONSONANTS))
_C_REGEX = rf"[{re.escape(_CONSONANT_CHARS)}]"

# Pre-base vowel signs
_PRE_BASE_CHARS = "".join(sorted(PRE_BASE_VOWEL_SIGNS))
_PRE_BASE_REGEX = rf"[{re.escape(_PRE_BASE_CHARS)}]"

# All vowel signs
_VOWEL_SIGN_CHARS = "".join(sorted(VOWEL_SIGNS))

# Regex for repairing misordered pre-base vowels
# Matches:
# (1) Non-consonant boundary before: start of string, whitespace, punctuation, or completed vowel
# (2) The mis-placed pre-base vowel sign (ি, ে, ৈ)
# (3) The consonant base or conjunct chain: C(্C)* with optional nuktas
# (4) Followed by non-vowel-sign (so we don't accidentally attach two matras)
_RE_MISORDERED_PREBASE: Final = re.compile(
    rf"(^|[^\u0980-\u09FF]|[\s\u0964\u0965{re.escape(_VOWEL_SIGN_CHARS)}])"
    rf"({_PRE_BASE_REGEX})"
    rf"({_C_REGEX}(?:{NUKTA})?(?:{HASANTA}{_C_REGEX}(?:{NUKTA})?)*)"
    rf"(?![{re.escape(_VOWEL_SIGN_CHARS)}])"
)


def repair_misordered_vowels(text: str) -> str:
    """Repair pre-base vowel signs (ি, ে, ৈ) typed before their consonant.

    E.g. 'িশক্ষা' -> 'শিক্ষা', 'েপ্রম' -> 'প্রেম', 'ৈতরি' -> 'তৈরি'.
    """
    if not any(v in text for v in PRE_BASE_VOWEL_SIGNS):
        return text

    def _swap(match: re.Match[str]) -> str:
        prefix = match.group(1)
        vowel_sign = match.group(2)
        consonant_cluster = match.group(3)
        return f"{prefix}{consonant_cluster}{vowel_sign}"

    # Perform replacement until fixed point (in case of consecutive repaired words)
    prev = None
    current = text
    while current != prev:
        prev = current
        current = _RE_MISORDERED_PREBASE.sub(_swap, current)
    return current


def canonicalize_nukta(text: str) -> str:
    """Convert decomposed character+nukta combinations to canonical atomic forms."""
    result = text
    for decomp, atomic in NUKTA_CANONICAL_MAP.items():
        if decomp in result:
            result = result.replace(decomp, atomic)
    return result


def compose_two_part_vowels(text: str) -> str:
    """Compose split two-part vowel signs (ে+া -> ো, ে+ৗ -> ৌ)."""
    result = text
    for split_form, composed_form in TWO_PART_VOWEL_MAP.items():
        if split_form in result:
            result = result.replace(split_form, composed_form)
    return result


def convert_digits_to_ascii(text: str) -> str:
    """Convert Bengali digits (০-৯) to ASCII digits (0-9)."""
    return "".join(BANGLA_TO_ASCII_DIGITS.get(ch, ch) for ch in text)


def normalize(
    text: str,
    digits_to_ascii: bool = True,
    preserve_formatting: bool = False,
    repair_vowels: bool = True,
) -> str:
    """Normalize Bengali and mixed script text into canonical form for indexing and retrieval.

    Steps applied:
    1. Unicode NFC normalization.
    2. Canonicalize decomposed nukta sequences (য়, ড়, ঢ়).
    3. Compose split two-part vowel signs (ো, ৌ).
    4. Collapse repeated hasanta (্+ -> ্).
    5. Standardize double danda (॥) to danda (।).
    6. Repair mis-ordered pre-base vowel signs (e.g. 'িশক্ষা' -> 'শিক্ষা') if repair_vowels=True.
    7. Convert Bengali digits to ASCII (০-৯ -> 0-9) if digits_to_ascii=True.
    8. Strip ZWJ and ZWNJ if preserve_formatting=False (per documented policy).

    Args:
        text: Input string.
        digits_to_ascii: Whether to map Bengali numerals to ASCII numerals (default True).
        preserve_formatting: If False (default), strips ZWJ/ZWNJ controls to ensure robust matching.
        repair_vowels: Whether to fix pre-base vowel signs typed before consonants (default True).

    Returns:
        Canonically normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode NFC composition
    normalized = unicodedata.normalize("NFC", text)

    # 2. Canonical forms for nukta characters (য়, ড়, ঢ়)
    normalized = canonicalize_nukta(normalized)

    # 3. Two-part vowel signs composition (ো, ৌ)
    normalized = compose_two_part_vowels(normalized)

    # 4. Collapse repeated hasantas
    normalized = _RE_REPEATED_HASANTA.sub(HASANTA, normalized)

    # 5. Danda normalization
    if DOUBLE_DANDA in normalized:
        normalized = normalized.replace(DOUBLE_DANDA, DANDA)

    # 6. Repair misordered pre-base vowels
    if repair_vowels:
        normalized = repair_misordered_vowels(normalized)

    # 7. Convert Bangla digits to ASCII
    if digits_to_ascii:
        normalized = convert_digits_to_ascii(normalized)

    # 8. ZWJ / ZWNJ Policy enforcement
    if not preserve_formatting:
        normalized = _RE_ZERO_WIDTH.sub("", normalized)

    return normalized
