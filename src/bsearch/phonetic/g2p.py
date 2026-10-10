"""Grapheme-aware Bengali G2P-lite (Grapheme-to-Phoneme) converter.

Maps orthographic Bengali tokens into phonemic representations, taking into account:
- Initial vs medial conjuncts (e.g. ক্ষ -> kh word-initial, kkh medial).
- Sibilant assimilation (শ, ষ, স -> sh by default, s before dental stops).
- Matras and inherent vowels.
- Nasal modifiers (ং -> ng).
"""

from __future__ import annotations

import re
from typing import Final

from bsearch.unicode.constants import (
    HASANTA,
    MODIFIERS,
    NUKTA,
    VOWEL_SIGNS,
)
from bsearch.unicode.grapheme import segment_graphemes
from bsearch.unicode.normalizer import normalize

# Consonant phoneme mappings
CONSONANT_MAP: Final[dict[str, str]] = {
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "T",
    "ঠ": "Th",
    "ড": "D",
    "ঢ": "Dh",
    "ণ": "N",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "ph",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "sh",
    "হ": "h",
    "ড়": "R",
    "ঢ়": "Rh",
    "য়": "y",
    "ৎ": "t",
}

# Vowel sign mappings
VOWEL_SIGN_MAP: Final[dict[str, str]] = {
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
}

# Independent vowel mappings
INDEPENDENT_VOWEL_MAP: Final[dict[str, str]] = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}


def bangla_to_phonemes(token: str) -> str:
    """Convert Bengali orthographic token to a phonemic string representation.

    Args:
        token: Normalized Bengali word token (e.g. 'শিক্ষা', 'ঢাকা', 'শিখা').

    Returns:
        Phonemic string (e.g. 'shikkha', 'dhaka', 'shikha').
    """
    if not token:
        return ""

    norm_token = normalize(token)
    graphemes = segment_graphemes(norm_token)
    if not graphemes:
        return ""

    phonemes: list[str] = []
    num_g = len(graphemes)

    for idx, g in enumerate(graphemes):
        # 1. Check independent vowels
        if g in INDEPENDENT_VOWEL_MAP:
            phonemes.append(INDEPENDENT_VOWEL_MAP[g])
            continue

        # Extract vowel signs and modifiers from grapheme
        vowel_part = ""
        has_hasanta_end = g.endswith(HASANTA)
        modifiers_part = ""

        # Identify any matra in the grapheme
        for matra, val in VOWEL_SIGN_MAP.items():
            if matra in g:
                vowel_part = val
                break

        if "ং" in g:
            modifiers_part += "ng"
        if "ঃ" in g:
            modifiers_part += "h"

        # Isolate consonant base (strip matras, modifiers, nukta)
        clean_base = re.sub(
            rf"[{re.escape(''.join(VOWEL_SIGNS | MODIFIERS))}]", "", g
        )

        # 2. Special conjunct phonological rules
        if clean_base == "ক্ষ":
            # Word-initial ক্ষ sounds like kh, medial/final sounds like kkh
            consonant_phoneme = "kh" if idx == 0 else "kkh"
        elif clean_base == "জ্ঞ":
            # Word-initial sounds like g, medial sounds like gg
            consonant_phoneme = "g" if idx == 0 else "gg"
        elif clean_base == "ত্র":
            consonant_phoneme = "tr"
        elif clean_base == "ত্ব":
            consonant_phoneme = "t" if idx == 0 else "tt"
        elif clean_base == "স্ব" or clean_base == "শ্ব":
            consonant_phoneme = "sh" if idx == 0 else "ssh"
        elif clean_base == "স্ম":
            consonant_phoneme = "sh" if idx == 0 else "shsh"
        elif clean_base.startswith("স") and HASANTA in clean_base:
            # Dental sibilant conjunct (e.g. স্ত -> st, স্প -> sp, স্ক -> sk)
            consonant_phoneme = "s" + clean_base.replace("স", "").replace(HASANTA, "")
            # Map following consonant
            consonant_phoneme = "".join(CONSONANT_MAP.get(c, c) for c in clean_base if c != HASANTA)
            if clean_base.startswith("স"):
                consonant_phoneme = "s" + consonant_phoneme[2:] if len(consonant_phoneme) > 2 else "s"
        else:
            # General conjunct decomposition: C1 + C2 + ...
            parts = clean_base.split(HASANTA)
            mapped_parts = [CONSONANT_MAP.get(p, p) for p in parts]
            consonant_phoneme = "".join(mapped_parts)

        phonemes.append(consonant_phoneme)

        # 3. Add vowel component
        if vowel_part:
            phonemes.append(vowel_part)
        elif modifiers_part and not has_hasanta_end:
            # Consonant with modifier (e.g. রং, সং) carries inherent vowel
            phonemes.append("o")
        elif not has_hasanta_end and idx < num_g - 1:
            # Medial consonant without explicit matra has inherent vowel
            # In Bengali, inherent vowel is 'o' (or 'a')
            phonemes.append("o")
        elif not has_hasanta_end and idx == num_g - 1:
            # Word-final consonant: optionally inherent vowel if preceded by certain patterns
            pass

        if modifiers_part:
            phonemes.append(modifiers_part)

    return "".join(phonemes)
