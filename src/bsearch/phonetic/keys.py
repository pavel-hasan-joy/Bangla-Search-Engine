"""Phonetic key generators for Bengali search: K1, K2, and K3.

Three key levels:
- K1 (fine): Lowercased, standardized fine phonetics.
- K2 (medium): Aspiration and gemination collapsed, s/sh merged to s,
  j/z merged to j, a/o merged to a, ee->i, oo->u.
- K3 (coarse): Consonant skeleton with vowels stripped and collapsed consonants.
"""

from __future__ import annotations

import re

# Aspirated consonant transformations for K2
_ASPIRATION_MAP: list[tuple[str, str]] = [
    ("kkh", "k"),
    ("cch", "c"),
    ("chh", "c"),
    ("ch", "c"),
    ("kh", "k"),
    ("gh", "g"),
    ("jh", "j"),
    ("th", "t"),
    ("dh", "d"),
    ("ph", "p"),
    ("bh", "b"),
    ("rh", "r"),
]

# Retroflex map for K1
_RETROFLEX_MAP: dict[str, str] = {
    "T": "t",
    "D": "d",
    "R": "r",
    "S": "sh",
    "N": "n",
}

# Vowel stripping set for K3
_VOWELS = set("aeiou")


def to_k1(phonemes: str) -> str:
    """Generate K1 fine phonetic key.

    - Converts retroflex hints (T, D, R, S, N) to standardized lowercase representations.
    - Normalizes common vowel length variants (ee -> i, oo -> u).
    - Preserves aspiration (kh != k), gemination (kk != k), and sibilant distinctions.
    """
    if not phonemes:
        return ""

    s = phonemes
    # Replace retroflex hints with lowercase equivalents
    for hint, target in _RETROFLEX_MAP.items():
        s = s.replace(hint, target)

    s = s.lower()
    s = s.replace("ee", "i").replace("oo", "u")
    # Clean any non-ascii letters
    s = re.sub(r"[^a-z]", "", s)
    return s


def to_k2(phonemes: str) -> str:
    """Generate K2 medium phonetic key.

    Collapses:
    - Aspiration: kh->k, gh->g, ch/chh->c, jh->j, th->t, dh->d, ph/f->p, bh/v->b
    - Gemination: kk->k, tt->t, pp->p, etc.
    - Sibilants: sh/s -> s
    - Affricates/Fricatives: j/z -> j
    - Vowels: a/o -> a, ee->i, oo->u
    """
    if not phonemes:
        return ""

    s = to_k1(phonemes)

    # 1. Sibilants & affricates
    s = s.replace("sh", "s")
    s = s.replace("z", "j")
    s = s.replace("f", "p")
    s = s.replace("v", "b")

    # 2. Aspirations
    for asp, base in _ASPIRATION_MAP:
        s = s.replace(asp, base)

    # 3. Vowels: a/o merged to a
    s = s.replace("o", "a")

    # 4. Gemination collapsed (repeated characters collapsed to 1)
    s = re.sub(r"([a-z])\1+", r"\1", s)

    return s


def to_k3(phonemes: str) -> str:
    """Generate K3 consonant skeleton.

    Builds on K2, strips all vowel characters (a, e, i, o, u),
    and collapses any adjacent identical consonants.
    """
    if not phonemes:
        return ""

    k2_val = to_k2(phonemes)
    # Strip vowels
    consonants = "".join(ch for ch in k2_val if ch not in _VOWELS)
    # Collapse adjacent duplicates
    return re.sub(r"([a-z])\1+", r"\1", consonants)
