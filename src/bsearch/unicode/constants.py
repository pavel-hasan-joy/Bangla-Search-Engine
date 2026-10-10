"""Constants and Unicode code-point ranges for Bengali script and symbols."""

from __future__ import annotations

# Zero Width Formatting Controls
ZWNJ = "\u200C"  # Zero Width Non-Joiner
ZWJ = "\u200D"   # Zero Width Joiner

# Bengali Punctuation
DANDA = "\u0964"         # । (Dari)
DOUBLE_DANDA = "\u0965"  # ॥ (Double Dari)

# Bengali Consonant Code Points (0x0995 - 0x09B9, plus 0x09DC-0x09DD, 0x09DF, 0x09CE)
CONSONANTS = frozenset(
    chr(cp)
    for cp in list(range(0x0995, 0x09A9))  # ক to ন
    + list(range(0x09AA, 0x09B1))         # প to র
    + [0x09B2, 0x09B6, 0x09B7, 0x09B8, 0x09B9]  # ল, শ, ষ, স, হ
    + [0x09DC, 0x09DD, 0x09DF]            # ড়, ঢ়, য়
    + [0x09CE]                            # ৎ (Khanda Ta)
)

# Independent Vowels
INDEPENDENT_VOWELS = frozenset(
    chr(cp)
    for cp in [
        0x0985, 0x0986, 0x0987, 0x0988, 0x0989, 0x098A,  # অ, আ, ই, ঈ, উ, ঊ
        0x098B, 0x098C, 0x098F, 0x0990, 0x0993, 0x0994   # ঋ, ঌ, এ, ঐ, ও, ঔ
    ]
)

# Dependent Vowel Signs (Matras)
VOWEL_SIGNS = frozenset(
    chr(cp)
    for cp in [
        0x09BE, 0x09BF, 0x09C0, 0x09C1, 0x09C2, 0x09C3, 0x09C4,  # া, ি, ী, ু, ূ, ৃ, ৄ
        0x09C7, 0x09C8, 0x09CB, 0x09CC, 0x09D7                   # ে, ৈ, ো, ৌ, ৗ
    ]
)

# Pre-base Vowel Signs (visually rendered before consonant, but phonetically typed after)
PRE_BASE_VOWEL_SIGNS = frozenset(["\u09BF", "\u09C7", "\u09C8"])  # ি, ে, ৈ

# Virama / Hasanta
HASANTA = "\u09CD"  # ্

# Nukta
NUKTA = "\u09BC"  # ়

# Modifiers (Anusvara, Visarga, Candrabindu)
MODIFIERS = frozenset([
    "\u0981",  # ঁ Candrabindu
    "\u0982",  # ং Anusvara
    "\u0983",  # ঃ Visarga
])

# Digits
BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
BANGLA_TO_ASCII_DIGITS = dict(zip(BANGLA_DIGITS, ASCII_DIGITS))
ASCII_TO_BANGLA_DIGITS = dict(zip(ASCII_DIGITS, BANGLA_DIGITS))

# Canonical forms for decomposed characters with Nukta
NUKTA_CANONICAL_MAP = {
    "\u09AF\u09BC": "\u09DF",  # য + ় -> য়
    "\u09A1\u09BC": "\u09DC",  # ড + ় -> ড়
    "\u09A2\u09BC": "\u09DD",  # ঢ + ় -> ঢ়
}

# Two-part vowel sign compositions
TWO_PART_VOWEL_MAP = {
    "\u09C7\u09BE": "\u09CB",  # ে + া -> ো
    "\u09C7\u09D7": "\u09CC",  # ে + ৗ -> ৌ
}
