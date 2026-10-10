"""Unicode normalization, character canonicalization, grapheme segmentation, and tokenization."""

from bsearch.unicode.constants import (
    BANGLA_DIGITS,
    CONSONANTS,
    DANDA,
    DOUBLE_DANDA,
    HASANTA,
    INDEPENDENT_VOWELS,
    MODIFIERS,
    NUKTA,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
)
from bsearch.unicode.grapheme import segment_graphemes
from bsearch.unicode.normalizer import (
    canonicalize_nukta,
    compose_two_part_vowels,
    convert_digits_to_ascii,
    normalize,
    repair_misordered_vowels,
)
from bsearch.unicode.stemmer import BanglaStemmer, get_default_stemmer, stem_word
from bsearch.unicode.tokenizer import tokenize

__all__ = [
    "BANGLA_DIGITS",
    "CONSONANTS",
    "DANDA",
    "DOUBLE_DANDA",
    "HASANTA",
    "INDEPENDENT_VOWELS",
    "MODIFIERS",
    "NUKTA",
    "VOWEL_SIGNS",
    "ZWJ",
    "ZWNJ",
    "segment_graphemes",
    "canonicalize_nukta",
    "compose_two_part_vowels",
    "convert_digits_to_ascii",
    "normalize",
    "repair_misordered_vowels",
    "BanglaStemmer",
    "get_default_stemmer",
    "stem_word",
    "tokenize",
]
