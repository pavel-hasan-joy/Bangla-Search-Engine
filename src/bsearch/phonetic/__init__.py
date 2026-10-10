"""Phonetic encoding, romanization normalization, and Banglish candidate generation."""

from bsearch.phonetic.distance import damerau_levenshtein
from bsearch.phonetic.g2p import bangla_to_phonemes
from bsearch.phonetic.index import PhoneticIndex, extract_trigrams
from bsearch.phonetic.keys import to_k1, to_k2, to_k3
from bsearch.phonetic.parser import (
    RomanPatternParser,
    get_default_parser,
    roman_to_phoneme_candidates,
)
from bsearch.phonetic.report import generate_rules_report

__all__ = [
    "PhoneticIndex",
    "RomanPatternParser",
    "bangla_to_phonemes",
    "damerau_levenshtein",
    "extract_trigrams",
    "generate_rules_report",
    "get_default_parser",
    "roman_to_phoneme_candidates",
    "to_k1",
    "to_k2",
    "to_k3",
]
