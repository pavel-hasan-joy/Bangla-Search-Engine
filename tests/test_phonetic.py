"""Tests for src/bsearch/phonetic/ module.

Covers:
- Damerau-Levenshtein distance (insertions, deletions, substitutions, transpositions).
- Greedy Roman pattern parsing with editable rules (rules/roman_patterns.tsv).
- Grapheme-aware Bengali G2P-lite (initial vs medial ক্ষ, জ্ঞ, sibilants, matras).
- Multi-level phonetic keys: K1 (fine), K2 (medium), K3 (consonant skeleton).
- Trigram index and adaptive fuzzy matching on K2 keys.
- Core ambiguity resolution: shikkha, sikkha, shikha all reach শিক্ষা; shikha reaches শিখা.
- Rules report generation.
"""

from __future__ import annotations

import pytest

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


# =====================================================================
# 1. Damerau-Levenshtein Tests
# =====================================================================


def test_damerau_levenshtein_identity_and_empty():
    assert damerau_levenshtein("", "") == 0
    assert damerau_levenshtein("a", "") == 1
    assert damerau_levenshtein("", "abc") == 3
    assert damerau_levenshtein("hello", "hello") == 0


def test_damerau_levenshtein_single_edits():
    # Insertion
    assert damerau_levenshtein("cat", "cats") == 1
    # Deletion
    assert damerau_levenshtein("cats", "cat") == 1
    # Substitution
    assert damerau_levenshtein("cat", "cot") == 1
    # Transposition (adjacent swap)
    assert damerau_levenshtein("ab", "ba") == 1
    assert damerau_levenshtein("sika", "saki") == 2
    assert damerau_levenshtein("shika", "sihka") == 1


def test_damerau_levenshtein_classic_benchmarks():
    assert damerau_levenshtein("kitten", "sitting") == 3
    assert damerau_levenshtein("saturday", "sunday") == 3
    assert damerau_levenshtein("ca", "abc") == 2  # delete c, transpose ab


# =====================================================================
# 2. Roman Pattern Parser Tests
# =====================================================================


def test_roman_parser_covers_required_patterns():
    parser = get_default_parser()
    required = [
        "sh", "s", "ch", "chh", "kh", "gh", "ng", "kkh", "gg",
        "ee", "oo", "oi", "ou", "w", "y", "z", "j", "f", "ph",
        "T", "D", "R", "S", "N",
    ]
    patterns_in_rules = {pat for pat, _ in parser.patterns}
    for req in required:
        assert req in patterns_in_rules, f"Pattern {req} missing from roman_patterns.tsv"


def test_roman_parser_greedy_longest_match():
    parser = get_default_parser()
    # 'kkh' should be parsed as kkh, not k + kh or k + k + h
    cands_kkh = parser.parse_to_candidates("kkh")
    assert any("kkh" in c for c in cands_kkh)

    # 'chh' should be parsed as chh, not ch + h
    cands_chh = parser.parse_to_candidates("chh")
    assert any("chh" in c for c in cands_chh)


def test_roman_parser_candidate_cap():
    # Long ambiguous query should not explode combinatorially beyond max_candidates
    cands = roman_to_phoneme_candidates("shikkhasikkha", max_candidates=8)
    assert len(cands) <= 8
    assert len(cands) > 0


def test_roman_parser_retroflex_hints():
    # Capital T vs lowercase t
    cands_upper_t = roman_to_phoneme_candidates("Taka")
    assert any("T" in c for c in cands_upper_t)

    # Capital D vs lowercase d
    cands_upper_d = roman_to_phoneme_candidates("Daka")
    assert any("D" in c for c in cands_upper_d)


# =====================================================================
# 3. Grapheme-Aware G2P Tests
# =====================================================================


def test_g2p_khiyo_conjunct_initial_vs_medial():
    # Word-initial ক্ষ sounds like kh
    assert bangla_to_phonemes("ক্ষতি") == "khoti"
    # Medial/final ক্ষ sounds like kkh
    assert bangla_to_phonemes("শিক্ষা") == "shikkha"


def test_g2p_gyan_conjunct_initial_vs_medial():
    # Word-initial জ্ঞ sounds like g
    assert bangla_to_phonemes("জ্ঞান") == "gan"
    # Medial জ্ঞ sounds like gg
    assert bangla_to_phonemes("বিজ্ঞান") == "biggan"


def test_g2p_sibilants_and_aspirations():
    assert bangla_to_phonemes("শিখা") == "shikha"
    assert bangla_to_phonemes("ঢাকা") == "Dhaka"
    assert bangla_to_phonemes("বাংলাদেশ") == "bangladesh"


def test_g2p_matras_and_modifiers():
    # Vowel signs
    assert bangla_to_phonemes("নদী") == "nodi"
    assert bangla_to_phonemes("ফুল") == "phul"
    # Nasal modifier anusvara ং -> ng
    assert bangla_to_phonemes("রং") == "rong"


# =====================================================================
# 4. Multi-Level Phonetic Keys Tests (K1, K2, K3)
# =====================================================================


def test_keys_k1_fine():
    assert to_k1("shikkha") == "shikkha"
    assert to_k1("shikha") == "shikha"
    assert to_k1("Dhaka") == "dhaka"
    assert to_k1("Taka") == "taka"


def test_keys_k2_medium():
    # Aspiration collapsed: kh->k, kkh->k
    # Sibilant merged: sh->s
    # a/o merged: o->a
    # Gemination collapsed: kk->k
    assert to_k2("shikkha") == "sika"
    assert to_k2("shikha") == "sika"
    assert to_k2("sikkha") == "sika"
    assert to_k2("sikha") == "sika"

    # j/z merged
    assert to_k2("zila") == to_k2("jila")
    # Aspiration in stops
    assert to_k2("Dhaka") == "daka"
    assert to_k2("daka") == "daka"
    assert to_k2("ghora") == "gara"


def test_keys_k3_consonant_skeleton():
    # Vowels stripped and consonants collapsed
    assert to_k3("shikkha") == "sk"
    assert to_k3("shikha") == "sk"
    assert to_k3("dhaka") == "dk"
    assert to_k3("bangladesh") == "bnglds"


# =====================================================================
# 5. Core Ambiguity Resolution Tests (shikkha, sikkha, shikha)
# =====================================================================


def test_core_ambiguity_resolution_shikkha_sikkha_shikha():
    index = PhoneticIndex()
    index.build_from_terms(["শিক্ষা", "শিখা", "ঢাকা", "বাংলাদেশ"])

    # 1. shikkha must reach শিক্ষা
    res_shikkha = [t for t, _ in index.lookup("shikkha")]
    assert "শিক্ষা" in res_shikkha, "shikkha must reach শিক্ষা"

    # 2. sikkha must reach শিক্ষা
    res_sikkha = [t for t, _ in index.lookup("sikkha")]
    assert "শিক্ষা" in res_sikkha, "sikkha must reach শিক্ষা"

    # 3. shikha must reach শিক্ষা (via K2) AND shikha must also reach শিখা (via K1/K2)
    res_shikha = [t for t, _ in index.lookup("shikha")]
    assert "শিক্ষা" in res_shikha, "shikha must reach শিক্ষা"
    assert "শিখা" in res_shikha, "shikha must also reach শিখা"

    # Top match for shikha should be শিখা
    assert res_shikha[0] == "শিখা"


# =====================================================================
# 6. Trigram Index & Fuzzy Candidate Retrieval Tests
# =====================================================================


def test_trigram_extraction():
    tg = extract_trigrams("sika")
    assert "$$s" in tg
    assert "$si" in tg
    assert "sik" in tg
    assert "ika" in tg
    assert "ka$" in tg
    assert "a$$" in tg


def test_fuzzy_matching_with_typo():
    index = PhoneticIndex()
    index.build_from_terms(["শিক্ষা", "শিখা", "ঢাকা", "বাংলাদেশ"])

    # 1. Typo transposition in Banglish: 'shikhka' instead of 'shikkha'
    # K2 for shikhka collapses to sika (dist 0 or 1)
    res_typo = [t for t, _ in index.lookup("shikhka")]
    assert "শিক্ষা" in res_typo

    # 2. Typo in dhakka -> matches ঢাকা
    res_dhkka = [t for t, _ in index.lookup("dhkka")]
    assert "ঢাকা" in res_dhkka


def test_fuzzy_distance_length_threshold():
    index = PhoneticIndex()
    # Add a short term and a long term
    index.build_from_terms(["বই", "বিশ্ববিদ্যালয়"])

    # Query key <= 5 chars allows distance <= 1
    # Query key > 5 chars allows distance <= 2
    short_k2 = to_k2(bangla_to_phonemes("বই"))
    assert len(short_k2) <= 5

    cands_short = index.get_fuzzy_k2_candidates(short_k2)
    for _, dist in cands_short:
        assert dist <= 1


# =====================================================================
# 7. Rules Report Generation Test
# =====================================================================


def test_generate_rules_report():
    report = generate_rules_report()
    assert "1. ROMAN PATTERNS COVERAGE" in report
    assert "2. G2P-LITE MAPPINGS" in report
    assert "3. THREE-LEVEL PHONETIC KEY TRANSFORMATIONS" in report
    assert "4. AMBIGUITY RESOLUTION" in report
    assert "5. FUZZY MATCHING SPECIFICATION" in report
    assert "শিক্ষা" in report
    assert "shikkha" in report
