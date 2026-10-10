"""Rules report generator for reviewing Roman-to-Bangla phonetic mapping tables.

Prints a comprehensive inspection report detailing:
- Roman pattern coverage (rules/roman_patterns.tsv).
- G2P rules for Bengali orthography to phonemes.
- K1 (fine), K2 (medium), K3 (skeleton) key transformations.
- Ambiguity resolution demonstration for 'শিক্ষা' vs 'শিখা'.
- Trigram and Damerau-Levenshtein fuzzy matching thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bsearch.phonetic.g2p import bangla_to_phonemes
from bsearch.phonetic.index import PhoneticIndex
from bsearch.phonetic.keys import to_k1, to_k2, to_k3
from bsearch.phonetic.parser import RomanPatternParser, roman_to_phoneme_candidates


def generate_rules_report(rules_path: Path | str | None = None) -> str:
    """Generate human-readable rules and mapping tables report."""
    parser = RomanPatternParser(rules_path=rules_path)
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  {title}")
        lines.append("=" * 80)

    section("1. ROMAN PATTERNS COVERAGE (rules/roman_patterns.tsv)")
    lines.append(f"Source file: {parser.rules_path}")
    lines.append(f"Total pattern rules: {len(parser.patterns)}")
    lines.append("")
    lines.append(f"{'Pattern':<12} {'Targets':<24} {'Rule Type / Notes':<36}")
    lines.append("-" * 76)

    covered_required = [
        "kkh", "chh", "sh", "ch", "kh", "gh", "ng", "gg",
        "ee", "oo", "oi", "ou", "w", "y", "z", "j", "f", "ph",
        "T", "D", "R", "S", "N",
    ]

    for pat, targets in parser.patterns:
        marker = " [REQUIRED]" if pat in covered_required else ""
        tgt_str = ",".join(targets)
        lines.append(f"{pat:<12} {tgt_str:<24} {marker:<36}")

    section("2. G2P-LITE MAPPINGS & PHONOLOGICAL CONJUNCT HANDLING")
    lines.append("Grapheme-aware Bengali G2P converts orthographic Bengali into canonical phonemes:")
    sample_bangla = [
        ("শিক্ষা", "Medial ক্ষ -> kkh, inherent/matra handling"),
        ("শিখা", "Single aspirated খ -> kh"),
        ("ক্ষতি", "Initial ক্ষ -> kh"),
        ("জ্ঞান", "Initial জ্ঞ -> g"),
        ("বিজ্ঞান", "Medial জ্ঞ -> gg"),
        ("ঢাকা", "Retroflex ঢ -> Dh"),
        ("বাংলাদেশ", "Nasal modifier ং -> ng, sibilant শ -> sh"),
        ("আত্মা", "Conjunct ত্ম -> tt"),
        ("স্বাধীন", "Conjunct স্ব -> sh"),
    ]
    lines.append(f"{'Bangla Word':<16} {'Phonemes (G2P)':<20} {'Phonological Rule':<40}")
    lines.append("-" * 76)
    for w, note in sample_bangla:
        ph = bangla_to_phonemes(w)
        lines.append(f"{w:<16} {ph:<20} {note:<40}")

    section("3. THREE-LEVEL PHONETIC KEY TRANSFORMATIONS (K1, K2, K3)")
    lines.append("K1 (Fine):     Exact phonetics, standardized lowercase, retroflex hints lowercased.")
    lines.append("K2 (Medium):   Aspiration & gemination collapsed, s/sh merged, j/z merged, a/o merged.")
    lines.append("K3 (Skeleton): Consonant skeleton (vowels stripped, collapsed consonants).")
    lines.append("")
    lines.append(f"{'Bangla Term':<16} {'Phonemes':<16} {'K1 (Fine)':<16} {'K2 (Medium)':<14} {'K3 (Skeleton)':<12}")
    lines.append("-" * 76)
    for w, _ in sample_bangla:
        ph = bangla_to_phonemes(w)
        k1 = to_k1(ph)
        k2 = to_k2(ph)
        k3 = to_k3(ph)
        lines.append(f"{w:<16} {ph:<16} {k1:<16} {k2:<14} {k3:<12}")

    section("4. AMBIGUITY RESOLUTION & TEST VALIDATION")
    test_vocab = ["শিক্ষা", "শিখা", "ঢাকা", "বাংলাদেশ", "জ্ঞান"]
    idx = PhoneticIndex(parser=parser)
    idx.build_from_terms(test_vocab)

    test_queries = ["shikkha", "sikkha", "shikha", "dhaka", "Daka", "shikhka"]
    lines.append(f"Indexed Vocabulary: {', '.join(test_vocab)}")
    lines.append("")
    lines.append(f"{'Roman Query':<14} {'Candidates (up to 8)':<28} {'Resolved Bangla Terms (Score)':<36}")
    lines.append("-" * 80)
    for q in test_queries:
        cands = parser.parse_to_candidates(q, max_candidates=8)
        cands_preview = ",".join(cands[:3]) + (f" (+{len(cands)-3})" if len(cands) > 3 else "")
        results = idx.lookup(q)
        res_str = ", ".join(f"{term} ({score:.2f})" for term, score in results[:3])
        lines.append(f"{q:<14} {cands_preview:<28} {res_str:<36}")

    section("5. FUZZY MATCHING SPECIFICATION")
    lines.append("- Character-trigram index: Over all indexed K2 keys with '$$' edge padding.")
    lines.append("- Distance metric: Hand-written Damerau-Levenshtein (insert, delete, substitute, transpose).")
    lines.append("- Adaptive cutoff: max_distance <= 1 for keys <= 5 chars, max_distance <= 2 for keys > 5 chars.")
    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint to print rules report to standard output."""
    # Ensure UTF-8 output on Windows terminals
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    report_text = generate_rules_report()
    print(report_text)


if __name__ == "__main__":
    main()
