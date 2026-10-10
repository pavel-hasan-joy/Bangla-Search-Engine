"""Greedy longest-match Romanization parser for Banglish to phoneme candidates.

Driven by editable rules in rules/roman_patterns.tsv.
Expands ambiguous phoneme segments up to 8 candidates.
Respects capital T, D, R, S, N as optional retroflex hints.
"""

from __future__ import annotations

import csv
import itertools
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger("bsearch.phonetic.parser")

DEFAULT_PATTERNS_PATH: Final = (
    Path(__file__).resolve().parent.parent.parent.parent / "rules" / "roman_patterns.tsv"
)


class RomanPatternParser:
    """Parses Roman/Banglish tokens into phoneme candidate sequences using greedy longest-match."""

    def __init__(self, rules_path: Path | str | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path else DEFAULT_PATTERNS_PATH
        self.patterns: list[tuple[str, list[str]]] = self._load_patterns()

    def _load_patterns(self) -> list[tuple[str, list[str]]]:
        if not self.rules_path.is_file():
            logger.warning("Roman patterns TSV not found at %s", self.rules_path)
            return []

        rules: list[tuple[str, list[str]]] = []
        with open(self.rules_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or not row[0].strip() or row[0].startswith("#") or row[0] == "pattern":
                    continue
                pattern = row[0].strip()
                targets = [t.strip() for t in row[1].split(",") if t.strip()] if len(row) > 1 else [pattern]
                rules.append((pattern, targets))

        # Sort patterns by length descending to enforce greedy longest match
        rules.sort(key=lambda r: len(r[0]), reverse=True)
        return rules

    def parse_to_candidates(self, token: str, max_candidates: int = 8) -> list[str]:
        """Parse Roman token into up to max_candidates phoneme string representations."""
        if not token:
            return []

        n = len(token)
        i = 0
        segment_options: list[list[str]] = []

        while i < n:
            matched = False
            for pat, targets in self.patterns:
                pat_len = len(pat)
                if i + pat_len <= n:
                    sub = token[i : i + pat_len]
                    # Direct match (case-sensitive to respect T, D, R, S, N hints, or lowercase match)
                    if sub == pat:
                        segment_options.append(targets)
                        i += pat_len
                        matched = True
                        break
                    elif sub.lower() == pat.lower() and not any(c in "TDRSN" for c in pat):
                        segment_options.append(targets)
                        i += pat_len
                        matched = True
                        break

            if not matched:
                # Fallback: single character
                char = token[i]
                segment_options.append([char.lower()])
                i += 1

        # Form Cartesian product of segment choices
        # Limit branching to prevent combinatorial explosion before product
        all_combinations: list[str] = []
        for combo in itertools.product(*segment_options):
            all_combinations.append("".join(combo))
            if len(all_combinations) >= max_candidates:
                break

        return all_combinations


# Module-level default parser
_default_parser: RomanPatternParser | None = None


def get_default_parser() -> RomanPatternParser:
    global _default_parser
    if _default_parser is None:
        _default_parser = RomanPatternParser()
    return _default_parser


def roman_to_phoneme_candidates(token: str, max_candidates: int = 8) -> list[str]:
    """Convenience function to parse a Roman/Banglish token into phoneme candidates."""
    return get_default_parser().parse_to_candidates(token, max_candidates=max_candidates)
