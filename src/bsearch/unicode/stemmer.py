"""Rule-driven inflectional stemmer for Bengali words.

Driven by editable configuration in rules/suffixes.tsv.
Strips common Bengali nominal and pronominal inflectional suffixes
(e.g. গুলো, গুলি, দের, কে, রা, ের, র, তে, টি, টা) while enforcing
minimum stem length constraints to protect base roots.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger("bsearch.unicode.stemmer")

DEFAULT_RULES_PATH: Final = (
    Path(__file__).resolve().parent.parent.parent.parent / "rules" / "suffixes.tsv"
)


@dataclass(frozen=True)
class SuffixRule:
    suffix: str
    min_stem_len: int
    description: str


class BanglaStemmer:
    """Configurable suffix-stripping stemmer for Bengali text."""

    def __init__(self, rules_path: Path | str | None = None) -> None:
        self.rules_path = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        self.rules: list[SuffixRule] = self._load_rules()

    def _load_rules(self) -> list[SuffixRule]:
        """Load suffix rules from TSV file, sorting by length descending."""
        if not self.rules_path.is_file():
            logger.warning(
                "Suffixes rule file not found at %s. Initializing empty stemmer.",
                self.rules_path,
            )
            return []

        rules: list[SuffixRule] = []
        with open(self.rules_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if not row or not row[0].strip() or row[0].startswith("#") or row[0] == "suffix":
                    continue
                suffix = row[0].strip()
                min_stem_len = int(row[1].strip()) if len(row) > 1 and row[1].strip().isdigit() else 2
                desc = row[2].strip() if len(row) > 2 else ""
                rules.append(SuffixRule(suffix=suffix, min_stem_len=min_stem_len, description=desc))

        # Sort by suffix length descending so that compound affixes match first
        rules.sort(key=lambda r: len(r.suffix), reverse=True)
        return rules

    def stem(self, word: str, enabled: bool = True) -> str:
        """Strip inflectional suffix if enabled and conditions are met.

        Args:
            word: Bengali word token.
            enabled: Whether stemming is active (behind a flag).

        Returns:
            Stemmed word or original word if disabled / no rule matched.
        """
        if not enabled or not word:
            return word

        for rule in self.rules:
            if word.endswith(rule.suffix):
                potential_stem = word[: -len(rule.suffix)]
                if len(potential_stem) >= rule.min_stem_len:
                    return potential_stem

        return word


# Module-level default instance
_default_stemmer: BanglaStemmer | None = None


def get_default_stemmer() -> BanglaStemmer:
    global _default_stemmer
    if _default_stemmer is None:
        _default_stemmer = BanglaStemmer()
    return _default_stemmer


def stem_word(word: str, enabled: bool = True) -> str:
    """Convenience function to stem a single Bengali word using default rules."""
    return get_default_stemmer().stem(word, enabled=enabled)
