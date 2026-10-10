"""Phonetic Index for Bengali Search.

Maintains multi-level phonetic mappings:
- K1 (fine): Exact phonetic representation.
- K2 (medium): Collapsed aspiration, gemination, sibilants, and vowels.
- K3 (skeleton): Consonant skeleton.

Provides:
- Inverted mapping: key -> list of Bangla vocabulary terms.
- Character-trigram inverted index over K2 keys.
- Fast candidate retrieval using Damerau-Levenshtein distance
  (distance <= 1 for keys <= 5 chars, else <= 2).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Final

from bsearch.phonetic.distance import damerau_levenshtein
from bsearch.phonetic.g2p import bangla_to_phonemes
from bsearch.phonetic.keys import to_k1, to_k2, to_k3
from bsearch.phonetic.parser import (
    RomanPatternParser,
    get_default_parser,
    roman_to_phoneme_candidates,
)

TRIGRAM_PAD: Final = "$$"


def extract_trigrams(text: str) -> set[str]:
    """Extract character trigrams with start/end padding."""
    if not text:
        return set()
    padded = f"{TRIGRAM_PAD}{text}{TRIGRAM_PAD}"
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


class PhoneticIndex:
    """Multi-level phonetic dictionary and fuzzy search index."""

    def __init__(self, parser: RomanPatternParser | None = None) -> None:
        self.parser = parser or get_default_parser()

        # Key -> list of Bangla terms
        self.k1_to_terms: dict[str, list[str]] = defaultdict(list)
        self.k2_to_terms: dict[str, list[str]] = defaultdict(list)
        self.k3_to_terms: dict[str, list[str]] = defaultdict(list)

        # Term -> (k1, k2, k3)
        self.term_to_keys: dict[str, tuple[str, str, str]] = {}

        # Trigram index over K2 keys: trigram -> set of k2_keys
        self.k2_trigram_index: dict[str, set[str]] = defaultdict(set)
        self.k2_keys: set[str] = set()

    def add_term(self, term: str) -> tuple[str, str, str]:
        """Index a single Bengali vocabulary term."""
        if not term or term in self.term_to_keys:
            return self.term_to_keys.get(term, ("", "", ""))

        phonemes = bangla_to_phonemes(term)
        k1 = to_k1(phonemes)
        k2 = to_k2(phonemes)
        k3 = to_k3(phonemes)

        if k1:
            self.k1_to_terms[k1].append(term)
        if k2:
            self.k2_to_terms[k2].append(term)
            if k2 not in self.k2_keys:
                self.k2_keys.add(k2)
                for tg in extract_trigrams(k2):
                    self.k2_trigram_index[tg].add(k2)
        if k3:
            self.k3_to_terms[k3].append(term)

        keys = (k1, k2, k3)
        self.term_to_keys[term] = keys
        return keys

    def build_from_terms(self, terms: Iterable[str]) -> int:
        """Batch index vocabulary terms. Returns number of indexed terms."""
        count = 0
        for term in terms:
            if term:
                self.add_term(term)
                count += 1
        return count

    def get_fuzzy_k2_candidates(self, query_k2: str) -> list[tuple[str, int]]:
        """Retrieve matching K2 keys using trigram filtering and Damerau-Levenshtein.

        Rule: distance <= 1 for keys <= 5 chars, else <= 2.
        """
        if not query_k2:
            return []

        q_len = len(query_k2)
        max_dist = 1 if q_len <= 5 else 2

        # 1. Exact match fast-path
        results: dict[str, int] = {}
        if query_k2 in self.k2_keys:
            results[query_k2] = 0

        # 2. Gather candidate K2 keys via trigram overlap
        q_trigrams = extract_trigrams(query_k2)
        candidate_k2_keys: set[str] = set()
        for tg in q_trigrams:
            candidate_k2_keys.update(self.k2_trigram_index.get(tg, ()))

        # 3. Verify candidates with Damerau-Levenshtein
        for cand in candidate_k2_keys:
            if cand in results:
                continue
            # Quick length difference filter
            if abs(len(cand) - q_len) > max_dist:
                continue

            dist = damerau_levenshtein(query_k2, cand)
            if dist <= max_dist:
                results[cand] = dist

        return sorted(results.items(), key=lambda x: x[1])

    def lookup(
        self,
        roman_token: str,
        max_candidates: int = 20,
        include_fuzzy: bool = True,
    ) -> list[tuple[str, float]]:
        """Resolve a Roman/Banglish token to candidate Bengali terms with relevance weights.

        Scoring tiers:
        - Exact K1 match: 1.00
        - Exact K2 match: 0.85
        - Fuzzy K2 match (dist 1): 0.70
        - Fuzzy K2 match (dist 2): 0.55
        - K3 skeleton match: 0.40
        """
        if not roman_token:
            return []

        phoneme_candidates = self.parser.parse_to_candidates(roman_token, max_candidates=8)
        term_scores: dict[str, float] = defaultdict(float)

        for ph in phoneme_candidates:
            q_k1 = to_k1(ph)
            q_k2 = to_k2(ph)
            q_k3 = to_k3(ph)

            # 1. K1 matches
            if q_k1 in self.k1_to_terms:
                for term in self.k1_to_terms[q_k1]:
                    term_scores[term] = max(term_scores[term], 1.00)

            # 2. K2 matches
            if q_k2 in self.k2_to_terms:
                for term in self.k2_to_terms[q_k2]:
                    term_scores[term] = max(term_scores[term], 0.85)

            # 3. Fuzzy K2 matches
            if include_fuzzy and q_k2:
                fuzzy_k2s = self.get_fuzzy_k2_candidates(q_k2)
                for cand_k2, dist in fuzzy_k2s:
                    score = 0.85 if dist == 0 else (0.70 if dist == 1 else 0.55)
                    for term in self.k2_to_terms[cand_k2]:
                        term_scores[term] = max(term_scores[term], score)

            # 4. K3 matches (only if no high-confidence K1/K2 match found yet)
            if q_k3 in self.k3_to_terms:
                for term in self.k3_to_terms[q_k3]:
                    if term not in term_scores:
                        term_scores[term] = 0.40

        # Sort by score descending
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:max_candidates]
