"""Query expansion and candidate weighting for Bengali and Banglish queries.

Builds weighted candidates {(bangla_term, weight)} per query token:
- K1 exact: ~1.0 (exact fine phonetic)
- K2 exact: ~0.7 (aspiration/gemination/sibilant/vowel collapsed)
- Fuzzy K2: ~0.4 scaled by normalized edit similarity
- K3 skeleton: ~0.2 (consonant skeleton)
- Native Bangla tokens: pass through directly with weight 1.0

Applies:
- Configurable weights via ExpansionConfig.
- Candidate capping (~20 candidates per token).
- Document-frequency (DF) prior for corpus-aware likelihood scoring.
- Comprehensive explanation details per token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Protocol

from bsearch.phonetic.index import PhoneticIndex
from bsearch.phonetic.keys import to_k1, to_k2, to_k3
from bsearch.phonetic.parser import RomanPatternParser, get_default_parser


class DFProvider(Protocol):
    """Protocol for querying term document frequency."""

    def get_df(self, term: str) -> int:
        ...


@dataclass(frozen=True)
class ExpansionConfig:
    """Configuration parameters for candidate expansion."""

    weight_k1: float = 1.0
    weight_k2: float = 0.7
    weight_fuzzy: float = 0.4
    weight_k3: float = 0.2
    max_candidates_per_token: int = 20
    df_prior_alpha: float = 0.15
    corpus_size_fallback: int = 100_000


@dataclass(frozen=True)
class TermCandidate:
    """Represents an expanded candidate Bangla term with its score components."""

    term: str
    weight: float
    base_weight: float
    level: str  # EXACT_BANGLA, K1_EXACT, K2_EXACT, K2_FUZZY, K3_SKELETON
    df: int = 0
    similarity: float = 1.0
    edit_distance: int = 0


@dataclass(frozen=True)
class TokenExpansion:
    """The full candidate expansion for a single query token."""

    token: str
    is_bangla: bool
    candidates: list[TermCandidate]

    def get_candidate_weights(self) -> dict[str, float]:
        """Convenience map of term -> final weight."""
        return {c.term: c.weight for c in self.candidates}


def is_bangla_word(token: str) -> bool:
    """Check if token contains Bengali script characters (Unicode block U+0980 - U+09FF)."""
    return any(0x0980 <= ord(ch) <= 0x09FF for ch in token)


class QueryExpander:
    """Expands query tokens into weighted Bengali candidate terms."""

    def __init__(
        self,
        phonetic_index: PhoneticIndex,
        df_provider: DFProvider | None = None,
        config: ExpansionConfig | None = None,
        parser: RomanPatternParser | None = None,
    ) -> None:
        self.phonetic_index = phonetic_index
        self.df_provider = df_provider
        self.config = config or ExpansionConfig()
        self.parser = parser or get_default_parser()

    def expand_token(self, token: str) -> TokenExpansion:
        """Expand a single query token into weighted candidate Bangla terms."""
        clean_token = token.strip()
        if not clean_token:
            return TokenExpansion(token=token, is_bangla=False, candidates=[])

        # 1. Native Bangla token pass-through
        if is_bangla_word(clean_token):
            df = self.df_provider.get_df(clean_token) if self.df_provider else 1
            cand = TermCandidate(
                term=clean_token,
                weight=1.0,
                base_weight=1.0,
                level="EXACT_BANGLA",
                df=df,
                similarity=1.0,
                edit_distance=0,
            )
            return TokenExpansion(token=clean_token, is_bangla=True, candidates=[cand])

        # 2. Roman / Banglish token expansion
        cfg = self.config
        roman_candidates = self.parser.parse_to_candidates(clean_token, max_candidates=8)

        # Map term -> dict of best candidate properties
        best_matches: dict[str, dict] = {}

        for ph in roman_candidates:
            q_k1 = to_k1(ph)
            q_k2 = to_k2(ph)
            q_k3 = to_k3(ph)

            # --- Tier 1: K1 Exact (~1.0) ---
            if q_k1 and q_k1 in self.phonetic_index.k1_to_terms:
                for term in self.phonetic_index.k1_to_terms[q_k1]:
                    self._update_candidate(
                        best_matches,
                        term=term,
                        base_weight=cfg.weight_k1,
                        level="K1_EXACT",
                        similarity=1.0,
                        dist=0,
                    )

            # --- Tier 2: K2 Exact (~0.7) ---
            if q_k2 and q_k2 in self.phonetic_index.k2_to_terms:
                for term in self.phonetic_index.k2_to_terms[q_k2]:
                    self._update_candidate(
                        best_matches,
                        term=term,
                        base_weight=cfg.weight_k2,
                        level="K2_EXACT",
                        similarity=1.0,
                        dist=0,
                    )

            # --- Tier 3: Fuzzy K2 (~0.4 scaled by similarity) ---
            if q_k2:
                fuzzy_k2s = self.phonetic_index.get_fuzzy_k2_candidates(q_k2)
                for cand_k2, dist in fuzzy_k2s:
                    if dist > 0:
                        denom = max(len(q_k2), len(cand_k2))
                        sim = max(0.0, 1.0 - (dist / denom if denom > 0 else 0.0))
                        fuzzy_weight = cfg.weight_fuzzy * sim
                        for term in self.phonetic_index.k2_to_terms[cand_k2]:
                            self._update_candidate(
                                best_matches,
                                term=term,
                                base_weight=fuzzy_weight,
                                level="K2_FUZZY",
                                similarity=sim,
                                dist=dist,
                            )

            # --- Tier 4: K3 Skeleton (~0.2) ---
            if q_k3 and q_k3 in self.phonetic_index.k3_to_terms:
                for term in self.phonetic_index.k3_to_terms[q_k3]:
                    self._update_candidate(
                        best_matches,
                        term=term,
                        base_weight=cfg.weight_k3,
                        level="K3_SKELETON",
                        similarity=0.5,
                        dist=0,
                    )

        # 3. Apply Document-Frequency (DF) prior and prune candidates
        candidates: list[TermCandidate] = []
        n_docs = cfg.corpus_size_fallback
        log_n = math.log(n_docs + 1.0)

        for term, data in best_matches.items():
            df = self.df_provider.get_df(term) if self.df_provider else 1
            # Corpus-frequency prior: log-smoothed DF factor
            # Prior ranges from 1.0 (df=0) to (1.0 + alpha) (df=N)
            prior = 1.0 + cfg.df_prior_alpha * (math.log(df + 1.0) / log_n if log_n > 0 else 0.0)
            final_weight = min(1.0, data["base_weight"] * prior)

            candidates.append(
                TermCandidate(
                    term=term,
                    weight=round(final_weight, 4),
                    base_weight=round(data["base_weight"], 4),
                    level=data["level"],
                    df=df,
                    similarity=round(data["similarity"], 4),
                    edit_distance=data["dist"],
                )
            )

        # Sort descending by final weight, breaking ties with DF then base_weight
        candidates.sort(key=lambda c: (c.weight, c.df, c.base_weight), reverse=True)

        # Cap candidates per token
        pruned_candidates = candidates[: cfg.max_candidates_per_token]

        return TokenExpansion(
            token=clean_token,
            is_bangla=False,
            candidates=pruned_candidates,
        )

    @staticmethod
    def _update_candidate(
        matches: dict[str, dict],
        term: str,
        base_weight: float,
        level: str,
        similarity: float,
        dist: int,
    ) -> None:
        """Update or insert candidate term, keeping the highest base weight."""
        if term not in matches or base_weight > matches[term]["base_weight"]:
            matches[term] = {
                "base_weight": base_weight,
                "level": level,
                "similarity": similarity,
                "dist": dist,
            }
