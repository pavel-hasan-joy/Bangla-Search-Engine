"""BM25 scoring function with configurable parameters and title boosting.

Implements Okapi BM25 scoring with:
- Configurable k1 (default 1.2) and b (default 0.75)
- Document length normalization via avgdl
- Robertson-Spärck Jones IDF (positive floor)
- Configurable title frequency boost
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.2
    b: float = 0.75
    title_boost: float = 2.0


def compute_idf(num_docs: int, df: int) -> float:
    """Compute smoothed, non-negative BM25 IDF.

    IDF(t) = ln(1.0 + (N - df + 0.5) / (df + 0.5))
    """
    if num_docs <= 0 or df <= 0:
        return 0.0
    return math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))


def score_term_bm25(
    body_tf: int,
    title_tf: int,
    doc_len: int,
    avgdl: float,
    idf: float,
    params: BM25Params | None = None,
) -> float:
    """Compute BM25 score for a single term in a document.

    Args:
        body_tf: Term occurrences in body text.
        title_tf: Term occurrences in document title.
        doc_len: Total token count of the document.
        avgdl: Average document length across corpus.
        idf: Inverse document frequency for the term.
        params: BM25Params(k1, b, title_boost).

    Returns:
        Calculated term relevance score.
    """
    if idf <= 0.0:
        return 0.0

    p = params or BM25Params()

    # Combine body and boosted title term frequencies
    eff_tf = float(body_tf) + p.title_boost * float(title_tf)
    if eff_tf <= 0.0:
        return 0.0

    # Length normalization factor
    len_norm = (1.0 - p.b) + p.b * (float(doc_len) / avgdl if avgdl > 0 else 1.0)
    tf_component = (eff_tf * (p.k1 + 1.0)) / (eff_tf + p.k1 * len_norm)

    return idf * tf_component


class BM25Scorer:
    """Scorer coordinating corpus statistics and BM25 scoring."""

    def __init__(
        self,
        num_docs: int,
        avgdl: float,
        params: BM25Params | None = None,
    ) -> None:
        self.num_docs = num_docs
        self.avgdl = avgdl
        self.params = params or BM25Params()

    def idf(self, df: int) -> float:
        return compute_idf(self.num_docs, df)

    def score_term(
        self,
        body_tf: int,
        title_tf: int,
        doc_len: int,
        df: int,
    ) -> float:
        return score_term_bm25(
            body_tf=body_tf,
            title_tf=title_tf,
            doc_len=doc_len,
            avgdl=self.avgdl,
            idf=self.idf(df),
            params=self.params,
        )
