"""Search execution engine with Banglish-to-Bangla expansion and configurable combiners.

Coordinates:
- Query tokenization.
- Query expansion into weighted candidate Bangla terms (K1, K2, Fuzzy, K3, Bangla pass-through).
- Document retrieval using InvertedIndexReader and DocStoreReader.
- Document-At-A-Time (DAAT) candidate scoring.
- Selectable combiners:
  1. 'max': Best candidate per token position.
  2. 'saturating_sum': Sum with diminishing returns per token position.
- Comprehensive explain mode detailing matched key levels, term frequencies, and scores.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader
from bsearch.index.postings import BlockPostingIterator
from bsearch.phonetic.index import PhoneticIndex
from bsearch.rank.bm25 import BM25Params, BM25Scorer, score_term_bm25
from bsearch.rank.expand import (
    ExpansionConfig,
    QueryExpander,
    TermCandidate,
    TokenExpansion,
)
from bsearch.rank.topk import ScoredDoc
from bsearch.unicode.tokenizer import tokenize


class CombinerType(str, Enum):
    """Combiner strategy for multiple candidate terms per query token."""

    MAX = "max"
    SATURATING_SUM = "saturating_sum"


@dataclass(frozen=True)
class SearchConfig:
    """Complete search configuration."""

    combiner: CombinerType = CombinerType.SATURATING_SUM
    saturating_lambda: float = 0.1
    bm25: BM25Params = field(default_factory=BM25Params)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    default_top_k: int = 10


@dataclass(frozen=True)
class TermMatchDetail:
    """Score breakdown for a single matched term in a document."""

    term: str
    body_tf: int
    title_tf: int
    idf: float
    raw_bm25: float
    candidate_weight: float
    weighted_bm25: float


@dataclass(frozen=True)
class SearchExplanation:
    """Full ranking and expansion audit trail for a search result."""

    query_tokens: list[str]
    expansions: list[TokenExpansion]
    token_scores: dict[str, float]
    matched_terms: list[TermMatchDetail]
    combiner: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked search result with document metadata and optional explanation."""

    rank: int
    doc_id: int
    score: float
    title: str
    snippet: str
    explanation: SearchExplanation | None = None


@dataclass(frozen=True)
class SearchResponse:
    """Response envelope for a search execution."""

    query: str
    tokens: list[str]
    expansions: list[TokenExpansion]
    total_hits: int
    results: list[SearchResult]


def apply_saturating_sum(scores: list[float], lambda_param: float = 0.1) -> float:
    """Saturating sum with diminishing returns: S / (1 + lambda * S) * (1 + lambda)."""
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0.0:
        return 0.0
    return (total / (1.0 + lambda_param * total)) * (1.0 + lambda_param)


class Searcher:
    """Coordinates phonetic expansion, inverted index traversal, and ranking."""

    def __init__(
        self,
        index_reader: InvertedIndexReader,
        docstore: DocStoreReader,
        phonetic_index: PhoneticIndex | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self.index_reader = index_reader
        self.docstore = docstore
        self.config = config or SearchConfig()

        # Build or attach phonetic index
        if phonetic_index is not None:
            self.phonetic_index = phonetic_index
        else:
            self.phonetic_index = PhoneticIndex()
            self.phonetic_index.build_from_terms(self.index_reader.terms.keys())

        # Initialize query expander
        self.expander = QueryExpander(
            phonetic_index=self.phonetic_index,
            df_provider=self.index_reader,
            config=self.config.expansion,
        )

        # Initialize BM25 scorer
        self.scorer = BM25Scorer(
            num_docs=self.index_reader.num_docs,
            avgdl=self.index_reader.avgdl,
            params=self.config.bm25,
        )

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path | str,
        config: SearchConfig | None = None,
        phonetic_index: PhoneticIndex | None = None,
    ) -> Searcher:
        """Construct Searcher by loading index and docstore from directory."""
        p = Path(index_dir)
        index_reader = InvertedIndexReader(p)
        docstore = DocStoreReader(p)
        return cls(
            index_reader=index_reader,
            docstore=docstore,
            phonetic_index=phonetic_index,
            config=config,
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        explain: bool = False,
    ) -> SearchResponse:
        """Execute search query with Banglish expansion, BM25 scoring, and ranking."""
        k = top_k if top_k is not None else self.config.default_top_k
        raw_tokens = tokenize(query)
        if not raw_tokens or k <= 0:
            return SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=[],
                total_hits=0,
                results=[],
            )

        # 1. Expand query tokens into weighted candidates
        expansions: list[TokenExpansion] = [
            self.expander.expand_token(tok) for tok in raw_tokens
        ]

        # 2. Gather unique candidate terms that exist in the index
        # Map: term -> list of (token_idx, weight)
        term_to_token_refs: dict[str, list[tuple[int, float]]] = {}
        unique_terms: set[str] = set()

        for token_idx, exp in enumerate(expansions):
            for cand in exp.candidates:
                if cand.term in self.index_reader.terms:
                    unique_terms.add(cand.term)
                    if cand.term not in term_to_token_refs:
                        term_to_token_refs[cand.term] = []
                    term_to_token_refs[cand.term].append((token_idx, cand.weight))

        if not unique_terms:
            return SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=expansions,
                total_hits=0,
                results=[],
            )

        # 3. Initialize postings iterators for active candidate terms
        # term -> (iterator, idf)
        streams: dict[str, tuple[BlockPostingIterator, float]] = {}
        for term in unique_terms:
            it = self.index_reader.get_iterator(term)
            if it and it.doc_id is not None:
                idf = self.scorer.idf(self.index_reader.get_df(term))
                streams[term] = (it, idf)

        if not streams:
            return SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=expansions,
                total_hits=0,
                results=[],
            )

        # 4. Document-At-A-Time (DAAT) evaluation with candidate combiner
        heap: list[ScoredDoc] = []
        total_hits = 0
        doc_details: dict[int, dict[str, Any]] = {} if explain else {}

        p = self.config.bm25
        avgdl = self.index_reader.avgdl
        combiner = self.config.combiner
        saturating_lambda = self.config.saturating_lambda
        num_tokens = len(expansions)

        while True:
            # Find next smallest doc_id across active streams
            min_doc: int | None = None
            for term, (it, _) in streams.items():
                cur_doc = it.doc_id
                if cur_doc is not None and (min_doc is None or cur_doc < min_doc):
                    min_doc = cur_doc

            if min_doc is None:
                break

            doc_len = self.docstore.get_doc_len(min_doc)

            # Compute term BM25 scores for terms present in min_doc
            term_scores: dict[str, tuple[float, int, int, float]] = {}
            for term, (it, idf) in streams.items():
                if it.doc_id == min_doc:
                    b_tf = it.body_tf
                    t_tf = it.title_tf
                    raw_bm25 = score_term_bm25(
                        body_tf=b_tf,
                        title_tf=t_tf,
                        doc_len=doc_len,
                        avgdl=avgdl,
                        idf=idf,
                        params=p,
                    )
                    term_scores[term] = (raw_bm25, b_tf, t_tf, idf)
                    it.next_doc()

            # Aggregate scores per token position
            token_scores_list: list[float] = [0.0] * num_tokens
            token_breakdown: dict[str, float] = {}
            match_details: list[TermMatchDetail] = []

            for token_idx in range(num_tokens):
                tok_str = raw_tokens[token_idx]
                token_candidate_scores: list[float] = []

                for cand in expansions[token_idx].candidates:
                    if cand.term in term_scores:
                        raw_bm25, b_tf, t_tf, idf = term_scores[cand.term]
                        w_score = cand.weight * raw_bm25
                        token_candidate_scores.append(w_score)

                        if explain:
                            match_details.append(
                                TermMatchDetail(
                                    term=cand.term,
                                    body_tf=b_tf,
                                    title_tf=t_tf,
                                    idf=round(idf, 4),
                                    raw_bm25=round(raw_bm25, 4),
                                    candidate_weight=cand.weight,
                                    weighted_bm25=round(w_score, 4),
                                )
                            )

                # Combine candidate scores for this token position
                if combiner == CombinerType.MAX:
                    tok_score = max(token_candidate_scores) if token_candidate_scores else 0.0
                else:  # CombinerType.SATURATING_SUM
                    tok_score = apply_saturating_sum(
                        token_candidate_scores, lambda_param=saturating_lambda
                    )

                token_scores_list[token_idx] = tok_score
                if explain:
                    token_breakdown[tok_str] = round(tok_score, 4)

            total_doc_score = sum(token_scores_list)

            if total_doc_score > 0.0:
                total_hits += 1
                item = ScoredDoc(score=total_doc_score, doc_id=min_doc)

                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)

                if explain:
                    doc_details[min_doc] = {
                        "token_scores": token_breakdown,
                        "matched_terms": match_details,
                    }

        # 5. Build ranked result set
        sorted_docs = sorted(heap, reverse=True)
        results: list[SearchResult] = []

        for rank_idx, item in enumerate(sorted_docs, start=1):
            doc_data = self.docstore.get(item.doc_id)
            title = doc_data.get("title", f"Doc #{item.doc_id}") if doc_data else f"Doc #{item.doc_id}"
            snippet = doc_data.get("snippet", "") if doc_data else ""
            if not snippet and doc_data:
                snippet = doc_data.get("text", "")[:200]

            explanation_obj: SearchExplanation | None = None
            if explain and item.doc_id in doc_details:
                details = doc_details[item.doc_id]
                explanation_obj = SearchExplanation(
                    query_tokens=raw_tokens,
                    expansions=expansions,
                    token_scores=details["token_scores"],
                    matched_terms=details["matched_terms"],
                    combiner=combiner.value,
                )

            results.append(
                SearchResult(
                    rank=rank_idx,
                    doc_id=item.doc_id,
                    score=round(item.score, 4),
                    title=title,
                    snippet=snippet,
                    explanation=explanation_obj,
                )
            )

        return SearchResponse(
            query=query,
            tokens=raw_tokens,
            expansions=expansions,
            total_hits=total_hits,
            results=results,
        )
