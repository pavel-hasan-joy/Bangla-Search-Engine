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
from bsearch.rank.topk import QueryTermStream, ScoredDoc
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
        resp, _ = self._search_internal(query, top_k=top_k, explain=explain, track_timings=False)
        return resp

    def search_with_stage_timings(
        self,
        query: str,
        top_k: int | None = None,
    ) -> tuple[SearchResponse, dict[str, float]]:
        """Execute search query returning both response and nanosecond-level per-stage timings."""
        return self._search_internal(query, top_k=top_k, explain=False, track_timings=True)

    def _search_internal(
        self,
        query: str,
        top_k: int | None = None,
        explain: bool = False,
        track_timings: bool = False,
    ) -> tuple[SearchResponse, dict[str, float]]:
        """Internal search core with optional high-resolution stage timing instrumentation."""
        import time

        k = top_k if top_k is not None else self.config.default_top_k
        t0 = time.perf_counter()
        raw_tokens = tokenize(query)
        t1 = time.perf_counter()
        t_tokenize_ms = (t1 - t0) * 1000.0 if track_timings else 0.0

        if not raw_tokens or k <= 0:
            resp = SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=[],
                total_hits=0,
                results=[],
            )
            return resp, {
                "tokenize_ms": t_tokenize_ms,
                "expand_ms": 0.0,
                "score_ms": 0.0,
                "fetch_ms": 0.0,
                "total_ms": t_tokenize_ms,
            }

        # 1. Expand query tokens into weighted candidates
        t1 = time.perf_counter()
        expansions: list[TokenExpansion] = [
            self.expander.expand_token(tok) for tok in raw_tokens
        ]
        t2 = time.perf_counter()
        t_expand_ms = (t2 - t1) * 1000.0 if track_timings else 0.0

        # Precompute term-to-candidate token mappings and max weights
        term_to_cands: dict[str, list[tuple[int, float]]] = {}
        for token_idx, exp in enumerate(expansions):
            for cand in exp.candidates:
                if cand.term in self.index_reader.terms:
                    if cand.term not in term_to_cands:
                        term_to_cands[cand.term] = []
                    term_to_cands[cand.term].append((token_idx, cand.weight))

        if not term_to_cands:
            resp = SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=expansions,
                total_hits=0,
                results=[],
            )
            return resp, {
                "tokenize_ms": t_tokenize_ms,
                "expand_ms": t_expand_ms,
                "score_ms": 0.0,
                "fetch_ms": 0.0,
                "total_ms": t_tokenize_ms + t_expand_ms,
            }

        # 2. WAND (Weak AND) dynamic pruning with candidate combiners
        t2 = time.perf_counter()
        p = self.config.bm25
        avgdl = self.index_reader.avgdl
        combiner = self.config.combiner
        saturating_lambda = self.config.saturating_lambda
        num_tokens = len(expansions)

        # Initialize active term streams
        streams: list[QueryTermStream] = []
        for term, cands in term_to_cands.items():
            it = self.index_reader.get_iterator(term)
            if it and it.doc_id is not None:
                idf = self.scorer.idf(self.index_reader.get_df(term))
                max_w = max(w for _, w in cands)
                max_s = self.index_reader.get_term_max_score(term)
                ub = max_w * max_s
                if ub <= 0.0:
                    ub = max_w * idf * (p.k1 + 1.0)
                streams.append(QueryTermStream(term, it, idf, max_w, ub))

        if not streams:
            resp = SearchResponse(
                query=query,
                tokens=raw_tokens,
                expansions=expansions,
                total_hits=0,
                results=[],
            )
            return resp, {
                "tokenize_ms": t_tokenize_ms,
                "expand_ms": t_expand_ms,
                "score_ms": 0.0,
                "fetch_ms": 0.0,
                "total_ms": t_tokenize_ms + t_expand_ms,
            }

        heap: list[ScoredDoc] = []
        threshold = 0.0
        total_hits = 0
        doc_details: dict[int, dict[str, Any]] = {} if explain else {}

        # Precompute BM25 coefficients for sub-microsecond evaluation
        k1 = p.k1
        k1_plus_1 = k1 + 1.0
        one_minus_b = 1.0 - p.b
        b_over_avgdl = p.b / avgdl if avgdl > 0.0 else 0.0
        title_boost = p.title_boost

        # Cap to top 24 streams by upper bound to avoid processing low-weight terms
        if len(streams) > 24:
            streams.sort(key=lambda s: s.upper_bound, reverse=True)
            streams = streams[:24]

        # Initial sort using C-level __lt__ on QueryTermStream
        import bisect
        streams.sort()

        while streams:
            # Find pivot stream where accumulated upper bound > threshold
            accumulated_bound = 0.0
            pivot_idx = -1
            for i, s in enumerate(streams):
                accumulated_bound += s.upper_bound
                if accumulated_bound > threshold:
                    pivot_idx = i
                    break

            if pivot_idx == -1:
                # No remaining document can possibly exceed threshold
                break

            pivot_doc = streams[pivot_idx].doc_id
            lead_doc = streams[0].doc_id

            if lead_doc == pivot_doc:
                # Collect matching streams currently at pivot_doc
                matching_streams: list[QueryTermStream] = []
                match_ub = 0.0
                while streams and streams[0].doc_id == pivot_doc:
                    s = streams.pop(0)
                    matching_streams.append(s)
                    match_ub += s.upper_bound

                # Quick upper bound check on matching streams
                if match_ub > threshold:
                    doc_len = self.docstore.get_doc_len(pivot_doc)  # type: ignore[arg-type]
                    len_norm = one_minus_b + b_over_avgdl * doc_len
                    token_candidate_scores: list[list[float]] = [[] for _ in range(num_tokens)]

                    if explain:
                        term_scores: dict[str, tuple[float, int, int, float]] = {}
                        for s in matching_streams:
                            b_tf = s.iterator.body_tf
                            t_tf = s.iterator.title_tf
                            eff_tf = b_tf + title_boost * t_tf
                            if eff_tf > 0.0:
                                tf_comp = (eff_tf * k1_plus_1) / (eff_tf + k1 * len_norm)
                                raw_bm25 = s.idf * tf_comp
                            else:
                                raw_bm25 = 0.0
                            term_scores[s.term] = (raw_bm25, b_tf, t_tf, s.idf)

                        match_details: list[TermMatchDetail] = []
                        for s_term, (raw_bm25, b_tf, t_tf, idf_val) in term_scores.items():
                            for tok_idx, w in term_to_cands[s_term]:
                                w_score = w * raw_bm25
                                token_candidate_scores[tok_idx].append(w_score)
                                match_details.append(
                                    TermMatchDetail(
                                        term=s_term,
                                        body_tf=b_tf,
                                        title_tf=t_tf,
                                        idf=round(idf_val, 4),
                                        raw_bm25=round(raw_bm25, 4),
                                        candidate_weight=w,
                                        weighted_bm25=round(w_score, 4),
                                    )
                                )
                    else:
                        for s in matching_streams:
                            b_tf = s.iterator.body_tf
                            t_tf = s.iterator.title_tf
                            eff_tf = b_tf + title_boost * t_tf
                            if eff_tf > 0.0:
                                raw_bm25 = s.idf * ((eff_tf * k1_plus_1) / (eff_tf + k1 * len_norm))
                                for tok_idx, w in term_to_cands[s.term]:
                                    token_candidate_scores[tok_idx].append(w * raw_bm25)

                    # Combine scores across query tokens
                    total_doc_score = 0.0
                    token_breakdown: dict[str, float] = {}

                    for tok_idx in range(num_tokens):
                        scores_for_tok = token_candidate_scores[tok_idx]
                        if not scores_for_tok:
                            tok_score = 0.0
                        elif combiner == CombinerType.MAX:
                            tok_score = max(scores_for_tok)
                        else:  # CombinerType.SATURATING_SUM
                            tok_score = apply_saturating_sum(scores_for_tok, saturating_lambda)
                        total_doc_score += tok_score
                        if explain:
                            token_breakdown[raw_tokens[tok_idx]] = round(tok_score, 4)

                    if total_doc_score > 0.0:
                        total_hits += 1

                    if total_doc_score > threshold:
                        item = ScoredDoc(score=total_doc_score, doc_id=pivot_doc)  # type: ignore[arg-type]
                        if len(heap) < k:
                            heapq.heappush(heap, item)
                            if len(heap) == k:
                                threshold = heap[0].score
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)
                            threshold = heap[0].score

                        if explain:
                            doc_details[pivot_doc] = {  # type: ignore[index]
                                "token_scores": token_breakdown,
                                "matched_terms": match_details,
                            }

                # Advance matching streams and reinsert using binary search in C
                for s in matching_streams:
                    s.next_doc()
                    if s.doc_id is not None:
                        bisect.insort(streams, s)
            else:
                # Advance lead stream to pivot_doc and reinsert in O(log N)
                lead_s = streams.pop(0)
                lead_s.advance(pivot_doc)  # type: ignore[arg-type]
                if lead_s.doc_id is not None:
                    bisect.insort(streams, lead_s)

        t3 = time.perf_counter()
        t_score_ms = (t3 - t2) * 1000.0 if track_timings else 0.0

        # 3. Build ranked result set
        t3 = time.perf_counter()
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

        t4 = time.perf_counter()
        t_fetch_ms = (t4 - t3) * 1000.0 if track_timings else 0.0
        t_total_ms = (t4 - t0) * 1000.0 if track_timings else 0.0

        resp = SearchResponse(
            query=query,
            tokens=raw_tokens,
            expansions=expansions,
            total_hits=total_hits,
            results=results,
        )

        return resp, {
            "tokenize_ms": t_tokenize_ms,
            "expand_ms": t_expand_ms,
            "score_ms": t_score_ms,
            "fetch_ms": t_fetch_ms,
            "total_ms": t_total_ms,
        }
