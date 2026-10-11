"""Top-K document retrieval algorithms: Exhaustive DAAT and WAND dynamic pruning.

Implements:
1. daat_top_k(): Document-At-A-Time exhaustive retrieval using a min-heap.
2. wand_top_k(): Weak AND (WAND) dynamic pruning with block skip-pointers.
Both algorithms guarantee mathematically identical top-k result sets.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader
from bsearch.index.postings import BlockPostingIterator
from bsearch.rank.bm25 import BM25Params, BM25Scorer, score_term_bm25


@dataclass(frozen=True)
class ScoredDoc:
    score: float
    doc_id: int

    def __lt__(self, other: ScoredDoc) -> bool:
        # Min-heap ordering: lower score is smaller.
        # On tie, higher doc_id is smaller (so smaller doc_id stays in top-k).
        if abs(self.score - other.score) < 1e-9:
            return self.doc_id > other.doc_id
        return self.score < other.score


class QueryTermStream:
    """Encapsulates a query term's posting iterator, IDF, and score upper-bound."""

    __slots__ = ("term", "iterator", "idf", "weight", "upper_bound", "doc_id")

    def __init__(
        self,
        term: str,
        iterator: BlockPostingIterator,
        idf: float,
        weight: float,
        max_score: float,
    ) -> None:
        self.term = term
        self.iterator = iterator
        self.idf = idf
        self.weight = weight
        # Upper bound score contribution: max_score * query_weight
        self.upper_bound = max_score * weight
        self.doc_id: int | None = iterator.doc_id

    def next_doc(self) -> int | None:
        self.doc_id = self.iterator.next_doc()
        return self.doc_id

    def advance(self, target: int) -> int | None:
        self.doc_id = self.iterator.advance(target)
        return self.doc_id

    def __lt__(self, other: QueryTermStream) -> bool:
        if self.doc_id is None:
            return False
        if other.doc_id is None:
            return True
        return self.doc_id < other.doc_id


def daat_top_k(
    query_terms: list[str],
    index: InvertedIndexReader,
    docstore: DocStoreReader,
    k: int = 10,
    params: BM25Params | None = None,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Exhaustive Document-At-A-Time (DAAT) top-k retrieval using min-heap.

    Returns:
        List of (doc_id, score) sorted descending by score.
    """
    if not query_terms or k <= 0:
        return []

    p = params or BM25Params()
    scorer = BM25Scorer(num_docs=index.num_docs, avgdl=index.avgdl, params=p)
    term_weights = weights or [1.0] * len(query_terms)

    # Initialize streams for terms that exist in index
    streams: list[QueryTermStream] = []
    for term, w in zip(query_terms, term_weights):
        it = index.get_iterator(term)
        if it and it.doc_id is not None:
            df = index.get_df(term)
            idf = scorer.idf(df)
            max_s = index.get_term_max_score(term)
            streams.append(QueryTermStream(term, it, idf, w, max_s))

    if not streams:
        return []

    heap: list[ScoredDoc] = []

    while True:
        # Find minimum doc_id across active streams
        min_doc = None
        for s in streams:
            doc = s.doc_id
            if doc is not None and (min_doc is None or doc < min_doc):
                min_doc = doc

        if min_doc is None:
            break

        # Accumulate score for min_doc
        total_score = 0.0
        doc_len = docstore.get_doc_len(min_doc)

        for s in streams:
            if s.doc_id == min_doc:
                term_score = score_term_bm25(
                    body_tf=s.iterator.body_tf,
                    title_tf=s.iterator.title_tf,
                    doc_len=doc_len,
                    avgdl=index.avgdl,
                    idf=s.idf,
                    params=p,
                )
                total_score += s.weight * term_score
                s.next_doc()

        if total_score > 0.0:
            item = ScoredDoc(score=total_score, doc_id=min_doc)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    # Sort descending
    result = [(item.doc_id, item.score) for item in sorted(heap, reverse=True)]
    return result


def wand_top_k(
    query_terms: list[str],
    index: InvertedIndexReader,
    docstore: DocStoreReader,
    k: int = 10,
    params: BM25Params | None = None,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Weak AND (WAND) top-k retrieval with block-skip acceleration.

    Returns:
        List of (doc_id, score) sorted descending by score.
    """
    if not query_terms or k <= 0:
        return []

    p = params or BM25Params()
    scorer = BM25Scorer(num_docs=index.num_docs, avgdl=index.avgdl, params=p)
    term_weights = weights or [1.0] * len(query_terms)

    # Initialize active streams
    streams: list[QueryTermStream] = []
    for term, w in zip(query_terms, term_weights):
        it = index.get_iterator(term)
        if it and it.doc_id is not None:
            df = index.get_df(term)
            idf = scorer.idf(df)
            max_s = index.get_term_max_score(term)
            streams.append(QueryTermStream(term, it, idf, w, max_s))

    if not streams:
        return []

    heap: list[ScoredDoc] = []
    threshold = 0.0

    while True:
        # Remove exhausted streams
        streams = [s for s in streams if s.doc_id is not None]
        if not streams:
            break

        # Sort streams by current doc_id ascending
        streams.sort(key=lambda s: s.doc_id)  # type: ignore[arg-type]

        # Find pivot term p where accumulated upper bound > threshold
        accumulated_bound = 0.0
        pivot_idx = -1
        for i, s in enumerate(streams):
            accumulated_bound += s.upper_bound
            if accumulated_bound > threshold:
                pivot_idx = i
                break

        # If even all remaining terms cannot exceed threshold, search terminates
        if pivot_idx == -1:
            break

        pivot_doc = streams[pivot_idx].doc_id
        lead_doc = streams[0].doc_id

        if lead_doc == pivot_doc:
            # Full evaluation of pivot_doc
            total_score = 0.0
            doc_len = docstore.get_doc_len(pivot_doc)  # type: ignore[arg-type]

            for s in streams:
                if s.doc_id == pivot_doc:
                    term_score = score_term_bm25(
                        body_tf=s.iterator.body_tf,
                        title_tf=s.iterator.title_tf,
                        doc_len=doc_len,
                        avgdl=index.avgdl,
                        idf=s.idf,
                        params=p,
                    )
                    total_score += s.weight * term_score
                    s.next_doc()

            if total_score > threshold:
                item = ScoredDoc(score=total_score, doc_id=pivot_doc)  # type: ignore[arg-type]
                if len(heap) < k:
                    heapq.heappush(heap, item)
                    if len(heap) == k:
                        threshold = heap[0].score
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
                    threshold = heap[0].score
        else:
            # Skip lead stream forward to at least pivot_doc
            streams[0].advance(pivot_doc)  # type: ignore[arg-type]

    result = [(item.doc_id, item.score) for item in sorted(heap, reverse=True)]
    return result
