"""Evaluation system implementations: B0, B1, B2, B3, and Oracle.

Systems:
- B0: Raw Roman string vs Bangla text (searches raw Roman tokens directly with plain BM25).
- B1: Static 1-to-1 transliteration + plain BM25 (naive baseline, deterministic fixed map).
- B2: Ours without fuzzy (K1, K2 exact, K3 skeleton, candidate combiner).
- B3: Full system (K1, K2, Fuzzy K2 with similarity scaling, K3, combiners).
- Oracle: Plain BM25 using human gold native Bengali query.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final

from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader
from bsearch.phonetic.index import PhoneticIndex
from bsearch.rank.bm25 import BM25Params
from bsearch.rank.expand import ExpansionConfig
from bsearch.rank.searcher import CombinerType, SearchConfig, Searcher
from bsearch.rank.topk import daat_top_k
from bsearch.unicode.tokenizer import tokenize

# Static 1-to-1 transliteration table for naive B1 baseline (greedy longest-match)
_STATIC_TRANSLIT_RULES: Final[list[tuple[str, str]]] = [
    ("kkh", "ক্ষ"),
    ("chh", "ছ"),
    ("sh", "শ"),
    ("ch", "চ"),
    ("kh", "খ"),
    ("gh", "ঘ"),
    ("jh", "ঝ"),
    ("th", "থ"),
    ("dh", "ধ"),
    ("ph", "ফ"),
    ("bh", "ভ"),
    ("ng", "ং"),
    ("ee", "ী"),
    ("oo", "ূ"),
    ("oi", "ৈ"),
    ("ou", "ৌ"),
    ("k", "ক"),
    ("g", "গ"),
    ("j", "জ"),
    ("T", "ট"),
    ("D", "ড"),
    ("R", "ড়"),
    ("t", "ত"),
    ("d", "দ"),
    ("n", "ন"),
    ("p", "প"),
    ("f", "ফ"),
    ("b", "ব"),
    ("v", "ভ"),
    ("m", "ম"),
    ("r", "র"),
    ("l", "ল"),
    ("s", "স"),
    ("h", "হ"),
    ("z", "জ"),
    ("y", "য়"),
    ("w", "ও"),
    ("a", "া"),
    ("i", "ি"),
    ("u", "ু"),
    ("e", "ে"),
    ("o", "ো"),
]


def static_transliterate(text: str) -> str:
    """Deterministic naive 1-to-1 character/chunk transliteration for B1 baseline."""
    n = len(text)
    i = 0
    out: list[str] = []

    while i < n:
        matched = False
        for pat, replacement in _STATIC_TRANSLIT_RULES:
            pat_len = len(pat)
            if i + pat_len <= n:
                sub = text[i : i + pat_len]
                if pat in ("T", "D", "R"):
                    if sub == pat:
                        out.append(replacement)
                        i += pat_len
                        matched = True
                        break
                else:
                    if sub.lower() == pat.lower():
                        out.append(replacement)
                        i += pat_len
                        matched = True
                        break
        if not matched:
            out.append(text[i])
            i += 1

    return "".join(out)


class BaseSystem(ABC):
    """Abstract base class for evaluation systems."""

    def __init__(
        self,
        index_reader: InvertedIndexReader,
        docstore: DocStoreReader,
        bm25_params: BM25Params | None = None,
    ) -> None:
        self.index_reader = index_reader
        self.docstore = docstore
        self.bm25_params = bm25_params or BM25Params()

    @abstractmethod
    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Retrieve ranked document IDs and scores for a query."""
        ...


class B0System(BaseSystem):
    """B0: Raw Roman string searched directly against inverted index with plain BM25."""

    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        tokens = tokenize(query, stem=False)
        return daat_top_k(
            query_terms=tokens,
            index=self.index_reader,
            docstore=self.docstore,
            k=top_k,
            params=self.bm25_params,
        )


class B1System(BaseSystem):
    """B1: Static 1-to-1 transliteration + plain BM25 (naive baseline)."""

    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        translit_text = static_transliterate(query)
        tokens = tokenize(translit_text, stem=False)
        return daat_top_k(
            query_terms=tokens,
            index=self.index_reader,
            docstore=self.docstore,
            k=top_k,
            params=self.bm25_params,
        )


class B2System(BaseSystem):
    """B2: Ours without fuzzy (K1, K2 exact, K3 skeleton with combiner)."""

    def __init__(
        self,
        index_reader: InvertedIndexReader,
        docstore: DocStoreReader,
        phonetic_index: PhoneticIndex | None = None,
        bm25_params: BM25Params | None = None,
    ) -> None:
        super().__init__(index_reader, docstore, bm25_params)
        cfg = SearchConfig(
            combiner=CombinerType.SATURATING_SUM,
            bm25=self.bm25_params,
            expansion=ExpansionConfig(weight_fuzzy=0.0),  # Fuzzy turned off
        )
        self.searcher = Searcher(
            index_reader=self.index_reader,
            docstore=self.docstore,
            phonetic_index=phonetic_index,
            config=cfg,
        )

    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        res = self.searcher.search(query, top_k=top_k, explain=False)
        return [(r.doc_id, r.score) for r in res.results]


class B3System(BaseSystem):
    """B3: Full system (K1, K2, Fuzzy K2 with similarity scaling, K3, combiners)."""

    def __init__(
        self,
        index_reader: InvertedIndexReader,
        docstore: DocStoreReader,
        phonetic_index: PhoneticIndex | None = None,
        bm25_params: BM25Params | None = None,
    ) -> None:
        super().__init__(index_reader, docstore, bm25_params)
        cfg = SearchConfig(
            combiner=CombinerType.SATURATING_SUM,
            bm25=self.bm25_params,
            expansion=ExpansionConfig(weight_fuzzy=0.4),
        )
        self.searcher = Searcher(
            index_reader=self.index_reader,
            docstore=self.docstore,
            phonetic_index=phonetic_index,
            config=cfg,
        )

    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        res = self.searcher.search(query, top_k=top_k, explain=False)
        return [(r.doc_id, r.score) for r in res.results]


class OracleSystem(BaseSystem):
    """Oracle: Plain BM25 using ground-truth native Bengali query."""

    def retrieve(
        self,
        query: str,
        gold_bangla_query: str = "",
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        target_query = gold_bangla_query.strip() or query.strip()
        tokens = tokenize(target_query, stem=False)
        return daat_top_k(
            query_terms=tokens,
            index=self.index_reader,
            docstore=self.docstore,
            k=top_k,
            params=self.bm25_params,
        )


def build_systems(
    index_reader: InvertedIndexReader,
    docstore: DocStoreReader,
    phonetic_index: PhoneticIndex | None = None,
    bm25_params: BM25Params | None = None,
) -> dict[str, BaseSystem]:
    """Factory creating all 5 evaluation systems."""
    p_idx = phonetic_index or PhoneticIndex()
    if phonetic_index is None:
        p_idx.build_from_terms(index_reader.terms.keys())

    return {
        "B0": B0System(index_reader, docstore, bm25_params),
        "B1": B1System(index_reader, docstore, bm25_params),
        "B2": B2System(index_reader, docstore, p_idx, bm25_params),
        "B3": B3System(index_reader, docstore, p_idx, bm25_params),
        "Oracle": OracleSystem(index_reader, docstore, bm25_params),
    }
