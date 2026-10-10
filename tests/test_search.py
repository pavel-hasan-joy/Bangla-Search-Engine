"""Tests for src/bsearch/rank/expand.py and src/bsearch/rank/searcher.py (Task 4).

Covers:
- Token expansion for native Bangla script (weight 1.0 pass-through).
- Token expansion for Banglish/Roman (K1 ~1.0, K2 ~0.7, Fuzzy ~0.4 * sim, K3 ~0.2).
- Candidate capping (~20 candidates) and Document-Frequency (DF) prior.
- Combiners: 'max' vs 'saturating_sum'.
- Explain mode audit trail.
- Full end-to-end Searcher retrieval on index.
"""

from __future__ import annotations

import pytest

from bsearch.index.builder import build_index_from_docs
from bsearch.phonetic.index import PhoneticIndex
from bsearch.rank.bm25 import BM25Params
from bsearch.rank.expand import (
    ExpansionConfig,
    QueryExpander,
    is_bangla_word,
)
from bsearch.rank.searcher import (
    CombinerType,
    SearchConfig,
    Searcher,
    apply_saturating_sum,
)


class MockDFProvider:
    """Mock document frequency provider."""

    def __init__(self, df_map: dict[str, int]) -> None:
        self.df_map = df_map

    def get_df(self, term: str) -> int:
        return self.df_map.get(term, 0)


# =====================================================================
# 1. Query Expander Tests
# =====================================================================


def test_is_bangla_word():
    assert is_bangla_word("শিক্ষা") is True
    assert is_bangla_word("ঢাকা") is True
    assert is_bangla_word("shikkha") is False
    assert is_bangla_word("dhaka") is False
    assert is_bangla_word("covid-19") is False


def test_bangla_token_pass_through():
    phonetic_idx = PhoneticIndex()
    expander = QueryExpander(phonetic_index=phonetic_idx)

    exp = expander.expand_token("শিক্ষা")
    assert exp.is_bangla is True
    assert len(exp.candidates) == 1
    c = exp.candidates[0]
    assert c.term == "শিক্ষা"
    assert c.weight == 1.0
    assert c.level == "EXACT_BANGLA"


def test_banglish_token_tiered_weights():
    phonetic_idx = PhoneticIndex()
    # Vocabulary contains:
    # 'শিক্ষা' (K1: shikkha, K2: sika)
    # 'শিখা' (K1: shikha, K2: sika)
    # 'শখ' (K1: shokh, K2: sak, K3: sk)
    phonetic_idx.build_from_terms(["শিক্ষা", "শিখা", "শখ"])

    cfg = ExpansionConfig(
        weight_k1=1.0,
        weight_k2=0.7,
        weight_fuzzy=0.4,
        weight_k3=0.2,
        df_prior_alpha=0.0,  # disable DF prior for pure base weight check
    )
    expander = QueryExpander(phonetic_index=phonetic_idx, config=cfg)

    # Query 'shikkha':
    # 'শিক্ষা' should match via K1_EXACT with weight 1.0
    # 'শিখা' should match via K2_EXACT with weight 0.7
    exp = expander.expand_token("shikkha")
    cand_map = {c.term: c for c in exp.candidates}

    assert "শিক্ষা" in cand_map
    assert cand_map["শিক্ষা"].level == "K1_EXACT"
    assert cand_map["শিক্ষা"].weight == 1.0

    assert "শিখা" in cand_map
    assert cand_map["শিখা"].level == "K2_EXACT"
    assert cand_map["শিখা"].weight == 0.7


def test_df_prior_prunes_and_boosts_frequent_terms():
    phonetic_idx = PhoneticIndex()
    # Two terms that have identical phonetic keys
    phonetic_idx.build_from_terms(["শিক্ষা", "শিখা"])

    # 'শিক্ষা' is common in corpus (df=500), 'শিখা' is rare (df=2)
    df_provider = MockDFProvider({"শিক্ষা": 500, "শিখা": 2})

    cfg = ExpansionConfig(
        weight_k1=0.8,
        weight_k2=0.8,
        df_prior_alpha=0.2,
        corpus_size_fallback=1000,
    )
    expander = QueryExpander(
        phonetic_index=phonetic_idx, df_provider=df_provider, config=cfg
    )

    exp = expander.expand_token("sika")
    assert exp.candidates[0].term == "শিক্ষা"
    assert exp.candidates[0].df == 500
    assert exp.candidates[0].weight > exp.candidates[1].weight


def test_candidate_capping():
    phonetic_idx = PhoneticIndex()
    # Add 30 terms
    terms = [f"শব্দ{i}" for i in range(30)]
    # All mapping to same K3 or similar
    phonetic_idx.build_from_terms(terms)

    cfg = ExpansionConfig(max_candidates_per_token=5)
    expander = QueryExpander(phonetic_index=phonetic_idx, config=cfg)
    exp = expander.expand_token("shobdo")

    assert len(exp.candidates) <= 5


# =====================================================================
# 2. Combiner Tests
# =====================================================================


def test_apply_saturating_sum():
    # Empty
    assert apply_saturating_sum([]) == 0.0

    # Single score
    s1 = apply_saturating_sum([2.0], lambda_param=0.1)
    # (2.0 / 1.2) * 1.1 = 1.833...
    assert 1.8 < s1 < 2.0

    # Two scores: diminishing returns
    s2 = apply_saturating_sum([2.0, 2.0], lambda_param=0.1)
    assert s2 > s1
    # Pure sum would be 4.0; saturated sum is less than 4.0
    assert s2 < 4.0


# =====================================================================
# 3. End-to-End Searcher Retrieval Tests
# =====================================================================


def test_searcher_retrieval_and_combiners(tmp_path):
    # Create small corpus
    docs = [
        {
            "id": "1",
            "title": "উচ্চ শিক্ষা ব্যবস্থা",
            "text": "বাংলাদেশে শিক্ষা বিস্তার ও বিশ্ববিদ্যালয়গুলোর মানোন্নয়ন।",
        },
        {
            "id": "2",
            "title": "প্রদীপের শিখা",
            "text": "অন্ধকার ঘরে উজ্জ্বল শিখা জ্বলছে।",
        },
        {
            "id": "3",
            "title": "খেলাধুলা ও বিনোদন",
            "text": "ফুটবল ও ক্রিকেট খেলা বাংলাদেশের মানুষের প্রিয় বিনোদন।",
        },
    ]

    index_dir = tmp_path / "test_idx"
    build_index_from_docs(docs, index_dir)

    # 1. Test search with Saturating Sum
    cfg_sat = SearchConfig(combiner=CombinerType.SATURATING_SUM)
    searcher_sat = Searcher.from_index_dir(index_dir, config=cfg_sat)

    # Search Banglish "shikkha"
    res_shikkha = searcher_sat.search("shikkha", top_k=5, explain=True)
    assert res_shikkha.total_hits > 0
    top_doc = res_shikkha.results[0]
    assert "শিক্ষা" in top_doc.title
    assert top_doc.doc_id == 0  # Doc 1 is index 0
    assert top_doc.explanation is not None
    assert top_doc.explanation.combiner == "saturating_sum"
    assert "shikkha" in top_doc.explanation.token_scores

    # 2. Test search with Max combiner
    cfg_max = SearchConfig(combiner=CombinerType.MAX)
    searcher_max = Searcher.from_index_dir(index_dir, config=cfg_max)
    res_max = searcher_max.search("shikkha", top_k=5, explain=True)
    assert res_max.total_hits > 0
    assert res_max.results[0].explanation.combiner == "max"

    # 3. Test native Bangla search "শিখা"
    res_bangla = searcher_sat.search("শিখা", top_k=5)
    assert res_bangla.total_hits > 0
    assert "শিখা" in res_bangla.results[0].title
    assert res_bangla.results[0].doc_id == 1  # Doc 2 is index 1


def test_explain_mode_contents(tmp_path):
    docs = [
        {"id": "1", "title": "ঢাকা বিশ্ববিদ্যালয়", "text": "শিক্ষা ও গবেষণার প্রাচীন কেন্দ্র।"}
    ]
    index_dir = tmp_path / "test_idx2"
    build_index_from_docs(docs, index_dir)

    searcher = Searcher.from_index_dir(index_dir)
    res = searcher.search("shikkha", top_k=1, explain=True)

    assert len(res.results) == 1
    expl = res.results[0].explanation
    assert expl is not None
    assert expl.query_tokens == ["shikkha"]
    assert len(expl.expansions) == 1
    assert any(c.term == "শিক্ষা" for c in expl.expansions[0].candidates)
    assert len(expl.matched_terms) >= 1
    m = expl.matched_terms[0]
    assert m.term == "শিক্ষা"
    assert m.candidate_weight == 1.0
    assert m.raw_bm25 > 0.0
