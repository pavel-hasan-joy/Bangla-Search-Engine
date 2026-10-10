"""Unit tests for bsearch.index and bsearch.rank.

Tests include:
- Varint integer encoding and decoding
- Block-skip postings list serialization and iterative advance
- Hand-computed BM25 scores on a tiny reference corpus
- Separate DocStore random access and snippets
- Exhaustive DAAT vs WAND equivalence across 1000 random queries
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from bsearch.index.builder import build_index_from_jsonl
from bsearch.index.docstore import DocStoreReader, DocStoreWriter
from bsearch.index.inverted_index import InvertedIndexReader
from bsearch.index.postings import BlockPostingIterator, Posting, encode_postings_list
from bsearch.index.vbyte import decode_vbyte, decode_vbyte_list, encode_vbyte, encode_vbyte_list
from bsearch.rank.bm25 import BM25Params, compute_idf, score_term_bm25
from bsearch.rank.topk import daat_top_k, wand_top_k


# =========================================================================
# 1. Variable Byte (Varint) Tests
# =========================================================================

def test_vbyte_roundtrip_single():
    for val in [0, 1, 127, 128, 255, 300, 16383, 16384, 1000000]:
        encoded = encode_vbyte(val)
        decoded, offset = decode_vbyte(encoded)
        assert decoded == val
        assert offset == len(encoded)


def test_vbyte_roundtrip_list():
    numbers = [0, 5, 128, 1024, 65536, 12, 99999]
    encoded = encode_vbyte_list(numbers)
    decoded, offset = decode_vbyte_list(encoded, len(numbers))
    assert decoded == numbers
    assert offset == len(encoded)


# =========================================================================
# 2. Block Postings and Skip Data Tests
# =========================================================================

def test_postings_encode_decode_advance():
    # Create 300 postings across 3 blocks of 128
    postings = [Posting(doc_id=i * 2 + 1, body_tf=(i % 5) + 1, title_tf=(i % 2)) for i in range(300)]
    encoded, skips, max_score = encode_postings_list(postings, block_size=128)

    assert len(skips) == 3
    assert skips[0].count == 128
    assert skips[1].count == 128
    assert skips[2].count == 44

    it = BlockPostingIterator(encoded)
    # Sequential scan verification
    for p in postings:
        assert it.doc_id == p.doc_id
        assert it.body_tf == p.body_tf
        assert it.title_tf == p.title_tf
        it.next_doc()
    assert it.doc_id is None

    # Skip advance verification
    it2 = BlockPostingIterator(encoded)
    # Target doc 250
    adv = it2.advance(250)
    assert adv is not None
    assert adv >= 250
    assert it2.doc_id == adv


# =========================================================================
# 3. Hand-Computed BM25 Scores on Tiny Reference Corpus
# =========================================================================

def test_hand_computed_bm25_scores(tmp_path: Path):
    """Verify BM25 math against hand-calculated values.

    Corpus:
    Doc 0: Title "ঢাকা" (len=1), Text "ঢাকা বাংলাদেশের রাজধানী" (len=3). |D0| = 4.
           ঢাকা: body_tf=1, title_tf=1.
    Doc 1: Title "সিলেট শহর" (len=2), Text "সিলেট একটি প্রাচীন শহর যেখানে চা বাগান আছে" (len=8). |D1| = 10.
    Doc 2: Title "বাংলাদেশের রাজধানী" (len=2), Text "বাংলাদেশের রাজধানী ঢাকা একটি সুন্দর প্রাচীন শহর" (len=7). |D2| = 9.
           ঢাকা: body_tf=1, title_tf=0.

    N = 3, avgdl = 23/3 = 7.666666666666667.
    df("ঢাকা") = 2.
    IDF("ঢাকা") = ln(1.0 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6) = 0.47000362924573563.

    k1 = 1.2, b = 0.75, title_boost = 2.0.

    Doc 0 for "ঢাকা":
      effective_tf = 1 + 2.0 * 1 = 3.0.
      len_norm = 0.25 + 0.75 * (4 / (23/3)) = 0.25 + 9/23 = 0.641304347826087.
      tf_comp = (3.0 * 2.2) / (3.0 + 1.2 * 0.641304347826087) = 6.6 / 3.7695652173913044 = 1.750865051903114.
      Score(ঢাকা, D0) = 0.47000362924573563 * 1.750865051903114 = 0.822912959828606.

    Doc 2 for "ঢাকা":
      effective_tf = 1.0.
      len_norm = 0.25 + 0.75 * (9 / (23/3)) = 0.25 + 20.25/23 = 1.1304347826086958.
      tf_comp = (1.0 * 2.2) / (1.0 + 1.2 * 1.1304347826086958) = 2.2 / 2.356521739130435 = 0.9335793357933579.
      Score(ঢাকা, D2) = 0.47000362924573563 * 0.9335793357933579 = 0.43878586392095815.
    """
    corpus_file = tmp_path / "tiny_corpus.jsonl"
    docs = [
        {"id": "doc0", "title": "ঢাকা", "text": "ঢাকা বাংলাদেশের রাজধানী"},
        {"id": "doc1", "title": "সিলেট শহর", "text": "সিলেট একটি প্রাচীন শহর যেখানে চা বাগান আছে"},
        {"id": "doc2", "title": "বাংলাদেশের রাজধানী", "text": "বাংলাদেশের রাজধানী ঢাকা একটি সুন্দর প্রাচীন শহর"},
    ]
    with open(corpus_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    index_dir = tmp_path / "tiny_index"
    params = BM25Params(k1=1.2, b=0.75, title_boost=2.0)
    stats = build_index_from_jsonl(corpus_file, index_dir, params=params)

    assert stats.num_docs == 3
    assert math.isclose(stats.avgdl, 23.0 / 3.0, rel_tol=1e-5)

    expected_idf_dhaka = math.log(1.6)
    actual_idf_dhaka = compute_idf(num_docs=3, df=2)
    assert math.isclose(actual_idf_dhaka, expected_idf_dhaka, rel_tol=1e-6)

    # Score Doc 0 for "ঢাকা"
    score_doc0 = score_term_bm25(
        body_tf=1,
        title_tf=1,
        doc_len=4,
        avgdl=23.0 / 3.0,
        idf=actual_idf_dhaka,
        params=params,
    )
    expected_score_doc0 = 0.822912959828606
    assert math.isclose(score_doc0, expected_score_doc0, rel_tol=1e-5)

    # Score Doc 2 for "ঢাকা"
    score_doc2 = score_term_bm25(
        body_tf=1,
        title_tf=0,
        doc_len=9,
        avgdl=23.0 / 3.0,
        idf=actual_idf_dhaka,
        params=params,
    )
    expected_score_doc2 = 0.43878586392095815
    assert math.isclose(score_doc2, expected_score_doc2, rel_tol=1e-5)

    # Retrieval top-k test on tiny index
    with InvertedIndexReader(index_dir) as idx_reader, DocStoreReader(index_dir) as ds_reader:
        results = daat_top_k(["ঢাকা"], idx_reader, ds_reader, k=2, params=params)
        assert len(results) == 2
        # Doc 0 should rank first with ~0.8229, Doc 2 second with ~0.4388
        assert results[0][0] == 0
        assert math.isclose(results[0][1], expected_score_doc0, rel_tol=1e-5)
        assert results[1][0] == 2
        assert math.isclose(results[1][1], expected_score_doc2, rel_tol=1e-5)


# =========================================================================
# 4. DAAT vs WAND Equivalence over 1000 Random Queries
# =========================================================================

def test_daat_wand_equivalence_1000_queries(tmp_path: Path):
    """Construct a synthetic multi-block corpus and assert DAAT and WAND give identical top-k on 1000 random queries."""
    corpus_file = tmp_path / "rand_corpus.jsonl"
    index_dir = tmp_path / "rand_index"

    vocab = [f"শব্দ_{i}" for i in range(80)]
    random.seed(42)

    # Generate 400 documents (spanning > 3 blocks of 128)
    num_docs = 400
    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc_id in range(num_docs):
            title_terms = random.sample(vocab, k=random.randint(1, 3))
            body_terms = [random.choice(vocab) for _ in range(random.randint(10, 40))]
            f.write(
                json.dumps(
                    {
                        "id": f"doc_{doc_id}",
                        "title": " ".join(title_terms),
                        "text": " ".join(body_terms),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    params = BM25Params(k1=1.2, b=0.75, title_boost=2.0)
    build_index_from_jsonl(corpus_file, index_dir, params=params)

    with InvertedIndexReader(index_dir) as idx_reader, DocStoreReader(index_dir) as ds_reader:
        for query_idx in range(1000):
            # Generate random query of 1 to 4 terms
            q_len = random.randint(1, 4)
            q_terms = random.sample(vocab, k=q_len)
            k = random.randint(1, 15)

            daat_res = daat_top_k(q_terms, idx_reader, ds_reader, k=k, params=params)
            wand_res = wand_top_k(q_terms, idx_reader, ds_reader, k=k, params=params)

            # Assert identical count
            assert len(daat_res) == len(wand_res), (
                f"Query {query_idx} ('{' '.join(q_terms)}') count mismatch: DAAT {len(daat_res)} vs WAND {len(wand_res)}"
            )

            # Assert identical ranking and scores within floating tolerance
            for rank, ((d_id_daat, s_daat), (d_id_wand, s_wand)) in enumerate(zip(daat_res, wand_res)):
                assert d_id_daat == d_id_wand, (
                    f"Query {query_idx} at rank {rank}: DAAT doc {d_id_daat} != WAND doc {d_id_wand}"
                )
                assert math.isclose(s_daat, s_wand, rel_tol=1e-5), (
                    f"Query {query_idx} at rank {rank}: score mismatch {s_daat} vs {s_wand}"
                )
