"""Tests for information retrieval evaluation tooling (Task 5).

Covers:
- Hand-worked mathematical example for MRR@10, nDCG@10, Success@1/5, Recall@20.
- Paired bootstrap 95% CI and significance testing.
- Static transliteration for B1 baseline.
- System initialization and retrieval for B0, B1, B2, B3, and Oracle.
- Ablation and category table formatting.
"""

from __future__ import annotations

import math
import pytest

from bsearch.eval.metrics import (
    aggregate_system_metrics,
    compute_dcg_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_success_at_k,
    evaluate_query,
    generate_ablation_table,
    generate_category_table,
    paired_bootstrap_test,
)
from bsearch.eval.systems import (
    build_systems,
    static_transliterate,
)
from bsearch.index.builder import build_index_from_docs
from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader


# =====================================================================
# 1. Exact Hand-Worked Example Test
# =====================================================================


def test_hand_worked_metrics_example():
    """Verify metrics against exact hand-calculated reference values.

    Setup:
    Ranked retrieved document IDs (10 docs):
      Rank 1: doc_10  (rel = 0)
      Rank 2: doc_20  (rel = 2)
      Rank 3: doc_30  (rel = 1)
      Rank 4..10: doc_40..doc_100 (rel = 0)

    Qrels (total 3 relevant documents in corpus):
      doc_20  -> 2
      doc_30  -> 1
      doc_999 -> 2  (not retrieved in top 10)
    """
    ranked_doc_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    qrels = {
        20: 2,
        30: 1,
        999: 2,
    }

    # 1. MRR@10:
    # First relevant doc is doc_20 at rank 2.
    # Expected RR = 1 / 2 = 0.5
    mrr = compute_mrr_at_k(ranked_doc_ids, qrels, k=10)
    assert mrr == 0.5

    # 2. Success@1 and Success@5:
    # Rank 1 is doc_10 (rel=0) -> Success@1 = 0.0
    # Rank 2 is doc_20 (rel=2) -> Success@5 = 1.0
    assert compute_success_at_k(ranked_doc_ids, qrels, k=1) == 0.0
    assert compute_success_at_k(ranked_doc_ids, qrels, k=5) == 1.0

    # 3. DCG@10:
    # rank 1: gain=0
    # rank 2: (2^2 - 1) / log2(3) = 3 / 1.584962500721156 = 1.8927892607143721
    # rank 3: (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    # rank 4..10: gain=0
    # Expected actual DCG = 1.8927892607143721 + 0.5 = 2.392789260714372
    actual_dcg = compute_dcg_at_k([0, 2, 1, 0, 0, 0, 0, 0, 0, 0], k=10)
    expected_dcg = 3.0 / math.log2(3.0) + 1.0 / math.log2(4.0)
    assert abs(actual_dcg - expected_dcg) < 1e-9

    # 4. IDCG@10:
    # Ideal grades: [2, 2, 1]
    # rank 1: (2^2 - 1) / log2(2) = 3 / 1.0 = 3.0
    # rank 2: (2^2 - 1) / log2(3) = 3 / 1.584962500721156 = 1.8927892607143721
    # rank 3: (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    # Expected ideal DCG = 3.0 + 1.8927892607143721 + 0.5 = 5.392789260714372
    expected_idcg = 3.0 / math.log2(2.0) + 3.0 / math.log2(3.0) + 1.0 / math.log2(4.0)

    # 5. nDCG@10:
    # Expected nDCG = actual_dcg / ideal_dcg = 0.4437016069...
    ndcg = compute_ndcg_at_k(ranked_doc_ids, qrels, k=10)
    expected_ndcg = expected_dcg / expected_idcg
    assert abs(ndcg - expected_ndcg) < 1e-9

    # 6. Recall@20:
    # Retrieved relevant docs in top 20: 2 (doc_20, doc_30)
    # Total relevant docs: 3 (doc_20, doc_30, doc_999)
    # Expected Recall@20 = 2 / 3 = 0.6666666666666666
    recall = compute_recall_at_k(ranked_doc_ids, qrels, k=20)
    assert abs(recall - (2.0 / 3.0)) < 1e-9


# =====================================================================
# 2. Paired Bootstrap CI Tests
# =====================================================================


def test_paired_bootstrap_test():
    scores_b3 = [0.8, 0.7, 0.9, 0.85, 0.95, 0.75, 0.88, 0.92, 0.78, 0.84]
    scores_b1 = [0.4, 0.3, 0.5, 0.45, 0.55, 0.35, 0.48, 0.52, 0.38, 0.44]

    bt = paired_bootstrap_test(
        scores_a=scores_b3,
        scores_b=scores_b1,
        system_a="B3",
        system_b="B1",
        metric_name="nDCG@10",
        num_bootstrap=1000,
        random_seed=42,
    )

    # Mean difference is exactly 0.4
    assert abs(bt.diff_mean - 0.4) < 1e-9
    # 95% CI lower should be <= upper
    assert bt.ci_lower_95 <= bt.ci_upper_95
    # Since B3 is strictly better on every single query, CI lower should be > 0 and p-value < 0.05
    assert bt.ci_lower_95 > 0.3
    assert bt.p_value < 0.01


# =====================================================================
# 3. Static Transliteration (B1 Baseline) Tests
# =====================================================================


def test_static_transliterate():
    # Naive deterministic 1-to-1 transliteration (without dictionary or phonetic collapse)
    assert static_transliterate("shikkha") == "শিক্ষা"
    assert static_transliterate("dhaka") == "ধাকা"  # Demonstrates naive baseline limitation (ধ vs ঢ)
    assert static_transliterate("bangladesh") == "বাংলাদেশ"


# =====================================================================
# 4. End-to-End System Retreival Tests (B0, B1, B2, B3, Oracle)
# =====================================================================


def test_systems_execution(tmp_path):
    docs = [
        {"id": "1", "title": "শিক্ষা ও সংস্কৃতি", "text": "বাঙালির শিক্ষা এবং গবেষণা সংস্কৃতি।"},
        {"id": "2", "title": "ঢাকার ইতিহাস", "text": "প্রাচীন ঢাকা শহরের ঐতিহাসিক বিবর্তন।"},
    ]
    index_dir = tmp_path / "sys_test_idx"
    build_index_from_docs(docs, index_dir)

    index_reader = InvertedIndexReader(index_dir)
    docstore = DocStoreReader(index_dir)
    systems = build_systems(index_reader, docstore)

    assert set(systems.keys()) == {"B0", "B1", "B2", "B3", "Oracle"}

    # B3 full search should retrieve doc 1 for "shikkha"
    res_b3 = systems["B3"].retrieve("shikkha", top_k=5)
    assert len(res_b3) > 0
    assert res_b3[0][0] == 0

    # Oracle search with gold Bangla query should retrieve doc 1
    res_oracle = systems["Oracle"].retrieve(query="shikkha", gold_bangla_query="শিক্ষা", top_k=5)
    assert len(res_oracle) > 0
    assert res_oracle[0][0] == 0

    # B0 with raw English token should return empty results on pure Bangla documents
    res_b0 = systems["B0"].retrieve("shikkha", top_k=5)
    assert len(res_b0) == 0


# =====================================================================
# 5. Table Formatter Tests
# =====================================================================


def test_table_formatters():
    q_metrics = evaluate_query("q1", [1, 2], {1: 2, 2: 1})
    agg_b3 = aggregate_system_metrics("B3", [q_metrics])
    agg_b2 = aggregate_system_metrics("B2", [q_metrics])

    ablation_tbl = generate_ablation_table({"B3": agg_b3, "B2": agg_b2})
    assert "| System |" in ablation_tbl
    assert "**B3**" in ablation_tbl
    assert "**B2**" in ablation_tbl

    cat_tbl = generate_category_table({"single_word": agg_b3})
    assert "| Category |" in cat_tbl
    assert "`single_word`" in cat_tbl
