"""Information Retrieval evaluation metrics and paired statistical testing.

Implements:
- MRR@10: Mean Reciprocal Rank at cutoff 10.
- nDCG@10: Graded Normalized Discounted Cumulative Gain at cutoff 10.
- Success@1 and Success@5: Binary success at ranks 1 and 5.
- Recall@20: Proportion of all known relevant documents retrieved in top-20.
- Paired bootstrap 95% Confidence Intervals and significance testing.
- Formatted ablation tables and per-category analysis tables.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Final, Sequence


@dataclass(frozen=True)
class QueryMetrics:
    """Evaluation metrics for a single query."""

    query_id: str
    mrr_10: float
    ndcg_10: float
    success_1: float
    success_5: float
    recall_20: float


@dataclass(frozen=True)
class SystemAggregateMetrics:
    """Aggregated evaluation metrics for a search system."""

    system_name: str
    num_queries: int
    mean_mrr_10: float
    mean_ndcg_10: float
    mean_success_1: float
    mean_success_5: float
    mean_recall_20: float
    per_query: list[QueryMetrics]


@dataclass(frozen=True)
class BootstrapComparison:
    """Paired bootstrap comparison between two systems."""

    system_a: str
    system_b: str
    metric_name: str
    diff_mean: float
    ci_lower_95: float
    ci_upper_95: float
    p_value: float


# =====================================================================
# 1. Core Metric Formulas
# =====================================================================


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[int],
    qrels: dict[int, int],
    k: int = 10,
    min_rel: int = 1,
) -> float:
    """Compute Reciprocal Rank at cutoff k for a single query."""
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) >= min_rel:
            return 1.0 / rank
    return 0.0


def compute_dcg_at_k(grades: Sequence[int], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain: sum((2^rel - 1) / log2(rank + 1))."""
    dcg = 0.0
    for rank, grade in enumerate(grades[:k], start=1):
        if grade > 0:
            gain = (2.0**grade) - 1.0
            discount = math.log2(rank + 1.0)
            dcg += gain / discount
    return dcg


def compute_ndcg_at_k(
    ranked_doc_ids: Sequence[int],
    qrels: dict[int, int],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at cutoff k."""
    # Actual DCG@k
    retrieved_grades = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    actual_dcg = compute_dcg_at_k(retrieved_grades, k=k)

    # Ideal DCG@k: sort all known positive grades descending
    known_positive_grades = sorted([g for g in qrels.values() if g > 0], reverse=True)
    if not known_positive_grades:
        return 0.0

    ideal_dcg = compute_dcg_at_k(known_positive_grades, k=k)
    if ideal_dcg <= 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def compute_success_at_k(
    ranked_doc_ids: Sequence[int],
    qrels: dict[int, int],
    k: int = 1,
    min_rel: int = 1,
) -> float:
    """Compute Success@k (1.0 if any doc in top-k is relevant, else 0.0)."""
    for doc_id in ranked_doc_ids[:k]:
        if qrels.get(doc_id, 0) >= min_rel:
            return 1.0
    return 0.0


def compute_recall_at_k(
    ranked_doc_ids: Sequence[int],
    qrels: dict[int, int],
    k: int = 20,
    min_rel: int = 1,
) -> float:
    """Compute Recall@k (proportion of all relevant documents retrieved in top-k)."""
    total_relevant = sum(1 for g in qrels.values() if g >= min_rel)
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        1 for doc_id in ranked_doc_ids[:k] if qrels.get(doc_id, 0) >= min_rel
    )
    return retrieved_relevant / total_relevant


def evaluate_query(
    query_id: str,
    ranked_doc_ids: Sequence[int],
    qrels: dict[int, int],
) -> QueryMetrics:
    """Compute all evaluation metrics for a single query."""
    return QueryMetrics(
        query_id=query_id,
        mrr_10=compute_mrr_at_k(ranked_doc_ids, qrels, k=10),
        ndcg_10=compute_ndcg_at_k(ranked_doc_ids, qrels, k=10),
        success_1=compute_success_at_k(ranked_doc_ids, qrels, k=1),
        success_5=compute_success_at_k(ranked_doc_ids, qrels, k=5),
        recall_20=compute_recall_at_k(ranked_doc_ids, qrels, k=20),
    )


def aggregate_system_metrics(
    system_name: str,
    query_metrics_list: list[QueryMetrics],
) -> SystemAggregateMetrics:
    """Aggregate per-query metrics into system-level means."""
    n = len(query_metrics_list)
    if n == 0:
        return SystemAggregateMetrics(
            system_name=system_name,
            num_queries=0,
            mean_mrr_10=0.0,
            mean_ndcg_10=0.0,
            mean_success_1=0.0,
            mean_success_5=0.0,
            mean_recall_20=0.0,
            per_query=[],
        )

    return SystemAggregateMetrics(
        system_name=system_name,
        num_queries=n,
        mean_mrr_10=sum(q.mrr_10 for q in query_metrics_list) / n,
        mean_ndcg_10=sum(q.ndcg_10 for q in query_metrics_list) / n,
        mean_success_1=sum(q.success_1 for q in query_metrics_list) / n,
        mean_success_5=sum(q.success_5 for q in query_metrics_list) / n,
        mean_recall_20=sum(q.recall_20 for q in query_metrics_list) / n,
        per_query=query_metrics_list,
    )


# =====================================================================
# 2. Paired Bootstrap 95% Confidence Interval
# =====================================================================


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    system_a: str = "System A",
    system_b: str = "System B",
    metric_name: str = "Metric",
    num_bootstrap: int = 1000,
    random_seed: int = 42,
) -> BootstrapComparison:
    """Perform paired bootstrap resampling to compute 95% CI and two-sided p-value.

    Delta = System_A - System_B.
    """
    n = len(scores_a)
    if n == 0 or len(scores_b) != n:
        raise ValueError("scores_a and scores_b must have the same non-zero length")

    deltas = [a - b for a, b in zip(scores_a, scores_b)]
    obs_diff = sum(deltas) / n

    rng = random.Random(random_seed)
    bootstrap_diffs: list[float] = []

    for _ in range(num_bootstrap):
        # Sample n indices with replacement
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_diff = sum(deltas[idx] for idx in sample_indices) / n
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs.sort()
    lower_idx = int(0.025 * num_bootstrap)
    upper_idx = int(0.975 * num_bootstrap)
    ci_lower = bootstrap_diffs[lower_idx]
    ci_upper = bootstrap_diffs[upper_idx]

    # Two-sided p-value
    if obs_diff >= 0.0:
        non_positive = sum(1 for d in bootstrap_diffs if d <= 0.0)
        p_val = min(1.0, 2.0 * (non_positive / num_bootstrap))
    else:
        non_negative = sum(1 for d in bootstrap_diffs if d >= 0.0)
        p_val = min(1.0, 2.0 * (non_negative / num_bootstrap))

    return BootstrapComparison(
        system_a=system_a,
        system_b=system_b,
        metric_name=metric_name,
        diff_mean=obs_diff,
        ci_lower_95=ci_lower,
        ci_upper_95=ci_upper,
        p_value=p_val,
    )


# =====================================================================
# 3. Formatted Tables: Ablation & Per-Category
# =====================================================================


def generate_ablation_table(
    systems_metrics: dict[str, SystemAggregateMetrics],
) -> str:
    """Format markdown comparison table of systems (B0, B1, B2, B3, Oracle)."""
    lines: list[str] = []
    lines.append("| System | Description | MRR@10 | nDCG@10 | Success@1 | Success@5 | Recall@20 |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    descriptions: dict[str, str] = {
        "B0": "Raw Roman string (no transliteration)",
        "B1": "Static 1-to-1 transliteration + plain BM25",
        "B2": "Ours without fuzzy (K1, K2 exact, K3)",
        "B3": "Full system (K1, K2, Fuzzy, K3, combiners)",
        "Oracle": "BM25 with gold native Bengali query",
    }

    order = ["B0", "B1", "B2", "B3", "Oracle"]
    for name in order:
        if name in systems_metrics:
            m = systems_metrics[name]
            desc = descriptions.get(name, "")
            lines.append(
                f"| **{name}** | {desc} | {m.mean_mrr_10:.4f} | {m.mean_ndcg_10:.4f} | "
                f"{m.mean_success_1:.4f} | {m.mean_success_5:.4f} | {m.mean_recall_20:.4f} |"
            )

    return "\n".join(lines)


def generate_category_table(
    category_metrics: dict[str, SystemAggregateMetrics],
) -> str:
    """Format markdown table breaking down performance across query categories."""
    lines: list[str] = []
    lines.append("| Category | Queries | MRR@10 | nDCG@10 | Success@1 | Success@5 | Recall@20 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cat_name, m in sorted(category_metrics.items()):
        lines.append(
            f"| `{cat_name}` | {m.num_queries} | {m.mean_mrr_10:.4f} | {m.mean_ndcg_10:.4f} | "
            f"{m.mean_success_1:.4f} | {m.mean_success_5:.4f} | {m.mean_recall_20:.4f} |"
        )

    return "\n".join(lines)
