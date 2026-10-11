"""Evaluation pipeline runner: runs systems, computes metrics, and formats reports.

Usage:
    python -m bsearch.eval.runner --queries eval/queries.tsv --qrels eval/qrels.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from bsearch.eval.label_tool import load_existing_qrels, load_queries
from bsearch.eval.metrics import (
    aggregate_system_metrics,
    evaluate_query,
    generate_ablation_table,
    generate_category_table,
    paired_bootstrap_test,
    QueryMetrics,
    SystemAggregateMetrics,
)
from bsearch.eval.systems import build_systems
from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader


def run_evaluation(
    queries_path: Path | str = "eval/queries.tsv",
    qrels_path: Path | str = "eval/qrels.tsv",
    index_dir: Path | str = "data/index",
    top_k: int = 20,
) -> str:
    """Run full evaluation suite across all systems and generate markdown report."""
    q_file = Path(queries_path)
    r_file = Path(qrels_path)
    idx_dir = Path(index_dir)

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  BANGLA SEARCH ENGINE - EVALUATION REPORT")
    lines.append("=" * 80)

    if not idx_dir.exists() or not (idx_dir / "index.meta").exists():
        return f"Error: Index not found at {idx_dir}. Please build the index first."

    all_queries = load_queries(q_file)
    qrels_flat = load_existing_qrels(r_file)

    # Group qrels by query_id: qid -> dict[doc_id, grade]
    qrels_by_query: dict[str, dict[int, int]] = defaultdict(dict)
    for (qid, did), grade in qrels_flat.items():
        qrels_by_query[qid][did] = grade

    # Filter to queries that have ground-truth judgments
    eval_queries = [q for q in all_queries if q["id"] in qrels_by_query and any(g > 0 for g in qrels_by_query[q["id"]].values())]

    lines.append(f"Loaded Total Queries:    {len(all_queries)}")
    lines.append(f"Queries with Judgments:  {len(eval_queries)}")
    lines.append(f"Total Judgments in Qrels: {len(qrels_flat)}")
    lines.append("-" * 80)

    if not eval_queries:
        lines.append("\n[!] No evaluated queries with positive judgments found in qrels.tsv.")
        lines.append("    Use 'python -m bsearch.eval.label_tool' to label documents first.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Initialize index and systems
    index_reader = InvertedIndexReader(idx_dir)
    docstore = DocStoreReader(idx_dir)
    systems = build_systems(index_reader, docstore)

    system_query_metrics: dict[str, list[QueryMetrics]] = defaultdict(list)
    system_cat_metrics: dict[str, dict[str, list[QueryMetrics]]] = defaultdict(lambda: defaultdict(list))

    for q_item in eval_queries:
        qid = q_item["id"]
        q_text = q_item["query"]
        cat = q_item.get("category", "unclassified")
        gold_q = q_item.get("gold_bangla_query", "")
        qrels_dict = qrels_by_query[qid]

        for sys_name, sys_obj in systems.items():
            res = sys_obj.retrieve(query=q_text, gold_bangla_query=gold_q, top_k=top_k)
            ranked_doc_ids = [did for did, _ in res]
            qm = evaluate_query(qid, ranked_doc_ids, qrels_dict)

            system_query_metrics[sys_name].append(qm)
            system_cat_metrics[sys_name][cat].append(qm)

    # Aggregate overall metrics per system
    system_aggregates: dict[str, SystemAggregateMetrics] = {}
    for sys_name, qm_list in system_query_metrics.items():
        system_aggregates[sys_name] = aggregate_system_metrics(sys_name, qm_list)

    # 1. Ablation Table
    lines.append("\n### 1. System Ablation Table")
    lines.append(generate_ablation_table(system_aggregates))

    # 2. Paired Bootstrap Hypothesis Testing
    lines.append("\n### 2. Paired Bootstrap Significance (B3 Full System vs Baselines, 95% CI)")
    b3_ndcg = [qm.ndcg_10 for qm in system_query_metrics["B3"]]
    b3_mrr = [qm.mrr_10 for qm in system_query_metrics["B3"]]

    lines.append("| Comparison | Metric | Mean Diff (Δ) | 95% Bootstrap CI | p-value | Significant (p < 0.05) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    baselines_to_compare = [("B2", "Ours w/o Fuzzy"), ("B1", "Static Transliteration"), ("B0", "Raw Roman String")]
    for b_name, b_desc in baselines_to_compare:
        if b_name in system_query_metrics:
            comp_ndcg = [qm.ndcg_10 for qm in system_query_metrics[b_name]]
            bt_ndcg = paired_bootstrap_test(b3_ndcg, comp_ndcg, system_a="B3", system_b=b_name, metric_name="nDCG@10")
            sig_ndcg = "Yes" if bt_ndcg.p_value < 0.05 else "No"
            lines.append(
                f"| B3 vs {b_name} ({b_desc}) | nDCG@10 | {bt_ndcg.diff_mean:+.4f} | "
                f"[{bt_ndcg.ci_lower_95:+.4f}, {bt_ndcg.ci_upper_95:+.4f}] | {bt_ndcg.p_value:.4f} | {sig_ndcg} |"
            )

            comp_mrr = [qm.mrr_10 for qm in system_query_metrics[b_name]]
            bt_mrr = paired_bootstrap_test(b3_mrr, comp_mrr, system_a="B3", system_b=b_name, metric_name="MRR@10")
            sig_mrr = "Yes" if bt_mrr.p_value < 0.05 else "No"
            lines.append(
                f"| B3 vs {b_name} ({b_desc}) | MRR@10 | {bt_mrr.diff_mean:+.4f} | "
                f"[{bt_mrr.ci_lower_95:+.4f}, {bt_mrr.ci_upper_95:+.4f}] | {bt_mrr.p_value:.4f} | {sig_mrr} |"
            )

    # 3. Per-Category Breakdown (B3 System)
    lines.append("\n### 3. Per-Category Performance Breakdown (B3 Full System)")
    b3_cat_aggregates: dict[str, SystemAggregateMetrics] = {}
    for cat_name, qm_list in system_cat_metrics["B3"].items():
        b3_cat_aggregates[cat_name] = aggregate_system_metrics(f"B3_{cat_name}", qm_list)

    lines.append(generate_category_table(b3_cat_aggregates))
    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def main() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Evaluate search systems on queries and qrels.")
    parser.add_argument("--queries", "-q", default="eval/queries.tsv", help="Path to queries.tsv.")
    parser.add_argument("--qrels", "-r", default="eval/qrels.tsv", help="Path to qrels.tsv.")
    parser.add_argument("--index", "-i", default="data/index", help="Index directory path.")
    parser.add_argument("--top-k", "-k", type=int, default=20, help="Retrieval cutoff.")
    args = parser.parse_args()

    report = run_evaluation(
        queries_path=args.queries,
        qrels_path=args.qrels,
        index_dir=args.index,
        top_k=args.top_k,
    )
    print(report)


if __name__ == "__main__":
    main()
