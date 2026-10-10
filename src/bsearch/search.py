"""Command-line search interface for Banglish and Bengali queries.

Usage:
    python -m bsearch.search "shikkha"
    python -m bsearch.search "shikkha" --explain
    python -m bsearch.search "dhaka" --combiner max -k 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from bsearch.rank.bm25 import BM25Params
from bsearch.rank.expand import ExpansionConfig
from bsearch.rank.searcher import (
    CombinerType,
    SearchConfig,
    SearchResponse,
    Searcher,
)


def format_search_results(response: SearchResponse, explain: bool = False) -> str:
    """Format search response for clean terminal display."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  BANGLA SEARCH ENGINE | Query: '{response.query}'")
    lines.append("=" * 80)

    # 1. Token Expansion Summary
    lines.append("Query Tokens & Candidate Expansions:")
    for exp in response.expansions:
        status = "Bangla Script (Direct)" if exp.is_bangla else "Banglish (Phonetic Expanded)"
        lines.append(f"  • Token: '{exp.token}' [{status}]")
        cands_preview = []
        for c in exp.candidates[:8]:
            cands_preview.append(f"{c.term} (w={c.weight:.2f}, {c.level}, df={c.df})")
        lines.append(f"    Candidates: {', '.join(cands_preview)}")
        if len(exp.candidates) > 8:
            lines.append(f"    ... and {len(exp.candidates) - 8} more candidates")
    lines.append("-" * 80)

    # 2. Results
    lines.append(f"Total Matches: {response.total_hits} | Showing Top: {len(response.results)}")
    lines.append("-" * 80)

    if not response.results:
        lines.append("  [No matching documents found]")
    else:
        for r in response.results:
            lines.append(f"[{r.rank}] Score: {r.score:.4f} | Doc #{r.doc_id}")
            lines.append(f"    Title: {r.title}")
            clean_snippet = r.snippet.replace("\n", " ").strip()
            if len(clean_snippet) > 160:
                clean_snippet = clean_snippet[:157] + "..."
            lines.append(f"    Snippet: {clean_snippet}")

            # Explain details
            if explain and r.explanation:
                lines.append("    Explanation:")
                lines.append(f"      Combiner: {r.explanation.combiner}")
                lines.append(f"      Token Scores: {r.explanation.token_scores}")
                for m in r.explanation.matched_terms:
                    lines.append(
                        f"      - Term '{m.term}': b_tf={m.body_tf}, t_tf={m.title_tf}, "
                        f"idf={m.idf:.3f}, BM25={m.raw_bm25:.3f}, weight={m.candidate_weight:.2f} "
                        f"-> Score={m.weighted_bm25:.3f}"
                    )
            lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint."""
    # Ensure UTF-8 output on Windows terminals
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Search Bengali corpus using Banglish or native Bengali queries."
    )
    parser.add_argument("query", type=str, help="Search query (e.g., 'shikkha', 'dhaka')")
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Path to inverted index directory (default: data/index)",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=10, help="Number of results to retrieve (default: 10)"
    )
    parser.add_argument(
        "-c",
        "--combiner",
        type=str,
        choices=["saturating_sum", "max"],
        default="saturating_sum",
        help="Candidate combiner strategy (default: saturating_sum)",
    )
    parser.add_argument(
        "-e",
        "--explain",
        action="store_true",
        help="Display detailed token candidate weights, key levels, and scoring components",
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter (default: 1.2)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter (default: 0.75)")
    parser.add_argument(
        "--title-boost", type=float, default=2.0, help="BM25 title boost (default: 2.0)"
    )

    args = parser.parse_args()
    index_path = Path(args.index_dir)

    if not index_path.exists() or not (index_path / "index.meta").exists():
        print(f"Error: Index directory not found at '{index_path}'.", file=sys.stderr)
        print("Please build the index first using Task 2 builder.", file=sys.stderr)
        sys.exit(1)

    # Initialize Searcher
    combiner_enum = CombinerType(args.combiner)
    search_cfg = SearchConfig(
        combiner=combiner_enum,
        bm25=BM25Params(k1=args.k1, b=args.b, title_boost=args.title_boost),
        expansion=ExpansionConfig(),
        default_top_k=args.top_k,
    )

    start_load = time.perf_counter()
    searcher = Searcher.from_index_dir(index_path, config=search_cfg)
    load_time = (time.perf_counter() - start_load) * 1000.0

    start_query = time.perf_counter()
    response = searcher.search(args.query, top_k=args.top_k, explain=args.explain)
    query_time = (time.perf_counter() - start_query) * 1000.0

    output = format_search_results(response, explain=args.explain)
    print(output)
    print(f"Index loaded in: {load_time:.2f} ms | Query executed in: {query_time:.2f} ms\n")


if __name__ == "__main__":
    main()
