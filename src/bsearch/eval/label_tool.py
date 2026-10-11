"""Interactive system-blind relevance judgment tool for human annotators.

Workflow:
1. Reads queries from eval/queries.tsv.
2. Runs systems B0, B1, B2, B3, and Oracle to retrieve top-20 documents per system.
3. Pools and deduplicates retrieved documents per query.
4. Shuffles documents system-blind (hides which system retrieved each document).
5. Shows query, intent note, document title, and snippet.
6. Prompts human evaluator to grade:
   - 0: Not Relevant
   - 1: Partially Relevant / Marginal
   - 2: Highly Relevant
7. Appends judgments atomically to eval/qrels.tsv.
8. Supports resuming without repeating previously graded documents.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

from bsearch.eval.systems import build_systems
from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader


def load_queries(queries_path: Path | str) -> list[dict[str, str]]:
    """Load query definitions from TSV."""
    path = Path(queries_path)
    if not path.is_file():
        raise FileNotFoundError(f"Queries file not found: {path}")

    queries: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        for row in reader:
            if not row or not row[0].strip() or row[0].startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                continue
            if len(row) >= len(header):
                item = {k: v.strip() for k, v in zip(header, row)}
                queries.append(item)
    return queries


def load_existing_qrels(qrels_path: Path | str) -> dict[tuple[str, int], int]:
    """Load existing relevance judgments: (query_id, doc_id) -> grade."""
    path = Path(qrels_path)
    if not path.is_file():
        return {}

    qrels: dict[tuple[str, int], int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or not row[0].strip() or row[0].startswith("#") or row[0] == "query_id":
                continue
            try:
                qid = row[0].strip()
                doc_id = int(row[1].strip())
                grade = int(row[2].strip())
                qrels[(qid, doc_id)] = grade
            except (ValueError, IndexError):
                continue
    return qrels


def record_judgment(
    qrels_path: Path | str, query_id: str, doc_id: int, grade: int
) -> None:
    """Append a single relevance judgment atomically to qrels.tsv."""
    path = Path(qrels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["query_id", "doc_id", "relevance"])
        writer.writerow([query_id, str(doc_id), str(grade)])
        f.flush()


def run_labeling_session(
    queries_path: Path | str = "eval/queries.tsv",
    qrels_path: Path | str = "eval/qrels.tsv",
    index_dir: Path | str = "data/index",
    pool_depth: int = 20,
    random_seed: int = 42,
) -> None:
    """Run interactive terminal labeling session."""
    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    q_file = Path(queries_path)
    r_file = Path(qrels_path)
    idx_dir = Path(index_dir)

    print("=" * 80)
    print("  BANGLA SEARCH ENGINE - RELEVANCE JUDGMENT TOOL")
    print("=" * 80)

    if not idx_dir.exists() or not (idx_dir / "index.meta").exists():
        print(f"Error: Index directory not found at {idx_dir}. Please build index first.")
        return

    queries = load_queries(q_file)
    if not queries:
        print(f"No queries found in {q_file}. Please add queries to {q_file} first.")
        return

    print(f"Loaded {len(queries)} queries from {q_file}.")
    existing_qrels = load_existing_qrels(r_file)
    print(f"Loaded {len(existing_qrels)} existing judgments from {r_file}.")

    # Load systems
    print("Initializing evaluation systems (B0, B1, B2, B3, Oracle)...")
    index_reader = InvertedIndexReader(idx_dir)
    docstore = DocStoreReader(idx_dir)
    systems = build_systems(index_reader, docstore)

    rng = random.Random(random_seed)
    total_graded_session = 0

    for q_idx, q_item in enumerate(queries, start=1):
        qid = q_item.get("id", f"q_{q_idx}")
        query_text = q_item.get("query", "")
        category = q_item.get("category", "unclassified")
        gold_bangla = q_item.get("gold_bangla_query", "")
        intent_note = q_item.get("intent_note", "")

        # 1. Pool top-N documents from all systems
        pooled_doc_ids: set[int] = set()
        for sys_name, sys_obj in systems.items():
            results = sys_obj.retrieve(query=query_text, gold_bangla_query=gold_bangla, top_k=pool_depth)
            for doc_id, _ in results:
                pooled_doc_ids.add(doc_id)

        # 2. Filter out already judged documents
        unjudged_docs = [did for did in pooled_doc_ids if (qid, did) not in existing_qrels]
        if not unjudged_docs:
            continue

        # 3. System-blind shuffle
        rng.shuffle(unjudged_docs)

        print("")
        print("=" * 80)
        print(f"Query [{q_idx}/{len(queries)}]: ID={qid} | Category={category}")
        print(f"Input Query:       '{query_text}'")
        if gold_bangla:
            print(f"Gold Bangla Query: '{gold_bangla}'")
        if intent_note:
            print(f"Intent Note:       '{intent_note}'")
        print(f"Unjudged Docs:     {len(unjudged_docs)} (Total Pooled: {len(pooled_doc_ids)})")
        print("=" * 80)

        skip_current_query = False

        for doc_pos, doc_id in enumerate(unjudged_docs, start=1):
            if skip_current_query:
                break

            doc_data = docstore.get(doc_id)
            title = doc_data.get("title", f"Doc #{doc_id}")
            snippet = doc_data.get("snippet", "")
            if not snippet:
                snippet = doc_data.get("text", "")[:250].replace("\n", " ").strip()

            print(f"\n[Doc {doc_pos}/{len(unjudged_docs)}] Doc ID: {doc_id}")
            print(f"  Title:   {title}")
            print(f"  Snippet: {snippet}")

            while True:
                prompt = (
                    "  Relevance grade -> [0] Not Relevant | [1] Partially Relevant | "
                    "[2] Highly Relevant | [s] Skip doc | [n] Next query | [q] Save & Quit\n"
                    "  Choice: "
                )
                choice = input(prompt).strip().lower()

                if choice in ("0", "1", "2"):
                    grade = int(choice)
                    record_judgment(r_file, qid, doc_id, grade)
                    existing_qrels[(qid, doc_id)] = grade
                    total_graded_session += 1
                    print(f"  -> Recorded grade {grade} for (query={qid}, doc={doc_id}).")
                    break
                elif choice == "s":
                    print("  -> Skipped document.")
                    break
                elif choice == "n":
                    print("  -> Moving to next query.")
                    skip_current_query = True
                    break
                elif choice == "q":
                    print(f"\nSession finished. {total_graded_session} new judgments saved to {r_file}.")
                    return
                else:
                    print("  Invalid choice! Please enter 0, 1, 2, s, n, or q.")

    print(f"\nAll queries judged! Total {total_graded_session} new judgments saved to {r_file}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="System-blind relevance judgment labeling tool.")
    parser.add_argument("--queries", "-q", default="eval/queries.tsv", help="Path to queries.tsv.")
    parser.add_argument("--qrels", "-r", default="eval/qrels.tsv", help="Path to output qrels.tsv.")
    parser.add_argument("--index", "-i", default="data/index", help="Index directory path.")
    parser.add_argument("--pool-depth", "-k", type=int, default=20, help="Pooling depth per system.")
    args = parser.parse_args()

    run_labeling_session(
        queries_path=args.queries,
        qrels_path=args.qrels,
        index_dir=args.index,
        pool_depth=args.pool_depth,
    )


if __name__ == "__main__":
    main()
