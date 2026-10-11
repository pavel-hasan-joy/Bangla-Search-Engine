"""Latency and throughput benchmark suite on 100,000 documents/passages.

Measures:
- Cold index load time (mmap)
- Index size on disk (bytes & MB)
- Process RSS memory consumption (MB)
- Single-thread warm retrieval latencies: p50, p90, p95, p99, mean, max
- Per-stage latency breakdown:
  1. Tokenization (t_tokenize)
  2. Phonetic Candidate Expansion (t_expand)
  3. Posting Traversal & BM25 Scoring (t_score)
  4. DocStore Fetch & Snippet (t_fetch)
  5. Total Query Latency (t_total)
- Target: p95 <= 100 ms
- Machine hardware specifications (CPU, cores, RAM, OS, Python)
- Saves full results to results/latency_benchmark.json and results/latency_benchmark.txt
"""

from __future__ import annotations

import argparse
import ctypes
import heapq
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from bsearch.index.builder import build_index_from_jsonl, get_process_rss_mb
from bsearch.index.docstore import DocStoreReader
from bsearch.index.inverted_index import InvertedIndexReader
from bsearch.phonetic.g2p import bangla_to_phonemes
from bsearch.phonetic.index import PhoneticIndex
from bsearch.rank.bm25 import BM25Params, score_term_bm25
from bsearch.rank.expand import ExpansionConfig, QueryExpander
from bsearch.rank.searcher import (
    CombinerType,
    SearchConfig,
    Searcher,
    apply_saturating_sum,
)
from bsearch.rank.topk import ScoredDoc
from bsearch.unicode.tokenizer import tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bsearch.bench")


def get_machine_specs() -> dict[str, Any]:
    """Retrieve machine hardware specifications using standard library only."""
    specs: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count() or 1,
        "python_version": f"{platform.python_implementation()} {platform.python_version()}",
    }

    # Total RAM via Windows ctypes
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram_gb = stat.ullTotalPhys / (1024**3)
            avail_ram_gb = stat.ullAvailPhys / (1024**3)
            specs["ram_total_gb"] = round(total_ram_gb, 2)
            specs["ram_avail_gb"] = round(avail_ram_gb, 2)
    except Exception:
        pass

    return specs


def ensure_100k_corpus(
    source_jsonl: Path | str = "data/bnwiki_articles.jsonl",
    output_jsonl: Path | str = "data/passages_100k.jsonl",
    target_count: int = 100_000,
) -> Path:
    """Generate or verify 100,000 distinct realistic passages from source articles."""
    out_path = Path(output_jsonl)
    if out_path.is_file():
        # Verify line count
        with open(out_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        if count >= target_count:
            logger.info("100k corpus already exists with %d documents at %s", count, out_path)
            return out_path

    src_path = Path(source_jsonl)
    if not src_path.is_file():
        raise FileNotFoundError(f"Source JSONL file not found at {src_path}")

    logger.info("Generating %d passages from %s...", target_count, src_path)
    articles: list[dict[str, str]] = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    if not articles:
        raise ValueError(f"No articles found in {src_path}")

    # Extract sentences and titles
    all_sentences: list[str] = []
    titles: list[str] = []
    re_sent = re.compile(r"[।?!॥\n]+")

    for art in articles:
        t = art.get("title", "")
        if t:
            titles.append(t)
        body = art.get("text", "")
        for s in re_sent.split(body):
            s_clean = s.strip()
            if len(s_clean) > 15:
                all_sentences.append(s_clean)

    if not all_sentences:
        all_sentences = [art.get("text", "") for art in articles if art.get("text")]

    rng = random.Random(1337)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for i in range(target_count):
            title = titles[i % len(titles)]
            # Pick 6-10 sentences to make ~120-160 words
            num_sents = rng.randint(6, 10)
            start_idx = (i * 7) % len(all_sentences)
            selected_sents = [all_sentences[(start_idx + j) % len(all_sentences)] for j in range(num_sents)]
            passage_text = "। ".join(selected_sents) + "।"

            doc_obj = {
                "id": f"p_{i}",
                "title": f"{title} (অংশ {i % 50 + 1})",
                "text": passage_text,
                "doc_len": len(passage_text.split()),
            }
            f_out.write(json.dumps(doc_obj, ensure_ascii=False) + "\n")

    logger.info("Generated %d passages at %s", target_count, out_path)
    return out_path


def generate_synthetic_banglish_queries(
    vocab_terms: list[str],
    user_queries_path: Path | str = "eval/queries.tsv",
    target_count: int = 1050,
) -> list[str]:
    """Generate 1000+ realistic Banglish queries for latency benchmarking only."""
    queries: list[str] = []

    # 1. Load user queries from queries.tsv if present
    u_path = Path(user_queries_path)
    if u_path.is_file():
        with open(u_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts and not parts[0].startswith("#") and parts[0] != "id" and len(parts) > 1:
                    q_val = parts[1].strip()
                    if q_val:
                        queries.append(q_val)

    logger.info("Loaded %d user queries for latency benchmark.", len(queries))

    # Filter good Bengali terms (len 3 to 10 chars)
    candidate_terms = [t for t in vocab_terms if 3 <= len(t) <= 10 and not t.isdigit()]
    if not candidate_terms:
        candidate_terms = ["শিক্ষা", "ঢাকা", "বিজ্ঞান", "বাংলাদেশ", "ইতিহাস", "সাহিত্য", "সরকার"]

    rng = random.Random(42)

    # 2. Generate phonetic Banglish single-word queries
    for term in candidate_terms:
        ph = bangla_to_phonemes(term)
        if ph:
            queries.append(ph)
            # Add spelling variants
            if "sh" in ph:
                queries.append(ph.replace("sh", "s"))
            if "kkh" in ph:
                queries.append(ph.replace("kkh", "kk"))
                queries.append(ph.replace("kkh", "kh"))
            if "ee" in ph:
                queries.append(ph.replace("ee", "i"))
        if len(queries) >= target_count // 2:
            break

    # 3. Generate multi-word Banglish queries (pairs of terms)
    while len(queries) < target_count:
        t1 = rng.choice(candidate_terms)
        t2 = rng.choice(candidate_terms)
        ph1 = bangla_to_phonemes(t1)
        ph2 = bangla_to_phonemes(t2)
        if ph1 and ph2:
            queries.append(f"{ph1} {ph2}")

    rng.shuffle(queries)
    return queries[:target_count]


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50, p90, p95, p99, mean, max for a list of timing values in ms."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}

    s = sorted(values)
    n = len(s)

    def p(pct: float) -> float:
        idx = int(pct * n / 100.0)
        idx = min(idx, n - 1)
        return s[idx]

    return {
        "p50": round(p(50), 3),
        "p90": round(p(90), 3),
        "p95": round(p(95), 3),
        "p99": round(p(99), 3),
        "mean": round(sum(s) / n, 3),
        "max": round(max(s), 3),
    }


def execute_query_timed(
    searcher: Searcher,
    query: str,
    top_k: int = 10,
) -> dict[str, float]:
    """Run searcher query with nanosecond-resolution per-stage breakdown."""
    _, timings = searcher.search_with_stage_timings(query, top_k=top_k)
    return timings


def run_benchmark(
    index_dir: Path | str = "data/index_100k",
    passages_jsonl: Path | str = "data/passages_100k.jsonl",
    results_dir: Path | str = "results",
    target_docs: int = 100_000,
    num_benchmark_queries: int = 1050,
) -> dict[str, Any]:
    """Execute complete 100k document benchmark suite."""
    out_results = Path(results_dir)
    out_results.mkdir(parents=True, exist_ok=True)
    idx_path = Path(index_dir)

    print("=" * 80)
    print("  BANGLA SEARCH ENGINE - 100K LATENCY BENCHMARK SUITE")
    print("=" * 80)

    # 1. Ensure 100k corpus
    corpus_file = ensure_100k_corpus(output_jsonl=passages_jsonl, target_count=target_docs)

    # 2. Build index if not present
    build_stats: dict[str, Any] = {}
    if not (idx_path / "index.meta").exists():
        logger.info("Building 100k inverted index from %s...", corpus_file)
        t_b0 = time.perf_counter()
        stats = build_index_from_jsonl(corpus_file, idx_path)
        build_time = time.perf_counter() - t_b0
        build_stats = {
            "num_docs": stats.num_docs,
            "vocab_size": stats.vocab_size,
            "build_time_seconds": round(build_time, 2),
            "index_size_bytes": stats.index_size_bytes,
            "rss_mb": round(stats.rss_mb, 2),
        }
    else:
        logger.info("Using existing index at %s", idx_path)
        with open(idx_path / "index.meta", "r", encoding="utf-8") as f:
            meta = json.load(f)
        total_size = sum(f.stat().st_size for f in idx_path.iterdir() if f.is_file())
        build_stats = {
            "num_docs": meta.get("num_docs", target_docs),
            "vocab_size": meta.get("vocab_size", 0),
            "build_time_seconds": meta.get("build_time_seconds", 0.0),
            "index_size_bytes": total_size,
            "rss_mb": round(get_process_rss_mb(), 2),
        }

    # 3. Measure cold load time
    logger.info("Measuring cold index load time...")
    t_c0 = time.perf_counter()
    index_reader = InvertedIndexReader(idx_path)
    docstore = DocStoreReader(idx_path)
    phonetic_idx = PhoneticIndex()
    phonetic_idx.build_from_terms(index_reader.terms.keys())
    searcher = Searcher(index_reader, docstore, phonetic_index=phonetic_idx)
    cold_load_time_ms = round((time.perf_counter() - t_c0) * 1000.0, 2)
    logger.info("Cold index loaded in %.2f ms", cold_load_time_ms)

    # 4. Generate queries
    queries = generate_synthetic_banglish_queries(
        vocab_terms=list(index_reader.terms.keys()),
        target_count=num_benchmark_queries,
    )
    logger.info("Prepared %d queries for latency measurement.", len(queries))

    # 5. Warm-up (50 queries)
    logger.info("Warming up search engine on 50 queries...")
    for q in queries[:50]:
        searcher.search(q, top_k=10)

    # 6. Single-thread warm latency benchmark
    logger.info("Running single-thread benchmark on %d queries...", len(queries))
    stage_tokenize: list[float] = []
    stage_expand: list[float] = []
    stage_score: list[float] = []
    stage_fetch: list[float] = []
    stage_total: list[float] = []

    t_bench_start = time.perf_counter()
    for q_idx, q in enumerate(queries, start=1):
        timing = execute_query_timed(searcher, q, top_k=10)
        stage_tokenize.append(timing["tokenize_ms"])
        stage_expand.append(timing["expand_ms"])
        stage_score.append(timing["score_ms"])
        stage_fetch.append(timing["fetch_ms"])
        stage_total.append(timing["total_ms"])

        if q_idx % 250 == 0:
            logger.info("Benchmarked %d / %d queries...", q_idx, len(queries))

    total_bench_duration = time.perf_counter() - t_bench_start
    qps = round(len(queries) / total_bench_duration, 2)

    # 7. Collect results
    specs = get_machine_specs()
    percentiles_tokenize = compute_percentiles(stage_tokenize)
    percentiles_expand = compute_percentiles(stage_expand)
    percentiles_score = compute_percentiles(stage_score)
    percentiles_fetch = compute_percentiles(stage_fetch)
    percentiles_total = compute_percentiles(stage_total)

    target_met = percentiles_total["p95"] <= 100.0

    benchmark_data: dict[str, Any] = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_specs": specs,
        "corpus_docs": build_stats["num_docs"],
        "vocab_size": build_stats["vocab_size"],
        "index_size_mb": round(build_stats["index_size_bytes"] / (1024**2), 2),
        "index_size_bytes": build_stats["index_size_bytes"],
        "build_time_seconds": build_stats["build_time_seconds"],
        "process_rss_mb": round(get_process_rss_mb(), 2),
        "cold_load_time_ms": cold_load_time_ms,
        "queries_evaluated": len(queries),
        "throughput_qps": qps,
        "target_p95_under_100ms": target_met,
        "latency_percentiles_ms": {
            "tokenization": percentiles_tokenize,
            "phonetic_expansion": percentiles_expand,
            "postings_bm25_scoring": percentiles_score,
            "docstore_snippet_fetch": percentiles_fetch,
            "total_query": percentiles_total,
        },
    }

    # Save JSON results
    json_path = out_results / "latency_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    # Format human-readable table
    txt_path = out_results / "latency_benchmark.txt"
    report_text = format_benchmark_report(benchmark_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    return benchmark_data


def format_benchmark_report(data: dict[str, Any]) -> str:
    """Format benchmark metrics into markdown table."""
    p = data["latency_percentiles_ms"]
    specs = data["machine_specs"]
    tot = p["total_query"]

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  LATENCY BENCHMARK REPORT (100,000 DOCUMENTS / PASSAGES)")
    lines.append("=" * 80)
    lines.append(f"Documents Indexed:    {data['corpus_docs']:,}")
    lines.append(f"Vocabulary Size:      {data['vocab_size']:,} terms")
    lines.append(f"Index Disk Size:      {data['index_size_mb']:.2f} MB")
    lines.append(f"Process RSS Memory:   {data['process_rss_mb']:.2f} MB")
    lines.append(f"Cold Load Time:       {data['cold_load_time_ms']:.2f} ms")
    lines.append(f"Queries Evaluated:    {data['queries_evaluated']:,}")
    lines.append(f"Throughput:           {data['throughput_qps']:.1f} QPS (single-thread)")
    lines.append(f"Target (p95 <= 100ms): {'[PASSED]' if data['target_p95_under_100ms'] else '[MISSED]'}")
    lines.append("-" * 80)
    lines.append("Hardware Specifications:")
    lines.append(f"  CPU:                {specs.get('processor', 'Unknown')}")
    lines.append(f"  Logical Cores:      {specs.get('cpu_count_logical', 'Unknown')}")
    lines.append(f"  System RAM:         {specs.get('ram_total_gb', 'Unknown')} GB")
    lines.append(f"  Operating System:   {specs.get('os', 'Unknown')}")
    lines.append(f"  Python Runtime:     {specs.get('python_version', 'Unknown')}")
    lines.append("-" * 80)
    lines.append("Per-Stage Latency Breakdown (Single-Thread Warm, ms):")
    lines.append(f"{'Pipeline Stage':<28} {'p50':<10} {'p90':<10} {'p95':<10} {'p99':<10} {'Mean':<10}")
    lines.append("-" * 78)

    stages = [
        ("1. Tokenization", p["tokenization"]),
        ("2. Phonetic Expansion", p["phonetic_expansion"]),
        ("3. BM25 Postings Scoring", p["postings_bm25_scoring"]),
        ("4. DocStore Snippet Fetch", p["docstore_snippet_fetch"]),
        ("TOTAL (End-to-End)", tot),
    ]

    for label, metrics in stages:
        lines.append(
            f"{label:<28} {metrics['p50']:<10.2f} {metrics['p90']:<10.2f} "
            f"{metrics['p95']:<10.2f} {metrics['p99']:<10.2f} {metrics['mean']:<10.2f}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="Run 100k document latency benchmark.")
    parser.add_argument("--index", default="data/index_100k", help="100k index directory.")
    parser.add_argument("--passages", default="data/passages_100k.jsonl", help="100k JSONL file.")
    parser.add_argument("--queries", type=int, default=1050, help="Number of benchmark queries.")
    args = parser.parse_args()

    run_benchmark(
        index_dir=args.index,
        passages_jsonl=args.passages,
        num_benchmark_queries=args.queries,
    )


if __name__ == "__main__":
    main()
