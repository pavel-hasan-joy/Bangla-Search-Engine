"""Single-pass index builder with block-based varint compression and separate docstore.

Builds:
- Separate DocStore (docstore.doci + docstore.docd)
- Block-compressed inverted index (postings.bin)
- Binary term dictionary (index.dict)
- Metadata (index.meta)

Measures and reports:
- Build time
- Total index disk size
- Process RSS memory consumption
- Memory-mapped load time
"""

from __future__ import annotations

import argparse
import collections
import ctypes
from ctypes import wintypes
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bsearch.index.docstore import DocStoreWriter
from bsearch.index.inverted_index import InvertedIndexReader, MAGIC_HEADER
from bsearch.index.postings import Posting, encode_postings_list
from bsearch.rank.bm25 import BM25Params, compute_idf, score_term_bm25
from bsearch.unicode.tokenizer import tokenize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bsearch.index.builder")


def get_process_rss_mb() -> float:
    """Get current process Resident Set Size (RSS) in MB using standard library ctypes."""
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        # Try kernel32 K32GetProcessMemoryInfo then psapi GetProcessMemoryInfo
        func = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None) or getattr(
            ctypes.windll.psapi, "GetProcessMemoryInfo", None
        )
        if func:
            func.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
            func.restype = ctypes.c_int
            ok = func(handle, ctypes.byref(counters), counters.cb)
            if ok:
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception as e:
        logger.info("Failed to fetch RSS via ctypes: %s", e)

    return 0.0


@dataclass
class IndexBuildStats:
    num_docs: int
    vocab_size: int
    total_tokens: int
    avgdl: float
    build_time_seconds: float
    index_size_bytes: int
    rss_mb: float
    load_time_seconds: float


def build_index_from_jsonl(
    jsonl_path: Path | str,
    output_dir: Path | str,
    max_docs: int | None = None,
    params: BM25Params | None = None,
) -> IndexBuildStats:
    """Single-pass index builder reading documents from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file with 'title' and 'text'.
        output_dir: Directory where index files will be created.
        max_docs: Optional maximum document limit.
        params: BM25 parameters for term upper-bound precomputation.

    Returns:
        IndexBuildStats with size, timing, and memory metrics.
    """
    p = params or BM25Params()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    docstore_writer = DocStoreWriter(out_dir)
    postings_map: dict[str, list[Posting]] = collections.defaultdict(list)
    doc_lengths: list[int] = []

    logger.info("Starting single-pass index construction from: %s", jsonl_path)
    total_tokens = 0
    doc_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            doc_obj = json.loads(line)
            title = doc_obj.get("title", "")
            text = doc_obj.get("text", "")
            ext_id = str(doc_obj.get("id", f"doc_{doc_count}"))

            title_tokens = tokenize(title, stem=False)
            body_tokens = tokenize(text, stem=False)

            doc_len = len(title_tokens) + len(body_tokens)
            doc_id = docstore_writer.append(ext_id=ext_id, title=title, text=text, doc_len=doc_len)

            doc_lengths.append(doc_len)
            total_tokens += doc_len
            doc_count += 1

            # Count term frequencies
            title_counts = collections.Counter(title_tokens)
            body_counts = collections.Counter(body_tokens)
            all_doc_terms = set(title_counts.keys()) | set(body_counts.keys())

            for term in all_doc_terms:
                b_tf = body_counts.get(term, 0)
                t_tf = title_counts.get(term, 0)
                postings_map[term].append(Posting(doc_id=doc_id, body_tf=b_tf, title_tf=t_tf))

            if doc_count % 10000 == 0:
                logger.info("Indexed %d documents...", doc_count)

            if max_docs and doc_count >= max_docs:
                break

    docstore_writer.close()
    avgdl = (total_tokens / doc_count) if doc_count > 0 else 0.0

    logger.info(
        "Corpus ingested: %d documents, %d distinct terms, avgdl=%.2f. Serializing postings...",
        doc_count,
        len(postings_map),
        avgdl,
    )

    # Serialize postings and term dictionary
    postings_path = out_dir / "postings.bin"
    dict_path = out_dir / "index.dict"

    sorted_terms = sorted(postings_map.keys())
    dict_entries = []

    with open(postings_path, "wb") as f_postings:
        for term in sorted_terms:
            postings = postings_map[term]
            df = len(postings)
            idf = compute_idf(doc_count, df)

            def _score_posting(post: Posting) -> float:
                d_len = doc_lengths[post.doc_id]
                return score_term_bm25(
                    body_tf=post.body_tf,
                    title_tf=post.title_tf,
                    doc_len=d_len,
                    avgdl=avgdl,
                    idf=idf,
                    params=p,
                )

            encoded_bytes, _, max_score = encode_postings_list(
                postings,
                max_score_calculator=_score_posting,
            )

            offset = f_postings.tell()
            f_postings.write(encoded_bytes)
            length = len(encoded_bytes)

            dict_entries.append((term, df, offset, length, max_score))

    # Write dictionary
    with open(dict_path, "wb") as f_dict:
        f_dict.write(MAGIC_HEADER)
        f_dict.write(struct.pack("<I", len(dict_entries)))
        for term, df, offset, length, max_score in dict_entries:
            term_bytes = term.encode("utf-8")
            f_dict.write(struct.pack("<H", len(term_bytes)))
            f_dict.write(term_bytes)
            f_dict.write(struct.pack("<IIIf", df, offset, length, max_score))

    # Write metadata
    meta = {
        "num_docs": doc_count,
        "total_tokens": total_tokens,
        "avgdl": avgdl,
        "vocab_size": len(sorted_terms),
    }
    meta_path = out_dir / "index.meta"
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)

    build_time = time.perf_counter() - start_time
    rss_mb = get_process_rss_mb()

    # Calculate index size on disk
    index_files = [
        meta_path,
        dict_path,
        postings_path,
        out_dir / "docstore.doci",
        out_dir / "docstore.docd",
    ]
    total_size = sum(os.path.getsize(p) for p in index_files if p.exists())

    # Measure mmap load time
    with InvertedIndexReader(out_dir) as reader:
        load_time = reader.load_time_seconds

    stats = IndexBuildStats(
        num_docs=doc_count,
        vocab_size=len(sorted_terms),
        total_tokens=total_tokens,
        avgdl=avgdl,
        build_time_seconds=build_time,
        index_size_bytes=total_size,
        rss_mb=rss_mb,
        load_time_seconds=load_time,
    )

    logger.info("Index build complete!")
    logger.info("Documents:    %d", stats.num_docs)
    logger.info("Vocabulary:   %d terms", stats.vocab_size)
    logger.info("Avg Doc Len:  %.2f tokens", stats.avgdl)
    logger.info("Build Time:   %.2f seconds", stats.build_time_seconds)
    logger.info("Total Size:   %.2f MB (%d bytes)", stats.index_size_bytes / (1024 * 1024), stats.index_size_bytes)
    logger.info("Process RSS:  %.2f MB", stats.rss_mb)
    logger.info("Load Time:    %.4f seconds (mmap)", stats.load_time_seconds)

    return stats


def build_index_from_docs(
    docs: list[dict[str, Any]],
    output_dir: Path | str,
    params: BM25Params | None = None,
) -> IndexBuildStats:
    """Convenience builder to index a list of in-memory document dictionaries."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = out_dir / "_temp_docs.jsonl"
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    try:
        return build_index_from_jsonl(tmp_jsonl, out_dir, params=params)
    finally:
        if tmp_jsonl.exists():
            tmp_jsonl.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bsearch inverted index from JSONL.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL path.")
    parser.add_argument("--output", "-o", type=str, default="data/index", help="Output index directory.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to index.")
    args = parser.parse_args()

    stats = build_index_from_jsonl(args.input, args.output, max_docs=args.limit)
    print("\n--- Index Build Summary ---")
    print(f"Documents:    {stats.num_docs:,}")
    print(f"Vocabulary:   {stats.vocab_size:,} terms")
    print(f"Index Size:   {stats.index_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Build Time:   {stats.build_time_seconds:.2f}s")
    print(f"Memory RSS:   {stats.rss_mb:.2f} MB")
    print(f"Load Time:    {stats.load_time_seconds * 1000:.2f} ms")
    print("---------------------------\n")


if __name__ == "__main__":
    main()
