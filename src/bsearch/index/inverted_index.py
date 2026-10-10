"""Inverted index binary serialization and fast memory-mapped reader.

Supports:
- Term dictionary with df, postings offset/length, and term max_score.
- Document lengths array and corpus avgdl.
- Zero-copy postings access via mmap.
- Fast, instant index loading.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bsearch.index.postings import BlockPostingIterator

MAGIC_HEADER: Final[bytes] = b"BIDX01"


@dataclass(frozen=True)
class TermInfo:
    df: int
    postings_offset: int
    postings_length: int
    max_score: float = 0.0


class InvertedIndexReader:
    """Zero-copy memory-mapped reader for the inverted index."""

    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self.meta_path = self.index_dir / "index.meta"
        self.dict_path = self.index_dir / "index.dict"
        self.postings_path = self.index_dir / "postings.bin"

        start_time = time.perf_counter()

        if not self.meta_path.is_file():
            raise FileNotFoundError(f"Missing index metadata: {self.meta_path}")

        # 1. Read metadata
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.num_docs: int = meta["num_docs"]
        self.total_tokens: int = meta["total_tokens"]
        self.avgdl: float = meta["avgdl"]
        self.vocab_size: int = meta["vocab_size"]

        # 2. Read term dictionary
        self.terms: dict[str, TermInfo] = {}
        with open(self.dict_path, "rb") as f:
            magic = f.read(len(MAGIC_HEADER))
            if magic != MAGIC_HEADER:
                raise ValueError("Invalid dictionary magic header")
            count = struct.unpack("<I", f.read(4))[0]
            for _ in range(count):
                term_len = struct.unpack("<H", f.read(2))[0]
                term = f.read(term_len).decode("utf-8")
                df, offset, length, max_score = struct.unpack("<IIIf", f.read(16))
                self.terms[term] = TermInfo(
                    df=df,
                    postings_offset=offset,
                    postings_length=length,
                    max_score=max_score,
                )

        # 3. Memory-map postings file
        self._postings_file = open(self.postings_path, "rb")
        postings_size = os.path.getsize(self.postings_path)
        if postings_size > 0:
            self._postings_mmap = mmap.mmap(
                self._postings_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        else:
            self._postings_mmap = None

        self.load_time_seconds: float = time.perf_counter() - start_time

    def get_iterator(self, term: str) -> BlockPostingIterator | None:
        """Obtain a posting iterator for the given term."""
        info = self.terms.get(term)
        if not info or not self._postings_mmap:
            return None
        return BlockPostingIterator(self._postings_mmap, base_offset=info.postings_offset)

    def get_df(self, term: str) -> int:
        info = self.terms.get(term)
        return info.df if info else 0

    def get_term_max_score(self, term: str) -> float:
        info = self.terms.get(term)
        return info.max_score if info else 0.0

    def contains(self, term: str) -> bool:
        return term in self.terms

    def close(self) -> None:
        if self._postings_mmap:
            self._postings_mmap.close()
        self._postings_file.close()

    def __enter__(self) -> InvertedIndexReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
