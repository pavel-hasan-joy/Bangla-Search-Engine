"""Separate document store for titles, texts, and snippets with fast O(1) random access.

Separates document presentation data from inverted index postings.
Uses binary offset indexing with mmap support for instant loads.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Any


class DocStoreWriter:
    """Writes document metadata, titles, and text to binary files."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "docstore.doci"
        self.data_path = self.output_dir / "docstore.docd"

        self._index_file = open(self.index_path, "wb")
        self._data_file = open(self.data_path, "wb")
        self.num_docs = 0

        # Reserve space for total_docs header in index file (4 bytes '<I')
        self._index_file.write(struct.pack("<I", 0))

    def append(self, ext_id: str, title: str, text: str, doc_len: int) -> int:
        """Append a document to store and return assigned integer doc_id."""
        doc_id = self.num_docs
        payload = json.dumps(
            {"id": ext_id, "title": title, "text": text},
            ensure_ascii=False,
        ).encode("utf-8")

        offset = self._data_file.tell()
        length = len(payload)
        self._data_file.write(payload)

        # Write index entry: offset (8 bytes '<Q'), length (4 bytes '<I'), doc_len (4 bytes '<I')
        self._index_file.write(struct.pack("<QII", offset, length, doc_len))

        self.num_docs += 1
        return doc_id

    def close(self) -> None:
        """Finalize files and write total document count header."""
        self._index_file.seek(0)
        self._index_file.write(struct.pack("<I", self.num_docs))
        self._index_file.close()
        self._data_file.close()


class DocStoreReader:
    """Fast random-access document store reader using memory-mapping."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.index_path = self.output_dir / "docstore.doci"
        self.data_path = self.output_dir / "docstore.docd"

        if not self.index_path.is_file() or not self.data_path.is_file():
            raise FileNotFoundError(f"DocStore files missing in {self.output_dir}")

        self._index_f = open(self.index_path, "rb")
        self._data_f = open(self.data_path, "rb")

        # Map index file
        self._index_mmap = mmap.mmap(self._index_f.fileno(), 0, access=mmap.ACCESS_READ)
        # Map data file if non-empty
        data_size = os.path.getsize(self.data_path)
        self._data_mmap = (
            mmap.mmap(self._data_f.fileno(), 0, access=mmap.ACCESS_READ)
            if data_size > 0
            else None
        )

        self.num_docs = struct.unpack_from("<I", self._index_mmap, 0)[0]
        # Entry format: offset (8 bytes), length (4 bytes), doc_len (4 bytes) = 16 bytes
        self.ENTRY_SIZE = struct.calcsize("<QII")

    def get(self, doc_id: int) -> dict[str, Any]:
        """Fetch document dictionary: {"id": str, "title": str, "text": str, "doc_len": int}."""
        if doc_id < 0 or doc_id >= self.num_docs:
            raise IndexError(f"doc_id {doc_id} out of bounds (total: {self.num_docs})")

        entry_pos = 4 + doc_id * self.ENTRY_SIZE
        offset, length, doc_len = struct.unpack_from("<QII", self._index_mmap, entry_pos)

        if not self._data_mmap:
            return {"id": str(doc_id), "title": "", "text": "", "doc_len": doc_len}

        raw_bytes = self._data_mmap[offset : offset + length]
        doc = json.loads(raw_bytes.decode("utf-8"))
        doc["doc_len"] = doc_len
        return doc

    def get_doc_len(self, doc_id: int) -> int:
        """Fetch document token length without decoding full document payload."""
        if doc_id < 0 or doc_id >= self.num_docs:
            raise IndexError(f"doc_id {doc_id} out of bounds")
        entry_pos = 4 + doc_id * self.ENTRY_SIZE + 12  # +8 (offset) +4 (length) = 12
        return struct.unpack_from("<I", self._index_mmap, entry_pos)[0]

    def get_snippet(self, doc_id: int, query_terms: list[str] | None = None, max_chars: int = 180) -> str:
        """Extract a snippet around matching query terms or first sentence."""
        doc = self.get(doc_id)
        text = doc.get("text", "")
        if not text:
            return ""

        if not query_terms:
            return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")

        # Look for first occurrence of any query term
        best_pos = -1
        lower_text = text.lower()
        for term in query_terms:
            pos = lower_text.find(term.lower())
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos

        if best_pos == -1:
            return text[:max_chars].strip() + ("..." if len(text) > max_chars else "")

        start = max(0, best_pos - 40)
        end = min(len(text), start + max_chars)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return prefix + text[start:end].strip() + suffix

    def close(self) -> None:
        if self._data_mmap:
            self._data_mmap.close()
        self._index_mmap.close()
        self._data_f.close()
        self._index_f.close()

    def __enter__(self) -> DocStoreReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
