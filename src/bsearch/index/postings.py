"""Block-based posting lists with delta-encoding, varint compression, and skip data.

Blocks contain up to 128 postings. Skip data records:
- max_doc_id in block
- byte offset of block payload
- number of postings in block
- max_score upper-bound for Block-Max WAND
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final

from bsearch.index.vbyte import decode_vbyte, decode_vbyte_list, encode_vbyte, encode_vbyte_list

BLOCK_SIZE: Final[int] = 128


@dataclass(frozen=True)
class Posting:
    doc_id: int
    body_tf: int
    title_tf: int = 0


@dataclass(frozen=True)
class BlockSkipInfo:
    max_doc_id: int
    offset: int
    count: int
    max_score: float = 0.0


def encode_postings_list(
    postings: list[Posting],
    block_size: int = BLOCK_SIZE,
    max_score_calculator=None,
) -> tuple[bytes, list[BlockSkipInfo], float]:
    """Encode a list of sorted postings into block-indexed varint format.

    Args:
        postings: Sorted list of Posting(doc_id, body_tf, title_tf).
        block_size: Number of postings per block (default 128).
        max_score_calculator: Optional function (Posting) -> float to compute BM25 score.

    Returns:
        (encoded_bytes, skip_list, term_max_score)
    """
    if not postings:
        return b"", [], 0.0

    num_postings = len(postings)
    num_blocks = (num_postings + block_size - 1) // block_size

    skips: list[BlockSkipInfo] = []
    blocks_payload = bytearray()
    term_max_score = 0.0

    last_doc_id = 0
    for b_idx in range(num_blocks):
        start = b_idx * block_size
        end = min(start + block_size, num_postings)
        block_postings = postings[start:end]
        count = len(block_postings)

        # Compute doc-id deltas
        deltas: list[int] = []
        body_tfs: list[int] = []
        title_tfs: list[int] = []

        block_max_score = 0.0
        for p in block_postings:
            delta = p.doc_id - last_doc_id
            deltas.append(delta)
            body_tfs.append(p.body_tf)
            title_tfs.append(p.title_tf)
            last_doc_id = p.doc_id

            if max_score_calculator:
                score = max_score_calculator(p)
                if score > block_max_score:
                    block_max_score = score
                if score > term_max_score:
                    term_max_score = score

        block_bytes = (
            encode_vbyte_list(deltas)
            + encode_vbyte_list(body_tfs)
            + encode_vbyte_list(title_tfs)
        )

        block_offset = len(blocks_payload)
        blocks_payload.extend(block_bytes)

        max_doc_in_block = block_postings[-1].doc_id
        skips.append(
            BlockSkipInfo(
                max_doc_id=max_doc_in_block,
                offset=block_offset,
                count=count,
                max_score=block_max_score,
            )
        )

    # Encode header:
    # num_blocks (vbyte)
    # for each block: max_doc_id (vbyte), offset (vbyte), count (vbyte), max_score (float 4-bytes '<f')
    header = bytearray()
    header.extend(encode_vbyte(num_blocks))
    for s in skips:
        header.extend(encode_vbyte(s.max_doc_id))
        header.extend(encode_vbyte(s.offset))
        header.extend(encode_vbyte(s.count))
        header.extend(struct.pack("<f", s.max_score))

    # Total serialized format: [header_length: 4 bytes '<I'] + [header] + [blocks_payload]
    full_encoded = bytearray()
    full_encoded.extend(struct.pack("<I", len(header)))
    full_encoded.extend(header)
    full_encoded.extend(blocks_payload)

    return bytes(full_encoded), skips, term_max_score


class BlockPostingIterator:
    """Fast iterator over block-indexed postings with O(1) skips and block-max bounds."""

    def __init__(self, data: bytes | memoryview, base_offset: int = 0) -> None:
        self.data = data
        self.base_offset = base_offset

        # Parse header length
        header_len = struct.unpack_from("<I", self.data, self.base_offset)[0]
        header_start = self.base_offset + 4
        self.payload_start = header_start + header_len

        # Parse skip table
        num_blocks, curr = decode_vbyte(self.data, header_start)
        self.skips: list[BlockSkipInfo] = []
        for _ in range(num_blocks):
            max_doc, curr = decode_vbyte(self.data, curr)
            b_offset, curr = decode_vbyte(self.data, curr)
            count, curr = decode_vbyte(self.data, curr)
            max_score = struct.unpack_from("<f", self.data, curr)[0]
            curr += 4
            self.skips.append(
                BlockSkipInfo(
                    max_doc_id=max_doc,
                    offset=b_offset,
                    count=count,
                    max_score=max_score,
                )
            )

        self.num_blocks = num_blocks
        self.curr_block_idx = -1
        self.curr_posting_idx = 0

        # Current block unpacked arrays
        self.curr_docs: list[int] = []
        self.curr_body_tfs: list[int] = []
        self.curr_title_tfs: list[int] = []
        self.last_decoded_doc_id = 0

        # Pointer to current posting
        self._doc_id: int | None = None
        self._body_tf: int = 0
        self._title_tf: int = 0

        # Load first block if exists
        if self.num_blocks > 0:
            self._load_block(0)
            self._update_current()

    def _load_block(self, block_idx: int) -> None:
        """Decode and decompress block at block_idx."""
        self.curr_block_idx = block_idx
        skip = self.skips[block_idx]
        b_start = self.payload_start + skip.offset
        count = skip.count

        deltas, next_off = decode_vbyte_list(self.data, count, b_start)
        body_tfs, next_off = decode_vbyte_list(self.data, count, next_off)
        title_tfs, _ = decode_vbyte_list(self.data, count, next_off)

        # Prefix sums to recover doc IDs
        docs = [0] * count
        curr_doc = self.last_decoded_doc_id
        for i, d in enumerate(deltas):
            curr_doc += d
            docs[i] = curr_doc
        self.last_decoded_doc_id = curr_doc

        self.curr_docs = docs
        self.curr_body_tfs = body_tfs
        self.curr_title_tfs = title_tfs
        self.curr_posting_idx = 0

    def _update_current(self) -> None:
        if self.curr_block_idx >= 0 and self.curr_posting_idx < len(self.curr_docs):
            self._doc_id = self.curr_docs[self.curr_posting_idx]
            self._body_tf = self.curr_body_tfs[self.curr_posting_idx]
            self._title_tf = self.curr_title_tfs[self.curr_posting_idx]
        else:
            self._doc_id = None
            self._body_tf = 0
            self._title_tf = 0

    @property
    def doc_id(self) -> int | None:
        return self._doc_id

    @property
    def body_tf(self) -> int:
        return self._body_tf

    @property
    def title_tf(self) -> int:
        return self._title_tf

    @property
    def block_max_score(self) -> float:
        if 0 <= self.curr_block_idx < self.num_blocks:
            return self.skips[self.curr_block_idx].max_score
        return 0.0

    def next_doc(self) -> int | None:
        """Advance iterator to next document."""
        if self._doc_id is None:
            return None

        self.curr_posting_idx += 1
        if self.curr_posting_idx < len(self.curr_docs):
            self._update_current()
            return self._doc_id

        # Need to move to next block
        next_block = self.curr_block_idx + 1
        if next_block < self.num_blocks:
            self._load_block(next_block)
            self._update_current()
            return self._doc_id

        # Exhausted
        self._doc_id = None
        return None

    def advance(self, target_doc_id: int) -> int | None:
        """Fast advance using skip pointers to first document >= target_doc_id."""
        if self._doc_id is None or self._doc_id >= target_doc_id:
            return self._doc_id

        # Skip blocks where max_doc_id < target_doc_id
        target_block = self.curr_block_idx
        while (
            target_block < self.num_blocks
            and self.skips[target_block].max_doc_id < target_doc_id
        ):
            target_block += 1

        if target_block >= self.num_blocks:
            self._doc_id = None
            return None

        if target_block != self.curr_block_idx:
            # Reconstruct last_decoded_doc_id up to target_block
            # Note: last_decoded_doc_id at end of block (b-1) is skips[b-1].max_doc_id!
            if target_block > 0:
                self.last_decoded_doc_id = self.skips[target_block - 1].max_doc_id
            else:
                self.last_decoded_doc_id = 0
            self._load_block(target_block)

        # Fast binary search within current block in C
        import bisect
        idx = bisect.bisect_left(self.curr_docs, target_doc_id, lo=self.curr_posting_idx)
        self.curr_posting_idx = idx
        if idx < len(self.curr_docs):
            self._update_current()
            return self._doc_id

        return self.next_doc()
