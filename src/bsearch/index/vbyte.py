"""Variable-byte (varint) integer compression for posting lists.

Encodes unsigned integers using 7 bits of payload per byte with the 8th bit
as a continuation flag. Compact, fast, and implemented with Python standard library.
"""

from __future__ import annotations


def encode_vbyte(value: int) -> bytes:
    """Encode a single non-negative integer using variable-byte encoding."""
    if value < 0:
        raise ValueError("vbyte only supports non-negative integers")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def encode_vbyte_list(values: list[int]) -> bytes:
    """Encode a sequence of non-negative integers into a continuous varint byte buffer."""
    out = bytearray()
    for val in values:
        if val < 0:
            raise ValueError("vbyte only supports non-negative integers")
        while val >= 0x80:
            out.append((val & 0x7F) | 0x80)
            val >>= 7
        out.append(val & 0x7F)
    return bytes(out)


def decode_vbyte(data: bytes | memoryview, offset: int = 0) -> tuple[int, int]:
    """Decode a single integer from data starting at offset.

    Returns:
        (decoded_value, new_offset)
    """
    val = 0
    shift = 0
    i = offset
    n = len(data)
    while i < n:
        b = data[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, i
        shift += 7
    raise EOFError("Unexpected end of data while decoding varint")


def decode_vbyte_list(data: bytes | memoryview, count: int, offset: int = 0) -> tuple[list[int], int]:
    """Decode exactly `count` integers from data starting at offset.

    Returns:
        (list_of_values, new_offset)
    """
    res: list[int] = [0] * count
    curr_offset = offset
    for idx in range(count):
        val = 0
        shift = 0
        while True:
            b = data[curr_offset]
            curr_offset += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        res[idx] = val
    return res, curr_offset
