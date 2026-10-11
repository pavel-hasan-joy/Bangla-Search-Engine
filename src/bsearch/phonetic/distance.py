"""Hand-written Damerau-Levenshtein edit distance using standard library only.

Computes the minimum edit operations (insertions, deletions, substitutions,
and transpositions of two adjacent characters) required to transform s1 into s2.
"""

from __future__ import annotations


def damerau_levenshtein(s1: str, s2: str, max_distance: int | None = None) -> int:
    """Compute exact Damerau-Levenshtein distance between strings s1 and s2.

    Supports:
    - Insertion (+1)
    - Deletion (+1)
    - Substitution (+1)
    - Adjacent transposition (+1)
    """
    len1 = len(s1)
    len2 = len(s2)

    if max_distance is not None and abs(len1 - len2) > max_distance:
        return max_distance + 1

    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    if s1 == s2:
        return 0

    # da dictionary stores the last seen position of each character
    da: dict[str, int] = {}

    # Distance matrix dimensions: (len1 + 2) x (len2 + 2)
    max_dist = len1 + len2
    h = [[0] * (len2 + 2) for _ in range(len1 + 2)]

    h[0][0] = max_dist
    for i in range(len1 + 1):
        h[i + 1][0] = max_dist
        h[i + 1][1] = i
    for j in range(len2 + 1):
        h[0][j + 1] = max_dist
        h[1][j + 1] = j

    for i in range(1, len1 + 1):
        db = 0
        ch1 = s1[i - 1]
        for j in range(1, len2 + 1):
            ch2 = s2[j - 1]
            k = da.get(ch2, 0)
            l = db

            cost = 0 if ch1 == ch2 else 1
            if cost == 0:
                db = j

            h[i + 1][j + 1] = min(
                h[i][j + 1] + 1,          # Deletion
                h[i + 1][j] + 1,          # Insertion
                h[i][j] + cost,           # Substitution
                h[k][l] + (i - k - 1) + 1 + (j - l - 1),  # Transposition
            )

        da[ch1] = i

    return h[len1 + 1][len2 + 1]
