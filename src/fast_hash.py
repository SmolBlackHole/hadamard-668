"""Fast invariant hash for GS4 sequence sets.

Covers negashift + reverse per sequence, then lexicographic sort.
Does NOT cover decimation — this is a diversity bound, not a full
canonicalizer.

Different hashes => different equivalence class.
Same hash => possibly same class (dedup signal).
"""

from __future__ import annotations

import hashlib

import numpy as np


def _canon_int(val: int, n: int) -> int:
    """Minimal integer among all negashift+reverse variants, O(n)."""
    best = val
    cur = val
    for _ in range(2 * n):
        mask = (1 << n) - 1
        shifted = (cur << 1) & mask
        cur = shifted | (1 ^ ((cur >> (n - 1)) & 1))
        if cur < best:
            best = cur
    rev = 0
    v = val
    for _ in range(n):
        rev = (rev << 1) | (v & 1)
        v >>= 1
    cur = rev
    for _ in range(2 * n):
        mask = (1 << n) - 1
        shifted = (cur << 1) & mask
        cur = shifted | (1 ^ ((cur >> (n - 1)) & 1))
        if cur < best:
            best = cur
    return best


def _to_int(seq: np.ndarray) -> int:
    result = 0
    for v in reversed(seq):
        result = (result << 1) | (1 if v == 1 else 0)
    return result


def gs4_class_hash(seqs: np.ndarray) -> str:
    """Fast invariant hash for four GS4 sequences.

    Canonizes each sequence under negashift+reverse, sorts them
    lexicographically, and returns sha256.
    """
    n = seqs.shape[1]
    ints = sorted(_canon_int(_to_int(seqs[s]), n) for s in range(4))
    payload = b"".join(v.to_bytes((n + 7) // 8, "big") for v in ints)
    return hashlib.sha256(payload).hexdigest()
