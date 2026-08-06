"""NGP canonicalizer: map an (a,b) pair to its equivalence class representative.

All BFS operations on Python ints (zero numpy allocations in the hot path).
LSB = a[0], MSB = a[n-1] for natural shift semantics.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

# --- int encoding (LSB = a[0]) -------------------------------------------------


def _to_int(seq: np.ndarray) -> int:
    """Pack ±1 int8 array to bit-packed Python int.  LSB = seq[0]."""
    result = 0
    for v in reversed(seq):
        result = (result << 1) | (1 if v == 1 else 0)
    return result


def _to_pair_int(a: np.ndarray, b: np.ndarray) -> int:
    """(a,b) → Python int: b in low n bits, a in high n bits."""
    n = len(a)
    return (_to_int(a) << n) | _to_int(b)


def _from_pair_int(val: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode back to arrays.  Only called for the best candidate."""
    mask = (1 << n) - 1
    b_int = val & mask
    a_int = (val >> n) & mask

    a = np.empty(n, dtype=np.int8)
    b = np.empty(n, dtype=np.int8)
    for i in range(n):
        a[i] = 1 if (a_int >> i) & 1 else -1
        b[i] = 1 if (b_int >> i) & 1 else -1
    return a, b


# --- fast int-level transformations (no allocations) ---------------------------


def _negashift(val: int, n: int) -> int:
    """Negacyclic shift of n bits: LSB = NOT old MSB."""
    mask = (1 << n) - 1
    shifted = (val << 1) & mask
    return shifted | (1 ^ ((val >> (n - 1)) & 1))


def _reverse(val: int, n: int) -> int:
    """Reverse n bits."""
    result = 0
    v = val
    for _ in range(n):
        result = (result << 1) | (v & 1)
        v >>= 1
    return result


def _negate_odd(val: int, n: int, dbl: int) -> int:
    """Toggle odd bits in both sequences (odd index bits in each n-bit half)."""
    # odd-index bits: positions 1,3,5,... and n+1, n+3, ...
    mask = 0
    for i in range(1, dbl, 2):
        mask |= 1 << i
    return val ^ mask


# --- decimation tables ---------------------------------------------------------


@lru_cache(maxsize=256)
def _decimation_maps(n: int) -> tuple[tuple[list[tuple[int, int]], int], ...]:
    """For each coprime k: (bit-perm pairs, xor_mask)."""
    result = []
    for k in range(1, n):
        if np.gcd(k, n) != 1:
            continue
        pairs: list[tuple[int, int]] = []
        xor = 0
        for i in range(n):
            j = (k * i) % n
            zi = 1 if (k * i) % (2 * n) < n else -1
            # a[i] ← sign * a[j]:  dst bit = i, src bit = j
            pairs.append((i, j))
            # b[i] ← sign * b[j]:  dst bit = n + i, src bit = n + j
            pairs.append((n + i, n + j))
            if zi == -1:
                xor |= 1 << i
                xor |= 1 << (n + i)
        result.append((pairs, xor))
    return tuple(result)


def _decimate(val: int, pairs: list[tuple[int, int]], xor_mask: int) -> int:
    """Apply precomputed decimation to a pair int."""
    out = 0
    for dst, src in pairs:
        if (val >> src) & 1:
            out |= 1 << dst
    return out ^ xor_mask


# --- canonical form (BFS over ints) --------------------------------------------


def canonical(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical (a,b) under NGP equivalence.

    BFS over Python ints.  10-50x faster than array-based BFS.
    """
    n = len(a)
    dbl = 2 * n
    maps = _decimation_maps(n)

    start = _to_pair_int(a, b)
    best = start

    seen: set[int] = {start}
    queue: list[int] = [start]

    while queue:
        cur = queue.pop()

        # extract a and b parts
        mask_seq = (1 << n) - 1
        cur_a = (cur >> n) & mask_seq
        cur_b = cur & mask_seq

        # 1-2: reverse / negashift individually
        variants = [
            (_reverse(cur_a, n) << n) | cur_b,
            (cur_a << n) | _reverse(cur_b, n),
            (_negashift(cur_a, n) << n) | cur_b,
            (cur_a << n) | _negashift(cur_b, n),
        ]
        for v in variants:
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if v < best:
                    best = v

        # 3: swap
        v = (cur_b << n) | cur_a
        if v not in seen:
            seen.add(v)
            queue.append(v)
            if v < best:
                best = v

        # 4: decimate
        for pairs, xor_mask in maps:
            v = _decimate(cur, pairs, xor_mask)
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if v < best:
                    best = v

        # 5: negate odd positions
        v = _negate_odd(cur, n, dbl)
        if v not in seen:
            seen.add(v)
            queue.append(v)
            if v < best:
                best = v

    return _from_pair_int(best, n)


def class_hash(a: np.ndarray, b: np.ndarray) -> str:
    """SHA-256 of canonical representative → equivalence class hash."""
    ca, cb = canonical(a, b)
    bits = np.concatenate((ca == 1, cb == 1)).astype(np.uint8)
    raw = np.packbits(bits, bitorder="big").tobytes()
    return hashlib.sha256(raw).hexdigest()
