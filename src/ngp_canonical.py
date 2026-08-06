"""NGP canonicalizer: BFS over Python ints (correct, no shortcuts).

Covers all 5 NGP symmetries via BFS closure over integer encodings.
O(|orbit|) time — sufficient for n <= 20 (Egan's reference range).
For n > 20 use fast_orbit_hash (shift+rev+swap only) as diversity bound.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

# --- int encoding (LSB = a[0]) -------------------------------------------------


def _to_int(seq: np.ndarray) -> int:
    result = 0
    for v in reversed(seq):
        result = (result << 1) | (1 if v == 1 else 0)
    return result


def _to_pair_int(a: np.ndarray, b: np.ndarray) -> int:
    n = len(a)
    return (_to_int(a) << n) | _to_int(b)


def _from_pair_int(val: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << n) - 1
    b_int = val & mask
    a_int = (val >> n) & mask
    a = np.empty(n, dtype=np.int8)
    b = np.empty(n, dtype=np.int8)
    for i in range(n):
        a[i] = 1 if (a_int >> i) & 1 else -1
        b[i] = 1 if (b_int >> i) & 1 else -1
    return a, b


def _encode_pair(a: np.ndarray, b: np.ndarray) -> bytes:
    bits = np.concatenate((a == 1, b == 1)).astype(np.uint8)
    return np.packbits(bits, bitorder="big").tobytes()


# --- O(1) int-level transformations -------------------------------------------


def _negashift(val: int, n: int) -> int:
    mask = (1 << n) - 1
    shifted = (val << 1) & mask
    return shifted | (1 ^ ((val >> (n - 1)) & 1))


def _reverse(val: int, n: int) -> int:
    result = 0
    v = val
    for _ in range(n):
        result = (result << 1) | (v & 1)
        v >>= 1
    return result


def _negate_odd(val: int, n: int) -> int:
    dbl = 2 * n
    mask = 0
    for i in range(1, dbl, 2):
        mask |= 1 << i
    return val ^ mask


# --- decimation tables --------------------------------------------------------


@lru_cache(maxsize=256)
def _decimation_maps(n: int) -> tuple[tuple[list[tuple[int, int]], int], ...]:
    """For each k coprime to n: (bit-perm-pairs, xor_mask)."""
    result = []
    for k in range(1, n):
        if np.gcd(k, n) != 1:
            continue
        pairs: list[tuple[int, int]] = []
        xor = 0
        for i in range(n):
            j = (k * i) % n
            zi = 1 if (k * i) % (2 * n) < n else -1
            pairs.append((i, j))
            pairs.append((n + i, n + j))
            if zi == -1:
                xor |= 1 << i
                xor |= 1 << (n + i)
        result.append((pairs, xor))
    return tuple(result)


def _decimate(val: int, pairs: list[tuple[int, int]], xor_mask: int) -> int:
    out = 0
    for dst, src in pairs:
        if (val >> src) & 1:
            out |= 1 << dst
    return out ^ xor_mask


# --- full BFS canonical form --------------------------------------------------


def canonical(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical (a,b) under full NGP equivalence.

    BFS visits the entire equivalence orbit.  Correct for all n but
    O(|orbit|) time — use only for n <= 20 in production sweeps.
    """
    n = len(a)
    maps = _decimation_maps(n)

    start = _to_pair_int(a, b)
    best = start
    seen: set[int] = {start}
    queue: list[int] = [start]

    while queue:
        cur = queue.pop()
        mask = (1 << n) - 1
        cur_a = (cur >> n) & mask
        cur_b = cur & mask

        for v in [
            (_reverse(cur_a, n) << n) | cur_b,
            (cur_a << n) | _reverse(cur_b, n),
            (_negashift(cur_a, n) << n) | cur_b,
            (cur_a << n) | _negashift(cur_b, n),
        ]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if v < best:
                    best = v

        v = (cur_b << n) | cur_a  # swap
        if v not in seen:
            seen.add(v)
            queue.append(v)
            if v < best:
                best = v

        for pairs, xor_mask in maps:
            v = _decimate(cur, pairs, xor_mask)
            if v not in seen:
                seen.add(v)
                queue.append(v)
                if v < best:
                    best = v

        v = _negate_odd(cur, n)
        if v not in seen:
            seen.add(v)
            queue.append(v)
            if v < best:
                best = v

    return _from_pair_int(best, n)


def class_hash(a: np.ndarray, b: np.ndarray) -> str:
    """SHA-256 of full BFS canonical representative."""
    ca, cb = canonical(a, b)
    return hashlib.sha256(_encode_pair(ca, cb)).hexdigest()


# --- fast subgroup signature (shift+rev+swap, no BFS) -------------------------


def _canon_int(val: int, n: int) -> int:
    """Minimal int among all shift+reverse variants (O(n), no BFS)."""
    best = val
    cur = val
    for _ in range(2 * n):
        cur = _negashift(cur, n)
        if cur < best:
            best = cur
    cur = _reverse(val, n)
    for _ in range(2 * n):
        cur = _negashift(cur, n)
        if cur < best:
            best = cur
    return best


def fast_orbit_hash(a: np.ndarray, b: np.ndarray) -> str:
    """Fast diversity bound: shift+reverse+swap only, NOT full NGP equivalence.

    Different hashes imply different NGP classes (upper bound on diversity).
    Same hash does NOT imply same class (lower bound, not equality).
    """
    n = len(a)

    def _int_to_arr(val):
        arr = np.empty(n, dtype=np.int8)
        for i in range(n):
            arr[i] = 1 if (val >> i) & 1 else -1
        return arr

    ca = _canon_int(_to_int(a), n)
    cb = _canon_int(_to_int(b), n)
    # lexicographic min of (ca,cb) and (cb,ca) = swap-invariant
    if ca < cb or (ca == cb and _to_int(a) < _to_int(b)):
        return hashlib.sha256(_encode_pair(_int_to_arr(ca), _int_to_arr(cb))).hexdigest()
    return hashlib.sha256(_encode_pair(_int_to_arr(cb), _int_to_arr(ca))).hexdigest()
