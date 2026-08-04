"""Known Turyn TT(n) solutions — lazy loader + equivalence-aware hashing.

All equivalence-class representatives live in ``data/tt_index.json``.
Build the index with::

    python data/setup.py

Sources:
  Uleth archive (n <= 32, all equivalence classes)
    https://www.cs.uleth.ca/~hadi/research/TurynType/
  Best, Djokovic, Kharaghani, Ramp (2012) — TT(34), TT(38)
    J. Combin. Designs 21: 24-35, 2013.  DOI: 10.1002/jcd.21318
  Djokovic, Kotsireas (2025) — TT(40), TT(42), TT(44)
    Cryptography and Communications, 2025.  DOI: 10.1007/S12095-025-00829-Z
  Kharaghani, Tayfeh-Rezaie (2005) — TT(36)
    J. Combin. Designs 13: 435-440, 2005.

Hex format: each digit encodes 4 bits (A,B,C,D); last digit encodes 3 bits (A,B,C).
Bit 0 -> +1, bit 1 -> -1.  D at position n is implicitly 0.
"""

from __future__ import annotations

import json as _json
import os as _os
from functools import cache as _cache

import numpy as np

_INDEX_PATH = _os.path.join(_os.path.dirname(__file__), "..", "data", "tt_index.json")


@_cache
def _load_index() -> dict[str, list[str]]:
    if not _os.path.isfile(_INDEX_PATH):
        raise FileNotFoundError(
            f"TT index not found: {_INDEX_PATH}\n"
            "Run: python data/setup.py"
        )
    with open(_INDEX_PATH, encoding="utf-8") as _f:
        return _json.load(_f)


@_cache
def _known_hex_set(n: int) -> frozenset[str]:
    """All canonical hex strings for TT(n).  O(1) lookup for solution detection."""
    idx = _load_index()
    key = str(n)
    if key not in idx:
        return frozenset()
    return frozenset(idx[key])


def _parse_hex(hex_str: str) -> tuple[tuple[int, ...], ...]:
    """Convert a TT(n) hex string (n digits) to 4 tuples of +-1."""
    seqs: list[list[int]] = [[], [], [], []]
    for d in hex_str[:-1]:
        for i, b in enumerate(f"{int(d, 16):04b}"):
            seqs[i].append(1 if b == "0" else -1)
    for i, b in enumerate(f"{int(hex_str[-1], 16):03b}"):
        seqs[i].append(1 if b == "0" else -1)
    return tuple(tuple(s) for s in seqs)


def _seq_to_hex(seq: np.ndarray, lengths: np.ndarray, n: int) -> str:
    """Encode (4, n) sequences as a canonical-format hex string."""
    chars: list[str] = []
    for pos in range(n - 1):
        val = 0
        for i in range(4):
            if pos < int(lengths[i]) and seq[i, pos] == -1:
                val |= 1 << (3 - i)
        chars.append(f"{val:x}")
    val = 0
    for i in range(3):
        if seq[i, n - 1] == -1:
            val |= 1 << (2 - i)
    chars.append(f"{val:x}")
    return "".join(chars)


def _all_transforms_hex(seq: np.ndarray, lengths: np.ndarray, n: int) -> frozenset[str]:
    """All 1024 transforms of ``seq``, hex-encoded.  2 sec per call for TT(30)."""
    alt_pattern = np.ones(n, dtype=np.int8)
    alt_pattern[1::2] = -1

    results: set[str] = set()
    for swap_xy in (False, True):
        s = seq.copy()
        if swap_xy:
            s[[0, 1]] = s[[1, 0]]
        for apply_t3 in (False, True):
            st = s.copy()
            if apply_t3:
                for i in range(4):
                    L = int(lengths[i])
                    st[i, :L] = st[i, :L] * alt_pattern[:L]
            for neg_mask in range(16):
                sn = st.copy()
                for i in range(4):
                    if neg_mask & (1 << i):
                        sn[i] = -sn[i]
                for rev_mask in range(16):
                    sr = sn.copy()
                    for i in range(4):
                        if rev_mask & (1 << i):
                            L = int(lengths[i])
                            sr[i, :L] = sr[i, L - 1 :: -1]
                    results.add(_seq_to_hex(sr, lengths, n))
    return frozenset(results)


# ── Public API ─────────────────────────────────────────────────────────────────


def all_representatives(n: int) -> list[tuple[tuple[int, ...], ...]]:
    """All known TT(n) equivalence-class representatives (parsed on demand)."""
    idx = _load_index()
    key = str(n)
    if key not in idx:
        return []
    return [_parse_hex(h) for h in idx[key]]


def tt_sequences(n: int, index: int = 0) -> np.ndarray:
    """Return the ``index``-th TT(n) representative as (4, n) int8."""
    reps = all_representatives(n)
    if not reps or index >= len(reps):
        raise IndexError(f"no representative {index} for TT({n})")
    rep = reps[index]
    arr = np.zeros((4, n), dtype=np.int8)
    for i, s in enumerate(rep):
        arr[i, : len(s)] = np.array(s, dtype=np.int8)
    return arr


def tt_lengths(n: int) -> np.ndarray:
    """Return TT(n) lengths: (n, n, n, n-1)."""
    return np.array((n, n, n, n - 1), dtype=np.int64)


def known_solution(n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sequences, lengths) for the first known TT(n), or None."""
    reps = all_representatives(n)
    if not reps:
        return None
    return tt_sequences(n), tt_lengths(n)


def hamming_distance(
    seq_a: np.ndarray, seq_b: np.ndarray, lengths: np.ndarray,
) -> int:
    """Count bits that differ between two padded TT(n) sequence arrays."""
    if seq_a.shape != seq_b.shape or seq_a.shape[0] != 4:
        raise ValueError("sequences must have shape (4, n)")
    mask = np.zeros(seq_a.shape, dtype=bool)
    for i, L in enumerate(lengths):
        mask[i, : int(L)] = True
    return int(np.sum((seq_a != seq_b) & mask))


# ── Solution detection ─────────────────────────────────────────────────────────


def is_known_solution(seq: np.ndarray, lengths: np.ndarray, n: int) -> bool:
    """Check whether ``seq`` is equivalent to any known TT(n) solution.

    Generates all 1024 transforms of ``seq``, hex-encodes them, and checks
    against the known canonical hex set via O(1) set intersection.
    """
    known = _known_hex_set(n)
    if not known:
        return False
    return not _all_transforms_hex(seq, lengths, n).isdisjoint(known)


# ── Hamming distance ───────────────────────────────────────────────────────────


def equiv_hamming(seq: np.ndarray, lengths: np.ndarray, n: int) -> int | None:
    """Minimum Hamming distance over ALL known TT(n) equivalence classes.

    Phase 1 (fast): hex-hash all 1024 transforms — if any matches a known
    canonical hex, return 0 immediately.
    Phase 2 (slow): scan known reps.  Only reached for non-solutions.
    """
    known = _known_hex_set(n)
    if not known:
        return None

    # Phase 1: hash lookup
    if not _all_transforms_hex(seq, lengths, n).isdisjoint(known):
        return 0

    # Phase 2: per-rep scan
    reps = all_representatives(n)
    best = 1_000_000
    ref = np.zeros((4, n), dtype=np.int8)
    mask = np.zeros((4, n), dtype=bool)
    for i, L in enumerate(lengths):
        mask[i, : int(L)] = True

    alt = np.ones(n, dtype=np.int8)
    alt[1::2] = -1

    for ref_tuple in reps:
        for i, s in enumerate(ref_tuple):
            ref[i, : len(s)] = np.array(s, dtype=np.int8)

        # Inline 1024-transform scan for speed
        for swap_xy in (False, True):
            s = seq.copy()
            if swap_xy:
                s[[0, 1]] = s[[1, 0]]
            for apply_t3 in (False, True):
                st = s.copy()
                if apply_t3:
                    for i2 in range(4):
                        L2 = int(lengths[i2])
                        st[i2, :L2] = st[i2, :L2] * alt[:L2]
                for neg_mask in range(16):
                    sn = st.copy()
                    for i2 in range(4):
                        if neg_mask & (1 << i2):
                            sn[i2] = -sn[i2]
                    for rev_mask in range(16):
                        sr = sn.copy()
                        for i2 in range(4):
                            if rev_mask & (1 << i2):
                                L2 = int(lengths[i2])
                                sr[i2, :L2] = sr[i2, L2 - 1 :: -1]
                        dist = int(np.sum((sr != ref) & mask))
                        if dist < best:
                            best = dist
                            if best == 0:
                                return 0
    return best
