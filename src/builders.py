"""Goethals-Seidel, Turyn, Group-GS4, and Williamson matrix assembly."""

from __future__ import annotations

import numpy as np

# ── Group-circulant diff table (cached) ───────────────────────────────────────

_DIFF_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def _diff_table(dims: tuple[int, ...]) -> np.ndarray:
    if dims not in _DIFF_CACHE:
        n = int(np.prod(dims))
        table = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            rem = i
            gi = []
            for d in reversed(dims):
                gi.insert(0, rem % d)
                rem //= d
            for j in range(n):
                rem = j
                gj = []
                for d in reversed(dims):
                    gj.insert(0, rem % d)
                    rem //= d
                diff = [(b - a) % d for a, b, d in zip(gi, gj, dims)]
                idx, stride = 0, 1
                for dv, ds in zip(reversed(diff), reversed(dims)):
                    idx += dv * stride
                    stride *= ds
                table[i, j] = idx
        _DIFF_CACHE[dims] = table
    return _DIFF_CACHE[dims]


# ── Circulant matrix ──────────────────────────────────────────────────────────


def _circulant(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int8)
    n = values.size
    return values[(np.arange(n)[None, :] - np.arange(n)[:, None]) % n]


# ── Goethals-Seidel ───────────────────────────────────────────────────────────


def build_goethals_seidel(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Build a Hadamard matrix from four Williamson sequences via GS4 array."""
    A, B, C, D = (_circulant(s) for s in (a, b, c, d))
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BtR, CtR, DtR = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DtR, CtR],
            [-CR, DtR, A, -BtR],
            [-DR, -CtR, BtR, A],
        ]
    ).astype(np.int8)


def build_williamson(seqs: np.ndarray) -> np.ndarray:
    """Build GS4 from (4, n) sequences. Shorthand for build_goethals_seidel."""
    n = seqs.shape[1]
    return build_goethals_seidel(
        seqs[0, :n].copy(),
        seqs[1, :n].copy(),
        seqs[2, :n].copy(),
        seqs[3, :n].copy(),
    )


# ── Group-GS4 ─────────────────────────────────────────────────────────────────


def build_group_gs4(seqs: np.ndarray, dims: list[int]) -> np.ndarray:
    """Build GS4 with group-based circulants. seqs: (4, n) int8, dims: factorisation."""
    n = int(np.prod(dims))
    dt = _diff_table(tuple(dims))

    def circ(flat):
        return flat[dt].astype(np.int8)

    A = circ(seqs[0, :n])
    B = circ(seqs[1, :n])
    C = circ(seqs[2, :n])
    D = circ(seqs[3, :n])
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BRt, CRt, DRt = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DRt, CRt],
            [-CR, DRt, A, -BRt],
            [-DR, -CRt, BRt, A],
        ]
    ).astype(np.int8)


# ── Turyn ─────────────────────────────────────────────────────────────────────


def _sign(values: np.ndarray, name: str) -> np.ndarray:
    s = np.asarray(values, dtype=np.int8)
    if s.ndim != 1 or not np.all(np.isin(s, (-1, 1))):
        raise ValueError(f"{name} must be a one-dimensional sign sequence")
    return s


def _turyn_to_base(x, y, z, w):
    x, y, z, w = (_sign(v, n) for v, n in zip((x, y, z, w), "XYZW", strict=False))
    size = len(x)
    if len(y) != size or len(z) != size or len(w) != size - 1:
        raise ValueError("Turyn sequences must have lengths (n, n, n, n-1)")
    return (np.concatenate((z, w)), np.concatenate((z, -w)), x.copy(), y.copy())


def _base_to_t(a, b, c, d):
    a, b, c, d = (_sign(v, n) for v, n in zip((a, b, c, d), "ABCD", strict=False))
    if len(a) != len(b) or len(c) != len(d) or len(a) != 2 * len(c) - 1:
        raise ValueError("base sequences must have lengths (2n-1, 2n-1, n, n)")
    zs = np.zeros(len(c), dtype=np.int8)
    zl = np.zeros(len(a), dtype=np.int8)
    return (
        np.concatenate(((a + b) // 2, zs)),
        np.concatenate(((a - b) // 2, zs)),
        np.concatenate((zl, (c + d) // 2)),
        np.concatenate((zl, (c - d) // 2)),
    )


def _t_to_sign(t1, t2, t3, t4):
    sequences = tuple(np.asarray(v, dtype=np.int8) for v in (t1, t2, t3, t4))
    if any(s.ndim != 1 for s in sequences) or len({len(s) for s in sequences}) != 1:
        raise ValueError("T-sequences must be one-dimensional and equally long")
    occupancy = np.sum(np.abs(np.stack(sequences)), axis=0)
    if not np.all(occupancy == 1):
        raise ValueError("T-sequences must have exactly one non-zero value per position")
    t1, t2, t3, t4 = sequences
    result = (
        t1 + t2 + t3 + t4,
        -t1 + t2 + t3 - t4,
        -t1 - t2 + t3 + t4,
        -t1 + t2 - t3 + t4,
    )
    if not all(np.all(np.isin(s, (-1, 1))) for s in result):
        raise ValueError("T-sequences did not produce sign sequences")
    return result


def build_turyn(x, y, z, w):
    """Build Hadamard via Turyn TT(n) → T-sequences → GS4."""
    return build_goethals_seidel(*_t_to_sign(*_base_to_t(*_turyn_to_base(x, y, z, w))))
