"""Known Turyn TT(n) solutions for experiments and testing."""
from __future__ import annotations
import numpy as np

TT_KNOWN: dict[int, tuple[tuple[int, ...], ...]] = {
    2: ((1, -1), (1, -1), (-1, -1), (-1,)),
    4: ((1, -1, 1, -1), (1, -1, -1, -1), (-1, 1, 1, -1), (-1, -1, -1)),
    6: ((-1, 1, -1, -1, 1, -1),
        (-1, 1, 1, -1, -1, -1),
        (1, -1, -1, -1, 1, -1),
        (1, -1, -1, -1, -1)),
    8: ((1, 1, -1, 1, -1, 1, -1, 1),
        (1, -1, -1, -1, -1, -1, -1, 1),
        (1, -1, -1, 1, 1, 1, 1, -1),
        (1, 1, 1, -1, 1, 1, -1)),
}

TT36_HEX = "060989975b685d8fc80750b21c0212eceb26"


def _parse_hex(hex_str: str):
    seqs = [[], [], [], []]
    for d in hex_str[:-1]:
        for i, b in enumerate(f"{int(d,16):04b}"):
            seqs[i].append(1 if b == "0" else -1)
    for i, b in enumerate(f"{int(hex_str[-1],16):03b}"):
        seqs[i].append(1 if b == "0" else -1)
    return tuple(tuple(s) for s in seqs)


TT_KNOWN[36] = _parse_hex(TT36_HEX)


def tt_sequences(n: int) -> np.ndarray:
    arr = np.zeros((4, n), dtype=np.int8)
    for i, s in enumerate(TT_KNOWN[n]):
        arr[i, : len(s)] = np.array(s, dtype=np.int8)
    return arr


def tt_lengths(n: int) -> np.ndarray:
    return np.array((n, n, n, n - 1), dtype=np.int64)


def known_solution(n: int) -> tuple[np.ndarray, np.ndarray] | None:
    if n in TT_KNOWN:
        return tt_sequences(n), tt_lengths(n)
    return None
