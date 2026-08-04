"""Difference-set seeded local search (Goethals-Seidel space)."""
from __future__ import annotations
import time
import numpy as np
from builders import build_goethals_seidel
from gpu import check_orthogonality
from .base import SearchStrategy


def _paf_state(sequences, weights=None):
    """Weighted periodic autocorrelation for equal-length sequences."""
    seqs = np.asarray(sequences, dtype=np.int8)
    w = np.ones(seqs.shape[0], dtype=np.int64) if weights is None else np.asarray(weights, dtype=np.int64)
    n = seqs.shape[1]
    total = np.zeros(n, dtype=np.int64)
    for si, seq in enumerate(seqs):
        for d in range(n):
            corr = 0
            for i in range(n):
                corr += int(seq[i]) * int(seq[(i + d) % n])
            total[d] += int(w[si]) * corr
    return total


def _paf_energy(correlations: np.ndarray) -> int:
    values = np.asarray(correlations, dtype=np.int64)[1:]
    return int(np.dot(values, values))


def _paf_flip(sequences, correlations, si, vi, weight=1):
    n = sequences.shape[1]
    old = int(sequences[si, vi])
    for d in range(1, n):
        fwd = int(sequences[si, (vi + d) % n])
        bwd = int(sequences[si, (vi - d) % n])
        correlations[d] -= 2 * weight * old * (fwd + bwd)
    sequences[si, vi] = -old
    return _paf_energy(correlations)


class DiffsetSearch(SearchStrategy):
    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 4 or order % 4:
            raise ValueError("diffset search requires an order divisible by four")
        self.ORDER = order
        self.K = order // 4

    @property
    def name(self) -> str: return "diffset"

    @property
    def construction(self) -> str: return f"goethals_seidel_k_{self.K}"

    def _seed(self) -> list[np.ndarray]:
        residues = {v * v % self.K for v in range(1, self.K)}
        base = -np.ones(self.K, dtype=np.int8)
        base[list(residues)] = 1
        return [np.roll(base, s).copy() for s in range(4)]

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        current = np.stack(self._seed())
        corr = _paf_state(current)
        energy = best_energy = _paf_energy(corr)
        best = current.copy()
        for _ in range(steps):
            si, vi = int(rng.integers(0, 4)), int(rng.integers(0, self.K))
            e = _paf_flip(current, corr, si, vi)
            if e <= energy:
                energy = e
                if e < best_energy:
                    best_energy = e; best = current.copy()
                    if e == 0: break
            else:
                _paf_flip(current, corr, si, vi)
        return build_goethals_seidel(*best), check_orthogonality(best), time.perf_counter() - started
