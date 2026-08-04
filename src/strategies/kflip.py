"""K-Flip repair: GPU batch enumeration of all C(B,k) flips, k ramped 1..max_k."""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from correlations import (
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_residual,
)
from gpu import to_numpy, xp

from .base import Result, TurynStrategy


def _exact_energy(sequences, lengths):
    return float(
        nonperiodic_correlation_energy(
            nonperiodic_autocorrelation_state(sequences, lengths=lengths, weights=TURYN_WEIGHTS)
        )
    )


class KFlipRepair(TurynStrategy):
    """GPU batch enumeration of all C(B,k) flips, k ramped 1..max_k."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        max_k: int = 5,
        top_k_for_large: int = 32,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.max_k = max_k
        self.top_k_for_large = top_k_for_large
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "kflip"

    def _kflip_scan(self, seq_f, positions, k, top_indices=None):
        pool = list(top_indices) if top_indices is not None else list(range(len(positions)))
        if len(pool) < k:
            return None
        combos = list(combinations(pool, k))
        if not combos:
            return None
        batch = xp.repeat(seq_f[None, ...], len(combos), axis=0)
        for idx, combo in enumerate(combos):
            for t in combo:
                r, c = positions[t]
                batch[idx, r, c] *= -1
        residuals = npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        energies = xp.sum(residuals**2, axis=1)
        best_idx = int(xp.argmin(energies))
        return combos[best_idx], float(energies[best_idx])

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = self.seed(rng)
        else:
            sequences = sequences.copy()
        positions = [(r, c) for r in range(4) for c in range(int(self.LENGTHS[r]))]
        B = len(positions)
        seq_f = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)
        best_e = _exact_energy(sequences, self.LENGTHS)
        tabu: set[tuple[int, ...]] = set()

        k_moves = 0
        for k in range(1, self.max_k + 1):
            if best_e == 0:
                break
            improved = True
            while improved and best_e > 0:
                improved = False
                if k >= 5 and B >= 30:
                    deltas, r0_cpu = self._singles_scan(seq_f, positions)
                    energies = np.sum((r0_cpu + deltas) ** 2, axis=1)
                    top = np.argsort(energies)[: self.top_k_for_large]
                    result = self._kflip_scan(seq_f, positions, k, list(top))
                else:
                    result = self._kflip_scan(seq_f, positions, k)

                if result is None:
                    break
                combo, gpu_energy = result
                if gpu_energy >= best_e - 0.5:
                    break

                for t in combo:
                    r, c = positions[t]
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1
                cpu_e = _exact_energy(sequences, self.LENGTHS)
                key = tuple(sorted(combo))
                if cpu_e < best_e and key not in tabu:
                    tabu.add(key)
                    best_e = cpu_e
                    k_moves += 1
                    improved = True
                else:
                    if cpu_e >= best_e:
                        tabu.add(key)
                    for t in combo:
                        r, c = positions[t]
                        sequences[r, c] *= -1
                        seq_f[r, c] *= -1

        matrix, metrics = self.build(sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_e={best_e:.0f} (k-moves={k_moves}) {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed, sequences)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)

    def _singles_scan(self, seq_f, positions):
        B = len(positions)
        batch = xp.repeat(seq_f[None, ...], B, axis=0)
        for idx, (r, c) in enumerate(positions):
            batch[idx, r, c] *= -1
        residuals = npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        r0 = npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
        r0_cpu = to_numpy(r0)
        return to_numpy(residuals) - r0_cpu[None, :], r0_cpu
