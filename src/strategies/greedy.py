"""Greedy single-flip descent in Turyn sequence space."""

from __future__ import annotations

import time

import numpy as np

from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)

from .base import Result, TurynStrategy


class TurynGreedySearch(TurynStrategy):
    """Accept any single-flip that does not increase the weighted NPAF energy."""

    @property
    def name(self) -> str:
        return "greedy"

    def _flip(self, sequences: np.ndarray, correlations: np.ndarray, si: int, vi: int) -> int:
        """Flip bit (si, vi); caller flips again to revert on rejection."""
        return apply_nonperiodic_flip(
            sequences, correlations, si, vi, lengths=self.LENGTHS, weight=int(self.WEIGHTS[si])
        )

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self.seed(rng)
        correlations = nonperiodic_autocorrelation_state(
            sequences, lengths=self.LENGTHS, weights=self.WEIGHTS
        )
        energy = best_energy = nonperiodic_correlation_energy(correlations)
        best_sequences = sequences.copy()
        accepted = best_at = 0
        n_seqs = len(self.LENGTHS)
        for step in range(steps):
            si = int(rng.integers(0, n_seqs))
            vi = int(rng.integers(0, self.LENGTHS[si]))
            new_energy = self._flip(sequences, correlations, si, vi)
            if new_energy > energy:
                self._flip(sequences, correlations, si, vi)  # undo rejected flip
                continue
            energy, accepted = new_energy, accepted + 1
            if new_energy < best_energy:
                best_energy, best_at = new_energy, step
                best_sequences = sequences.copy()
                if not best_energy:
                    break
        matrix, metrics = self.build(best_sequences)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s"
        )
        return Result(matrix, metrics, elapsed)
