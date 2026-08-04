"""Greedy local search in the Turyn-type sequence space."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from builders import build_turyn
from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from gpu import check_orthogonality
from sieve import seed_turyn_batch
from .base import SearchStrategy


class TurynGreedySearch(SearchStrategy):
    """Greedily minimize the weighted non-periodic Turyn energy.

    Zweck: Sucht TT(n)-Folgen der Längen ``(n,n,n,n-1)``.
    Mechanik: Akzeptiert nur exakte, nicht verschlechternde Einzelbit-Flips.
    Grundlage: ``N_X + N_Y + 2N_Z + 2N_W`` muss für jeden Shift verschwinden.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Lokale Minima werden ohne Temperatur oder Restart nicht verlassen.
    """

    N = 56
    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, *, n: int = N, sieve: bool = True) -> None:
        if n < 2:
            raise ValueError("Turyn type requires n >= 2")
        self.N = n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.ORDER = 4 * (3 * n - 1)
        self.sieve = sieve

    @property
    def name(self) -> str:
        return "greedy"

    @property
    def construction(self) -> str:
        return f"turyn_tt_{self.N}"

    def _seed(self, rng: np.random.Generator) -> np.ndarray:
        if self.sieve:
            return seed_turyn_batch(self.N, 1, rng)[0]
        sequences = np.zeros((4, self.N), dtype=np.int8)
        for index, length in enumerate(self.LENGTHS):
            sequences[index, :length] = rng.choice((-1, 1), size=length)
        return sequences

    def _build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        lengths = self.LENGTHS
        matrix = build_turyn(*(sequences[index, :lengths[index]]
                               for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self._seed(rng)
        correlations = nonperiodic_autocorrelation_state(
            sequences, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energy = best_energy = nonperiodic_correlation_energy(correlations)
        best_sequences = sequences.copy()
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                sequence_index = int(rng.integers(0, 4))
                value_index = int(rng.integers(
                    0, self.LENGTHS[sequence_index]))
                new_energy = apply_nonperiodic_flip(
                    sequences, correlations, sequence_index, value_index,
                    lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                if new_energy <= energy:
                    energy, accepted = new_energy, accepted + 1
                    if new_energy < best_energy:
                        best_energy, best_at = new_energy, step
                        best_sequences = sequences.copy()
                        if not best_energy:
                            break
                else:
                    apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                bar.update(1)
        matrix, metrics = self._build(best_sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_energy={best_energy} found@step={best_at} "
              f"accepted={accepted} {elapsed:.1f}s")
        return matrix, metrics, elapsed
