"""Sampled steepest-descent search in the Turyn-type sequence space."""
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
from .base import SearchStrategy


class TurynSteepestSearch(SearchStrategy):
    """Choose the best of a small random sample of exact Turyn flips.

    Zweck: Behält den Best-Candidate-Stil der entfernten Baumert-Suche.
    Mechanik: Bewertet eine zufällige Flip-Stichprobe und nimmt den besten Gewinn.
    Grundlage: Alle Werte sind exakte gewichtete nichtperiodische Korrelationen.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Die Stichprobe ist absichtlich kein vollständiger Steepest-Descent-Scan.
    """

    N = 56
    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, *, n: int = N, candidates: int = 32) -> None:
        if n < 2:
            raise ValueError("Turyn type requires n >= 2")
        if candidates < 1:
            raise ValueError("candidates must be positive")
        self.N = n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.ORDER = 4 * (3 * n - 1)
        self.candidates = candidates

    @property
    def name(self) -> str:
        return "turyn_steepest"

    def _seed(self, rng: np.random.Generator) -> np.ndarray:
        sequences = np.zeros((4, self.N), dtype=np.int8)
        for index, length in enumerate(self.LENGTHS):
            sequences[index, :length] = rng.choice((-1, 1), size=length)
        return sequences

    def _build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        matrix = build_turyn(*(sequences[index, :self.LENGTHS[index]]
                               for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def _sample_positions(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        count = min(self.candidates, int(self.LENGTHS.sum()))
        flat = rng.choice(int(self.LENGTHS.sum()), size=count, replace=False)
        starts = np.concatenate(([0], np.cumsum(self.LENGTHS)))
        return [(int(np.searchsorted(starts, value, side="right") - 1),
                 int(value - starts[np.searchsorted(starts, value, side="right") - 1]))
                for value in flat]

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self._seed(rng)
        correlations = nonperiodic_autocorrelation_state(
            sequences, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energy = best_energy = nonperiodic_correlation_energy(correlations)
        best_sequences = sequences.copy()
        best_at = accepted = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                selected: tuple[int, int] | None = None
                selected_energy = energy
                for sequence_index, value_index in self._sample_positions(rng):
                    candidate_energy = apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                    apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                    if candidate_energy < selected_energy:
                        selected, selected_energy = (
                            sequence_index, value_index), candidate_energy
                if selected is not None:
                    sequence_index, value_index = selected
                    apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                    energy, accepted = selected_energy, accepted + 1
                    if energy < best_energy:
                        best_energy, best_at = energy, step
                        best_sequences = sequences.copy()
                if not best_energy:
                    break
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                bar.update(1)
        matrix, metrics = self._build(best_sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_energy={best_energy} found@step={best_at} "
              f"accepted={accepted} {elapsed:.1f}s")
        return matrix, metrics, elapsed
