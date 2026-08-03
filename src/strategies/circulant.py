"""Zirkulante Vierblock-Suche mit gespiegelten Sequenzen."""
from __future__ import annotations


import time

import numpy as np
from tqdm import tqdm

from constructions import (
    apply_symmetric_flip,
    autocorrelation_state,
    build_goethals_seidel,
    build_williamson,
    correlation_energy,
    symmetric_circulant,
)
from gpu import check_orthogonality
from .base import SearchStrategy


class CirculantSearch(SearchStrategy):
    """Sucht vier gespiegelte Sequenzen für zirkulante Blockmatrizen.

    Zweck: Erzeugt Kandidaten über Williamson und optional Goethals-Seidel.
    Mechanik: Flipt 336 unabhängige Halbsequenzeinträge und aktualisiert deren Autokorrelation inkrementell.
    Grundlage: Vier Sequenzen sind komplementär, wenn ihre periodischen Autokorrelationen für jeden Nichtnull-Shift zu null summieren.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Betrachtet ausschließlich symmetrische zirkulante Blöcke der gewählten Konstruktion.
    """

    ORDER = 668
    K = 167
    HALF = 84

    def __init__(self, constructions: str = "williamson", *, ORDER: int = ORDER, K: int = K, HALF: int = HALF) -> None:
        if constructions not in {"williamson", "all"}:
            raise ValueError("constructions must be 'williamson' or 'all'")
        if ORDER != 4 * K or HALF != (K + 1) // 2:
            raise ValueError(
                "ORDER, K, and HALF must describe symmetric 4-block sequences")
        self.constructions = constructions
        self.ORDER = ORDER
        self.K = K
        self.HALF = HALF

    @property
    def name(self) -> str:
        return "hybrid" if self.constructions == "all" else "circulant"

    def _builders(self) -> tuple:
        return (build_williamson,) if self.constructions == "williamson" else (build_williamson, build_goethals_seidel)

    def _build_best(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        candidates = (
            (matrix, check_orthogonality(matrix))
            for matrix in (build(*sequences) for build in self._builders())
        )
        return min(candidates, key=lambda candidate: candidate[1]["energy"])

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        halves = [rng.choice([-1, 1], size=self.HALF).astype(np.int8)
                  for _ in range(4)]
        sequences = np.stack([symmetric_circulant(half) for half in halves])
        correlations = autocorrelation_state(sequences)
        energy = best_energy = correlation_energy(correlations) // 2
        best_sequences = sequences.copy()
        best_at = accepted = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                bar.update(1)
                sequence_index, value_index = rng.integers(
                    0, 4), rng.integers(0, self.HALF)
                new_energy = apply_symmetric_flip(
                    sequences, correlations, int(sequence_index),
                    int(value_index)) // 2
                if new_energy <= energy:
                    energy, accepted = new_energy, accepted + 1
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_sequences = sequences.copy()
                        best_at = step
                        if best_energy == 0:
                            break
                else:
                    apply_symmetric_flip(
                        sequences, correlations, int(sequence_index),
                        int(value_index))
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
        best_matrix, best_metrics = self._build_best(best_sequences)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s")
        return best_matrix, best_metrics, elapsed
