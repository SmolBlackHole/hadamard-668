"""Temperaturgesteuerte Suche über symmetrische zirkulante Vierblock-Sequenzen."""
from __future__ import annotations

import math
import time

import numpy as np

from gpu import check_orthogonality
from .base import SearchStrategy
from builders import build_goethals_seidel
from correlations import (
    apply_symmetric_flip,
    autocorrelation_state,
    correlation_energy,
    expand_symmetric_sequence,
)


class AnnealingSearch(SearchStrategy):
    """Verfeinert zirkulante Kandidaten mit temperaturgesteuerten Flips.

    Zweck: Durchbricht lokale Minima der zirkulanten Hill-Climb-Suche.
    Mechanik: Aktualisiert die Autokorrelation inkrementell und akzeptiert schlechtere Flips gemäß exponentieller Temperaturschedule.
    Grundlage: Ein Anstieg ``ΔE`` wird mit ``exp(-ΔE / (T · E_scale))`` akzeptiert; ``T`` sinkt exponentiell.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Betrachtet ausschließlich symmetrische Goethals-Seidel-Blöcke.
    """

    ORDER = 668
    K = 167
    HALF = 84

    def __init__(self, t_start: float = 5.0, t_end: float = 0.001, *, ORDER: int = ORDER, K: int = K, HALF: int = HALF) -> None:
        if K % 2 == 0 or ORDER != 4 * K or HALF != (K + 1) // 2:
            raise ValueError(
                "symmetric 4-block sequences require odd K, ORDER = 4*K, and HALF = (K+1)//2")
        self.t_start = t_start
        self.t_end = t_end
        self.ORDER = ORDER
        self.K = K
        self.HALF = HALF

    @property
    def name(self) -> str:
        return "annealing"

    def _search(self, current: list[np.ndarray], steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = np.stack(
            [expand_symmetric_sequence(sequence) for sequence in current])
        correlations = autocorrelation_state(sequences)
        energy = best_energy = correlation_energy(correlations) // 2
        best_sequences = sequences.copy()

        def matrices() -> tuple[np.ndarray, dict[str, int], bool]:
            best_matrix = build_goethals_seidel(*best_sequences)
            best_metrics = check_orthogonality(best_matrix)
            return best_matrix, best_metrics, best_metrics["energy"] == 0

        accepted = uphill = best_at = 0
        energy_scale = max(best_energy, 1)
        for step in range(steps):
            temperature = self.t_start * \
                (self.t_end / self.t_start) ** (step / steps)
            sequence_index, value_index = rng.integers(
                0, 4), rng.integers(0, self.HALF)
            new_energy = apply_symmetric_flip(
                sequences, correlations, int(sequence_index),
                int(value_index)) // 2
            delta = new_energy - energy
            if delta <= 0 or (temperature > 0.0001 and rng.random() < math.exp(-delta / (temperature * energy_scale))):
                if delta > 0:
                    uphill += 1
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
        best_matrix, best_metrics, _ = matrices()
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best={best_energy} found@step={best_at} accept={accepted} uphill={uphill} {elapsed:.1f}s")
        return best_matrix, best_metrics, elapsed

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        current = [rng.choice([-1, 1], size=self.HALF).astype(np.int8)
                   for _ in range(4)]
        return self._search(current, steps, seed)
