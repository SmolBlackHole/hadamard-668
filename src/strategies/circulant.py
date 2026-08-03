"""Zirkulante Vierblock-Suche mit gespiegelten Sequenzen."""
from __future__ import annotations


import time

import numpy as np
from tqdm import tqdm

from constructions import autocorrelation_energy, build_goethals_seidel, build_propus, build_williamson, symmetric_circulant
from gpu import check_orthogonality
from .base import SearchStrategy


class CirculantSearch(SearchStrategy):
    """Sucht vier gespiegelte Sequenzen für zirkulante Blockmatrizen.

    Zweck: Erzeugt Kandidaten über Williamson, optional auch Propus und Goethals-Seidel.
    Mechanik: Flipt 336 unabhängige Halbsequenzeinträge und minimiert Autokorrelationsenergie.
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
        return (build_williamson,) if self.constructions == "williamson" else (build_williamson, build_propus, build_goethals_seidel)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        current = [rng.choice([-1, 1], size=self.HALF).astype(np.int8)
                   for _ in range(4)]
        best_half = [sequence.copy() for sequence in current]
        energy = best_energy = autocorrelation_energy(
            tuple(symmetric_circulant(sequence) for sequence in current))
        best_matrix = np.empty((self.ORDER, self.ORDER), dtype=np.int8)
        best_metrics = {"energy": 2**63,
                        "orthogonal_pairs": 0, "max_abs_correlation": 0}
        best_at = accepted = 0

        def check_builds() -> bool:
            nonlocal best_matrix, best_metrics
            sequences = [symmetric_circulant(sequence)
                         for sequence in best_half]
            for build in self._builders():
                matrix = build(*sequences)
                metrics = check_orthogonality(matrix)
                if metrics["energy"] < best_metrics["energy"]:
                    best_matrix, best_metrics = matrix, metrics
                if metrics["energy"] == 0:
                    return True
            return False

        if check_builds():
            return best_matrix, best_metrics, time.perf_counter() - started
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                sequence_index, value_index = rng.integers(
                    0, 4), rng.integers(0, self.HALF)
                current[sequence_index][value_index] *= -1
                new_energy = autocorrelation_energy(
                    tuple(symmetric_circulant(sequence) for sequence in current))
                if new_energy <= energy:
                    energy, accepted = new_energy, accepted + 1
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_half = [sequence.copy() for sequence in current]
                        best_at = step
                        if check_builds():
                            break
                else:
                    current[sequence_index][value_index] *= -1
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                    bar.update(50)
        check_builds()
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s")
        return best_matrix, best_metrics, elapsed
