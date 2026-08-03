"""Kontinuierliche Mean-Field-Heuristik für die Gram-Matrix-Energie."""
from __future__ import annotations

import time

import numpy as np

from gpu import check_orthogonality
from .base import SearchStrategy


class IsingSearch(SearchStrategy):
    """Optimiert eine kontinuierliche Gram-Energie-Relaxation und rundet auf Vorzeichen.

    Zweck: Erprobt Mean-Field-artige Dynamik im vollständigen Matrixraum.
    Mechanik: Aktualisiert reelle Zustände per Gram-Gradient und bewertet gerundete Kandidaten.
    Grundlage: Mit ``G = M Mᵀ`` ohne Diagonale nutzt der Code ``M ← tanh(β(M - G M / n))`` und rundet danach auf ``±1``.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die Relaxation ist heuristisch; Rundung und Temperatur liefern keine Konvergenzgarantie.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, beta_end: float = 3.0) -> None:
        if order < 2 or order % 2:
            raise ValueError("ising search requires a positive even order")
        self.ORDER = order
        self.beta_end = beta_end

    @property
    def name(self) -> str:
        return "ising"

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        state = rng.uniform(-1, 1, size=(self.ORDER, self.ORDER))
        best = np.where(state >= 0, 1, -1).astype(np.int8)
        best_metrics = check_orthogonality(best)
        for step in range(max(1, steps)):
            beta = self.beta_end * (step + 1) / max(1, steps)
            gram = state @ state.T
            np.fill_diagonal(gram, 0)
            gradient = (gram @ state) / self.ORDER
            state = np.tanh(beta * (state - gradient))
            candidate = np.where(state >= 0, 1, -1).astype(np.int8)
            metrics = check_orthogonality(candidate)
            if metrics["energy"] < best_metrics["energy"]:
                best, best_metrics = candidate, metrics
                if metrics["energy"] == 0:
                    break
        return best, best_metrics, time.perf_counter() - started
