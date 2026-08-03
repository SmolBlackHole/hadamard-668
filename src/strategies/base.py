"""Gemeinsamer Vertrag und sequentielle Pipeline für Suchstrategien."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from gpu import check_orthogonality


class SearchStrategy(ABC):
    """Gemeinsamer Vertrag für Kandidatensuchen.

    Zweck: Vereinheitlicht Name, Startsuche und Ergebnisformat aller Strategien.
    Mechanik: ``search`` liefert eine Vorzeichenmatrix, Metriken und Laufzeit.
    Grundlage: Eine Lösung der Ordnung ``n`` erfüllt ``H Hᵀ = n I``; bewertet wird ihre Off-Diagonal-Energie.
    Pipeline: ``refine`` ist optional; ohne Implementierung wird klar abgebrochen.
    Grenzen: Der Vertrag beweist keine Hadamard-Eigenschaft; nur Energie null tut das.
    """

    ORDER = 668

    @abstractmethod
    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        """Fuehre die Suche aus.

        Args:
            steps: Anzahl Suchschritte (strategie-abhaengig)
            seed: RNG-Seed fuer Reproduzierbarkeit

        Returns:
            (matrix, metrics, elapsed_seconds)
            matrix: 668x668 int8 +-1-Matrix
            metrics: {"energy": int, "orthogonal_pairs": int, "max_abs_correlation": int}
            elapsed_seconds: float
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Eindeutiger Name fuer --strategy."""

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        """Optionales Verfeinern einer existierenden Matrix.

        Nicht alle Strategien unterstuetzen das — default wirft
        NotImplementedError. Ueberschreiben in Strategien die von
        einer Startmatrix aus weitersuchen koennen (z.B. RepairSearch).
        """
        raise NotImplementedError(
            f"{self.name} cannot refine an existing matrix")


class Pipeline(SearchStrategy):
    """Führt Suchstrategien mit einem gemeinsamen Kandidaten nacheinander aus.

    Zweck: Kombiniert Grobsuche und nachfolgende Verfeinerung zu einem Lauf.
    Mechanik: Die erste Stufe ruft ``search`` auf, jede weitere ``refine``; Form, Typ und Vorzeichen werden an jeder Übergabe geprüft.
    Grundlage: Alle Stufen vergleichen denselben Metrikvertrag mit Energie null als exaktem Ziel.
    Pipeline: Nur die erste Stufe darf keine ``refine``-Implementierung besitzen.
    Grenzen: Nicht verfeinerbare spätere Stufen und abweichende Matrixordnungen werden bereits beim Aufbau abgelehnt.
    """

    def __init__(self, stages: list[tuple[SearchStrategy, int]]) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        if any(stage_steps < 0 for _, stage_steps in stages):
            raise ValueError("pipeline step budgets must be non-negative")
        order = stages[0][0].ORDER
        if any(strategy.ORDER != order for strategy, _ in stages):
            raise ValueError("all pipeline stages must use the same order")
        for strategy, _ in stages[1:]:
            if type(strategy).refine is SearchStrategy.refine:
                raise ValueError(
                    f"pipeline stage {strategy.name} cannot refine a candidate")
        self._stages = stages
        self.ORDER = order

    @property
    def name(self) -> str:
        return "->".join(strategy.name for strategy, _ in self._stages)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        matrix = np.empty((1, 1), dtype=np.int8)
        metrics = {"energy": 2**63, "orthogonal_pairs": 0,
                   "max_abs_correlation": 0}
        elapsed_total = 0.0
        for index, (strategy, stage_steps) in enumerate(self._stages):
            if index == 0:
                matrix, metrics, elapsed = strategy.search(
                    stage_steps, seed + index)
            else:
                matrix, metrics, elapsed = strategy.refine(
                    matrix, stage_steps, seed + index)
            self._validate_result(matrix, metrics, strategy)
            elapsed_total += elapsed
            if metrics["energy"] == 0:
                metrics = check_orthogonality(matrix)
                if metrics["energy"] == 0:
                    break
        return matrix, metrics, elapsed_total

    def _validate_result(
        self,
        matrix: np.ndarray,
        metrics: dict[str, int],
        strategy: SearchStrategy,
    ) -> None:
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"pipeline stage {strategy.name} returned a non-host matrix")
        if matrix.shape != (self.ORDER, self.ORDER):
            raise ValueError(
                f"pipeline stage {strategy.name} returned shape {matrix.shape}")
        if matrix.dtype != np.int8 or not np.all(np.isin(matrix, (-1, 1))):
            raise ValueError(
                f"pipeline stage {strategy.name} must return an int8 sign matrix")
        if not isinstance(metrics.get("energy"), int):
            raise TypeError(
                f"pipeline stage {strategy.name} returned invalid metrics")
