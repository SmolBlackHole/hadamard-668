"""Restart-Wrapper mit Checkpoint-Speicherung fuer Suchstrategien."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from .base import SearchStrategy


class BacktrackSearch(SearchStrategy):
    """Wiederholt eine Strategie in begrenzten Suchabschnitten.

    Zweck: Verteilt ein Budget auf mehrere deterministisch geseedete Neustarts.
    Mechanik: Teilt das Schrittbudget exakt auf und verfeinert, wenn möglich, perturbierte Bestwert-Checkpoints.
    Grundlage: Mehrere unabhängige Startwerte können unterschiedliche lokale Minima einer diskreten Energieheuristik erreichen.
    Pipeline: Ist selbst nur eine erste Pipeline-Stufe, weil keine ``refine``-Methode implementiert ist.
    Grenzen: Strategien ohne eigene ``refine``-Methode erhalten unabhängige Restarts statt Checkpoint-Zuständen.
    """

    def __init__(
        self,
        inner: SearchStrategy,
        perturbation: float = 0.05,
        max_backtracks: int = 10,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        if not 0 <= perturbation <= 1:
            raise ValueError("perturbation must be in [0, 1]")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be non-negative")
        self._inner = inner
        self.ORDER = inner.ORDER
        self.perturbation = perturbation
        self.max_backtracks = max_backtracks
        self.checkpoint_dir = Path(checkpoint_dir)

    @property
    def name(self) -> str:
        return f"backtrack_{self._inner.name}"

    def _step_budgets(self, steps: int) -> tuple[int, ...]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return (0,)
        restart_count = min(self.max_backtracks + 1, steps)
        chunk, remainder = divmod(steps, restart_count)
        return tuple(chunk + (index < remainder) for index in range(restart_count))

    def _can_refine(self) -> bool:
        return type(self._inner).refine is not SearchStrategy.refine

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_matrix = np.empty((self.ORDER, self.ORDER), dtype=np.int8)
        best_metrics: dict[str, int] = {"energy": 2 ** 63}
        for restart, inner_steps in enumerate(self._step_budgets(steps)):
            checkpoint = self._load_checkpoint(seed) if restart else None
            if checkpoint is not None and self._can_refine():
                start = self._perturb(checkpoint, rng)
                matrix, metrics, _ = self._inner.refine(
                    start, inner_steps, seed + restart)
            else:
                matrix, metrics, _ = self._inner.search(
                    inner_steps, seed + restart)

            if metrics["energy"] < best_metrics["energy"]:
                best_matrix, best_metrics = matrix.copy(), metrics.copy()
                self._save_checkpoint(best_matrix, seed, restart)
                if metrics["energy"] == 0:
                    break

        elapsed = time.perf_counter() - t0
        return best_matrix, best_metrics, elapsed

    def _save_checkpoint(self, matrix: np.ndarray, seed: int, iteration: int) -> None:
        path = self.checkpoint_dir / f"ckpt_{self.name}_s{seed}_i{iteration}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"matrix": matrix.tobytes(), "shape": matrix.shape, "dtype": str(matrix.dtype)}, f)
        # Keep only last 5 checkpoints
        ckpts = sorted(
            self.checkpoint_dir.glob(f"ckpt_{self.name}_s{seed}_*.pkl"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        for old in ckpts[:-5]:
            old.unlink(missing_ok=True)

    def _load_checkpoint(self, seed: int) -> np.ndarray | None:
        ckpts = sorted(
            self.checkpoint_dir.glob(f"ckpt_{self.name}_s{seed}_*.pkl"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not ckpts:
            return None
        with open(ckpts[-1], "rb") as f:
            data = pickle.load(f)
        return np.frombuffer(data["matrix"], dtype=np.dtype(data["dtype"])).reshape(data["shape"])

    def _perturb(self, matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Flip perturbation% of entries while leaving the first row untouched."""
        M = matrix.copy()
        n = self.ORDER
        total_flips = int(n * n * self.perturbation)
        for _ in range(total_flips):
            i = rng.integers(1, n)  # never touch first row (normalised)
            j = rng.integers(0, n)
            M[i, j] *= -1
        return M
