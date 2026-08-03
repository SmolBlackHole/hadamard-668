"""Restart-Wrapper mit Checkpoint-Speicherung fuer Suchstrategien."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from gpu import check_orthogonality
from .base import SearchStrategy


class BacktrackSearch(SearchStrategy):
    """Wiederholt eine Strategie in begrenzten Suchabschnitten.

    Zweck: Verteilt ein Budget auf mehrere deterministisch geseedete Neustarts.
    Mechanik: Ruft ``inner.search`` bis zu ``max_backtracks + 1`` mal auf und speichert jeden neuen globalen Bestwert als Pickle-Checkpoint.
    Grundlage: Mehrere unabhängige Startwerte können unterschiedliche lokale Minima einer diskreten Energieheuristik erreichen.
    Pipeline: Ist selbst nur eine erste Pipeline-Stufe, weil keine ``refine``-Methode implementiert ist.
    Grenzen: Checkpoint und perturbierte Matrix dienen derzeit nicht als Startzustand des nächsten ``inner.search``-Aufrufs; ``patience`` und ``min_improvement`` beeinflussen nur die Diagnoseausgabe, nicht die Restart-Steuerung.
    """

    def __init__(
        self,
        inner: SearchStrategy,
        patience: int = 5000,
        min_improvement: float = 0.01,
        perturbation: float = 0.05,
        max_backtracks: int = 10,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self._inner = inner
        self.ORDER = inner.ORDER
        self.patience = patience
        self.min_improvement = min_improvement
        self.perturbation = perturbation
        self.max_backtracks = max_backtracks
        self.checkpoint_dir = Path(checkpoint_dir)

    @property
    def name(self) -> str:
        return f"backtrack_{self._inner.name}"

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_matrix = np.empty((self.ORDER, self.ORDER), dtype=np.int8)
        best_metrics: dict[str, int] = {"energy": 2 ** 63}
        total_accepted = 0

        for restart in range(self.max_backtracks + 1):
            chunk = steps // (self.max_backtracks + 1)
            inner_steps = max(1000, chunk)

            if restart == 0:
                matrix, metrics, elapsed = self._inner.search(inner_steps, seed + restart)
            else:
                # Create a perturbed checkpoint candidate for diagnostics.
                ckpt = self._load_checkpoint(seed)
                if ckpt is not None:
                    matrix = self._perturb(ckpt, rng)
                else:
                    matrix = rng.choice([-1, 1], size=(self.ORDER, self.ORDER)).astype(np.int8)
                # The inner strategy currently starts independently.
                matrix, metrics, elapsed = self._inner.search(inner_steps, seed + restart)

            total_accepted += 1

            if metrics["energy"] < best_metrics["energy"]:
                best_matrix, best_metrics = matrix.copy(), metrics
                self._save_checkpoint(best_matrix, seed, restart)
                if metrics["energy"] == 0:
                    break

            # Check if we improved enough since last checkpoint
            if restart > 0 and restart < self.max_backtracks:
                prev_energy = best_metrics.get("energy", 2 ** 63)
                delta = prev_energy - metrics["energy"]
                if delta <= 0 or delta / max(prev_energy, 1) < self.min_improvement:
                    print(f"  plateau restart {restart}/{self.max_backtracks}  "
                          f"energy={metrics['energy']}  best={best_metrics['energy']}  no improvement")
                else:
                    print(f"  restart {restart} improved: {prev_energy} -> {metrics['energy']}  "
                          f"delta={delta}")

        elapsed = time.perf_counter() - t0
        return best_matrix, best_metrics, elapsed

    def _save_checkpoint(self, matrix: np.ndarray, seed: int, iteration: int) -> None:
        path = self.checkpoint_dir / f"ckpt_{self.name}_s{seed}_i{iteration}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"matrix": matrix.tobytes(), "shape": matrix.shape, "dtype": str(matrix.dtype)}, f)
        # Keep only last 5 checkpoints
        ckpts = sorted(self.checkpoint_dir.glob(f"ckpt_{self.name}_s{seed}_*.pkl"))
        for old in ckpts[:-5]:
            old.unlink(missing_ok=True)

    def _load_checkpoint(self, seed: int) -> np.ndarray | None:
        ckpts = sorted(self.checkpoint_dir.glob(f"ckpt_{self.name}_s{seed}_*.pkl"))
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
