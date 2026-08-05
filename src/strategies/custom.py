"""CustomSolver — konfigurierbare Hadamard-Suche via KFlip + Konstruktion."""

from __future__ import annotations

import time

import numpy as np

from builders import build_goethals_seidel, build_group_gs4
from energy import group_gs4_energy, npaf_energy_exact
from gpu import check_orthogonality
from strategies.base import Result, SearchStrategy

from .kflip import search as ils_search


class CustomSolver(SearchStrategy):
    """Konfiguriert NPAF-Suche, delegiert an kflip_search, baut Matrix.

    Classmethods: williamson(n), group(n), tensor(n1, n2), from_order(order).
    """

    def __init__(
        self,
        *,
        seq_lengths: tuple[int, ...],
        weights: tuple[float, ...],
        group_dims: list[int] | None = None,
        order: int | None = None,
    ):
        if len(seq_lengths) != len(weights):
            raise ValueError("seq_lengths and weights must have same length")
        self.N_SEQS = len(seq_lengths)
        self.LENGTHS = np.array(seq_lengths, dtype=np.int64)
        self.WEIGHTS = np.array(weights, dtype=np.float64)
        self.N = int(self.LENGTHS.max())
        self.ORDER = order if order is not None else 4 * self.N
        self.group_dims = group_dims
        self._last_seq: np.ndarray | None = None
        self.gpu_exclusive = False

    @classmethod
    def williamson(cls, *, n: int) -> CustomSolver:
        """Williamson: 4 gleiche Sequenzen, unit weights, GS4-Matrix."""
        return cls(
            seq_lengths=(n, n, n, n),
            weights=(1.0, 1.0, 1.0, 1.0),
            order=4 * n,
        )

    @classmethod
    def group(cls, *, n: int) -> CustomSolver:
        """Group-based GS4 mit auto-detected factorisation."""
        dims = _best_factorization(n)
        return cls(
            seq_lengths=(n, n, n, n),
            weights=(1.0, 1.0, 1.0, 1.0),
            order=4 * n,
            group_dims=dims,
        )

    @classmethod
    def from_order(cls, order: int) -> CustomSolver:
        n = order // 4
        return cls.group(n=n)

    @property
    def name(self) -> str:
        if self.group_dims is not None:
            label = "x".join(map(str, self.group_dims))
            return f"custom[{label}]"
        return "custom"

    @property
    def construction(self) -> str:
        if self.group_dims is not None:
            return f"group_gs4_{'x'.join(map(str, self.group_dims))}"
        return "williamson"

    def search(
        self, steps: int | None = None, seed: int = 0, sequences: np.ndarray | None = None
    ) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        if sequences is None:
            sequences = np.zeros((self.N_SEQS, self.N), dtype=np.int8)
            for i in range(self.N_SEQS):
                sequences[i, : int(self.LENGTHS[i])] = rng.choice(
                    (-1, 1), size=int(self.LENGTHS[i])
                ).astype(np.int8)
        else:
            sequences = sequences.copy()

        if self.group_dims is not None:
            dims = self.group_dims

            def energy_fn(s):
                return group_gs4_energy(s, dims)
        else:

            def energy_fn(s):
                return npaf_energy_exact(s, self.LENGTHS, self.WEIGHTS)

        budget = steps if steps is not None else None  # None = kflip decides, 0 = no search
        best_seq, best_e = ils_search(sequences, energy_fn, rng, budget=budget)
        elapsed = time.perf_counter() - started

        # ── Build matrix ──
        dims_label = ""
        if best_e == 0:
            if self.group_dims is not None:
                matrix = build_group_gs4(best_seq, self.group_dims)
                dims_label = f" [{self.group_dims}]"
            elif self.N_SEQS == 4 and all(w == 1.0 for w in self.WEIGHTS):
                n = self.N
                A = best_seq[0, :n].copy()
                B = best_seq[1, :n].copy()
                C = best_seq[2, :n].copy()
                D = best_seq[3, :n].copy()
                matrix = build_goethals_seidel(A, B, C, D)
            else:
                matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
        else:
            matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)

        metrics = check_orthogonality(matrix)
        if best_e == 0 and metrics.energy == 0:
            print(
                f"  seed={seed} VALID {matrix.shape[0]}x{matrix.shape[1]} Hadamard"
                f"{dims_label} {elapsed:.1f}s"
            )
        else:
            print(f"  seed={seed} best_e={best_e}{dims_label} {elapsed:.1f}s")

        return Result(
            matrix=matrix,
            metrics=metrics,
            elapsed=elapsed,
            seed=seed,
            sequences=best_seq,
        )

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)

    def hamming(self) -> int | None:
        if self._last_seq is None:
            return None
        from fixtures import equiv_hamming

        return equiv_hamming(self._last_seq, self.LENGTHS, self.N)


def _best_factorization(n: int) -> list[int]:
    """Best group dims for n: factors closest to sqrt(n), both >= 3."""
    best = [n]
    best_diff = n
    for p in range(3, int(n**0.5) + 1):
        if n % p == 0:
            q = n // p
            if q >= 3:
                diff = abs(p - q)
                if diff < best_diff:
                    best_diff = diff
                    best = sorted([p, q], reverse=True)
    return best


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cs = CustomSolver.group(n=args.n)
    r = cs.search(seed=args.seed)
    print(f"\n  RESULT: {r.matrix.shape[0]}x{r.matrix.shape[1]}  E={r.metrics.energy}")
