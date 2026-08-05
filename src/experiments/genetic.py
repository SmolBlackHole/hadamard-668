"""GPU genetic search for TT(n) — CuPy RawKernel NPAF evaluation.

Population energy is computed via a hand-written CUDA kernel
that evaluates all individuals in parallel.

Usage::

    PYTHONPATH=src python -m experiments.genetic --tt 12 --popsize 4096 --gen 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from fixtures import equiv_hamming, hamming_distance  # noqa: E402
from gpu import to_numpy, xp  # noqa: E402
from strategies.base import Result, TurynStrategy  # noqa: E402

# ── CuPy RawKernel — compiled once on import ────────────────────────────────

_KERNEL = r"""
extern "C" __global__
void npaf_energy(
    const signed char* batch,   // (pop, 4, n)
    const long long* lens,      // (4,) int64: n, n, n, n-1
    double* energies             // (pop,) float64 output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pop = gridDim.x * blockDim.x;
    if (idx >= pop) return;

    int n = (int)lens[0];
    int L0 = n;
    int L1 = n;
    int L2 = n;
    int L3 = n - 1;

    int base = idx * 4 * n;
    double energy = 0.0;

    for (int k = 1; k < n; ++k) {
        long long corr = 0;

        if (k < L0) {
            int s = 0;
            for (int i = 0; i < L0 - k; ++i)
                s += batch[base + 0 * n + i] * batch[base + 0 * n + i + k];
            corr += s;
        }
        if (k < L1) {
            int s = 0;
            for (int i = 0; i < L1 - k; ++i)
                s += batch[base + 1 * n + i] * batch[base + 1 * n + i + k];
            corr += s;
        }
        if (k < L2) {
            int s = 0;
            for (int i = 0; i < L2 - k; ++i)
                s += batch[base + 2 * n + i] * batch[base + 2 * n + i + k];
            corr += 2 * s;
        }
        if (k < L3) {
            int s = 0;
            for (int i = 0; i < L3 - k; ++i)
                s += batch[base + 3 * n + i] * batch[base + 3 * n + i + k];
            corr += 2 * s;
        }

        double cfd = (double)corr;
        energy += cfd * cfd;
    }

    energies[idx] = energy;
}
"""


def _compile():
    if xp.__name__ != "cupy":
        return None
    return xp.RawKernel(_KERNEL, "npaf_energy")


_npaf_kernel = _compile()


def batch_npaf_energy(pop: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """NPAF energy for (pop, 4, n) int8 on GPU. Falls back to FFT if no CuPy."""
    if _npaf_kernel is None:
        from correlations import TURYN_WEIGHTS, npa_f_residual

        r = npa_f_residual(pop.astype(np.float32), lengths=lengths, weights=TURYN_WEIGHTS)
        return to_numpy(xp.sum(r**2, axis=1))

    import cupy as cp

    pop_gpu = cp.asarray(pop.astype(np.int8))
    lengths_gpu = cp.asarray(lengths.astype(np.int64))
    energies_gpu = cp.zeros(pop.shape[0], dtype=cp.float64)

    tpb = 256
    nb = (pop.shape[0] + tpb - 1) // tpb
    _npaf_kernel((nb,), (tpb,), (pop_gpu, lengths_gpu, energies_gpu))

    return cp.asnumpy(energies_gpu)


def _random_population_batch(n, size, rng):
    """Generate population via DR sieve. Returns (size, 4, n) int8."""
    from sieve import seed_turyn_batch

    batch = seed_turyn_batch(n, size, rng, module=xp)
    if hasattr(batch, "get"):
        batch = to_numpy(batch)
    batch[:, 3, -1] = 0
    return batch


class BatchedGeneticSearch(TurynStrategy):
    """GPU genetic algorithm with RawKernel NPAF evaluation."""

    gpu_exclusive = True

    def __init__(
        self,
        *,
        n: int = 56,
        pop_size: int = 256,
        generations: int = 100,
        elite_count: int = 8,
        mutation_rate: float = 0.3,
        mutation_strength: float = 0.05,
        crossover_rate: float = 0.8,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.pop_size = pop_size
        self.generations = generations
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate

    @classmethod
    def from_order(cls, order: int) -> BatchedGeneticSearch:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(f"order {order} has no TT(n) construction")
        return cls(n=n)

    @property
    def name(self) -> str:
        return "genetic-batch"

    def _crossover_gpu(self, pop_gpu, rng):
        """GPU crossover: vectorized split-and-swap for (size, 4, N)."""
        size, _, N = pop_gpu.shape
        perm = xp.asarray(rng.permutation(size), dtype=xp.int32)
        partners = pop_gpu[perm]
        result = pop_gpu.copy()

        for s in range(4):
            L = N - (1 if s == 3 else 0)
            if L <= 1:
                continue
            splits = xp.asarray(rng.integers(1, L, size=size), dtype=xp.int32)
            cols = xp.arange(N, dtype=xp.int32)
            mask = cols[None, :] >= splits[:, None]
            result[:, s, :] = xp.where(mask, partners[:, s, :], result[:, s, :])

        result[:, 3, -1] = 0
        return result

    def _mutate_gpu(self, pop_gpu, rng):
        """GPU mutation: flip random bits via vectorized mask. Mutates in-place."""
        size, _, N = pop_gpu.shape
        B = 4 * N - 1
        n_max = max(1, int(B * self.mutation_strength))
        n_flips = rng.integers(0, n_max + 1, size=size).astype(np.int32)
        total_flips = int(n_flips.sum())
        if total_flips == 0:
            return pop_gpu
        indiv_idx = np.repeat(np.arange(size, dtype=np.int32), n_flips)
        flat_idx = np.array([], dtype=np.int32)
        for i in range(size):
            nf = int(n_flips[i])
            if nf:
                flat_idx = np.concatenate(
                    [flat_idx, rng.choice(B, size=nf, replace=False).astype(np.int32)]
                )
        s_idx = flat_idx // N
        c_idx = flat_idx % N
        valid = ~((s_idx == 3) & (c_idx == N - 1))
        pop_gpu[
            xp.asarray(indiv_idx[valid]), xp.asarray(s_idx[valid]), xp.asarray(c_idx[valid])
        ] *= -1
        pop_gpu[:, 3, -1] = 0
        return pop_gpu

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        lengths = self.LENGTHS

        pop = _random_population_batch(self.N, self.pop_size, rng)
        pop_gpu = xp.asarray(pop, dtype=xp.int8)
        from energy import turyn_energy
        from strategies.kflip import search as kflip_search

        best_e = float("inf")
        best_ind = pop[0].copy()

        for gen in range(self.generations):
            # ── CUDA kernel: evaluate population ──
            pop_np = to_numpy(pop_gpu)
            energies = batch_npaf_energy(pop_np, lengths)
            pop_best_idx = int(np.argmin(energies))
            pop_best_e = float(energies[pop_best_idx])

            if pop_best_e < best_e:
                best_e = pop_best_e
                best_ind = pop_np[pop_best_idx].copy()

            # ── Elitism ──
            order = np.argsort(energies)
            elite_gpu = pop_gpu[xp.asarray(order[: self.elite_count])].copy()

            # ── GPU crossover + mutation ──
            new_pop_gpu = self._crossover_gpu(pop_gpu, rng)
            new_pop_gpu = self._mutate_gpu(new_pop_gpu, rng)

            # Replace worst with elite
            new_np = to_numpy(new_pop_gpu)
            new_energies = batch_npaf_energy(new_np, lengths)
            worst_idx = xp.asarray(np.argsort(new_energies)[::-1][: self.elite_count])
            new_pop_gpu[worst_idx] = elite_gpu

            pop_gpu = new_pop_gpu
            pop = new_np

            # ── KFlip polishing ──
            if gen % 5 == 4 or gen == self.generations - 1:
                top_n = min(8, self.pop_size)
                top_indices = np.argsort(energies)[:top_n]
                for idx in top_indices:
                    seq = pop[idx].copy()
                    rng_kf = np.random.default_rng(rng.integers(0, 2**31))
                    _, e = kflip_search(seq, turyn_energy, rng_kf)
                    if e < best_e:
                        best_e = e
                        best_ind = seq.copy()
                    if best_e == 0:
                        break

            # ── Basin collapse detection + restart ──
            if gen % 10 == 9 and len(pop) >= 8 and best_e > 0:
                top8 = pop[np.argsort(energies)[:8]]
                hamm = np.zeros((8, 8), dtype=np.int32)
                for i in range(8):
                    for j in range(i + 1, 8):
                        hamm[i, j] = hamming_distance(top8[i], top8[j], lengths)
                avg_dist = hamm.sum() / 28
                if avg_dist < 3:
                    n_restart = self.pop_size // 2
                    fresh = _random_population_batch(self.N, n_restart, rng)
                    half_idx = self.pop_size // 2
                    pop_gpu[xp.arange(half_idx, self.pop_size)] = xp.asarray(fresh)

            if gen % 10 == 0 or gen == self.generations - 1 or best_e == 0:
                avg_e = float(np.mean(energies))
                print(
                    f"  gen {gen:>3}: best={best_e:.0f} avg={avg_e:.0f} pop_best={pop_best_e:.0f}"
                )

            if best_e == 0:
                break

        matrix, metrics = self.build(best_ind)
        elapsed = time.perf_counter() - started
        h = equiv_hamming(best_ind, lengths, self.N)
        print(f"  seed={seed} best_e={metrics.energy:.0f} hamming={h} {elapsed:.1f}s")
        return Result(
            matrix=matrix, metrics=metrics, elapsed=elapsed, seed=seed, sequences=best_ind
        )

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tt", type=int, default=12)
    parser.add_argument("--popsize", type=int, default=4096)
    parser.add_argument("--gen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elite", type=int, default=8)
    args = parser.parse_args()

    ga = BatchedGeneticSearch(
        n=args.tt,
        pop_size=args.popsize,
        generations=args.gen,
        elite_count=args.elite,
    )
    result = ga.search(steps=0, seed=args.seed)
    print(f"\nFinal: E={result.metrics.energy}")
    return result.metrics.energy == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
