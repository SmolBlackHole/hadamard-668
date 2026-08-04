"""Experimental repair with triples, tabu detection, and plateau escape.

Mirrors ``strategies/repair.py`` API (TurynStrategy, works in pipeline/CLI)
but adds rich diagnostics, cycle prevention, and controlled-noise benchmarks.

Usage as CLI strategy::

    python run.py --strategy "spectral_descent:100,repair-exp:400" --order 140

Usage as benchmark::

    python src/experiments/syndrome_repair.py
"""

from __future__ import annotations

import time
from collections import deque
from itertools import combinations

import numpy as np

from correlations import (
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_residual,
)
from gpu import to_numpy, xp
from strategies.base import Result, TurynStrategy


class RepairExperiment(TurynStrategy):
    """Experimental repair: singles + pairs + triples + tabu + plateau escape.

    Parameters
    ----------
    n : int
        Turyn parameter (TT(n)).
    sieve : bool
        Use DR sieve for seeding (ignored when sequences are provided).
    pair_interval, pair_top : int
        Run pair model every N steps, considering the top-K singles.
    triple_interval, triple_top : int
        Run triple scan every N steps, considering the top-K singles.
    tabu_size : int
        Number of recent state hashes to remember to prevent cycles.
    plateau_threshold : int
        Number of consecutive non-improving pair/triple scans before
        injecting random noise to escape a local minimum.
    plateau_noise : int
        Number of random bits to flip during a plateau escape.
    """

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        sieve: bool = True,
        pair_interval: int = 5,
        pair_top: int = 64,
        triple_interval: int = 25,
        triple_top: int = 20,
        tabu_size: int = 256,
        plateau_threshold: int = 8,
        plateau_noise: int = 4,
        verbose: bool = True,
        temperature: float = 0.0,
        cooling: float = 1.0,
    ):
        super().__init__(n=n, sieve=sieve)
        self.pair_interval = pair_interval
        self.pair_top = pair_top
        self.triple_interval = triple_interval
        self.triple_top = triple_top
        self.tabu_size = tabu_size
        self.plateau_threshold = plateau_threshold
        self.plateau_noise = plateau_noise
        self.verbose = verbose
        self.temperature = temperature
        self.cooling = cooling
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "repair-exp"

    # ── GPU primitives ──────────────────────────────────────────────────────

    def _single_deltas(self, seq_f, positions):
        B = len(positions)
        batch = xp.repeat(seq_f[None, ...], B, axis=0)
        for idx, (r, c) in enumerate(positions):
            batch[idx, r, c] *= -1
        residuals = npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        r0 = npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
        r0_cpu = to_numpy(r0)
        residuals_cpu = to_numpy(residuals)
        return residuals_cpu - r0_cpu[None, :], r0_cpu

    def _q_pair_model(self, seq_f, positions, top_indices, deltas_top):
        K = len(top_indices)
        if K < 2:
            return np.zeros((K, K, self.N - 1), dtype=np.float64)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        batch = xp.repeat(seq_f[None, ...], len(pairs), axis=0)
        for idx, (pi, pj) in enumerate(pairs):
            for t in (pi, pj):
                r, c = positions[top_indices[t]]
                batch[idx, r, c] *= -1
        residuals = npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        r0 = npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
        residuals_cpu = to_numpy(residuals)
        r0_cpu = to_numpy(r0)
        Q = np.zeros((K, K, self.N - 1), dtype=np.float64)
        for idx, (pi, pj) in enumerate(pairs):
            q = residuals_cpu[idx] - r0_cpu - deltas_top[pi] - deltas_top[pj]
            Q[pi, pj] = q
            Q[pj, pi] = q
        return Q

    def _triple_scan(self, seq_f, positions, top_indices):
        K = len(top_indices)
        if K < 3:
            return None
        triples = list(combinations(range(K), 3))
        batch = xp.repeat(seq_f[None, ...], len(triples), axis=0)
        for idx, (ti, tj, tk) in enumerate(triples):
            for t in (ti, tj, tk):
                r, c = positions[top_indices[t]]
                batch[idx, r, c] *= -1
        residuals = npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        energies = xp.sum(residuals**2, axis=1)
        best_idx = int(xp.argmin(energies))
        return (triples[best_idx], float(energies[best_idx]))

    # ── State hashing (tabu) ────────────────────────────────────────────────

    def _state_key(self, sequences: np.ndarray) -> int:
        """Fast 64-bit hash of the full (4, n) sequence array."""
        return hash(sequences.tobytes())

    def _random_kick(
        self, sequences: np.ndarray, seq_f, positions, rng: np.random.Generator, k: int
    ):
        """Flip k random bits to escape a plateau."""
        indices = rng.choice(len(positions), size=k, replace=False)
        for idx in indices:
            r, c = positions[idx]
            sequences[r, c] *= -1
            seq_f[r, c] *= -1

    # ── Exact energy helper ──────────────────────────────────────────────────

    def _exact_energy(self, sequences: np.ndarray) -> float:
        return float(
            nonperiodic_correlation_energy(
                nonperiodic_autocorrelation_state(
                    sequences, lengths=self.LENGTHS, weights=TURYN_WEIGHTS
                )
            )
        )

    def _should_accept(self, cpu_e: float, current_e: float, metro_rng) -> bool:
        """Accept if energy improves, or via Metropolis if temperature > 0."""
        return cpu_e < current_e or (
            self.temperature > 0
            and cpu_e > current_e
            and metro_rng.random() < np.exp(-(cpu_e - current_e) / self.temperature)
        )

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed + 1)
        if sequences is None:
            rng_seed = np.random.default_rng(seed)
            sequences = self.seed(rng_seed)
        else:
            sequences = sequences.copy()
        positions = [(r, c) for r in range(4) for c in range(int(self.LENGTHS[r]))]
        seq_f = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)

        best_e = self._exact_energy(sequences)
        best_seq = sequences.copy()
        best_e_ever = best_e
        best_at = 0

        tabu: deque[int] = deque(maxlen=self.tabu_size)
        tabu.append(self._state_key(sequences))

        singles_used = 0
        pairs_used = 0
        triples_used = 0
        kicks_used = 0
        metro_used = 0
        plateau_strikes = 0
        metro_rng = np.random.default_rng(seed + 2) if self.temperature > 0 else None

        for step in range(steps):
            if best_e == 0:
                break

            # Phase 1: singles — GPU full scan, CPU-confirmed acceptance
            deltas, r0 = self._single_deltas(seq_f, positions)
            energies = np.sum((r0 + deltas) ** 2, axis=1)
            best_idx = int(np.argmin(energies))
            best_single_e = float(energies[best_idx])

            single_accepted = False
            if best_single_e < best_e - 0.5:
                r, c = positions[best_idx]
                sequences[r, c] *= -1
                seq_f[r, c] *= -1
                cpu_e = self._exact_energy(sequences)
                if self._should_accept(cpu_e, best_e, metro_rng):
                    key = self._state_key(sequences)
                    if key in tabu:
                        sequences[r, c] *= -1
                        seq_f[r, c] *= -1
                    else:
                        tabu.append(key)
                        if cpu_e >= best_e:
                            metro_used += 1
                        best_e = cpu_e
                        singles_used += 1
                        single_accepted = True
                        if best_e < best_e_ever:
                            best_e_ever = best_e
                            best_seq = sequences.copy()
                            best_at = step
                            plateau_strikes = 0
                        if best_e == 0:
                            break
                else:
                    key = self._state_key(sequences)
                    tabu.append(key)
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1

            if single_accepted:
                continue

            # Phase 2: pairs — on interval tick OR after single stall
            should_try_pairs = (
                step % self.pair_interval == self.pair_interval - 1
                or best_single_e >= best_e - 0.5  # singles stalled
            )
            if should_try_pairs and best_e > 0:
                # Recompute deltas for current state (not stale)
                deltas, r0 = self._single_deltas(seq_f, positions)
                energies = np.sum((r0 + deltas) ** 2, axis=1)
                top_idx = np.argsort(energies)[: self.pair_top]
                Q = self._q_pair_model(seq_f, positions, top_idx, deltas[top_idx])
                r0_now = to_numpy(
                    npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
                )
                best_pair_e = best_e
                best_pair = None
                for i in range(len(top_idx)):
                    for j in range(i + 1, len(top_idx)):
                        e = float(
                            np.sum(
                                (r0_now + deltas[top_idx[i]] + deltas[top_idx[j]] + Q[i, j]) ** 2
                            )
                        )
                        if e < best_pair_e:
                            best_pair_e = e
                            best_pair = (top_idx[i], top_idx[j])
                if best_pair is not None and best_pair_e < best_e - 0.5:
                    for pi in best_pair:
                        r, c = positions[pi]
                        sequences[r, c] *= -1
                        seq_f[r, c] *= -1
                    cpu_e = self._exact_energy(sequences)
                    key = self._state_key(sequences)
                    if self._should_accept(cpu_e, best_e, metro_rng):
                        if key in tabu:
                            for pi in best_pair:
                                r, c = positions[pi]
                                sequences[r, c] *= -1
                                seq_f[r, c] *= -1
                        else:
                            tabu.append(key)
                            if cpu_e >= best_e:
                                metro_used += 1
                            best_e = cpu_e
                            pairs_used += 1
                            if best_e < best_e_ever:
                                best_e_ever = best_e
                                best_seq = sequences.copy()
                                best_at = step
                                plateau_strikes = 0
                            if best_e == 0:
                                break
                    else:
                        tabu.append(key)
                        for pi in best_pair:
                            r, c = positions[pi]
                            sequences[r, c] *= -1
                            seq_f[r, c] *= -1
                else:
                    plateau_strikes += 1

            # Phase 3: triples
            if step % self.triple_interval == self.triple_interval - 1 and best_e > 0:
                triple_candidates = np.argsort(energies)[: self.triple_top]
                result = self._triple_scan(seq_f, positions, triple_candidates)
                if result is not None:
                    triple_indices, triple_energy = result
                    if triple_energy < best_e - 0.5:
                        for ti in triple_indices:
                            r, c = positions[triple_candidates[ti]]
                            sequences[r, c] *= -1
                            seq_f[r, c] *= -1
                        cpu_e = self._exact_energy(sequences)
                        key = self._state_key(sequences)
                        if self._should_accept(cpu_e, best_e, metro_rng):
                            if key in tabu:
                                for ti in triple_indices:
                                    r, c = positions[triple_candidates[ti]]
                                    sequences[r, c] *= -1
                                    seq_f[r, c] *= -1
                            else:
                                tabu.append(key)
                                if cpu_e >= best_e:
                                    metro_used += 1
                                best_e = cpu_e
                                triples_used += 1
                                if best_e < best_e_ever:
                                    best_e_ever = best_e
                                    best_seq = sequences.copy()
                                    best_at = step
                                    plateau_strikes = 0
                                if best_e == 0:
                                    break
                        else:
                            tabu.append(key)
                            for ti in triple_indices:
                                r, c = positions[triple_candidates[ti]]
                                sequences[r, c] *= -1
                                seq_f[r, c] *= -1
                else:
                    plateau_strikes += 1

            # Phase 4: plateau escape
            if plateau_strikes >= self.plateau_threshold and best_e > 0:
                self._random_kick(sequences, seq_f, positions, rng, self.plateau_noise)
                best_e = self._exact_energy(sequences)
                kicks_used += 1
                plateau_strikes = 0
                if best_e < best_e_ever:
                    best_e_ever = best_e
                    best_seq = sequences.copy()
                    best_at = step

            if self.cooling < 1.0:
                self.temperature *= self.cooling

        matrix, metrics = self.build(best_seq)
        elapsed = time.perf_counter() - started
        stats_parts = [f"singles={singles_used}"]
        if pairs_used:
            stats_parts.append(f"pairs={pairs_used}")
        if triples_used:
            stats_parts.append(f"triples={triples_used}")
        if kicks_used:
            stats_parts.append(f"kicks={kicks_used}")
        if metro_used:
            stats_parts.append(f"metro={metro_used}")
        stats_parts.append(f"plateau_strikes={plateau_strikes}")
        if self.verbose:
            print(
                f"  seed={seed} best_e={best_e_ever:.0f}"
                f" ({', '.join(stats_parts)})"
                f" best@step={best_at}"
                f" {elapsed:.1f}s"
            )
        return Result(matrix, metrics, elapsed, best_seq)

    def refine(
        self, matrix: np.ndarray, steps: int, seed: int, sequences: np.ndarray | None = None
    ) -> Result:
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)


# ── Controlled-noise benchmark ─────────────────────────────────────────────────


def _benchmark(
    tt_name: str,
    sol: np.ndarray,
    lengths: np.ndarray,
    noise_levels: tuple[float, ...] = (1, 2, 5, 10, 15, 20),
    trials: int = 10,
) -> None:
    """Flip ``noise%`` of bits in a known solution and measure repair success."""
    from fixtures import equiv_hamming

    total_bits = int(lengths.sum())
    n = int(lengths[0])
    backend = "CUPY" if xp.__name__ == "cupy" else "NUMPY"

    print(f"\n{'=' * 65}")
    print(f"  {tt_name}  n={n}  bits={total_bits}  ({backend})")
    print(f"{'=' * 65}")
    print(f"  {'noise':>5} {'flips':>6} {'repaired':>10} {'hamming':>8} {'time':>8}")
    print(f"  {'-' * 41}")

    all_positions = [(r, c) for r in range(4) for c in range(int(lengths[r]))]
    rng = np.random.default_rng(2026)
    for pct in noise_levels:
        n_flips = max(1, int(total_bits * pct / 100))
        steps_budget = max(200, n_flips * 10)
        repaired = 0
        best_h = 999
        total_time = 0.0
        for trial in range(trials):
            damaged = sol.copy()
            chosen = rng.choice(len(all_positions), size=n_flips, replace=False)
            for idx in chosen:
                r, c = all_positions[int(idx)]
                damaged[r, c] *= -1

            rp = RepairExperiment(
                n=n,
                sieve=False,
                pair_interval=5,
                pair_top=64,
                triple_interval=25,
                triple_top=20,
                tabu_size=256,
                plateau_threshold=8,
                plateau_noise=4,
            )
            t0 = time.perf_counter()
            result = rp.search(steps=steps_budget, seed=2026 + trial, sequences=damaged)
            total_time += time.perf_counter() - t0
            h = (
                equiv_hamming(result.sequences, lengths, n)
                if result.sequences is not None
                else None
            )
            if result.metrics["energy"] == 0:
                repaired += 1
            if h is not None and h < best_h:
                best_h = h

        avg_t = total_time / trials
        print(f"  {pct:>4}% {n_flips:>6} {repaired:>8}/{trials} {best_h:>8} {avg_t:>7.1f}s")


if __name__ == "__main__":
    from fixtures import tt_sequences

    for n, label in [(6, "TT(6)"), (8, "TT(8)"), (10, "TT(10)"), (12, "TT(12)")]:
        sol = tt_sequences(n)
        lengths = np.array((n, n, n, n - 1), dtype=np.int64)
        _benchmark(label, sol, lengths)
