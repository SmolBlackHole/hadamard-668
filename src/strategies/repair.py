"""Sequence-level repair: GPU exact single-flip scan + CPU pair model via Q[i,j]."""

from __future__ import annotations

import time

import numpy as np

from correlations import (
    TURYN_WEIGHTS_F,
    TURYN_WEIGHTS_I,
    _npa_f_residual,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from gpu import xp

from .base import Result, TurynStrategy


class RepairSearch(TurynStrategy):
    """GPU singles scan + pair model on TT(n) sequences. Pipeline stage."""

    def __init__(self, *, n: int = TurynStrategy.DEFAULT_N, sieve: bool = True,
                 pair_interval: int = 5, pair_top: int = 64):
        super().__init__(n=n, sieve=sieve)
        self.pair_interval = pair_interval
        self.pair_top = pair_top
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "repair"

    def _q_pair_model(self, seq_f, positions, top_indices, deltas_top):
        """GPU batch: Q[i][j] = residual(i,j) - r0 - delta_i - delta_j."""
        K = len(top_indices)
        if K < 2:
            return np.zeros((K, K, self.N - 1), dtype=np.float64)
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        batch = xp.repeat(seq_f[None, ...], len(pairs), axis=0)
        for idx, (pi, pj) in enumerate(pairs):
            for _pos_idx, (r, c) in enumerate([positions[top_indices[pi]],
                                              positions[top_indices[pj]]]):
                batch[idx, r, c] *= -1
        residuals = _npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS_F, module=xp)
        r0 = _npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS_F, module=xp)[0]
        Q = np.zeros((K, K, self.N - 1), dtype=np.float64)
        for idx, (pi, pj) in enumerate(pairs):
            q = xp.asnumpy(residuals[idx]) - xp.asnumpy(r0) - deltas_top[pi] - deltas_top[pj]
            Q[pi, pj] = q
            Q[pj, pi] = q
        return Q

    def _single_deltas(self, seq_f, positions):
        """GPU batch: delta vectors for all positions."""
        B = len(positions)
        batch = xp.repeat(seq_f[None, ...], B, axis=0)
        for idx, (r, c) in enumerate(positions):
            batch[idx, r, c] *= -1
        residuals = _npa_f_residual(batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS_F, module=xp)
        r0 = _npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS_F, module=xp)[0]
        return np.array(
            [xp.asnumpy(residuals[i] - r0) for i in range(B)], dtype=np.float64
        ), xp.asnumpy(r0)

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self.seed(rng)
        positions = [(r, c) for r in range(4) for c in range(int(self.LENGTHS[r]))]
        seq_f = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)

        best_e = float(
            nonperiodic_correlation_energy(
                nonperiodic_autocorrelation_state(
                    sequences, lengths=self.LENGTHS, weights=TURYN_WEIGHTS_I)))

        for step in range(steps):
            if best_e == 0:
                break
            deltas, r0 = self._single_deltas(seq_f, positions)
            energies = np.sum((r0 + deltas) ** 2, axis=1)
            best_idx = int(np.argmin(energies))
            best_single_e = float(energies[best_idx])

            if best_single_e < best_e:
                r, c = positions[best_idx]
                sequences[r, c] *= -1
                seq_f[r, c] *= -1
                best_e = float(
                    nonperiodic_correlation_energy(
                        nonperiodic_autocorrelation_state(
                            sequences, lengths=self.LENGTHS, weights=TURYN_WEIGHTS_I)))
                if best_e == 0:
                    break

            if step % self.pair_interval == self.pair_interval - 1 and best_e > 0:
                top_idx = np.argsort(energies)[:self.pair_top]
                Q = self._q_pair_model(seq_f, positions, top_idx, deltas[top_idx])
                r0_now = xp.asnumpy(
                    _npa_f_residual(seq_f[None, ...], lengths=self.LENGTHS,
                                    weights=TURYN_WEIGHTS_F, module=xp)[0])
                best_pair_e = best_e
                best_pair = None
                for i in range(len(top_idx)):
                    for j in range(i + 1, len(top_idx)):
                        e = float(np.sum(
                            (r0_now + deltas[top_idx[i]] + deltas[top_idx[j]] + Q[i, j]) ** 2))
                        if e < best_pair_e:
                            best_pair_e = e
                            best_pair = (top_idx[i], top_idx[j])
                if best_pair is not None and best_pair_e < best_e:
                    for pi in best_pair:
                        r, c = positions[pi]
                        sequences[r, c] *= -1
                        seq_f[r, c] *= -1
                    best_e = float(
                        nonperiodic_correlation_energy(
                            nonperiodic_autocorrelation_state(
                                sequences, lengths=self.LENGTHS, weights=TURYN_WEIGHTS_I)))
                    if best_e == 0:
                        break

        matrix, metrics = self.build(sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_energy={best_e:.0f} {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> Result:
        """Repair an existing matrix: extract sequences, repair, rebuild."""
        # Matrix -> sequences is lossy (we can't perfectly extract sequences).
        # For now, re-run search on the same seed.
        return self.search(steps, seed)
