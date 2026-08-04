"""CUDA-batch search over Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, xp
from sieve import seed_turyn_batch
from .base import SearchStrategy


class MonteCarloSearch(SearchStrategy):
    """Mutate many TT(n) candidates in parallel on CUDA.

    Zweck: Führt unabhängige Turyn-Restarts als GPU-Batch aus.
    Mechanik: Testet einen Flip je Kandidat über batched zero-padded FFTs.
    Grundlage: Fitness ist die gewichtete nichtperiodische Turyn-Energie.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Benötigt CuPy; ein Flip-Schritt berechnet den kleinen FFT-Batch neu.
    """

    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)
    _STEP_KERNEL = None

    def __init__(self, order: int = SearchStrategy.ORDER, batch_size: int = 256, sieve: bool = True) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(
                "montecarlo search requires an order with a TT(n) construction")
        self.ORDER, self.N, self.batch_size, self.sieve = order, n, batch_size, sieve
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)

    @property
    def name(self) -> str:
        return "montecarlo"

    @property
    def construction(self) -> str:
        return f"turyn_tt_{self.N}"

    def _energies(self, batch):
        return self._energies_from_correlations(self._correlations(batch))

    def _correlations(self, batch):
        fft_size = 2 * self.N - 1
        spectrum = xp.fft.fft(batch, n=fft_size, axis=2)
        correlations = xp.rint(xp.fft.ifft(
            xp.abs(spectrum) ** 2, axis=2).real[:, :, :self.N])
        return xp.sum(correlations * xp.asarray(
            self.WEIGHTS, dtype=xp.int32)[None, :, None], axis=1).astype(xp.int32)

    @staticmethod
    def _energies_from_correlations(correlations):
        values = correlations[:, 1:].astype(xp.int64, copy=False)
        return xp.sum(values * values, axis=1)

    @classmethod
    def _apply_flips(cls, batch, correlations, energies, sequence, column, lengths, weights) -> None:
        """Apply non-worsening weighted NPAF flips with one CUDA launch."""
        if cls._STEP_KERNEL is None:
            cls._STEP_KERNEL = xp.RawKernel(r'''
            extern "C" __global__ void turyn_montecarlo_step(
                signed char* batch, int* correlations, long long* energies,
                const long long* sequences, const long long* columns,
                const long long* lengths, const long long* weights,
                int batch_size, int width
            ) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;
                if (row >= batch_size) return;
                int sequence = (int)sequences[row];
                int column = (int)columns[row];
                int length = (int)lengths[sequence];
                int weight = (int)weights[sequence];
                int batch_offset = (row * 4 + sequence) * width;
                int correlation_offset = row * width;
                int old_value = batch[batch_offset + column];
                long long candidate_energy = energies[row];
                for (int shift = 1; shift < length; ++shift) {
                    int neighbour = 0;
                    if (column + shift < length) neighbour += batch[batch_offset + column + shift];
                    if (column >= shift) neighbour += batch[batch_offset + column - shift];
                    int delta = -2 * weight * old_value * neighbour;
                    long long value = correlations[correlation_offset + shift];
                    candidate_energy += 2LL * value * delta + 1LL * delta * delta;
                }
                if (candidate_energy > energies[row]) return;
                batch[batch_offset + column] = -old_value;
                energies[row] = candidate_energy;
                for (int shift = 1; shift < length; ++shift) {
                    int neighbour = 0;
                    if (column + shift < length) neighbour += batch[batch_offset + column + shift];
                    if (column >= shift) neighbour += batch[batch_offset + column - shift];
                    correlations[correlation_offset + shift] += -2 * weight * old_value * neighbour;
                }
            }
            ''', "turyn_montecarlo_step")
        threads = 128
        cls._STEP_KERNEL(
            ((len(batch) + threads - 1) // threads,), (threads,),
            (batch, correlations, energies, sequence, column,
             xp.asarray(lengths, dtype=xp.int64), xp.asarray(
                 weights, dtype=xp.int64),
             np.int32(len(batch)), np.int32(batch.shape[2])),
        )

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        if xp.__name__ != "cupy":
            raise RuntimeError(
                "montecarlo search requires an active CuPy backend")
        rng = xp.random.default_rng(seed)
        batch = (seed_turyn_batch(self.N, self.batch_size, rng, module=xp)
                 if self.sieve else rng.integers(
                     0, 2, size=(self.batch_size, 4, self.N), dtype=xp.int8) * 2 - 1)
        batch[:, 3, -1] = 0
        correlations = self._correlations(batch)
        energies = self._energies_from_correlations(correlations)
        for _ in range(steps):
            position = rng.integers(
                0, int(self.LENGTHS.sum()), size=self.batch_size)
            sequence = xp.minimum(position // self.N, 3).astype(xp.int64)
            column = (position - sequence * self.N).astype(xp.int64)
            self._apply_flips(
                batch, correlations, energies, sequence, column,
                self.LENGTHS, self.WEIGHTS)
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix = build_turyn(
            *(best[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
