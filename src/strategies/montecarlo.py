"""CUDA-Batch-Suche über viele zyklische Vier-Sequenzen-Kandidaten."""
from __future__ import annotations

import time

import numpy as np

from builders import build_goethals_seidel
from gpu import check_orthogonality, xp
from .base import SearchStrategy


class MonteCarloSearch(SearchStrategy):
    """Mutiert viele Vier-Sequenzen-Kandidaten parallel auf CUDA.

    Zweck: Nutzt die GPU für einen Batch zirkulanter Goethals-Seidel-Kandidaten.
    Mechanik: Initialisiert die Batch-Korrelation per FFT und bewertet danach einen Flip je Kandidat mit exakten Deltas.
    Grundlage: Wiener-Khinchin liefert den Startzustand; ein Flip ändert jede periodische Korrelation in konstanter Zeit.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Benötigt einen aktiven CuPy-Backend; ohne CUDA wird kein CPU-Fallback angeboten.
    """

    _STEP_KERNEL = None

    def __init__(self, order: int = SearchStrategy.ORDER, batch_size: int = 256) -> None:
        if order < 4 or order % 4:
            raise ValueError(
                "montecarlo search requires an order divisible by four")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ORDER = order
        self.K = order // 4
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "montecarlo"

    @staticmethod
    def _correlations(batch):
        spectrum = xp.fft.fft(batch, axis=2)
        return xp.rint(xp.fft.ifft(
            xp.abs(spectrum) ** 2, axis=2).real).sum(
                axis=1).astype(xp.int32)

    @staticmethod
    def _energies(correlations):
        values = correlations[:, 1:].astype(xp.int64, copy=False)
        return xp.sum(values * values, axis=1)

    @staticmethod
    def _shift_indices(size: int):
        positions = xp.arange(size, dtype=xp.int32)[:, None]
        shifts = xp.arange(size, dtype=xp.int32)[None, :]
        return (positions + shifts) % size, (positions - shifts) % size

    @classmethod
    def _apply_flips(
        cls, batch, correlations, energies, sequence, column,
        forward_indices, backward_indices,
    ) -> None:
        """Evaluate and apply improving flips with one CUDA kernel launch."""
        if cls._STEP_KERNEL is None:
            cls._STEP_KERNEL = xp.RawKernel(r'''
            extern "C" __global__ void montecarlo_step(
                signed char* batch,
                int* correlations,
                long long* energies,
                const long long* sequences,
                const long long* columns,
                const int* forward,
                const int* backward,
                int batch_size,
                int size
            ) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;
                if (row >= batch_size) return;

                long long sequence = sequences[row];
                long long column = columns[row];
                long long batch_offset = (row * 4 + sequence) * size;
                long long correlation_offset = row * size;
                int old_value = batch[batch_offset + column];
                long long candidate_energy = energies[row];

                for (int shift = 1; shift < size; ++shift) {
                    int neighbor =
                        batch[batch_offset + forward[column * size + shift]] +
                        batch[batch_offset + backward[column * size + shift]];
                    int delta = -2 * old_value * neighbor;
                    candidate_energy +=
                        2LL * correlations[correlation_offset + shift] * delta +
                        1LL * delta * delta;
                }

                if (candidate_energy > energies[row]) return;
                batch[batch_offset + column] = -old_value;
                energies[row] = candidate_energy;
                for (int shift = 1; shift < size; ++shift) {
                    int neighbor =
                        batch[batch_offset + forward[column * size + shift]] +
                        batch[batch_offset + backward[column * size + shift]];
                    correlations[correlation_offset + shift] +=
                        -2 * old_value * neighbor;
                }
            }
            ''', "montecarlo_step")
        threads = 128
        cls._STEP_KERNEL(
            ((len(batch) + threads - 1) // threads,), (threads,),
            (batch, correlations, energies, sequence, column,
             forward_indices, backward_indices,
             np.int32(len(batch)), np.int32(batch.shape[2])),
        )

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        if xp.__name__ != "cupy":
            raise RuntimeError(
                "montecarlo search requires an active CuPy backend")
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = rng.integers(0, 2, size=(
            self.batch_size, 4, self.K), dtype=xp.int8) * 2 - 1
        correlations = self._correlations(batch)
        energies = self._energies(correlations)
        forward_indices, backward_indices = self._shift_indices(self.K)
        for step in range(steps):
            sequence = rng.integers(0, 4, size=self.batch_size)
            column = rng.integers(0, self.K, size=self.batch_size)
            self._apply_flips(
                batch, correlations, energies, sequence, column,
                forward_indices, backward_indices)
            if (step + 1) % 1000 == 0 and int(energies.min().get()) == 0:
                break
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
