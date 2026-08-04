"""FFT-guided local search in the Turyn-type sequence space."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from builders import build_turyn
from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from gpu import check_orthogonality, xp
from sieve import seed_turyn_batch
from .montecarlo import MonteCarloSearch
from .base import SearchStrategy


class TurynPocsSearch(SearchStrategy):
    """Use a zero-padded FFT gradient to rank exact Turyn bit flips.

    Zweck: Priorisiert aussichtsreiche Turyn-Flips mit einem FFT-Gradienten.
    Mechanik: Prüft jeden vorgeschlagenen Flip gegen die exakte NPAF-Energie.
    Grundlage: Zero-Padding berechnet lineare statt zyklischer Korrelationen.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Der Gradient ordnet Kandidaten nur vor; er entscheidet nicht über Annahme.
    """

    N = 56
    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, *, n: int = N, gradient_interval: int = 7, candidates: int = 8,
                 batch_size: int = 256, sieve: bool = True) -> None:
        if n < 2:
            raise ValueError("Turyn type requires n >= 2")
        if gradient_interval < 1 or candidates < 1:
            raise ValueError(
                "gradient_interval and candidates must be positive")
        self.N = n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.ORDER = 4 * (3 * n - 1)
        self.gradient_interval, self.candidates = gradient_interval, candidates
        self.batch_size, self.sieve = batch_size, sieve

    @property
    def name(self) -> str:
        return "turyn_pocs"

    def _seed(self, rng: np.random.Generator) -> np.ndarray:
        if self.sieve:
            return seed_turyn_batch(self.N, 1, rng)[0]
        sequences = np.zeros((4, self.N), dtype=np.int8)
        for index, length in enumerate(self.LENGTHS):
            sequences[index, :length] = rng.choice((-1, 1), size=length)
        return sequences

    def _build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        matrix = build_turyn(*(sequences[index, :self.LENGTHS[index]]
                               for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def _gradient(self, sequences: np.ndarray) -> np.ndarray:
        """Return the exact continuous NPAF gradient via zero-padded FFTs."""
        fft_size = 2 * self.N - 1
        spectrum = np.fft.fft(sequences, n=fft_size, axis=1)
        correlations = np.fft.ifft(spectrum * spectrum.conj(), axis=1).real
        total = np.tensordot(self.WEIGHTS, correlations, axes=1)
        coefficients = np.zeros(fft_size, dtype=np.float64)
        coefficients[1:self.N] = total[1:self.N]
        coefficients[-(self.N - 1):] = total[1:self.N][::-1]
        gradient = 2.0 * self.WEIGHTS[:, None] * np.fft.ifft(
            np.fft.fft(coefficients)[None, :] * spectrum, axis=1).real
        return gradient[:, :self.N]

    def _search_cpu(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self._seed(rng)
        correlations = nonperiodic_autocorrelation_state(
            sequences, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energy = best_energy = nonperiodic_correlation_energy(correlations)
        best_sequences = sequences.copy()
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                choices: list[tuple[int, int]] = []
                if step % self.gradient_interval == 0:
                    gradient = self._gradient(sequences)
                    score = -2.0 * sequences * gradient
                    score[3, self.N - 1] = np.inf
                    selected = np.argpartition(
                        score.ravel(), self.candidates - 1)[:self.candidates]
                    choices = [tuple(map(int, divmod(index, self.N)))
                               for index in selected]
                else:
                    sequence_index = int(rng.integers(0, 4))
                    choices = [(sequence_index, int(
                        rng.integers(0, self.LENGTHS[sequence_index])))]
                improved = False
                for sequence_index, value_index in choices:
                    new_energy = apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                    if new_energy < energy:
                        energy, accepted, improved = new_energy, accepted + 1, True
                        if new_energy < best_energy:
                            best_energy, best_at = new_energy, step
                            best_sequences = sequences.copy()
                        break
                    apply_nonperiodic_flip(
                        sequences, correlations, sequence_index, value_index,
                        lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
                if not best_energy:
                    break
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                bar.update(1)
        matrix, metrics = self._build(best_sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_energy={best_energy} found@step={best_at} "
              f"accepted={accepted} {elapsed:.1f}s")
        return matrix, metrics, elapsed

    def _search_gpu(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        """Run independent POCS trajectories with persistent GPU NPAF state."""
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = (seed_turyn_batch(self.N, self.batch_size, rng, module=xp)
                 if self.sieve else rng.integers(
                     0, 2, size=(self.batch_size, 4, self.N), dtype=xp.int8) * 2 - 1)
        batch[:, 3, -1] = 0
        delta = MonteCarloSearch(self.ORDER, batch_size=self.batch_size)
        correlations = delta._correlations(batch)
        energies = delta._energies_from_correlations(correlations)
        rows = xp.arange(self.batch_size)
        for step in range(steps):
            if step % self.gradient_interval == 0:
                fft_size = 2 * self.N - 1
                spectrum = xp.fft.fft(batch, n=fft_size, axis=2)
                autocorrelations = xp.fft.ifft(
                    xp.abs(spectrum) ** 2, axis=2).real
                total = xp.sum(xp.asarray(self.WEIGHTS)[
                               None, :, None] * autocorrelations, axis=1)
                coefficients = xp.zeros_like(total)
                coefficients[:, 1:self.N] = total[:, 1:self.N]
                coefficients[:, -(self.N - 1):] = total[:, 1:self.N][:, ::-1]
                gradient = 2 * xp.asarray(self.WEIGHTS)[None, :, None] * xp.fft.ifft(
                    xp.fft.fft(coefficients, axis=1)[:, None, :] * spectrum, axis=2).real[:, :, :self.N]
                score = -2 * batch * gradient
                score[:, 3, -1] = xp.inf
                choice = xp.argmin(score.reshape(self.batch_size, -1), axis=1)
                sequence = (choice // self.N).astype(xp.int64)
                column = (choice % self.N).astype(xp.int64)
            else:
                position = rng.integers(
                    0, int(self.LENGTHS.sum()), size=self.batch_size)
                sequence = xp.minimum(position // self.N, 3).astype(xp.int64)
                column = (position - sequence * self.N).astype(xp.int64)
            delta._apply_flips(batch, correlations, energies, sequence, column,
                               self.LENGTHS, self.WEIGHTS)
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix, metrics = self._build(best)
        return matrix, metrics, time.perf_counter() - started

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        return self._search_gpu(steps, seed) if xp.__name__ == "cupy" else self._search_cpu(steps, seed)
