"""Douglas-Rachford projection for Turyn-type sequences."""
from __future__ import annotations
import time
import numpy as np
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, xp
from sieve import seed_turyn_batch
from .base import TurynStrategy


def _project_weighted_nonperiodic_power(sequences, *, lengths, weights, module=np):
    state = module.asarray(sequences, dtype=module.float32)
    if state.ndim not in (2, 3) or state.shape[-2] != len(lengths):
        raise ValueError("shape must be (seq, val) or (batch, seq, val)")
    fft_size = 2 * state.shape[-1] - 1
    spectrum = module.fft.fft(state, n=fft_size, axis=-1)
    target = module.sum(module.asarray(lengths) * module.asarray(weights, dtype=state.dtype))
    shape = (1,) * (state.ndim - 2) + (len(weights), 1)
    power = module.sum(module.asarray(weights, dtype=state.dtype).reshape(shape) *
                       module.abs(spectrum) ** 2, axis=-2)
    proj = module.fft.ifft(
        spectrum * module.sqrt(target / module.maximum(power, 1e-12))[..., None, :],
        axis=-1).real[..., :state.shape[-1]]
    for idx, L in enumerate(lengths):
        proj[..., idx, int(L):] = 0.0
    return proj


class PocsSearch(TurynStrategy):
    """Douglas-Rachford splitting: project alternately between
    weighted NPAF power shell and {+-1} constraint."""

    W_F = np.array((1, 1, 2, 2), dtype=np.float64)

    def __init__(self, inner_steps: int = 50, *, n: int = TurynStrategy.DEFAULT_N,
                 batch_size: int = 256, sieve: bool = True):
        super().__init__(n=n, sieve=sieve)
        self.inner_steps = inner_steps
        self.batch_size = batch_size

    @property
    def name(self) -> str: return "pocs"

    def _project_fourier(self, state, *, module=np):
        return _project_weighted_nonperiodic_power(
            state, lengths=self.LENGTHS, weights=self.W_F, module=module)

    def _project_sign(self, state, *, module=np):
        signs = module.where(state >= 0, 1.0, -1.0).astype(module.float32)
        signs[..., 3, -1] = 0.0
        return signs

    def _step(self, state, *, module=np):
        proj = self._project_fourier(state, module=module)
        refl = 2 * proj - state
        return (0.5 * (state + 2 * self._project_sign(refl, module=module) - refl)).astype(module.float32)

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        module = xp
        batch_size = self.batch_size if module.__name__ == "cupy" else 1
        rng = module.random.default_rng(seed)
        state = (seed_turyn_batch(self.N, batch_size, rng, module=module).astype(module.float32)
                 if self.sieve else rng.standard_normal((batch_size, 4, self.N)).astype(module.float32))
        state[:, 3, -1] = 0.0
        best = self._project_sign(state, module=module)
        best_energy = nonperiodic_batch_energy(
            best, lengths=self.LENGTHS, weights=self.W_F, module=module)
        for _ in range(steps):
            for _ in range(self.inner_steps):
                state = self._step(state, module=module)
            candidate = self._project_sign(
                self._project_fourier(state, module=module), module=module)
            energy = nonperiodic_batch_energy(
                candidate, lengths=self.LENGTHS, weights=self.W_F, module=module)
            improved = energy < best_energy
            best = module.where(improved[:, None, None], candidate, best)
            best_energy = module.minimum(best_energy, energy)
        best_np = best[int(module.argmin(best_energy).item())]
        best_np = best_np if isinstance(best_np, np.ndarray) else best_np.get()
        matrix = self.build(best_np)[0]
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
