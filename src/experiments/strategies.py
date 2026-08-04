"""Experimental repair strategies — relaxation, phase retrieval, hybrid.

All strategies inherit ``TurynStrategy`` and are CLI-/pipeline-compatible.
Note: KFlipRepair has been promoted to ``strategies.kflip``.
"""

from __future__ import annotations

import time

import numpy as np

from correlations import (
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_gradient,
    npa_f_residual,
)
from gpu import to_numpy, xp
from strategies.base import Result, TurynStrategy
from strategies.kflip import KFlipRepair


def _exact_energy(sequences, lengths):
    return float(
        nonperiodic_correlation_energy(
            nonperiodic_autocorrelation_state(sequences, lengths=lengths, weights=TURYN_WEIGHTS)
        )
    )


# ── Continuous relaxation repair ─────────────────────────────────────────────


class RelaxRepair(TurynStrategy):
    """Continuous relaxation + L-BFGS + sign-round + tabu polish."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        lbfgs_iters: int = 2000,
        restarts: int = 5,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.lbfgs_iters = lbfgs_iters
        self.restarts = restarts
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "relax"

    def _energy_and_grad(self, v_flat):
        v = v_flat.reshape(4, self.N)
        # Energy
        corr = npa_f_residual(
            v.astype(np.float32)[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS
        )
        energy = float(xp.sum(corr**2))
        # Gradient on CPU
        grad = npa_f_gradient(v.astype(np.float64), lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        return energy, grad.ravel()

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        from scipy.optimize import minimize

        started = time.perf_counter()
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = self.seed(rng)
        v0 = sequences.astype(np.float64)

        best_x = None
        best_e = float("inf")
        bounds = [(-1.0, 1.0)] * (4 * self.N)

        for restart in range(self.restarts):
            if restart > 0:
                noise = np.random.default_rng(seed + restart + 1000).uniform(-0.3, 0.3, v0.shape)
                v0 = np.clip(v0 + noise, -1.0, 1.0)
            res = minimize(
                self._energy_and_grad,
                v0.ravel(),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.lbfgs_iters, "ftol": 1e-12},
            )
            x = np.sign(res.x.reshape(4, self.N))
            x[3, self.N - 1] = 0  # only W is padded
            e = _exact_energy(x.astype(np.int8), self.LENGTHS)
            if e < best_e:
                best_e = e
                best_x = x.copy()
            if best_e == 0:
                break

        # Polish with singles-only repair
        from experiments.syndrome_repair import RepairExperiment

        polished = np.sign(best_x).astype(np.int8)
        polished[3, self.N - 1] = 0
        rp = RepairExperiment(
            n=self.N, sieve=False, pair_top=0, triple_top=0, plateau_threshold=10**9, verbose=False
        )
        polish_result = rp.search(steps=200, seed=seed + 2000, sequences=polished)
        seq = polish_result.sequences if polish_result.sequences is not None else polished
        matrix, metrics = self.build(seq)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_e={metrics['energy']:.0f} (relax) {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed, seq)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)


# ── Phase retrieval repair ────────────────────────────────────────────────────


class PhaseRetrievalRepair(TurynStrategy):
    """Gerchberg-Saxton / Fienup-HIO phase retrieval on NPAF constraint."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        hio_iters: int = 300,
        beta: float = 0.9,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.hio_iters = hio_iters
        self.beta = beta
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "phase-ret"

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        from strategies.pocs import _project_fourier, _project_sign

        started = time.perf_counter()
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = self.seed(rng)
        state = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)
        best_x = sequences.copy()
        best_e = _exact_energy(sequences, self.LENGTHS)

        for it in range(self.hio_iters):
            state_b = state[None, ...]  # batch dim for projections
            proj_f = _project_fourier(state_b, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
            reflected = _project_sign(2 * proj_f[None, ...] - state_b)[0]
            # HIO support: where state conforms to sign constraint
            sign_proj = _project_sign(state_b)[0]
            in_support = state == sign_proj
            if isinstance(state, np.ndarray):
                new_state = reflected
                new_state[in_support] = ((1 - self.beta) * reflected + self.beta * state)[
                    in_support
                ]
            else:
                # CuPy path
                new_state = xp.where(
                    in_support, (1 - self.beta) * reflected + self.beta * state, reflected
                )
            state = new_state.astype(xp.float32)

            # Every 20 iters: check exact energy of rounded state
            if it % 20 == 19 or it == self.hio_iters - 1:
                x = np.sign(to_numpy(state)).astype(np.int8)
                x[3, self.N - 1] = 0  # only W is padded
                e = _exact_energy(x, self.LENGTHS)
                if e < best_e:
                    best_e = e
                    best_x = x.copy()
                if best_e == 0:
                    break

        # Polish
        from experiments.syndrome_repair import RepairExperiment

        polished = np.sign(best_x).astype(np.int8)
        polished[3, self.N - 1] = 0
        rp = RepairExperiment(
            n=self.N, sieve=False, pair_top=0, triple_top=0, plateau_threshold=10**9, verbose=False
        )
        polish_result = rp.search(steps=200, seed=seed + 3000, sequences=polished)
        matrix, metrics = self.build(polish_result.sequences)
        elapsed = time.perf_counter() - started
        seq = polish_result.sequences if polish_result.sequences is not None else polished
        print(f"  seed={seed} best_e={metrics['energy']:.0f} (phase-ret) {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed, seq)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)


# ── Hybrid repair: phase retrieval pre-relaxation + k-flip finish ────────────


class HybridRepair(TurynStrategy):
    """Phase retrieval pre-step (50 HIO iters) + K-Flip finish."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        max_k: int = 5,
        hio_iters: int = 50,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.max_k = max_k
        self.hio_iters = hio_iters
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = self.seed(rng)

        # Phase 1: Phase retrieval pre-relaxation
        pr = PhaseRetrievalRepair(n=self.N, hio_iters=self.hio_iters, beta=0.9, sieve=False)
        pr_result = pr.search(steps=0, seed=seed, sequences=sequences)
        relaxed = pr_result.sequences if pr_result.sequences is not None else sequences

        # Phase 2: K-Flip finish
        kf = KFlipRepair(n=self.N, max_k=self.max_k, sieve=False)
        kf_result = kf.search(steps=steps, seed=seed + 1, sequences=relaxed)

        elapsed = time.perf_counter() - started
        seq = kf_result.sequences
        print(f"  seed={seed} best_e={kf_result.metrics['energy']:.0f} (hybrid) {elapsed:.1f}s")
        return Result(kf_result.matrix, kf_result.metrics, elapsed, seq)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)
