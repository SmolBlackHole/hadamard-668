"""Energy functions for Hadamard search — all in one place, no strategy deps."""

from __future__ import annotations

import numpy as np

from correlations import (
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_residual,
)
from gpu import xp

# ── NPAF energies ────────────────────────────────────────────────────────────


def npaf_energy_f32(seqs: np.ndarray, lengths: np.ndarray, weights: np.ndarray) -> float:
    """NPAF energy via FFT on GPU. Fast, float32. For singles-scan batches."""
    r = npa_f_residual(seqs, lengths=lengths, weights=weights)
    return float(xp.sum(r**2))


def npaf_energy_exact(seqs: np.ndarray, lengths: np.ndarray, weights: np.ndarray) -> int:
    """NPAF energy via exact integer arithmetic (numba JIT)."""
    corr = nonperiodic_autocorrelation_state(seqs, lengths=lengths, weights=weights)
    return int(nonperiodic_correlation_energy(corr))


# ── Group-GS4 energy ──────────────────────────────────────────────────────────


def group_gs4_energy(seqs: np.ndarray, dims: list[int]) -> int:
    """Gram off-diagonal energy for group-based GS4. Lower = better."""
    from builders import build_group_gs4

    M = build_group_gs4(seqs, dims)
    G = M.astype(np.int32) @ M.astype(np.int32).T
    np.fill_diagonal(G, 0)
    return int((G.astype(np.int64) ** 2).sum())


# ── Matrix energy ─────────────────────────────────────────────────────────────


def gram_energy(matrix: np.ndarray) -> int:
    """Squared sum of off-diagonal Gram entries. 0 = orthogonal."""
    G = matrix.astype(np.int32) @ matrix.astype(np.int32).T
    np.fill_diagonal(G, 0)
    return int((G.astype(np.int64) ** 2).sum())


# ── Convenience ───────────────────────────────────────────────────────────────


def turyn_energy(seqs: np.ndarray) -> int:
    """TT(n) NPAF energy with standard Turyn weights."""
    lengths = np.array((seqs.shape[1],) * 3 + (seqs.shape[1] - 1,), dtype=np.int64)
    return npaf_energy_exact(seqs, lengths, TURYN_WEIGHTS)
