"""Seed generation with sum constraint + PSD filtering via Douglas-Rachford."""

from __future__ import annotations

from functools import cache

import numpy as np

from gpu import to_numpy, xp


def turyn_lengths(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("Turyn type requires n >= 2")
    return np.array((n, n, n, n - 1), dtype=np.int64)


@cache
def turyn_sum_patterns(n: int) -> np.ndarray:
    """Return signed TT(n) row-sum patterns satisfying the DC power constraint."""
    lengths = turyn_lengths(n)
    target = 6 * n - 2
    sums = [range(-int(L), int(L) + 1, 2) for L in lengths]
    sw_squares: dict[int, int] = {}
    for sw in sums[3]:
        val = 2 * sw * sw
        if val not in sw_squares:
            sw_squares[val] = sw
    patterns: list[tuple[int, int, int, int]] = []
    for sx in sums[0]:
        for sy in sums[1]:
            for sz in sums[2]:
                remainder = target - sx * sx - sy * sy - 2 * sz * sz
                if remainder in sw_squares:
                    sw = sw_squares[remainder]
                    patterns.append((sx, sy, sz, sw))
                    if sw != -sw:
                        patterns.append((sx, sy, sz, -sw))
    if not patterns:
        raise ValueError(f"TT({n}) has no compatible row-sum pattern")
    return np.asarray(patterns, dtype=np.int16)


def _sum_seeds(n: int, count: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Batch of sum-constrained seeds via argsort trick. Returns (count, 4, n) float32."""
    patterns = turyn_sum_patterns(n)
    lengths = turyn_lengths(n)
    size = min(count, 32_768)
    survivors: list[np.ndarray] = []
    remaining = count
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(20):
        indices = to_numpy(rng.integers(0, len(patterns), size=size))
        selected = patterns[indices]
        plus = (np.asarray(lengths)[None, :] + selected) // 2
        keys = to_numpy(rng.random((size, 4, n), dtype=np.float32))
        keys[:, 3, -1] = np.inf
        ranks = np.argsort(keys, axis=2).argsort(axis=2)
        batch = np.where(ranks < plus[:, :, None], 1, -1).astype(np.float32)
        batch[:, 3, -1] = 0.0
        survivors.append(batch[:remaining])
        remaining -= min(size, remaining)
        if not remaining:
            return np.concatenate(survivors, axis=0)
    raise RuntimeError("sum seed generation failed")


def _dr_project(
    batch: np.ndarray, lengths: np.ndarray, weights: np.ndarray, n_iter: int = 10
) -> np.ndarray:
    """Push a batch through DR towards the PSD power shell. Returns int8 array."""
    from strategies.pocs import _project_fourier, _project_sign

    state = xp.asarray(batch, dtype=xp.float32)
    for _ in range(n_iter):
        proj = _project_fourier(state, lengths=lengths, weights=weights)
        refl = 2 * proj - state
        state = (state + _project_sign(refl) - proj).astype(xp.float32)
    result = to_numpy(state)
    result[:, 3, -1] = 0
    dr_batch = np.where(result >= 0, 1, -1).astype(np.int8)
    dr_batch[:, 3, -1] = 0
    return dr_batch


def turyn_psd_mask(batch: np.ndarray, *, n: int, module=np) -> np.ndarray:
    """Return boolean mask: candidates obeying the weighted TT(n) PSD caps."""
    values = module.asarray(batch)
    if values.ndim != 3 or values.shape[1:] != (4, n):
        raise ValueError("batch must have shape (batch, 4, n)")
    if bool(module.any(values[:, 3, -1] != 0)):
        raise ValueError("the padded final W entry must be zero")
    target = 6 * n - 2
    spectrum = module.fft.fft(values, n=2 * n - 1, axis=-1)
    peak = module.max(module.abs(spectrum) ** 2, axis=-1)
    caps = module.asarray((target, target, target // 2, target // 2))
    return module.all(peak <= caps[None, :], axis=1)


def seed_turyn_batch(
    n: int,
    count: int,
    rng: np.random.Generator,
    *,
    module=np,
) -> np.ndarray:
    """Return ``count`` PSD-valid TT(n) seeds via DR projection + PSD filter."""
    if count < 1:
        raise ValueError("count must be positive")

    lengths = turyn_lengths(n)
    weights = np.array((1, 1, 2, 2), dtype=np.float64)

    # Oversample: DR yield drops with n (76% at n=8, 19% at n=36, 13% at n=56)
    overprovision = 30 if n >= 56 else 15 if n >= 36 else 2
    dr_iters = 20 if n >= 56 else 10 if n >= 36 else 5
    batch_size = min(count * overprovision, 32_768)
    survivors: list[np.ndarray] = []
    remaining = count
    for _ in range(50):
        batch = _sum_seeds(n, batch_size, rng)
        dr_batch = _dr_project(batch, lengths=lengths,
                               weights=weights, n_iter=dr_iters)
        mask = turyn_psd_mask(dr_batch, n=n)
        valid = dr_batch[mask]
        if len(valid):
            survivors.append(valid[:remaining])
            remaining -= min(len(valid), remaining)
            if not remaining:
                result = np.concatenate(survivors, axis=0)
                return module.asarray(result, dtype=module.int8) if module != np else result
    raise RuntimeError(
        f"DR sieve could not produce {count} TT({n}) candidates")
