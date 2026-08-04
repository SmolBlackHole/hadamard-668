"""Necessary spectral and sum constraints for Turyn-type sequence seeds."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def turyn_lengths(n: int) -> np.ndarray:
    """Return the padded TT(n) sequence lengths."""
    if n < 2:
        raise ValueError("Turyn type requires n >= 2")
    return np.array((n, n, n, n - 1), dtype=np.int64)


@lru_cache(maxsize=None)
def turyn_sum_patterns(n: int) -> np.ndarray:
    """Return signed TT(n) row-sum patterns satisfying the DC power constraint."""
    lengths = turyn_lengths(n)
    target = 6 * n - 2
    sums = [range(-int(length), int(length) + 1, 2) for length in lengths]
    patterns: list[tuple[int, int, int, int]] = []
    for sx in sums[0]:
        for sy in sums[1]:
            for sz in sums[2]:
                remainder = target - sx * sx - sy * sy - 2 * sz * sz
                for sw in sums[3]:
                    if remainder == 2 * sw * sw:
                        patterns.append((sx, sy, sz, sw))
    if not patterns:
        raise ValueError(f"TT({n}) has no compatible row-sum pattern")
    return np.asarray(patterns, dtype=np.int16)


def turyn_psd_mask(batch, *, n: int, module=np):
    """Return candidates obeying the necessary weighted TT(n) PSD caps."""
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
    rng,
    *,
    module=np,
    chunk_size: int | None = None,
):
    """Draw ``count`` padded TT(n) starts satisfying sum and PSD constraints."""
    if count < 1:
        raise ValueError("count must be positive")
    lengths = turyn_lengths(n)
    patterns = module.asarray(turyn_sum_patterns(n))
    size = min(65_536, max(4_096, count * 256))
    if chunk_size is not None:
        size = chunk_size
    survivors = []
    remaining = count
    for _ in range(100):
        selected = patterns[rng.integers(0, len(patterns), size=size)]
        plus = (module.asarray(lengths)[None, :] + selected) // 2
        keys = rng.random((size, 4, n))
        keys[:, 3, -1] = module.inf
        ranks = module.argsort(module.argsort(keys, axis=2), axis=2)
        batch = module.where(
            ranks < plus[:, :, None], 1, -1).astype(module.int8)
        batch[:, 3, -1] = 0
        valid = batch[turyn_psd_mask(batch, n=n, module=module)]
        if len(valid):
            survivors.append(valid[:remaining])
            remaining -= min(len(valid), remaining)
            if not remaining:
                return module.concatenate(survivors, axis=0)
    raise RuntimeError(f"PSD sieve could not seed {count} TT({n}) candidates")
