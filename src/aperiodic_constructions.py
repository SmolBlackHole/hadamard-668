"""Initializers and exact embeddings for aperiodic sequence families."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Int8Array = npt.NDArray[np.int8]


def random_bs(
    m: int,
    n: int,
    rng: np.random.Generator,
    *,
    sum_filter: bool = True,
    batch_size: int = 16_384,
    max_batches: int = 1_024,
) -> tuple[Int8Array, ...]:
    """Generate a BS(m,n)-shaped state satisfying cheap exact constraints."""
    if m < n or n < 1:
        raise ValueError("BS requires m >= n >= 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    alternating_m = np.where(np.arange(m) % 2 == 0, 1, -1)
    alternating_n = np.where(np.arange(n) % 2 == 0, 1, -1)
    target = 2 * (m + n)
    for _ in range(max_batches):
        sequences = [
            rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(batch_size, length))
            for length in (m, m, n, n)
        ]
        for sequence in sequences:
            sequence *= sequence[:, :1]

        if m > n:
            sequences[1][:, -1] = -sequences[0][:, -1]
            endpoint_valid = np.ones(batch_size, dtype=np.bool_)
        else:
            endpoint_valid = sum(sequence[:, -1] for sequence in sequences) == 0
        if not sum_filter:
            valid = np.flatnonzero(endpoint_valid)
        else:
            plus = sum(np.sum(sequence, axis=1, dtype=np.int64) ** 2 for sequence in sequences)
            minus = (
                (sequences[0] @ alternating_m) ** 2
                + (sequences[1] @ alternating_m) ** 2
                + (sequences[2] @ alternating_n) ** 2
                + (sequences[3] @ alternating_n) ** 2
            )
            valid = np.flatnonzero(endpoint_valid & (plus == target) & (minus == target))
        if valid.size:
            index = int(valid[0])
            return tuple(sequence[index].copy() for sequence in sequences)
    raise RuntimeError(f"no BS seed passed the sum filter in {max_batches} batches")


def random_tt(
    n: int,
    rng: np.random.Generator,
    *,
    sum_filter: bool = True,
    batch_size: int = 16_384,
    max_batches: int = 1_024,
) -> tuple[Int8Array, ...]:
    """Generate a TT-shaped state satisfying cheap exact constraints."""
    if n < 2 or n % 2:
        raise ValueError("TT requires an even n >= 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")

    for _ in range(max_batches):
        sequences = [
            rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(batch_size, length))
            for length in (n, n, n, n - 1)
        ]
        for sequence in sequences:
            sequence *= sequence[:, :1]

        endpoint_product = sequences[0][:, -1].copy()
        sequences[1][:, -1] = endpoint_product
        sequences[2][:, -1] = -endpoint_product
        if not sum_filter:
            return tuple(sequence[0].copy() for sequence in sequences)

        alternating_n = np.where(np.arange(n) % 2 == 0, 1, -1)
        alternating_d = alternating_n[:-1]
        plus = (
            np.sum(sequences[0], axis=1, dtype=np.int64) ** 2
            + np.sum(sequences[1], axis=1, dtype=np.int64) ** 2
            + 2 * np.sum(sequences[2], axis=1, dtype=np.int64) ** 2
            + 2 * np.sum(sequences[3], axis=1, dtype=np.int64) ** 2
        )
        minus = (
            (sequences[0] @ alternating_n) ** 2
            + (sequences[1] @ alternating_n) ** 2
            + 2 * (sequences[2] @ alternating_n) ** 2
            + 2 * (sequences[3] @ alternating_d) ** 2
        )
        valid = np.flatnonzero((plus == 6 * n - 2) & (minus == 6 * n - 2))
        if valid.size:
            index = int(valid[0])
            return tuple(sequence[index].copy() for sequence in sequences)
    raise RuntimeError(f"no TT seed passed the sum filter in {max_batches} batches")


def tt_to_base(sequences: Sequence[npt.ArrayLike]) -> tuple[Int8Array, ...]:
    """Apply the classical TT(n) to BS(2n-1,n) concatenation."""
    a, b, c, d = _tt_arrays(sequences)
    return (
        np.concatenate((c, d)),
        np.concatenate((c, -d)),
        a.copy(),
        b.copy(),
    )


def base_to_gs4(sequences: Sequence[npt.ArrayLike]) -> Int8Array:
    """Embed BS(m,n) parameters into four binary GS4 rows of length m+n."""
    if len(sequences) != 4:
        raise ValueError("base embedding requires four sequences")
    a, b, c, d = (np.asarray(sequence, dtype=np.int8) for sequence in sequences)
    if a.ndim != 1 or b.shape != a.shape or c.ndim != 1 or d.shape != c.shape:
        raise ValueError("base sequences must have pairwise lengths (m,m,n,n)")
    if any(np.any(np.abs(sequence) != 1) for sequence in (a, b, c, d)):
        raise ValueError("base sequences must contain only -1 and +1")

    m = a.size
    n = c.size
    zero_m = np.zeros(m, dtype=np.int8)
    zero_n = np.zeros(n, dtype=np.int8)
    ternary = np.stack(
        (
            np.concatenate(((a + b) // 2, zero_n)),
            np.concatenate(((a - b) // 2, zero_n)),
            np.concatenate((zero_m, (c + d) // 2)),
            np.concatenate((zero_m, (c - d) // 2)),
        )
    )
    hadamard_4 = np.asarray(
        ((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)),
        dtype=np.int8,
    )
    result = np.asarray(hadamard_4 @ ternary, dtype=np.int8)
    if np.any(np.abs(result) != 1):
        raise ValueError("base embedding did not produce binary GS4 rows")
    return result


def tt_to_gs4(sequences: Sequence[npt.ArrayLike]) -> Int8Array:
    """Embed a TT-shaped state into its exact GS4 representation."""
    return base_to_gs4(tt_to_base(sequences))


def _tt_arrays(sequences: Sequence[npt.ArrayLike]) -> tuple[Int8Array, ...]:
    if len(sequences) != 4:
        raise ValueError("TT requires four sequences")
    arrays = tuple(np.asarray(sequence, dtype=np.int8) for sequence in sequences)
    n = arrays[0].size
    if n < 2 or tuple(array.size for array in arrays) != (n, n, n, n - 1):
        raise ValueError("TT sequence lengths must be (n,n,n,n-1)")
    if any(array.ndim != 1 or np.any(np.abs(array) != 1) for array in arrays):
        raise ValueError("TT sequences must be one-dimensional and binary")
    return arrays
