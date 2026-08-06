"""GS4 energy tracker via negaperiodic autocorrelation residuals.

For four negacyclic sequences of length ``n``, let ``r_t`` be the sum of
their negaperiodic autocorrelations.  Antisymmetry gives ``r_{n-t} = -r_t``
and every residual is divisible by four.  The tracker therefore stores only
``u_t = r_t / 4`` for ``t = 1..floor(n / 2)``:

    E = 64 n * sum(u_t**2)

The delta cache has the same reduced coordinates.  Its construction, all
single-flip scores, and cache updates are NumPy array operations.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt


class Tracker:
    """Tracks GS4 orthogonality energy with reduced NAF residuals."""

    def __init__(self) -> None:
        self._n = 0
        self._seqs: npt.NDArray[np.int8] | None = None
        self._u: npt.NDArray[np.int32] | None = None
        self._q = 0
        self._e = 0
        self._delta: npt.NDArray[np.int8] | None = None
        self._norm2: npt.NDArray[np.int32] | None = None
        self._update_cols: npt.NDArray[np.intp] | None = None
        self._update_lags: npt.NDArray[np.intp] | None = None
        self._update_signs: npt.NDArray[np.int8] | None = None

    def build(self, seqs: npt.NDArray[np.int8]) -> None:
        self._n = seqs.shape[1]
        self._seqs = seqs.copy()
        m = self._n // 2
        residual = self._compute_residual(self._seqs)
        self._u = (residual[1 : m + 1] // 4).astype(np.int32)
        self._q = int(np.dot(self._u, self._u))
        self._e = 64 * self._n * self._q
        self._delta = _build_delta_cache(self._seqs)
        self._norm2 = np.asarray(
            np.sum(self._delta * self._delta, axis=1, dtype=np.int32), dtype=np.int32
        )
        self._update_cols, self._update_lags, self._update_signs = _update_geometry(self._n)

    def energy(self) -> int:
        return self._e

    def flip(self, s: int, c: int) -> int:
        """Return the energy after one flip without changing the tracker."""
        return int(self.flip_batch(s * self._n + c, s * self._n + c + 1)[0])

    def flip_batch(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Return energies for a contiguous range of single flips."""
        assert self._u is not None and self._delta is not None and self._norm2 is not None
        d = self._delta[start:stop]
        delta_q = 2 * (d @ self._u) + self._norm2[start:stop]
        return (64 * self._n * (self._q + delta_q.astype(np.int64))).astype(np.int64)

    def flip_energies(self) -> npt.NDArray[np.int64]:
        """Return energies for all 4n single flips in solver scan order."""
        return self.flip_batch(0, 4 * self._n)

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        """Apply one flip and update the exact delta cache in O(n)."""
        assert (
            self._seqs is not None
            and self._u is not None
            and self._delta is not None
            and self._norm2 is not None
            and self._update_cols is not None
            and self._update_lags is not None
            and self._update_signs is not None
        )
        index = s * self._n + c
        d = self._delta[index]
        self._q += self._single_delta_q(d, int(self._norm2[index]))
        self._u += d

        a = self._seqs[s]
        value = int(a[c])
        _update_delta_slice(
            self._delta,
            self._norm2,
            a,
            s,
            c,
            value,
            self._update_cols[c],
            self._update_lags[c],
            self._update_signs[c],
        )
        a[c] *= -1
        seqs[s, c] *= -1

        self._e = 64 * self._n * self._q
        return self._e

    def pair_energies(self, candidates: list[tuple[int, int]]) -> npt.NDArray[np.int64]:
        """Return all pair energies in the candidates' combination order."""
        assert (
            self._seqs is not None
            and self._u is not None
            and self._delta is not None
            and self._norm2 is not None
        )
        positions = np.asarray(candidates, dtype=np.intp)
        seq_ids, cols = positions[:, 0], positions[:, 1]
        indices = seq_ids * self._n + cols
        d = self._delta[indices]

        single = 2 * (d @ self._u) + self._norm2[indices]
        change = (
            single[:, None] + single[None, :] + 2 * np.einsum("ik,jk->ij", d, d, dtype=np.int32)
        )

        distance = np.abs(cols[:, None] - cols[None, :])
        lag = np.minimum(distance, self._n - distance)
        valid = (seq_ids[:, None] == seq_ids[None, :]) & (lag > 0)
        valid &= distance != self._n - distance
        values = self._seqs[seq_ids, cols]
        correction = np.where(distance < self._n - distance, 1, -1)
        correction *= values[:, None] * values[None, :]

        padded_u = np.concatenate((np.zeros(1, dtype=np.int32), self._u))
        padded_d = np.pad(d, ((0, 0), (1, 0)))
        base = (
            padded_u[lag]
            + padded_d[np.arange(len(candidates))[:, None], lag]
            + padded_d[np.arange(len(candidates))[None, :], lag]
        )
        change += valid * (2 * base * correction + correction * correction)
        return (64 * self._n * (self._q + change.astype(np.int64))).astype(np.int64)

    def combo_energy(self, flips: list[tuple[int, int]]) -> int:
        """Return the exact energy after any set of bit flips.

        NAF is quadratic in one sequence: singleton deltas plus one pair
        correction for every same-sequence pair are complete, even for three
        or more flips.
        """
        assert self._seqs is not None and self._delta is not None
        if not flips:
            return self._e

        raw = np.asarray(flips, dtype=np.intp)
        flat = raw[:, 0] * self._n + raw[:, 1]
        unique, counts = np.unique(flat, return_counts=True)
        indices = unique[counts % 2 == 1]
        if indices.size == 0:
            return self._e

        seq_ids, cols = np.divmod(indices, self._n)
        d = np.asarray(np.sum(self._delta[indices], axis=0, dtype=np.int32), dtype=np.int32)
        if indices.size > 1:
            distance = np.abs(cols[:, None] - cols[None, :])
            lag = np.minimum(distance, self._n - distance)
            same_sequence = seq_ids[:, None] == seq_ids[None, :]
            upper = np.triu(np.ones((indices.size, indices.size), dtype=bool), 1)
            valid = same_sequence & upper & (lag > 0) & (distance != self._n - distance)
            if np.any(valid):
                values = self._seqs[seq_ids, cols]
                correction = np.where(distance < self._n - distance, 1, -1)
                correction = correction * values[:, None] * values[None, :]
                np.add.at(d, lag[valid] - 1, correction[valid])
        return self._energy_after_delta(d)

    def _single_delta_q(self, d: npt.NDArray[np.int8], norm2: int) -> int:
        assert self._u is not None
        return int(2 * np.dot(self._u, d) + norm2)

    def _energy_after_delta(self, d: npt.NDArray[np.int32]) -> int:
        assert self._u is not None
        q = self._q + int(2 * np.dot(self._u, d) + np.dot(d, d))
        return 64 * self._n * q

    @classmethod
    def _compute_residual(cls, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
        n = seqs.shape[1]
        j = np.arange(n)[:, None]
        t = np.arange(n)[None, :]
        shifted = (j + t) % n
        sign = np.where(j + t < n, 1, -1).astype(np.int32)
        values = seqs.astype(np.int32)
        return np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))


def _build_delta_cache(seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """Build every reduced singleton delta without Python loops."""
    n = seqs.shape[1]
    m = n // 2
    c = np.arange(n)[:, None]
    t = np.arange(1, m + 1)[None, :]
    forward = (c + t) % n
    backward = (c - t) % n
    forward_sign = np.where(c + t < n, 1, -1)
    backward_sign = np.where(backward + t < n, 1, -1)
    values = seqs.astype(np.int16)
    delta = (
        -2
        * values[:, :, None]
        * (forward_sign * values[:, forward] + backward_sign * values[:, backward])
    )
    return (delta // 4).reshape(4 * n, m).astype(np.int8)


@cache
def _update_geometry(
    n: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.int8]]:
    """Return the affected cache coordinates for each flipped column."""
    centers = np.arange(n)[:, None]
    columns = np.broadcast_to(np.arange(n), (n, n))
    distance = np.abs(columns - centers)
    lag = np.minimum(distance, n - distance)
    valid = (lag > 0) & (distance != n - distance)
    width = int(np.sum(valid[0]))
    order = np.argsort(~valid, axis=1, kind="stable")[:, :width]
    selected_columns = np.take_along_axis(columns, order, axis=1).astype(np.intp)
    selected_lags = np.take_along_axis(lag, order, axis=1).astype(np.intp) - 1
    signs = np.where(distance < n - distance, 1, -1)
    selected_signs = np.take_along_axis(signs, order, axis=1).astype(np.int8)
    return selected_columns, selected_lags, selected_signs


def _update_delta_slice(
    cache: npt.NDArray[np.int8],
    norm2: npt.NDArray[np.int32],
    a: npt.NDArray[np.int8],
    s: int,
    c: int,
    value: int,
    columns: npt.NDArray[np.intp],
    lags: npt.NDArray[np.intp],
    signs: npt.NDArray[np.int8],
) -> None:
    """Update one sequence's delta rows after flipping column ``c``."""
    offset = s * len(a)
    rows = offset + columns
    old = cache[rows, lags]
    correction = signs * value * a[columns]
    new = old.astype(np.int16) + correction.astype(np.int16)
    norm2[rows] = norm2[rows] + new * new - old * old
    cache[rows, lags] = new.astype(np.int8)
    cache[offset + c] *= -1
