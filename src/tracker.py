"""GS4 energy tracker via negaperiodic autocorrelation residuals.

For four negacyclic sequences of length ``n``, let ``r_t`` be the sum of
their negaperiodic autocorrelations.  Antisymmetry gives ``r_{n-t} = -r_t``
and every residual is divisible by four.  The tracker therefore stores only
``u_t = r_t / 4`` for ``t = 1..floor((n - 1) / 2)``:

    E = 64 n * sum(u_t**2)

The delta cache has the same reduced coordinates.  Its construction, all
single-flip scores, and cache updates are NumPy array operations.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt


class Tracker:
    """Track exact GS4 energy and single-flip deltas for one mutable state.

    The tracker owns a copy of the current ``(4, n)`` sequence state. Calls to
    :meth:`accept` keep that copy, the caller's state, and all residual caches
    synchronized.
    """

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
        """Initialize all exact caches from four binary sequences.

        Args:
            seqs: Current state with shape ``(4, n)`` and entries ``+1`` or
                ``-1``. The tracker copies this array.

        Note:
            The reduced objective is ``Q = ||u||^2``; :meth:`energy` exposes
            the equivalent matrix energy ``E = 64nQ``. Building the tracker is
            not charged to the solver's candidate budget. Inputs must satisfy
            the documented shape and alphabet; verification happens at the
            pipeline acceptance boundary.
        """
        self._n = seqs.shape[1]
        self._seqs = seqs.copy()
        m = (self._n - 1) // 2
        residual = self._compute_residual(self._seqs)
        self._u = (residual[1 : m + 1] // 4).astype(np.int32)
        self._q = int(np.dot(self._u, self._u))
        self._e = 64 * self._n * self._q
        self._delta = _build_delta_cache(self._seqs)
        self._norm2 = np.asarray(
            np.sum(self._delta * self._delta, axis=1, dtype=np.int32), dtype=np.int32
        )
        self._update_cols, self._update_lags, self._update_signs = _update_geometry(self._n)

    def _adopt(
        self,
        seqs: npt.NDArray[np.int8],
        u: npt.NDArray[np.int32],
        q: int,
        delta: npt.NDArray[np.int8],
        norm2: npt.NDArray[np.int32],
    ) -> None:
        """Adopt an exact cache snapshot without copying its arrays.

        Note:
            Ownership of all array arguments transfers to this tracker. The
            caller must not mutate them after adoption.
        """
        self._seqs = seqs
        self._u = u
        self._q = q
        self._e = 64 * self._n * q
        self._delta = delta
        self._norm2 = norm2

    def energy(self) -> int:
        """Return the exact matrix energy ``E = 64nQ`` of the current state."""
        return self._e

    def flip(self, s: int, c: int) -> int:
        """Return the exact energy after one hypothetical bit flip.

        Args:
            s: Sequence index in ``0..3``.
            c: Column index in ``0..n-1``.

        Returns:
            The matrix energy ``E = 64nQ`` after the flip.

        Note:
            Neither the tracker nor its source state is modified.
        """
        return int(self.flip_batch(s * self._n + c, s * self._n + c + 1)[0])

    def flip_batch(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        """Evaluate a contiguous range of hypothetical single flips.

        Args:
            start: Inclusive flat index in sequence-major order.
            stop: Exclusive flat index, at most ``4n``.

        Returns:
            Exact matrix energies for ``stop - start`` flips.

        Note:
            The tracker remains unchanged. The solver charges one candidate
            evaluation for each returned entry.
        """
        return 64 * self._n * self._flip_q_batch(start, stop)

    def flip_qs(self) -> npt.NDArray[np.int64]:
        """Return exact reduced ``Q`` values after all ``4n`` single flips."""
        return self._flip_q_batch(0, 4 * self._n)

    def _flip_q_batch(self, start: int, stop: int) -> npt.NDArray[np.int64]:
        assert self._u is not None and self._delta is not None and self._norm2 is not None
        d = self._delta[start:stop]
        delta_q = 2 * (d @ self._u) + self._norm2[start:stop]
        return self._q + delta_q.astype(np.int64)

    def flip_energies(self) -> npt.NDArray[np.int64]:
        """Return exact energies for all ``4n`` flips in sequence-major order."""
        return self.flip_batch(0, 4 * self._n)

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        """Accept one bit flip and update every exact cache in ``O(n)``.

        Args:
            seqs: Caller-owned current state with shape ``(4, n)``. It must
                equal the state used by the preceding :meth:`build` and is
                mutated in place.
            s: Sequence index in ``0..3``.
            c: Column index in ``0..n-1``.

        Returns:
            The exact matrix energy after the accepted flip.

        Note:
            Both ``seqs`` and the tracker's private state change. Passing a
            stale or different array breaks the synchronization contract.
        """
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

    def combo_energy(self, flips: list[tuple[int, int]]) -> int:
        """Return the exact energy after any set of bit flips.

        Args:
            flips: Sequence and column pairs. Repeated pairs cancel modulo two.

        Returns:
            The matrix energy after applying the normalized flip set.

        Note:
            The tracker is not modified. NAF is quadratic in one sequence, so
            singleton deltas plus one correction for every same-sequence pair
            are exact even for three or more flips.
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
        """Compute all combined negaperiodic autocorrelation residuals."""
        n = seqs.shape[1]
        j: npt.NDArray[np.intp] = np.arange(n, dtype=np.intp)[:, None]
        t: npt.NDArray[np.intp] = np.arange(n, dtype=np.intp)[None, :]
        shifted = (j + t) % n
        sign = np.where(j + t < n, 1, -1).astype(np.int32)
        values = seqs.astype(np.int32)
        return np.sum(values[:, :, None] * values[:, shifted] * sign, axis=(0, 1))


def _build_delta_cache(seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """Build all ``4n`` reduced singleton deltas without Python loops."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    c: npt.NDArray[np.intp] = np.arange(n, dtype=np.intp)[:, None]
    t: npt.NDArray[np.intp] = np.arange(1, m + 1, dtype=np.intp)[None, :]
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
    """Return cached delta coordinates and signs affected by each column."""
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
    """Update one sequence's delta rows before flipping column ``c``."""
    offset = s * len(a)
    rows = offset + columns
    old = cache[rows, lags]
    correction = signs * value * a[columns]
    new = old + correction
    norm2[rows] = norm2[rows] + new * new - old * old
    cache[rows, lags] = new
    cache[offset + c] *= -1
