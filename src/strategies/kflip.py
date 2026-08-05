"""K-Flip repair: GPU batch enumeration of all C(B,k) flips, k ramped 1..max_k."""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from correlations import (
    TURYN_WEIGHTS,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_residual,
)
from gpu import to_numpy, xp

from .base import Result, TurynStrategy


def _exact_energy(sequences, lengths):
    return float(
        nonperiodic_correlation_energy(
            nonperiodic_autocorrelation_state(
                sequences, lengths=lengths, weights=TURYN_WEIGHTS)
        )
    )


def _pair_scan_q(deltas_top, r0_cpu, Q):
    """CPU scan: find best pair energy using precomputed Q model. O(K²)."""
    K = Q.shape[0]
    best_e = float("inf")
    best_pair = None
    for i in range(K):
        di = deltas_top[i]
        for j in range(i + 1, K):
            e = float(np.sum((r0_cpu + di + deltas_top[j] + Q[i, j]) ** 2))
            if e < best_e:
                best_e = e
                best_pair = (i, j)  # indices into top_for_pairs
    if best_pair is None:
        return None
    return best_pair, best_e


class KFlipRepair(TurynStrategy):
    """GPU batch enumeration of all C(B,k) flips, k ramped 1..max_k.

    """

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        max_k: int = 5,
        top_k_for_large: int = 32,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.max_k = max_k
        self.top_k_for_large = top_k_for_large

    @property
    def name(self) -> str:
        return "kflip"

    # ── GPU primitives ──────────────────────────────────────────────────────

    def _singles_scan(self, seq_f, positions):
        """Return (deltas: (B, n-1) float64, r0: (n-1,) float64) on CPU."""
        B = len(positions)
        pos = np.array(positions, dtype=np.int32)
        _rows = xp.asarray(pos[:, 0])
        _cols = xp.asarray(pos[:, 1])
        batch = xp.repeat(seq_f[None, ...], B, axis=0)
        batch[xp.arange(B), _rows, _cols] *= -1
        residuals = npa_f_residual(
            batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        r0 = npa_f_residual(
            seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
        r0_cpu = to_numpy(r0)
        return to_numpy(residuals) - r0_cpu[None, :], r0_cpu

    def _build_q_model(self, seq_f, positions, top_indices, deltas_top):
        """Q[i,j] = residual(i,j) - r0 - delta_i - delta_j via GPU batch.

        top_indices: position indices into positions[].
        deltas_top: (K, n-1) deltas pre-indexed from full deltas array.
        Returns Q (K, K, n-1) float64 on CPU. Only upper triangle is valid.
        """
        K = len(top_indices)
        if K < 2:
            return np.zeros((K, K, self.N - 1), dtype=np.float64)

        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        n_pairs = len(pairs)
        pair_arr = np.array(pairs, dtype=np.int32)
        pos_arr = np.array(positions, dtype=np.int32)

        # Map pair entries to position coordinates via top_indices lookup
        ti = np.array(top_indices, dtype=np.int32)
        coords = pos_arr[ti]  # (K, 2): (r, c) for each top position
        flat0 = pair_arr[:, 0]  # indices into coords: 0..K-1
        flat1 = pair_arr[:, 1]

        batch = xp.repeat(seq_f[None, ...], n_pairs, axis=0)
        batch[xp.arange(n_pairs), xp.asarray(coords[flat0, 0]),
              xp.asarray(coords[flat0, 1])] *= -1
        batch[xp.arange(n_pairs), xp.asarray(coords[flat1, 0]),
              xp.asarray(coords[flat1, 1])] *= -1

        residuals = npa_f_residual(
            batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        r0 = npa_f_residual(
            seq_f[None, ...], lengths=self.LENGTHS, weights=TURYN_WEIGHTS)[0]
        residuals_cpu = to_numpy(residuals)
        r0_cpu = to_numpy(r0)

        Q = np.zeros((K, K, self.N - 1), dtype=np.float64)
        for idx, (pi, pj) in enumerate(pairs):
            q = residuals_cpu[idx] - r0_cpu - deltas_top[pi] - deltas_top[pj]
            Q[pi, pj] = q
            Q[pj, pi] = q
        return Q

    def _kflip_scan(self, seq_f, positions, k, pool):
        """Enumerate all C(|pool|,k) flips via vectorized GPU batch. k>=2."""
        n_combos = 1
        for i in range(k):
            n_combos = n_combos * (len(pool) - i) // (i + 1)
        if n_combos > 500_000:
            return None  # would be too large, caller should narrow pool

        combos = list(combinations(pool, k))
        if not combos:
            return None

        combo_arr = np.array(combos, dtype=np.int32)
        pos_arr = np.array(positions, dtype=np.int32)
        flat_t = combo_arr.ravel()
        n = len(combos)
        batch_idx = np.repeat(np.arange(n, dtype=np.int32), k)

        batch = xp.repeat(seq_f[None, ...], n, axis=0)
        batch[xp.asarray(batch_idx), xp.asarray(
            pos_arr[flat_t, 0]), xp.asarray(pos_arr[flat_t, 1])] *= -1
        residuals = npa_f_residual(
            batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
        energies = xp.sum(residuals**2, axis=1)
        best_idx = int(xp.argmin(energies))
        return combos[best_idx], float(energies[best_idx])

    def _twoply_scan(self, seq_f, positions, singles_energies, r0_cpu):
        """2-ply look-ahead: for each of top-N first flips, find best second flip.

        Tests all second flips per candidate first flip via batched GPU scan.
        Returns (combo, energy) for best improving 2-step sequence, or None.
        """
        B = len(positions)
        top_n = min(self.top_k_for_large, B)
        top_first = np.argsort(singles_energies)[:top_n]

        best_combo = None
        best_energy = float("inf")

        for idx1 in top_first:
            r1, c1 = positions[idx1]

            # Build batch: state after flip idx1, then flip each j
            batch = xp.repeat(seq_f[None, ...], B, axis=0)
            batch[:, r1, c1] *= -1  # first flip applied to all
            for j, (rj, cj) in enumerate(positions):
                batch[j, rj, cj] *= -1  # second flip: each position tested
            residuals = npa_f_residual(
                batch, lengths=self.LENGTHS, weights=TURYN_WEIGHTS)
            e2 = to_numpy(xp.sum(residuals**2, axis=1))

            best_j = int(np.argmin(e2))
            best_e2 = float(e2[best_j])

            if best_e2 < best_energy:
                best_energy = best_e2
                best_combo = (int(idx1), best_j)

        if best_combo is None:
            return None
        return best_combo, best_energy

    # ── Search ──────────────────────────────────────────────────────────────

    def _systematic_rescue(self, seqs, positions, singles_energies, best_e, rng):
        """2-bit then 3-bit systematic rescue among top-K singles candidates.

        Modifies seqs in-place. Caller must resync GPU state afterwards.
        """
        B = len(positions)
        K = min(B, max(16, B // 3))  # dynamic pool size
        top = np.argsort(singles_energies)[:K]
        improved = False
        best_local = best_e

        # 2-bit rescue: test C(K,2) pairs
        for i in range(K):
            ti = top[i]
            si, ii = positions[ti]
            orig_i = seqs[si, ii]
            seqs[si, ii] *= -1
            for j in range(i + 1, K):
                tj = top[j]
                sj, ij = positions[tj]
                orig_j = seqs[sj, ij]
                seqs[sj, ij] *= -1
                e = _exact_energy(seqs, self.LENGTHS)
                if e < best_local:
                    best_local = e
                    improved = True
                    break
                seqs[sj, ij] = orig_j
            if improved:
                break
            seqs[si, ii] = orig_i

        # 3-bit rescue: narrower pool
        if not improved and best_e > 0:
            K3 = min(K // 2, 12)
            for i in range(K3):
                ti = top[i]
                si, ii = positions[ti]
                orig_i = seqs[si, ii]
                seqs[si, ii] *= -1
                for j in range(i + 1, K3):
                    tj = top[j]
                    sj, ij = positions[tj]
                    orig_j = seqs[sj, ij]
                    seqs[sj, ij] *= -1
                    for k in range(j + 1, K3):
                        tk = top[k]
                        sk, ik = positions[tk]
                        orig_k = seqs[sk, ik]
                        seqs[sk, ik] *= -1
                        e = _exact_energy(seqs, self.LENGTHS)
                        if e < best_local:
                            best_local = e
                            improved = True
                            break
                        seqs[sk, ik] = orig_k
                    if improved:
                        break
                    seqs[sj, ij] = orig_j
                if improved:
                    break
                seqs[si, ii] = orig_i

        # Kick: flip one random bit per sequence to escape basin
        if not improved:
            n_pos = seqs.shape[1]
            for s in range(4):
                idx = rng.integers(0, min(int(self.LENGTHS[s]), n_pos))
                seqs[s, idx] *= -1

        return best_local

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = self.seed(rng)
        else:
            sequences = sequences.copy()
        positions = [(r, c) for r in range(4)
                     for c in range(int(self.LENGTHS[r]))]
        B = len(positions)
        seq_f = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)
        best_e = _exact_energy(sequences, self.LENGTHS)
        best_seq = sequences.copy()
        tabu: set[tuple[int, ...]] = set()
        k_moves = 0
        K3_MAX_FAILS = 3
        k3_fails = 0  # start unlocked; block k>=3 only after wasted passes

        step_iter = range(steps) if steps > 0 else range(0)
        for _step in step_iter:
            if best_e == 0:
                break
            moves_before = k_moves
            effective_max_k = self.max_k if k3_fails < K3_MAX_FAILS else 2

            # Cache singles deltas once per pass (only worth it when k>=3 uses top-k filter)
            need_singles_cache = self.max_k >= 3 and B >= 40 and effective_max_k > 2
            _empty = np.array([], dtype=np.float64)
            _empty_q = np.zeros((0, 0, self.N - 1), dtype=np.float64)
            if need_singles_cache:
                deltas, r0_cpu = self._singles_scan(seq_f, positions)
                singles_energies = np.sum((r0_cpu + deltas) ** 2, axis=1)
                # k=2: CPU pair-scan via Q model only when B justifies avoiding C(B,2) GPU batch
                use_q_for_pairs = self.top_k_for_large + 16 < B
                if use_q_for_pairs:
                    top_for_pairs = np.argsort(singles_energies)[
                        : min(B, max(self.top_k_for_large, 48))]
                    deltas_top = deltas[top_for_pairs]
                    q_model = self._build_q_model(
                        seq_f, positions, top_for_pairs, deltas_top)
                else:
                    top_for_pairs = np.array([], dtype=np.int64)
                    deltas_top = _empty
                    q_model = _empty_q
            else:
                deltas = _empty
                r0_cpu = _empty
                singles_energies = _empty
                use_q_for_pairs = False
                top_for_pairs = np.array([], dtype=np.int64)
                deltas_top = _empty
                q_model = _empty_q

            k = 1
            while k >= 1 and k <= effective_max_k and best_e > 0:
                if k == 1:
                    if need_singles_cache:
                        best_idx = int(np.argmin(singles_energies))
                        combo = (best_idx,)
                        gpu_energy = float(singles_energies[best_idx])
                    else:
                        result = self._kflip_scan(
                            seq_f, positions, 1, list(range(B)))
                        if result is None:
                            k += 1
                            continue
                        combo, gpu_energy = result
                elif k == 2:
                    if use_q_for_pairs:
                        result = _pair_scan_q(deltas_top, r0_cpu, q_model)
                        if result is None:
                            k += 1
                            continue
                        pair_local, gpu_energy = result
                        combo = (top_for_pairs[pair_local[0]],
                                 top_for_pairs[pair_local[1]])
                    elif need_singles_cache:
                        pool = list(np.argsort(singles_energies)[
                                    : min(B, self.top_k_for_large)])
                        result = self._kflip_scan(seq_f, positions, 2, pool)
                        if result is None:
                            k += 1
                            continue
                        combo, gpu_energy = result
                    else:
                        result = self._kflip_scan(
                            seq_f, positions, 2, list(range(B)))
                        if result is None:
                            k += 1
                            continue
                        combo, gpu_energy = result
                else:
                    use_filter = k >= 3 and need_singles_cache and B >= 40
                    pool = list(
                        np.argsort(singles_energies)[: self.top_k_for_large]
                        if use_filter
                        else range(B)
                    )
                    result = self._kflip_scan(seq_f, positions, k, pool)
                    if result is None:
                        k += 1
                        continue
                    combo, gpu_energy = result

                if gpu_energy >= best_e - 0.5:
                    # Singles stalled — try 2-ply look-ahead rescue
                    if k == 1 and need_singles_cache:
                        result = self._twoply_scan(
                            seq_f, positions, singles_energies, r0_cpu)
                        if result is not None:
                            combo, gpu_energy = result
                            if gpu_energy < best_e - 0.5:
                                pass  # fall through to acceptance below
                            else:
                                k += 1
                                continue
                        else:
                            k += 1
                            continue
                    else:
                        k += 1
                        continue

                # Apply candidate flip
                for t in combo:
                    r, c = positions[t]
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1
                cpu_e = _exact_energy(sequences, self.LENGTHS)
                key = tuple(sorted(combo))
                if cpu_e < best_e and key not in tabu:
                    tabu.add(key)
                    best_e = cpu_e
                    best_seq = sequences.copy()
                    k_moves += 1
                    k = 1
                    break  # state changed — re-scan singles next pass
                if cpu_e >= best_e:
                    tabu.add(key)
                for t in combo:
                    r, c = positions[t]
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1
                k += 1

            if k_moves == moves_before:
                # VND exhausted — try systematic rescue before giving up
                if need_singles_cache and best_e > 0:
                    rescue_e = self._systematic_rescue(
                        sequences, positions, singles_energies, best_e,
                        np.random.default_rng(seed + _step)
                    )
                    seq_f = xp.asarray(sequences.astype(
                        np.float32), dtype=xp.float32)
                    if rescue_e < best_e:
                        best_e = rescue_e
                        best_seq = sequences.copy()
                        k_moves += 1
                        k3_fails = 0
                        continue
                if effective_max_k == self.max_k:
                    k3_fails += 1
                else:
                    k3_fails = 0
                break
            k3_fails = 0  # any success resets

        matrix, metrics = self.build(best_seq)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_e={best_e:.0f} (k-moves={k_moves}) {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed, best_seq)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)
