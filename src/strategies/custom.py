"""Custom Hadamard solver — end-to-end from sequences to matrix.

Usage::

    from strategies.custom import CustomSolver

    # Single Williamson
    cs = CustomSolver.williamson(n=11, max_k=4)
    result = cs.search(steps=500, seed=42)
    # result.matrix is a 44x44 Hadamard matrix

    # Tensor product of two Williamsons
    cs = CustomSolver.tensor(n1=11, n2=13, max_k=4)
    result = cs.search(steps=500, seed=42)
    # result.matrix is a 2288x2288 Hadamard matrix

    # Custom configuration (any N_SEQS, any lengths, any weights)
    cs = CustomSolver(seq_lengths=(8,8,8,8), weights=(1,1,1,1), max_k=4)
"""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np

from gpu import to_numpy, xp

from .base import Result, SearchStrategy


def _exact_npaf_energy(sequences, lengths, weights):
    from correlations import (
        nonperiodic_autocorrelation_state,
        nonperiodic_correlation_energy,
    )

    return float(
        nonperiodic_correlation_energy(
            nonperiodic_autocorrelation_state(
                sequences, lengths=lengths, weights=weights)
        )
    )


def _build_williamson(seqs, n):
    """Build valid Hadamard via Goethals-Seidel. Raises if NPAF != 0."""
    from builders import build_goethals_seidel
    from gpu import check_orthogonality

    A = np.array(seqs[0, :n], dtype=np.int8)
    B = np.array(seqs[1, :n], dtype=np.int8)
    C = np.array(seqs[2, :n], dtype=np.int8)
    D = np.array(seqs[3, :n], dtype=np.int8)
    H = build_goethals_seidel(A, B, C, D)
    return H, check_orthogonality(H)


_DIFF_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def _get_diff_table(dims):
    key = tuple(dims)
    if key not in _DIFF_CACHE:
        n = int(np.prod(dims))
        diff_table = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            rem = i
            gi = []
            for d in reversed(dims):
                gi.insert(0, rem % d)
                rem //= d
            for j in range(n):
                rem = j
                gj = []
                for d in reversed(dims):
                    gj.insert(0, rem % d)
                    rem //= d
                diff = [(b - a) % d for a, b, d in zip(gi, gj, dims)]
                idx, stride = 0, 1
                for dv, ds in zip(reversed(diff), reversed(dims)):
                    idx += dv * stride
                    stride *= ds
                diff_table[i, j] = idx
        _DIFF_CACHE[key] = diff_table
    return _DIFF_CACHE[key]


def _build_group_gs4(seqs, dims):
    """Build GS4 matrix with group-based circulants. seqs: (4, total) int8."""
    n = int(np.prod(dims))
    diff_table = _get_diff_table(dims)

    def circ(seq_flat):
        return seq_flat[diff_table].astype(np.int8)

    A = circ(seqs[0, :n])
    B = circ(seqs[1, :n])
    C = circ(seqs[2, :n])
    D = circ(seqs[3, :n])
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BRt, CRt, DRt = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DRt, CRt],
            [-CR, DRt, A, -BRt],
            [-DR, -CRt, BRt, A],
        ]
    ).astype(np.int8)  # skip orthogonality check in hot path


def _group_gs4_energy(seqs, dims):
    """Gram off-diagonal energy for group-based GS4. Lower = better."""
    M = _build_group_gs4(seqs, dims)
    G = M.astype(np.int32) @ M.astype(np.int32).T
    np.fill_diagonal(G, 0)
    return int((G.astype(np.int64) ** 2).sum())


def _kflip_group_search(seqs, dims, steps, rng):
    """KFlip for group-based circulant GS4. Singles + systematic 2-bit rescue."""
    n = int(np.prod(dims))
    B = 4 * n
    best_e = _group_gs4_energy(seqs, dims)
    best_seq = seqs.copy()

    for _step in range(steps):
        if best_e == 0:
            break

        # Full singles scan — track all energies for pair selection
        singles_e = np.full(B, np.inf)
        improved = False
        idxs = rng.permutation(B)
        for idx in idxs:
            s, i = divmod(int(idx), n)
            seqs[s, i] *= -1
            e = _group_gs4_energy(seqs, dims)
            singles_e[idx] = e
            if e < best_e:
                best_e = e
                best_seq = seqs.copy()
                improved = True
                break
            seqs[s, i] *= -1

        # Systematic 2-bit rescue: test pairs among top-K closest singles
        top = np.argsort(singles_e)[: min(B, 24)]
        K = len(top)
        if not improved:
            for i in range(K):
                ti = top[i]
                si, ii = divmod(int(ti), n)
                seqs[si, ii] *= -1
                for j in range(i + 1, K):
                    tj = top[j]
                    sj, ij = divmod(int(tj), n)
                    seqs[sj, ij] *= -1
                    e = _group_gs4_energy(seqs, dims)
                    if e < best_e:
                        best_e = e
                        best_seq = seqs.copy()
                        improved = True
                        break
                    seqs[sj, ij] *= -1
                if improved:
                    break
                seqs[si, ii] *= -1

        # 3-bit rescue when 2-bit fails (narrower pool for speed)
        if not improved and best_e > 0:
            K3 = min(len(top) // 2, 14)
            for i in range(K3):
                ti = top[i]
                si, ii = divmod(int(ti), n)
                seqs[si, ii] *= -1
                for j in range(i + 1, K3):
                    tj = top[j]
                    sj, ij = divmod(int(tj), n)
                    seqs[sj, ij] *= -1
                    for k in range(j + 1, K3):
                        tk = top[k]
                        sk, ik = divmod(int(tk), n)
                        seqs[sk, ik] *= -1
                        e = _group_gs4_energy(seqs, dims)
                        if e < best_e:
                            best_e = e
                            best_seq = seqs.copy()
                            improved = True
                            break
                        seqs[sk, ik] *= -1
                    if improved:
                        break
                    seqs[sj, ij] *= -1
                if improved:
                    break
                seqs[si, ii] *= -1

        if not improved:
            # Kick: flip one random bit per sequence to escape basin
            for s in range(4):
                seqs[s, rng.integers(0, n)] *= -1
    return best_seq, best_e


def _tensor_build(solutions):
    """Tensor product of a list of (matrix, _) tuples."""
    H = solutions[0][0]
    from gpu import check_orthogonality

    for H2, _ in solutions[1:]:
        H = np.kron(H, H2)
    return H, check_orthogonality(H)


class CustomSolver(SearchStrategy):
    """K-Flip repair for arbitrary NPAF configurations.

    Class methods provide canned setups:

    - ``CustomSolver.williamson(n=11)`` for standard Williamson
    - ``CustomSolver.tensor(n1=11, n2=13)`` for tensor product
    - ``CustomSolver(seq_lengths=..., weights=...)`` for arbitrary configs
    """

    def __init__(
        self,
        *,
        seq_lengths: tuple[int, ...],
        weights: tuple[float, ...],
        max_k: int = 4,
        order: int | None = None,
        group_dims: list[int] | None = None,
    ):
        if len(seq_lengths) != len(weights):
            raise ValueError("seq_lengths and weights must have same length")
        self.N_SEQS = len(seq_lengths)
        self.LENGTHS = np.array(seq_lengths, dtype=np.int64)
        self.WEIGHTS = np.array(weights, dtype=np.float64)
        self.max_k = max_k
        self.N = int(self.LENGTHS.max())
        self.ORDER = order if order is not None else 4 * self.N
        self.group_dims = group_dims  # non-None = group-based GS4 mode

    @classmethod
    def williamson(cls, *, n: int, max_k: int = 4):
        """Standard Williamson: 4 equal-length sequences, unit weights."""
        return cls(seq_lengths=(n, n, n, n), weights=(1, 1, 1, 1), max_k=max_k, order=4 * n)

    @classmethod
    def group(cls, *, n: int, max_k: int = 4):
        """Group-based GS4 with auto-detected best factorization."""
        dims = cls._best_factorization(n)
        return cls(
            seq_lengths=(n, n, n, n),
            weights=(1, 1, 1, 1),
            max_k=max_k,
            order=4 * n,
            group_dims=dims,
        )

    @staticmethod
    def _best_factorization(n: int) -> list[int]:
        """Best group dims for n: factors closest to sqrt(n), both >= 3."""
        best = [n]
        best_diff = n
        for p in range(3, int(n**0.5) + 1):
            if n % p == 0:
                q = n // p
                if q >= 3:
                    diff = abs(p - q)
                    if diff < best_diff:
                        best_diff = diff
                        best = sorted([p, q], reverse=True)
        return best

    @classmethod
    def tensor(cls, *, n1: int, n2: int, max_k: int = 4):
        """Tensor-product search: finds W(n1) and W(n2), returns kron."""
        return cls(
            seq_lengths=(max(n1, n2),) * 4,
            weights=(1, 1, 1, 1),
            max_k=max_k,
            order=4 * n1 * 4 * n2,
        )

    @property
    def name(self) -> str:
        return "custom"

    # ── GPU primitives ──────────────────────────────────────────────────

    def _singles_scan(self, seq_f, positions):
        B = len(positions)
        pos = np.array(positions, dtype=np.int32)
        batch = xp.repeat(seq_f[None, ...], B, axis=0)
        batch[xp.arange(B), xp.asarray(pos[:, 0]), xp.asarray(pos[:, 1])] *= -1
        from correlations import npa_f_residual

        residuals = npa_f_residual(
            batch, lengths=self.LENGTHS, weights=self.WEIGHTS)
        r0 = npa_f_residual(
            seq_f[None, ...], lengths=self.LENGTHS, weights=self.WEIGHTS)[0]
        r0_cpu = to_numpy(r0)
        return to_numpy(residuals) - r0_cpu[None, :], r0_cpu

    def _kflip_scan(self, seq_f, positions, k, pool):
        from correlations import npa_f_residual

        n_combos = 1
        for i in range(k):
            n_combos = n_combos * (len(pool) - i) // (i + 1)
        if n_combos > 500_000:
            return None
        combos = list(combinations(pool, k))
        if not combos:
            return None
        combo_arr = np.array(combos, dtype=np.int32)
        pos_arr = np.array(positions, dtype=np.int32)
        flat_t = combo_arr.ravel()
        n = len(combos)
        batch_idx = np.repeat(np.arange(n, dtype=np.int32), k)
        batch = xp.repeat(seq_f[None, ...], n, axis=0)
        batch[
            xp.asarray(batch_idx),
            xp.asarray(pos_arr[flat_t, 0]),
            xp.asarray(pos_arr[flat_t, 1]),
        ] *= -1
        residuals = npa_f_residual(
            batch, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energies = xp.sum(residuals**2, axis=1)
        best_idx = int(xp.argmin(energies))
        return combos[best_idx], float(energies[best_idx])

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        started = time.perf_counter()

        # ── Group-based GS4 mode ──
        if self.group_dims is not None:
            rng = np.random.default_rng(seed)
            if sequences is None:
                sequences = np.zeros((4, self.N), dtype=np.int8)
                for i in range(4):
                    sequences[i, : int(self.LENGTHS[i])] = rng.choice(
                        (-1, 1), size=int(self.LENGTHS[i])).astype(np.int8)
            else:
                sequences = sequences.copy()

            best_seq, best_e = _kflip_group_search(
                sequences, self.group_dims, steps, rng)

            if best_e == 0:
                from gpu import check_orthogonality
                matrix = _build_group_gs4(best_seq, self.group_dims)
                metrics = check_orthogonality(matrix)
            else:
                matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
                metrics = {"energy": int(
                    best_e), "orthogonal_pairs": 0, "max_abs_correlation": 0}

            elapsed = time.perf_counter() - started
            label = "x".join(map(str, self.group_dims))
            if best_e == 0 and metrics["energy"] == 0:
                print(
                    f"  seed={seed} VALID {matrix.shape[0]}x{matrix.shape[1]} Hadamard [{label}] {elapsed:.1f}s")
            else:
                print(
                    f"  seed={seed} best_e={best_e:.0f} [{label}] {elapsed:.1f}s")
            return Result(matrix, metrics, elapsed, best_seq)

        # ── Standard NPAF mode ──
        if sequences is None:
            rng = np.random.default_rng(seed)
            sequences = np.zeros((self.N_SEQS, self.N), dtype=np.int8)
            for i, L in enumerate(self.LENGTHS):
                sequences[i, : int(L)] = rng.choice(
                    (-1, 1), size=int(L)).astype(np.int8)
        else:
            sequences = sequences.copy()

        positions = [
            (r, c) for r in range(self.N_SEQS) for c in range(int(self.LENGTHS[r]))
        ]
        B = len(positions)
        seq_f = xp.asarray(sequences.astype(np.float32), dtype=xp.float32)
        best_e = _exact_npaf_energy(sequences, self.LENGTHS, self.WEIGHTS)
        best_seq = sequences.copy()
        tabu: set[tuple[int, ...]] = set()
        k_moves = 0

        for _step in range(steps):
            if best_e == 0:
                break
            moves_before = k_moves

            deltas, r0_cpu = self._singles_scan(seq_f, positions)
            singles_energies = np.sum((r0_cpu + deltas) ** 2, axis=1)

            k = 1
            while k <= self.max_k and best_e > 0:
                if k == 1:
                    best_idx = int(np.argmin(singles_energies))
                    combo = (best_idx,)
                    gpu_energy = float(singles_energies[best_idx])
                else:
                    pool = (
                        list(range(B))
                        if (k == 2 and B <= 128)
                        else list(np.argsort(singles_energies)[: min(B, 32)])
                    )
                    result = self._kflip_scan(seq_f, positions, k, pool)
                    if result is None:
                        k += 1
                        continue
                    combo, gpu_energy = result

                if gpu_energy >= best_e - 0.5:
                    k += 1
                    continue

                for t in combo:
                    r, c = positions[t]
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1
                cpu_e = _exact_npaf_energy(
                    sequences, self.LENGTHS, self.WEIGHTS)
                key = tuple(sorted(combo))
                if cpu_e < best_e and key not in tabu:
                    tabu.add(key)
                    best_e = cpu_e
                    best_seq = sequences.copy()
                    k_moves += 1
                    k = 1
                    break
                if cpu_e >= best_e:
                    tabu.add(key)
                for t in combo:
                    r, c = positions[t]
                    sequences[r, c] *= -1
                    seq_f[r, c] *= -1
                k += 1

            if k_moves == moves_before:
                break

        # ── Build matrix ─────────────────────────────────────────────────

        if best_e == 0 and self.N_SEQS == 4 and all(w == 1.0 for w in self.WEIGHTS):
            matrix, metrics = _build_williamson(best_seq, self.N)
        else:
            matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
            metrics = {
                "energy": int(best_e),
                "orthogonal_pairs": 0,
                "max_abs_correlation": 0,
            }

        elapsed = time.perf_counter() - started
        if best_e == 0 and metrics["energy"] == 0:
            print(
                f"  seed={seed} VALID {matrix.shape[0]}x{matrix.shape[1]} Hadamard"
                f" (k-moves={k_moves}) {elapsed:.1f}s"
            )
        else:
            print(
                f"  seed={seed} best_e={best_e:.0f} (k-moves={k_moves})"
                f" seqs={self.N_SEQS}x{self.LENGTHS.tolist()} w={self.WEIGHTS.tolist()} {elapsed:.1f}s"
            )
        return Result(matrix, metrics, elapsed, best_seq)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)

    # ── Tensor product pipeline ─────────────────────────────────────────

    @staticmethod
    def kron(H1: np.ndarray, H2: np.ndarray):
        """Tensor product of two Hadamard matrices — always valid."""
        from gpu import check_orthogonality

        H = np.kron(H1, H2)
        return H, check_orthogonality(H)


# ── Quick CLI test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor", type=str, default=None,
                        help="n1,n2 for tensor product")
    args = parser.parse_args()

    if args.tensor:
        n1, n2 = map(int, args.tensor.split(","))
        cs = CustomSolver.tensor(n1=n1, n2=n2)
        import sys

        print(
            f"\n{'=' * 60}\n  Tensor product search: W({n1}) ⊗ W({n2})"
            f"\n  Target: {4*n1*4*n2}x{4*n1*4*n2}\n{'=' * 60}"
        )
        # Find first
        cs1 = CustomSolver.williamson(n=n1)
        r1 = cs1.search(steps=args.steps, seed=args.seed)
        if r1.metrics["energy"] != 0:
            print(f"  W({n1}) failed. Increase steps?")
            sys.exit(1)
        cs2 = CustomSolver.williamson(n=n2)
        r2 = cs2.search(steps=args.steps, seed=args.seed + 1)
        if r2.metrics["energy"] != 0:
            print(f"  W({n2}) failed. Increase steps?")
            sys.exit(1)
        H, m = CustomSolver.kron(r1.matrix, r2.matrix)
        print(
            f"\n  RESULT: {H.shape[0]}x{H.shape[1]} Hadamard  E={m['energy']}")
    else:
        cs = CustomSolver.williamson(n=args.n)
        r = cs.search(steps=args.steps, seed=args.seed)
        print(
            f"\n  RESULT: {r.matrix.shape[0]}x{r.matrix.shape[1]}  E={r.metrics['energy']}")
