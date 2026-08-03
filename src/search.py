"""Hadamard-668 search: direct local search + Williamson construction.

Tweak STEPS, SEED, STRATEGY below. Run: python src/search.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# --- verifier als library ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "verifier"))
from verify import load_matrix_with_hash, independent_audit, normalized_sha256  # noqa: E402

# --- config ------------------------------------------------------------------
ORDER = 668
K = 167  # Williamson sub-order: 668 = 4 * 167
H = (K + 1) // 2  # half-length of symmetric circulant first row (84)

STEPS = 200_000
SEED = 42
STRATEGY = "williamson"  # "direct", "williamson", or "both"


# --- metrics ----------------------------------------------------------------


def energy_off_diagonal(matrix: np.ndarray) -> int:
    """Sum of squared off-diagonal entries of H @ H^T."""
    n = matrix.shape[0]
    gram = matrix @ matrix.T.astype(np.int64)
    upper = gram[np.triu_indices(n, k=1)]
    return int(np.dot(upper, upper))


def check_orthogonality(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    gram = matrix @ matrix.T.astype(np.int64)
    upper = gram[np.triu_indices(n, k=1)]
    return {
        "energy": int(np.dot(upper, upper)),
        "orthogonal_pairs": int(np.count_nonzero(upper == 0)),
        "max_abs_correlation": int(np.abs(upper).max()),
    }


# --- output ----------------------------------------------------------------


def save(matrix: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [",".join(str(int(v)) for v in row) for row in matrix]
    (directory / "candidate.csv").write_text("\n".join(lines) + "\n")
    sha = normalized_sha256([[int(v) for v in row] for row in matrix])
    print(f"  saved -> {directory}/candidate.csv  sha256={sha}")


# --- strategy: direct local search on 668x668 ------------------------------


def direct_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    M = rng.choice([-1, 1], size=(ORDER, ORDER)).astype(np.int8)
    best = M.copy()
    e = best_e = energy_off_diagonal(M)
    accepted = best_at = 0
    t0 = time.perf_counter()

    print(f"direct  steps={steps}  seed={seed}  energy_start={e}")
    for step in range(steps):
        i, j = rng.integers(0, ORDER), rng.integers(0, ORDER)
        if i == j:
            continue
        M[i, j] *= -1
        M[j, i] *= -1
        ne = energy_off_diagonal(M)
        if ne <= e:
            e = ne
            accepted += 1
            if ne < best_e:
                best_e, best, best_at = ne, M.copy(), step
                if ne == 0:
                    print(f"  FOUND at step {step}!")
                    break
        else:
            M[i, j] *= -1
            M[j, i] *= -1
        if step % (max(1, steps // 40)) == 0:
            pct = (step + 1) / steps * 100
            rate = (step + 1) / max(time.perf_counter() - t0, 0.001)
            print(
                f"  [{step + 1:>7d}/{steps}]  energy={e:>10d}  best={best_e:>10d}  rate={rate:>6.0f}/s  {pct:>5.1f}%", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"  done. best_energy={best_e}  found@step={best_at}  {elapsed:.1f}s  accepted={accepted}")
    return best, check_orthogonality(best)


# --- strategy: Williamson (4 circulant blocks of order 167) ----------------


def _symmetric_circulant(half: np.ndarray) -> np.ndarray:
    n = 2 * len(half) - 1
    result = np.zeros(n, dtype=np.int8)
    result[:len(half)] = half
    result[len(half):] = half[n - len(half) - 1::-1]
    return result


def _circulant(v: np.ndarray) -> np.ndarray:
    n = len(v)
    return np.array([np.roll(v, i) for i in range(n)], dtype=np.int8)


def _autocorrelation_energy(seqs: tuple[np.ndarray, ...]) -> int:
    n = len(seqs[0])
    total = np.zeros(n, dtype=np.int64)
    for s in seqs:
        for d in range(n):
            total[d] += int(np.dot(s, np.roll(s, -d)))
    return int(np.sum(total[1:(n + 1) // 2] ** 2))


def _build_williamson(a, b, c, d) -> np.ndarray:
    A, B, C, D = (_circulant(s) for s in (a, b, c, d))
    return np.block([
        [A,  B,  C,  D],
        [-B,  A, -D,  C],
        [-C,  D,  A, -B],
        [-D, -C,  B,  A],
    ]).astype(np.int8)


def williamson_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    half_len = H
    current = [rng.choice([-1, 1], size=half_len).astype(np.int8)
               for _ in range(4)]
    best_half = [m.copy() for m in current]
    e = best_e = _autocorrelation_energy(
        tuple(_symmetric_circulant(m) for m in current))
    accepted = best_at = 0
    t0 = time.perf_counter()

    print(
        f"williamson  steps={steps}  seed={seed}  sub_order={K}  vars={4*half_len}  constraints={half_len-1}  energy_start={e}")
    for step in range(steps):
        mi, pi = rng.integers(0, 4), rng.integers(0, half_len)
        current[mi][pi] *= -1
        ne = _autocorrelation_energy(
            tuple(_symmetric_circulant(m) for m in current))
        if ne <= e:
            e, accepted = ne, accepted + 1
            if ne < best_e:
                best_e, best_half, best_at = ne, [
                    m.copy() for m in current], step
                if ne == 0:
                    print(f"  FOUND at step {step}!")
                    break
        else:
            current[mi][pi] *= -1
        if step % (max(1, steps // 40)) == 0:
            pct = (step + 1) / steps * 100
            rate = (step + 1) / max(time.perf_counter() - t0, 0.001)
            print(
                f"  [{step + 1:>7d}/{steps}]  energy={e:>10d}  best={best_e:>10d}  rate={rate:>6.0f}/s  {pct:>5.1f}%", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"  done. best_energy={best_e}  found@step={best_at}  {elapsed:.1f}s  accepted={accepted}")
    full = [_symmetric_circulant(m) for m in best_half]
    M = _build_williamson(*full)
    return M, check_orthogonality(M)


# --- runner ----------------------------------------------------------------


def run(strategy: str = STRATEGY, steps: int = STEPS, seed: int = SEED) -> dict:
    """Run search, return best result. Saves to output/ on exact solution."""
    out = Path("output")

    if strategy == "direct":
        matrix, metrics = direct_search(steps=steps, seed=seed)
    elif strategy == "williamson":
        matrix, metrics = williamson_search(steps=steps, seed=seed)
    else:  # both
        methods = [direct_search, williamson_search]
        best_matrix, best_metrics = None, {"energy": 2**63}
        for i, method in enumerate(methods):
            matrix, metrics = method(steps=steps, seed=seed + i)
            if metrics["energy"] < best_metrics["energy"]:
                best_matrix, best_metrics = matrix, metrics
            if metrics["energy"] == 0:
                break
        matrix, metrics = best_matrix, best_metrics

    exact = metrics["energy"] == 0
    print(f"\n{'*** HADAMARD! ***' if exact else 'Kein Hadamard.'}")
    print(
        f"energy={metrics['energy']}  orth_pairs={metrics['orthogonal_pairs']}/222778  max_corr={metrics['max_abs_correlation']}")

    if exact:
        verify = independent_audit([[int(v) for v in row] for row in matrix])
        save(matrix, out)
    return metrics


if __name__ == "__main__":
    run()
