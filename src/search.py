"""Hadamard-668 search: Williamson construction + direct local search."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from verifier.verify import independent_audit, normalized_sha256

# --- gpu setup --------------------------------------------------------------
_xp: Any = np
try:
    import os as _os
    import glob as _glob
    import site as _site
    _nvidia = _os.path.join(_site.getsitepackages()[0], "nvidia")
    for _dll in set(_os.path.dirname(p) for p in _glob.glob(_os.path.join(_nvidia, "**", "*.dll"), recursive=True)):
        _os.environ["PATH"] = _dll + ";" + _os.environ.get("PATH", "")
    _os.environ.setdefault("CUDA_PATH", _nvidia)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as _cp
    _cp.dot(_cp.array([1]), _cp.array([1]))  # force cublas load
    _xp = _cp
except Exception:
    pass


# --- config -----------------------------------------------------------------
ORDER = 668
K = 167
HALF = 84

STEPS = 200_000
SEED = 42
STRATEGY = "williamson"


# --- metrics ----------------------------------------------------------------


def _gram_off_diagonal_energy(M: np.ndarray) -> int:
    """sum((H @ H^T - n*I)^2) for off-diagonals, GPU if available."""
    n = M.shape[0]
    if n <= 1:
        return 0
    xp = _xp
    g = xp.asarray(M, dtype=xp.int64)
    g = g @ g.T
    upper = g[xp.triu_indices(n, k=1)]
    val = xp.dot(upper, upper)
    return int(val.get() if xp is not np else val)


def check_orthogonality(matrix: np.ndarray) -> dict[str, int]:
    n = matrix.shape[0]
    total = n * (n - 1) // 2
    if total == 0:
        return {"energy": 0, "orthogonal_pairs": 0, "max_abs_correlation": 0}
    xp = _xp
    g = xp.asarray(matrix, dtype=xp.int64)
    g = g @ g.T
    upper = g[xp.triu_indices(n, k=1)]
    e = xp.dot(upper, upper)
    o = xp.count_nonzero(upper == 0)
    m = xp.abs(upper).max()
    if xp is not np:
        e, o, m = e.get(), o.get(), m.get()
    return {"energy": int(e), "orthogonal_pairs": int(o), "max_abs_correlation": int(m)}


# --- output -----------------------------------------------------------------


def save(matrix: np.ndarray, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [",".join(str(int(v)) for v in row) for row in matrix]
    (directory / "candidate.csv").write_text("\n".join(lines) + "\n")
    sha = normalized_sha256([[int(v) for v in row] for row in matrix])
    print(f"  saved -> {directory}/candidate.csv  sha256={sha}")


# --- direct search ----------------------------------------------------------


def direct_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    M = rng.choice([-1, 1], size=(ORDER, ORDER)).astype(np.int8)
    best = M.copy()
    e = best_e = _gram_off_diagonal_energy(M)
    accepted = best_at = 0

    pbar = tqdm(total=steps, desc="direct", unit="steps")
    for step in range(steps):
        i, j = rng.integers(0, ORDER), rng.integers(0, ORDER)
        if i == j:
            continue
        M[i, j] *= -1
        M[j, i] *= -1
        ne = _gram_off_diagonal_energy(M)
        if ne <= e:
            e, accepted = ne, accepted + 1
            if ne < best_e:
                best_e, best, best_at = ne, M.copy(), step
                if ne == 0:
                    break
        else:
            M[i, j] *= -1
            M[j, i] *= -1
        if step % 50 == 0:
            pbar.set_postfix(e=e, best=best_e, acc=accepted)
            pbar.update(50)
    pbar.set_postfix(e=e, best=best_e, acc=accepted)
    pbar.close()

    print(f"  best_energy={best_e}  found@step={best_at}  accepted={accepted}")
    return best, check_orthogonality(best)


# --- Williamson construction ------------------------------------------------


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


def _build_williamson(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    A, B, C, D = (_circulant(s) for s in (a, b, c, d))
    return np.block([
        [A,  B,  C,  D],
        [-B,  A, -D,  C],
        [-C,  D,  A, -B],
        [-D, -C,  B,  A],
    ]).astype(np.int8)


def williamson_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    half = HALF
    current = [rng.choice([-1, 1], size=half).astype(np.int8)
               for _ in range(4)]
    best_half = [m.copy() for m in current]
    e = best_e = _autocorrelation_energy(
        tuple(_symmetric_circulant(m) for m in current))
    accepted = best_at = 0

    tqdm.write(f"  sub_order={K}  vars={4 * half}  energy_start={e}")
    pbar = tqdm(total=steps, desc="williamson", unit="steps")
    for step in range(steps):
        mi, pi = rng.integers(0, 4), rng.integers(0, half)
        current[mi][pi] *= -1
        ne = _autocorrelation_energy(
            tuple(_symmetric_circulant(m) for m in current))
        if ne <= e:
            e, accepted = ne, accepted + 1
            if ne < best_e:
                best_e, best_half, best_at = ne, [
                    m.copy() for m in current], step
                if ne == 0:
                    break
        else:
            current[mi][pi] *= -1
        if step % 50 == 0:
            pbar.set_postfix(e=e, best=best_e, acc=accepted)
            pbar.update(50)
    pbar.set_postfix(e=e, best=best_e, acc=accepted)
    pbar.close()

    print(f"  best_energy={best_e}  found@step={best_at}  accepted={accepted}")
    full = [_symmetric_circulant(m) for m in best_half]
    M = _build_williamson(*full)
    return M, check_orthogonality(M)


# --- runner -----------------------------------------------------------------


def run(strategy: str = STRATEGY, steps: int = STEPS, seed: int = SEED) -> dict[str, int]:
    out = Path("output")

    if strategy == "direct":
        matrix, metrics = direct_search(steps=steps, seed=seed)
    elif strategy == "williamson":
        matrix, metrics = williamson_search(steps=steps, seed=seed)
    else:
        best_metrics: dict[str, int] = {"energy": 2**63}
        matrix = None
        for _name, fn in [("direct", direct_search), ("williamson", williamson_search)]:
            m, met = fn(steps=steps, seed=seed)
            if met["energy"] < best_metrics["energy"]:
                matrix, best_metrics = m, met
            if met["energy"] == 0:
                break
        metrics = best_metrics

    exact = metrics["energy"] == 0
    total = ORDER * (ORDER - 1) // 2
    print(f"\n{'*** HADAMARD! ***' if exact else 'Kein Hadamard.'}")
    print(
        f"energy={metrics['energy']}  orth_pairs={metrics['orthogonal_pairs']}/{total}  max_corr={metrics['max_abs_correlation']}")

    if exact and matrix is not None:
        independent_audit([[int(v) for v in row] for row in matrix])
        save(matrix, out)
    return metrics


if __name__ == "__main__":
    run()
