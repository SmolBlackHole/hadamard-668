"""Hadamard-668 search: Williamson construction + direct local search."""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from verifier.verify import independent_audit

# --- gpu setup --------------------------------------------------------------
_xp: Any = np
try:
    import glob as _glob
    import site as _site
    _nvidia = os.path.join(_site.getsitepackages()[0], "nvidia")
    for _dll in set(os.path.dirname(p) for p in _glob.glob(os.path.join(_nvidia, "**", "*.dll"), recursive=True)):
        os.environ["PATH"] = _dll + ";" + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_PATH", _nvidia)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import cupy as _cp
    _cp.dot(_cp.array([1]), _cp.array([1]))  # force cublas load
    _xp = _cp
except Exception:
    pass


def _progress(total: int, desc: str) -> tqdm | AbstractContextManager:
    """tqdm bar, or no-op in batch mode."""
    if os.environ.get("BATCH"):
        from contextlib import nullcontext
        return nullcontext()
    return tqdm(total=total, desc=desc, unit="steps", dynamic_ncols=True)


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


def save_run(
    matrix: np.ndarray,
    metrics: dict[str, int],
    directory: Path,
    *,
    method_family: str,
    search_scope: str,
    seed: int,
    steps: int,
    wall_seconds: float,
) -> str:
    """Write candidate.csv + run.json. Returns canonical SHA-256."""
    directory.mkdir(parents=True, exist_ok=True)

    lines = [",".join(str(int(v)) for v in row) for row in matrix]
    csv_text = "\n".join(lines) + "\n"
    (directory / "candidate.csv").write_text(csv_text)

    raw_sha256 = hashlib.sha256(
        (directory / "candidate.csv").read_bytes()).hexdigest()

    run_data = {
        "schema_version": "h668-run-v1",
        "result_type": "exact_solution" if metrics["energy"] == 0 else "checkpoint",
        "method_family": method_family,
        "search_scope": search_scope,
        "coverage_kind": "heuristic",
        "seed_derivation": f"fixed-seed-{seed}",
        "seeds": [str(seed)],
        "evaluations": steps,
        "wall_seconds": round(wall_seconds, 3),
        "hardware_summary": "cpu+gpu" if _xp is not np else "cpu",
        "model_summary": None,
        "code_url": None,
        "code_commit": None,
        "parent_candidate_sha256": None,
        "candidate_sha256": raw_sha256,
        "metrics": {
            "off_diagonal_energy": metrics["energy"],
            "orthogonal_row_pairs": metrics["orthogonal_pairs"],
            "max_absolute_off_diagonal": metrics["max_abs_correlation"],
        },
        "publication_consent": True,
    }
    (directory / "run.json").write_text(json.dumps(run_data,
                                                   indent=2) + "\n", encoding="utf-8")
    print(f"  saved -> {directory}/  sha256={raw_sha256}")
    return raw_sha256


# --- direct search ----------------------------------------------------------


def direct_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict[str, int], float]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    M = rng.choice([-1, 1], size=(ORDER, ORDER)).astype(np.int8)
    best = M.copy()
    e = best_e = _gram_off_diagonal_energy(M)
    accepted = best_at = 0

    with _progress(total=steps, desc="direct") as pbar:
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
            if step % 50 == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix(e=e, best=best_e, acc=accepted)
                pbar.update(50)

    elapsed = time.perf_counter() - t0
    print(f"  seed={seed}  best_energy={best_e}  found@step={best_at}  accepted={accepted}  {elapsed:.1f}s")
    return best, check_orthogonality(best), elapsed


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


def williamson_search(steps: int = STEPS, seed: int = SEED) -> tuple[np.ndarray, dict[str, int], float]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    half = HALF
    current = [rng.choice([-1, 1], size=half).astype(np.int8)
               for _ in range(4)]
    best_half = [m.copy() for m in current]
    e = best_e = _autocorrelation_energy(
        tuple(_symmetric_circulant(m) for m in current))
    accepted = best_at = 0

    print(f"  sub_order={K}  vars={4 * half}  energy_start={e}")
    with _progress(total=steps, desc="williamson") as pbar:
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
            if step % 50 == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix(e=e, best=best_e, acc=accepted)
                pbar.update(50)

    elapsed = time.perf_counter() - t0
    print(f"  seed={seed}  best_energy={best_e}  found@step={best_at}  accepted={accepted}  {elapsed:.1f}s")
    full = [_symmetric_circulant(m) for m in best_half]
    M = _build_williamson(*full)
    return M, check_orthogonality(M), elapsed


# --- runner -----------------------------------------------------------------


def run(strategy: str = STRATEGY, steps: int = STEPS, seed: int = SEED) -> dict[str, int]:
    out = Path("output")
    method = strategy
    elapsed = 0.0

    if strategy == "direct":
        matrix, metrics, elapsed = direct_search(steps=steps, seed=seed)
    elif strategy == "williamson":
        matrix, metrics, elapsed = williamson_search(steps=steps, seed=seed)
    else:
        best_metrics: dict[str, int] = {"energy": 2**63}
        for _name, fn in [("direct", direct_search), ("williamson", williamson_search)]:
            m, met, e = fn(steps=steps, seed=seed)
            if met["energy"] < best_metrics["energy"]:
                matrix, best_metrics, elapsed = m, met, e
                method = _name
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

    mf = {"direct": "local_search",
          "williamson": "williamson_propus"}.get(method, "other")
    save_run(
        matrix, metrics, out,
        method_family=mf,
        search_scope=f"{method} search, {steps} iterations",
        seed=seed,
        steps=steps,
        wall_seconds=elapsed,
    )
    return metrics
