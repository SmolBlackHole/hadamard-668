"""Persist a search result as JSON (run metadata) and optionally a dataset entry."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from metrics import Metrics
from verify import independent_audit


def _seqs_to_b64(seqs: np.ndarray) -> str:
    return base64.b64encode(seqs.astype(np.int8, copy=False).tobytes()).decode("ascii")


def _seqs_from_b64(b64: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int8).reshape(shape)


def save(
    matrix: np.ndarray,
    metrics: Metrics,
    directory: Path,
    *,
    strategy: str,
    seed: int,
    steps: int,
    elapsed: float,
    order: int,
    iterations: int = 0,
) -> str:
    """Write run.json (no CSV). Audits via pure Python if energy==0."""
    if metrics.energy == 0:
        independent_audit(matrix.tolist())

    directory.mkdir(parents=True, exist_ok=True)
    info = {
        "order": order,
        "strategy": strategy,
        "seed": seed,
        "steps": steps,
        "elapsed": round(elapsed, 3),
        "energy": metrics.energy,
        "orthogonal_pairs": metrics.orthogonal_pairs,
        "max_abs_correlation": metrics.max_abs_correlation,
        "is_solution": metrics.energy == 0,
        "iterations": iterations,
    }
    run_json = directory / "run.json"
    run_json.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(run_json.read_bytes()).hexdigest()


def append_solution(
    dataset_path: Path,
    strategy: str,
    n: int,
    seed: int,
    sequences: np.ndarray | None,
    elapsed: float,
    iterations: int = 0,
    stats: dict[str, int] | None = None,
) -> None:
    """Append a verified solution to the dataset."""
    if sequences is None:
        return
    s64 = _seqs_to_b64(sequences)
    sha = hashlib.sha256(sequences.astype(np.int8, copy=False).tobytes()).hexdigest()
    entry: dict[str, Any] = {
        "seed": seed,
        "sha256": sha,
        "seqs_b64": s64,
        "elapsed_s": round(elapsed, 3),
        "iterations": iterations,
    }
    if stats:
        entry["stats"] = stats
    if strategy == "golay_2n":
        from ngp_canonical import class_hash

        entry["ngp_class"] = class_hash(sequences[0], sequences[1])
    existing: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if dataset_path.exists():
        existing = json.loads(dataset_path.read_text(encoding="utf-8"))
    existing.setdefault(strategy, {}).setdefault(str(n), []).append(entry)
    existing[strategy][str(n)].sort(key=lambda r: r["seed"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def check_dataset(path: Path) -> int:
    """Verify all records via Generator (CLI path) + independent_audit."""
    from generator import Generator

    if not path.exists():
        print(f"{path} not found")
        return 0
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    total = 0
    ok = 0
    for strategy, ns in data.items():
        for n_str, entries in ns.items():
            n = int(n_str)
            gen = Generator(kind=strategy, n=n)
            for r in entries:
                total += 1
                seqs = _seqs_from_b64(r["seqs_b64"], (gen._builder.k, n))

                # integrity: check sha256 of raw bytes
                actual_sha = hashlib.sha256(seqs.astype(np.int8, copy=False).tobytes()).hexdigest()
                if actual_sha != r.get("sha256", ""):
                    print(
                        f"  CORRUPT {strategy} n={n_str} seed={r['seed']}: sha256 mismatch ({actual_sha} vs {r.get('sha256')})"
                    )
                    continue

                H = gen._builder.build(seqs)
                try:
                    independent_audit(H.tolist())
                    ok += 1
                except Exception as e:
                    print(f"  FAIL {strategy} n={n_str} seed={r['seed']}: {e}")
    print(f"{ok}/{total} valid.")
    return ok
