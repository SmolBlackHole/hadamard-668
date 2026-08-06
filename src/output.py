"""Persist a search result as CSV + JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from metrics import Metrics
from verify import independent_audit


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
    """Write matrix.csv and run.json. Audits via pure Python if energy==0."""
    if metrics.energy == 0:
        independent_audit(matrix.tolist())

    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "matrix.csv"
    csv_path.write_text(
        "\n".join(",".join(str(int(v)) for v in row) for row in matrix) + "\n",
        encoding="utf-8",
    )

    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
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
        "sha256": sha,
        "iterations": iterations,
    }
    (directory / "run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return sha
