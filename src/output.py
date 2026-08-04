"""Write reproducible, verifier-compatible search results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from verifier.verify import independent_audit


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
    hardware_summary: str,
    order: int,
    construction: str,
) -> str:
    """Write canonical candidate.csv and schema-compatible run.json."""
    if metrics["energy"] == 0:
        independent_audit(matrix.tolist())
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / "candidate.csv"
    candidate_path.write_text(
        "\n".join(",".join(str(int(value)) for value in row)
                  for row in matrix) + "\n",
        encoding="utf-8",
    )
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "h668-run-v1",
        "result_type": "exact_solution" if metrics["energy"] == 0 else "checkpoint",
        "order": order,
        "construction": construction,
        "method_family": method_family,
        "search_scope": search_scope,
        "coverage_kind": "heuristic",
        "seed_derivation": f"fixed-seed-{seed}",
        "seeds": [str(seed)],
        "evaluations": steps,
        "wall_seconds": round(wall_seconds, 3),
        "hardware_summary": hardware_summary,
        "model_summary": None,
        "code_url": None,
        "code_commit": None,
        "parent_candidate_sha256": None,
        "candidate_sha256": candidate_sha256,
        "metrics": {
            "off_diagonal_energy": metrics["energy"],
            "orthogonal_row_pairs": metrics["orthogonal_pairs"],
            "max_absolute_off_diagonal": metrics["max_abs_correlation"],
        },
        "publication_consent": True,
    }
    (directory / "run.json").write_text(json.dumps(manifest,
                                                   indent=2) + "\n", encoding="utf-8")
    return candidate_sha256
