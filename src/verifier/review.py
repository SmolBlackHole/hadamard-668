#!/usr/bin/env python3
"""Score one Hadamard candidate bundle.

``candidate.csv`` is the matrix and is always examined first.
``run.json`` is telemetry: required for a checkpoint, but incapable of
invalidating an otherwise exact Hadamard matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .validate import InvalidManifest, load_and_validate_manifest
from .verify import InvalidMatrix, independent_audit, load_matrix_with_hash, normalized_sha256


def review_bundle(directory: Path, order: int) -> dict[str, object]:
    started = time.perf_counter()
    if not directory.is_dir():
        raise ValueError("bundle path must be a directory")
    entries = {entry.name for entry in directory.iterdir()
               if not entry.name.startswith(".")}
    if "candidate.csv" not in entries:
        raise ValueError("submission must contain candidate.csv")
    unexpected_files = sorted(entries - {"candidate.csv", "run.json"})

    # The matrix is authoritative. Never allow optional telemetry to hide an
    # exact mathematical result.
    rows, matrix, raw_sha256 = load_matrix_with_hash(
        directory / "candidate.csv", order)
    gram = matrix @ matrix.T
    upper = gram[np.triu_indices(order, k=1)]
    absolute = np.abs(upper)
    histogram = Counter(int(value) for value in absolute)
    total_pairs = order * (order - 1) // 2
    orthogonal_pairs = int(np.count_nonzero(upper == 0))
    energy = int(np.dot(upper, upper))
    max_correlation = int(absolute.max(initial=0))
    exact = energy == 0
    if exact:
        # Deliberately separate from the NumPy Gram computation above.
        independent_audit(rows)

    recomputed = {
        "off_diagonal_energy": energy,
        "orthogonal_row_pairs": orthogonal_pairs,
        "max_absolute_off_diagonal": max_correlation,
    }
    manifest: dict[str, object] | None = None
    manifest_valid = False
    manifest_error: str | None = None
    claim_mismatches: dict[str, object] = {}
    if "run.json" not in entries:
        manifest_error = "run.json was not submitted"
    else:
        try:
            manifest = load_and_validate_manifest(directory / "run.json")
            if manifest["candidate_sha256"] != raw_sha256:
                raise InvalidManifest(
                    "run.json candidate_sha256 does not match candidate.csv raw bytes"
                )
            manifest_valid = True
            raw_metrics = manifest.get("metrics", {})
            if not isinstance(raw_metrics, dict):
                raise InvalidManifest("metrics must be an object")
            claimed: dict[str, object] = raw_metrics
            claim_mismatches = {
                key: {"claimed": claimed[key], "recomputed": value}
                for key, value in recomputed.items()
                if claimed[key] is not None and claimed[key] != value
            }
        except InvalidManifest as exc:
            manifest_error = str(exc)

    checkpoint_valid = bool(not exact and manifest_valid)
    return {
        "bundle_valid": bool(exact or checkpoint_valid),
        "bundle_valid_meaning": (
            "exact solution or valid checkpoint; "
            "use exact_solution for mathematical success"
        ),
        "candidate_valid": True,
        "exact_solution": exact,
        "checkpoint_valid": checkpoint_valid,
        "manifest_valid": manifest_valid,
        "manifest_error": manifest_error,
        "unexpected_files_ignored": unexpected_files,
        "order": order,
        "result_type": manifest["result_type"] if manifest else None,
        "method_family": manifest["method_family"] if manifest else None,
        "raw_sha256": raw_sha256,
        "canonical_sha256": normalized_sha256(rows),
        **recomputed,
        "total_row_pairs": total_pairs,
        "absolute_correlation_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "metric_mismatches": claim_mismatches,
        "independent_audit": exact,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review an H668 candidate matrix and optional run manifest."
    )
    parser.add_argument("bundle_directory", type=Path)
    parser.add_argument("--order", type=int, default=668)
    args = parser.parse_args()
    try:
        if args.order <= 0:
            raise ValueError("order must be positive")
        report = review_bundle(args.bundle_directory, args.order)
    except (InvalidManifest, InvalidMatrix, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "bundle_valid": False,
                    "candidate_valid": False,
                    "exact_solution": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
