#!/usr/bin/env python3
"""Report exact progress metrics for a square ±1 candidate matrix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .verify import InvalidMatrix, load_matrix, normalized_sha256


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a ±1 matrix by its exact off-diagonal correlations."
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--order", type=int, required=True)
    args = parser.parse_args()

    try:
        rows, matrix = load_matrix(args.candidate, args.order)
    except InvalidMatrix as exc:
        print(json.dumps({"scorable": False, "error": str(exc)}, indent=2))
        return 1

    gram = matrix @ matrix.T
    upper = gram[np.triu_indices(args.order, k=1)]
    absolute = np.abs(upper)
    histogram = Counter(int(value) for value in absolute)
    total_pairs = args.order * (args.order - 1) // 2
    orthogonal_pairs = int(np.count_nonzero(upper == 0))
    energy = int(np.dot(upper, upper))

    print(
        json.dumps(
            {
                "scorable": True,
                "valid_hadamard": energy == 0,
                "order": args.order,
                "sha256": normalized_sha256(rows),
                "off_diagonal_energy": energy,
                "max_absolute_correlation": int(absolute.max(initial=0)),
                "orthogonal_pairs": orthogonal_pairs,
                "total_row_pairs": total_pairs,
                "orthogonal_pair_fraction": (
                    round(orthogonal_pairs / total_pairs,
                          8) if total_pairs else 1.0
                ),
                "absolute_correlation_histogram": {
                    str(key): histogram[key] for key in sorted(histogram)
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
