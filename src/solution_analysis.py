"""Versioned structural features for verified GS4 solutions."""

from __future__ import annotations

import json
from collections import Counter
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .canonical import orbit_hash, quick_orbit_size
from .constructions import paley_ng_sequences, supports_paley_ng
from .models import Int8Array
from .output import load_valid_solutions, store_solution_features
from .tracker import Tracker

FEATURE_VERSION = "gs4-solution-features-v1"
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def _naf(sequence: Int8Array) -> list[int]:
    """Compute all nonzero-lag negaperiodic autocorrelations of one row."""
    values = sequence.astype(np.int64)
    return [
        int(values[:-lag] @ values[lag:]) - int(values[-lag:] @ values[:lag])
        for lag in range(1, sequence.size)
    ]


@cache
def _paley_orbit(n: int) -> str | None:
    """Return the direct Paley construction's quick-orbit hash when supported."""
    return orbit_hash(paley_ng_sequences(n)) if supports_paley_ng(n) else None


def solution_features(sequences: Int8Array) -> dict[str, Any]:
    """Compute the versioned structural features of one verified GS4 solution.

    Args:
        sequences: Four verified binary sequences with shape ``(4, n)``.

    Returns:
        A JSON-compatible mapping containing sequence sums, NAF signatures,
        pair separators, dual and basis measures, the complete single-flip
        neighbor-Q histogram, quick-orbit data, and Paley-orbit membership.

    Raises:
        ValueError: If the state fails the tracker's or canonicalizer's input
            contract.

    Note:
        The input is not mutated. Feature computation does not independently
        establish that the state is a GS4 solution.
    """
    n = sequences.shape[1]
    nafs = [_naf(row) for row in sequences]
    pair_norms = [
        sum((nafs[left][lag] + nafs[right][lag]) ** 2 for lag in range(n - 1))
        for (left, right), _ in PAIRINGS
    ]
    beta = np.prod(sequences, axis=0, dtype=np.int8)
    dual = beta * sequences
    dual_tracker = Tracker()
    dual_tracker.build(dual)
    tracker = Tracker()
    tracker.build(sequences)
    neighbor_qs = tracker.flip_qs()
    q_counts = Counter(map(int, neighbor_qs))
    symbols = np.sum(
        (sequences > 0).astype(np.int8) * (1 << np.arange(4, dtype=np.int8))[:, None], axis=0
    )
    symbol_counts = np.bincount(symbols, minlength=16)
    orbit_size = quick_orbit_size(sequences)
    group_size = 24 * (4 * n) ** 4
    return {
        "sequence_sums_abs": sorted(abs(int(row.sum())) for row in sequences),
        "single_naf_signatures": [list(signature) for signature in sorted(map(tuple, nafs))],
        "pair_separator_norms_full": sorted(pair_norms),
        "min_pair_separator_norm_full": min(pair_norms),
        "has_complementary_pairing": min(pair_norms) == 0,
        "dual_q": dual_tracker.energy() // (64 * n),
        "strong_split": dual_tracker.energy() == 0,
        "basis_minus": int(np.count_nonzero(beta < 0)),
        "basis_plus": int(np.count_nonzero(beta > 0)),
        "basis_transitions_cyclic": int(np.count_nonzero(beta != np.roll(beta, 1))),
        "symbol_histogram": symbol_counts.tolist(),
        "neighbor_q_min": int(neighbor_qs.min()),
        "neighbor_q_histogram": {str(q): count for q, count in sorted(q_counts.items())},
        "quick_orbit_size": orbit_size,
        "quick_orbit_stabilizer": group_size // orbit_size,
        "paley_orbit": orbit_hash(sequences) == _paley_orbit(n),
    }


def analyze_solution_database(db_path: Path, output: Path) -> dict[str, Any]:
    """Persist features for valid solutions and rebuild the public catalog.

    Args:
        db_path: SQLite database whose ``valid = 1`` solution rows are analyzed.
        output: JSON catalog path to replace with the regenerated snapshot.

    Returns:
        The same versioned catalog mapping written to ``output``.

    Raises:
        ValueError: If a stored solution fails feature input validation.
        OSError: If the catalog cannot be written or atomically replaced.

    Note:
        Feature rows are upserted in SQLite before the catalog is written. The
        JSON file itself is written through a sibling temporary file and then
        atomically replaced; the database and file are not one transaction.
    """
    records = load_valid_solutions(db_path)
    catalog: list[dict[str, Any]] = []
    stored: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        sequences = np.asarray(record.pop("sequences"), dtype=np.int8)
        features = solution_features(sequences)
        solution_id = int(record["id"])
        stored.append((solution_id, features))
        catalog.append({**record, "features": features})
    store_solution_features(db_path, FEATURE_VERSION, stored)

    by_n: dict[str, dict[str, Any]] = {}
    for n in sorted({int(record["n"]) for record in catalog}):
        selected = [record for record in catalog if record["n"] == n]
        basis_minus = [record["features"]["basis_minus"] for record in selected]
        by_n[str(n)] = {
            "solutions": len(selected),
            "quick_orbits": len({record["orbit_hash"] for record in selected}),
            "strong_split": sum(record["features"]["strong_split"] for record in selected),
            "complementary_pairing": sum(
                record["features"]["has_complementary_pairing"] for record in selected
            ),
            "paley_orbit": sum(record["features"]["paley_orbit"] for record in selected),
            "basis_minus_min_median_max": [
                min(basis_minus),
                median(basis_minus),
                max(basis_minus),
            ],
        }
    payload = {
        "feature_version": FEATURE_VERSION,
        "database": str(db_path),
        "valid_solutions": len(catalog),
        "quick_orbits": len({record["orbit_hash"] for record in catalog}),
        "by_n": by_n,
        "solutions": catalog,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    return payload
