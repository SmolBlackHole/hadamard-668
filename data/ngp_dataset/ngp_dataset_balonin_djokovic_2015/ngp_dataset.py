"""NGP fixtures reconstructed from Balonin & Djokovic, arXiv:1508.00640v1.

Data source:
- Appendix C: second Paley series
- Appendix D: Ito series

The JSON file contains the full reconstructed ±1 sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DATA_PATH = Path(__file__).with_name("ngp_pairs.json")


def load_records() -> list[dict[str, Any]]:
    """Load all reconstructed NGP records."""
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))["records"]


def get_pairs(length: int, family: str | None = None) -> list[dict[str, Any]]:
    """Return all records for a length, optionally filtered by family."""
    result = [r for r in load_records() if r["length"] == length]
    if family is not None:
        result = [r for r in result if r["family"] == family]
    return result


def get_pair(length: int, family: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return the first matching pair as int8 NumPy arrays."""
    records = get_pairs(length, family)
    if not records:
        suffix = f" in family {family!r}" if family else ""
        raise KeyError(f"No NGP fixture for length {length}{suffix}")
    record = records[0]
    return (
        np.asarray(record["a"], dtype=np.int8),
        np.asarray(record["b"], dtype=np.int8),
    )


def negaperiodic_autocorrelation(sequence: np.ndarray) -> np.ndarray:
    """NAF(k) = AF(k) - AF(n-k), k=0,...,n-1."""
    sequence = np.asarray(sequence, dtype=np.int64)
    if sequence.ndim != 1:
        raise ValueError("sequence must be one-dimensional")

    n = sequence.size
    result = np.empty(n, dtype=np.int64)
    for k in range(n):
        result[k] = (
            np.dot(sequence[: n - k], sequence[k:])
            - np.dot(sequence[:k], sequence[n - k :])
        )
    return result


def ngp_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return NAF_a + NAF_b."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("a and b must have equal shapes")
    return negaperiodic_autocorrelation(a) + negaperiodic_autocorrelation(b)


def ngp_energy(a: np.ndarray, b: np.ndarray) -> int:
    """Squared residual energy, excluding lag zero."""
    residual = ngp_residual(a, b)[1:]
    return int(np.dot(residual, residual))


def negacyclic_matrix(first_row: np.ndarray) -> np.ndarray:
    """Build the negacyclic matrix using the paper's shift convention."""
    first_row = np.asarray(first_row, dtype=np.int64)
    if first_row.ndim != 1:
        raise ValueError("first_row must be one-dimensional")

    n = first_row.size
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    indices = (cols - rows) % n
    signs = np.where(cols >= rows, 1, -1)
    return signs * first_row[indices]


def build_2n_negaperiodic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build H = [[A, B], [-B.T, A.T]]."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be one-dimensional and equally long")

    A = negacyclic_matrix(a)
    B = negacyclic_matrix(b)
    return np.block([[A, B], [-B.T, A.T]])


def validate_pair(a: np.ndarray, b: np.ndarray) -> None:
    """Raise AssertionError unless the pair and resulting 2N matrix are valid."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)

    assert ngp_energy(a, b) == 0

    H = build_2n_negaperiodic(a, b)
    order = H.shape[0]
    assert np.array_equal(
        H @ H.T,
        order * np.eye(order, dtype=np.int64),
    )


def validate_all() -> int:
    """Validate all records and return their count."""
    records = load_records()
    for record in records:
        validate_pair(record["a"], record["b"])
    return len(records)


if __name__ == "__main__":
    count = validate_all()
    print(f"Validated {count}/{count} NGP fixtures.")
