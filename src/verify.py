"""Independent reference verification for Hadamard constructions.

Pure-Python, deliberately avoids the solver's NumPy/CuPy code paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .builder import build_gs4
from .models import AuditReport, Int8Array


class InvalidMatrix(ValueError):
    """Raised when a candidate does not satisfy the acceptance contract."""


def independent_audit(rows: list[list[int]]) -> None:
    """Verify rows form a Hadamard matrix: pure-Python, no NumPy."""
    order = len(rows)
    for i in range(order):
        if any(v not in (-1, 1) for v in rows[i]):
            raise InvalidMatrix(f"row {i + 1} contains a value other than -1 or 1")
        if sum(v * v for v in rows[i]) != order:
            raise InvalidMatrix(f"row {i + 1} has incorrect squared length")
        for j in range(i):
            dot = sum(a * b for a, b in zip(rows[i], rows[j]))
            if dot != 0:
                raise InvalidMatrix(
                    f"independent audit failed for rows {j + 1} and {i + 1}: dot product is {dot}"
                )


def verify_candidate(sequences: Int8Array) -> None:
    if sequences.ndim != 2 or sequences.shape[0] != 4:
        raise InvalidMatrix("candidate must have shape (4, n)")
    if not np.all(np.abs(sequences) == 1):
        raise InvalidMatrix("candidate sequences contain a value other than -1 or 1")
    independent_audit(build_gs4(sequences).tolist())


def audit_database(path: Path) -> AuditReport:
    from .output import load_solved_records

    failures: list[str] = []
    records = load_solved_records(path)
    for record in records:
        label = f"{record['strategy']} n={record['n']} seed={record['seed']}"
        sequences: Int8Array = np.asarray(record["sequences"], dtype=np.int8)
        digest = hashlib.sha256(sequences.tobytes()).hexdigest()
        if digest != record["sha256"]:
            failures.append(f"{label}: sha256 mismatch")
            continue
        try:
            verify_candidate(sequences)
        except InvalidMatrix as error:
            failures.append(f"{label}: {error}")

    return AuditReport(
        checked=len(records),
        valid=len(records) - len(failures),
        failures=tuple(failures),
    )
