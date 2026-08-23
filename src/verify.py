"""Independent reference verification for Hadamard constructions.

Pure-Python, deliberately avoids the solver's NumPy/CuPy code paths.
"""

from __future__ import annotations

import hashlib
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .builder import build_gs4
from .canonical import CANONICALIZER, StateIdentity, identify_state
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
    verify_gs4_sequences(sequences)
    independent_audit(build_gs4(sequences).tolist())


def verify_gs4_sequences(sequences: Int8Array) -> None:
    """Independently verify the exact negaperiodic GS4 equations."""
    if sequences.ndim != 2 or sequences.shape[0] != 4:
        raise InvalidMatrix("candidate must have shape (4, n)")
    if not np.all(np.abs(sequences) == 1):
        raise InvalidMatrix("candidate sequences contain a value other than -1 or 1")
    values = sequences.astype(np.int64)
    for lag in range(1, sequences.shape[1]):
        residual = sum(int(row[:-lag] @ row[lag:]) - int(row[-lag:] @ row[:lag]) for row in values)
        if residual:
            raise InvalidMatrix(f"negaperiodic residual at lag {lag} is {residual}")


def validate_database_record(
    record: Mapping[str, Any], *, check_stored_identity: bool = True
) -> tuple[str | None, StateIdentity | None]:
    try:
        raw = b64decode(str(record["seqs_b64"]), validate=True)
    except (Base64Error, ValueError):
        return "invalid base64 payload", None
    n = int(record["n"])
    if len(raw) != 4 * n:
        return f"payload has {len(raw)} bytes, expected {4 * n}", None
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["sha256"]:
        return "sha256 mismatch", None
    sequences = np.frombuffer(raw, dtype=np.int8).reshape(4, n)
    try:
        identity = identify_state(sequences)
    except ValueError as error:
        return str(error), None
    if check_stored_identity:
        if record.get("validation_hash") != identity.validation_hash:
            return "validation_hash mismatch", identity
        if record.get("orbit_hash") != identity.orbit_hash:
            return "orbit_hash mismatch", identity
        if record.get("canonical_b64") != identity.canonical_b64:
            return "canonical payload mismatch", identity
        if record.get("canonicalizer") != CANONICALIZER:
            return "canonicalizer mismatch", identity
    if bool(record["solved"]):
        try:
            verify_gs4_sequences(sequences)
        except InvalidMatrix as error:
            return str(error), identity
    return None, identity


def audit_database(path: Path, progress: Callable[[int, int], None] | None = None) -> AuditReport:
    from .output import load_audit_records

    failures: list[str] = []
    quarantined = 0
    valid = 0
    records = load_audit_records(path)
    for checked, record in enumerate(records, 1):
        label = (
            f"{record['source_table']} id={record['id']} {record['strategy']}"
            f" n={record['n']} seed={record['seed']}"
        )
        error, _ = validate_database_record(record)
        expected_valid = record["valid"]
        expected_error = record["validation_error"]
        if error is None and expected_valid == 1:
            valid += 1
        elif error is not None and expected_valid == 0 and error == expected_error:
            quarantined += 1
        elif error is None:
            failures.append(f"{label}: valid record is quarantined")
        else:
            failures.append(f"{label}: {error}")
        if progress is not None:
            progress(checked, len(records))

    return AuditReport(
        checked=len(records),
        valid=valid,
        quarantined=quarantined,
        failures=tuple(failures),
    )
