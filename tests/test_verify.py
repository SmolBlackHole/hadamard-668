"""Tests for pure-Python Hadamard verification."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.models import RunResult, SearchStats
from src.output import migrate_database, save_run
from src.verify import InvalidMatrix, audit_database, independent_audit


def _hadamard2() -> list[list[int]]:
    return [[1, 1], [1, -1]]


def test_valid_hadamard_passes() -> None:
    independent_audit(_hadamard2())


def test_non_pm1_value_raises() -> None:
    with pytest.raises(InvalidMatrix, match="contains a value"):
        independent_audit([[1, 0], [1, -1]])


def test_non_orthogonal_rows_raises() -> None:
    with pytest.raises(InvalidMatrix, match="dot product"):
        independent_audit([[1, 1], [1, 1]])


def test_database_audit_fails_when_only_some_rows_are_valid(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    for seed in (1, 2):
        save_run(
            path,
            RunResult(
                strategy="gs4",
                n=1,
                seed=seed,
                sequences=np.ones((4, 1), dtype=np.int8),
                energy=0,
                candidate_evals=0,
                elapsed_seconds=0.0,
                stats=SearchStats(),
                verified=True,
            ),
        )
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE runs SET sha256 = 'corrupt' WHERE seed = 2")

    report = audit_database(path)

    assert report.checked == 3
    assert report.valid == 2
    assert not report.ok
    assert report.failures == ("runs id=2 gs4 n=1 seed=2: sha256 mismatch",)


def test_database_audit_does_not_create_a_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    report = audit_database(path)
    assert not report.ok
    assert not path.exists()


def test_migration_quarantines_invalid_legacy_solution(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    save_run(
        path,
        RunResult(
            strategy="gs4",
            n=1,
            seed=1,
            sequences=np.ones((4, 1), dtype=np.int8),
            energy=0,
            candidate_evals=0,
            elapsed_seconds=0.0,
            stats=SearchStats(),
            verified=True,
        ),
    )
    with sqlite3.connect(path) as connection:
        raw = np.zeros((4, 1), dtype=np.int8).tobytes()
        import base64
        import hashlib

        payload = base64.b64encode(raw).decode("ascii")
        digest = hashlib.sha256(raw).hexdigest()
        connection.execute(
            "INSERT INTO solutions (strategy,n,seed,sha256,seqs_b64,class_hash)"
            " VALUES ('gs4',1,2,?,?,?)",
            (digest, payload, digest),
        )

    migration = migrate_database(path)
    report = audit_database(path)

    assert migration.checked == 3
    assert migration.valid == 2
    assert migration.quarantined == 1
    assert report.ok
    assert report.valid == 2
    assert report.quarantined == 1
