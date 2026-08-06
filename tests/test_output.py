"""Tests for output persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from metrics import Metrics
from output import save


def test_save_writes_csv_and_json(tmp_path: Path) -> None:
    matrix = np.array([[1, 1], [1, -1]], dtype=np.int8)
    sha = save(
        matrix,
        Metrics(energy=0, orthogonal_pairs=0, max_abs_correlation=0),
        tmp_path / "run",
        strategy="gs4",
        seed=42,
        steps=5000,
        elapsed=1.5,
        order=2,
        iterations=10,
    )
    assert len(sha) == 64
    assert (tmp_path / "run" / "matrix.csv").exists()
    assert (tmp_path / "run" / "run.json").exists()

    data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert data["strategy"] == "gs4"
    assert data["seed"] == 42
    assert data["steps"] == 5000
    assert data["elapsed"] == 1.5
    assert data["energy"] == 0
    assert data["max_abs_correlation"] == 0
    assert data["is_solution"] is True
    assert data["sha256"] == sha
    assert data["iterations"] == 10


def test_save_non_solution_skips_audit(tmp_path: Path) -> None:
    matrix = np.ones((4, 4), dtype=np.int8)
    sha = save(
        matrix,
        Metrics(energy=16, orthogonal_pairs=0, max_abs_correlation=0),
        tmp_path / "run",
        strategy="gs4",
        seed=1,
        steps=100,
        elapsed=0.1,
        order=4,
    )
    data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert data["is_solution"] is False
    assert data["energy"] == 16
    assert len(sha) == 64
