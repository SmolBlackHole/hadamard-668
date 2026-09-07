"""Tests for versioned GS4 solution features."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from lab.catalog import main as analyze_main
from src.models import RunResult, SearchStats
from src.output import save_run
from src.solution_analysis import FEATURE_VERSION, analyze_solution_database, solution_features


def test_solution_features_for_order_four() -> None:
    sequences = np.ones((4, 1), dtype=np.int8)

    features = solution_features(sequences)

    assert features["strong_split"]
    assert features["has_complementary_pairing"]
    assert features["basis_minus"] == 0
    assert features["basis_transitions_cyclic"] == 0
    assert sum(features["symbol_histogram"]) == 1


def test_analysis_persists_versioned_features_and_catalog(tmp_path: Path) -> None:
    database = tmp_path / "runs.db"
    output = tmp_path / "catalog.json"
    save_run(
        database,
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

    result = analyze_solution_database(database, output)

    assert result["valid_solutions"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["feature_version"] == FEATURE_VERSION
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT feature_version, features_json FROM solution_features"
        ).fetchone()
    assert row is not None
    assert row[0] == FEATURE_VERSION
    assert json.loads(row[1])["strong_split"]


def test_analysis_cli_does_not_replace_catalog_without_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "missing.db"
    output = tmp_path / "catalog.json"

    with pytest.raises(SystemExit) as error:
        analyze_main(["--database", str(database), "--output", str(output)])

    assert error.value.code == 2
    assert "database does not exist" in capsys.readouterr().err
    assert not output.exists()
