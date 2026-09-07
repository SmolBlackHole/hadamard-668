"""Checks for provenance separation and unbiased effort aggregation."""

import sqlite3
from pathlib import Path

import pytest

from lab.plot import load_cohorts, main, summarize


def test_cohorts_separate_provenance_and_deduplicate_seeds(tmp_path: Path) -> None:
    database = tmp_path / "runs.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE runs (id,n,seed,energy,solved,candidate_evals,"
            "candidate_budget,code_revision,construction,solver_config_json,strategy,valid)"
        )
        for identifier, n, revision, valid in [
            (1, 30, "a", 1),
            (2, 30, "a", 1),
            (3, 30, "b", 1),
            (4, 31, "a", 1),
            (5, 30, "a", 0),
        ]:
            connection.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (identifier, n, 42, 64 * n, 0, 100, 100, revision, "random", "{}", "gs4", valid),
            )
    cohorts = load_cohorts(database, 1, 5)
    assert len(cohorts) == 3
    assert sum(group["duplicates"] for group in cohorts) == 1
    assert {row["id"] for group in cohorts for row in group["records"]} == {2, 3, 4}
    main(["--database", str(database), "--output", str(tmp_path / "plots")])
    assert len(list((tmp_path / "plots").glob("*.png"))) == 3
    assert (tmp_path / "plots" / "manifest.json").is_file()


def test_effort_includes_failures_and_q_is_normalized() -> None:
    summary = summarize(
        [
            {"n": 30, "energy": 0, "solved": 1, "candidate_evals": 10},
            {"n": 30, "energy": 3840, "solved": 0, "candidate_evals": 100},
        ]
    )[0]
    assert summary["solved"] == 1
    assert summary["q_median"] == 1
    assert summary["eval_median"] == 55


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    with pytest.raises(SystemExit):
        main(["--database", str(path)])
    assert not path.exists()
