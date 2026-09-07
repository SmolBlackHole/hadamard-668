"""Lab entry points preserve inputs and retain reproducible experiment outputs."""

import base64
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from lab import compare, recover
from src.constructions import paley_ng_sequences


def test_recovery_retains_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    database = tmp_path / "input.db"
    sequences = paley_ng_sequences(6)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE runs (id,n,seqs_b64,solved,valid)")
        connection.execute(
            "INSERT INTO runs VALUES (1,6,?,1,1)", (base64.b64encode(sequences.tobytes()).decode(),)
        )
    before = database.read_bytes()
    output = tmp_path / "recovery.json"
    args = [
        "--database",
        str(database),
        "--parent-ids",
        "1",
        "--radii",
        "1",
        "--repeats",
        "2",
        "--candidate-budget",
        "24",
        "--output",
        str(output),
    ]
    recover.main(args)
    report = json.loads(output.read_text())
    assert len(report["runs"]) == 2
    assert all(row["end_q"] == 0 and row["candidate_evals"] == 24 for row in report["runs"])
    assert report["parents"][0]["seqs_b64"] == base64.b64encode(sequences.tobytes()).decode()
    assert (tmp_path / "recovery.png").is_file()
    assert (tmp_path / "recovery-source" / "src" / "solver" / "engine.py").is_file()
    assert database.read_bytes() == before
    previous = output.read_bytes()
    with pytest.raises(SystemExit):
        recover.main(args)
    assert output.read_bytes() == previous


def test_start_comparison_pairs_seeds_and_keeps_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "starts.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--compare-starts",
            "--n",
            "4",
            "--seeds",
            "2",
            "--candidate-budget",
            "20",
            "--workers",
            "1",
            "--output",
            str(output),
        ],
    )
    compare.main()
    report = json.loads(output.read_text())
    assert report["starts"] == {"random": "random", "cyclic": "cyclic"}
    assert len(report["runs"]) == 4
    assert (tmp_path / "starts.png").is_file()
    with pytest.raises(SystemExit):
        compare.main()
