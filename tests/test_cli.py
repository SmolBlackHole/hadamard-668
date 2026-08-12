"""CLI integration tests — keep `--workers 1` to avoid process storms."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run import _execute_single, _format_time
from src.solver import SolverConfig


def test_run_help_works() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--symmetric-bs" not in result.stdout
    assert "SQLite" in result.stdout


def test_format_time() -> None:
    assert _format_time(0.0005) == "500us"
    assert _format_time(0.5) == "500ms"
    assert _format_time(5.0) == "5.0s"


def test_execute_single_smoke() -> None:
    result = _execute_single("gs4", 5, 2_000, 42, SolverConfig(), "random")
    assert result.solved


def test_execute_single_paley_ng_smoke() -> None:
    result = _execute_single("paley-ng", 52, 1, 42, SolverConfig(), "random")
    assert result.solved
    assert result.order == 208


def test_execute_single_construct_smoke() -> None:
    result = _execute_single("construct", 104, 1, 42, SolverConfig(), "random")
    assert result.solved
    assert result.order == 416
