"""CLI integration tests — keep `--workers 1` to avoid process storms."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from run import derive_seeds


def test_run_help_works_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    src = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = src
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_derive_seeds() -> None:
    assert derive_seeds(40, 3) == [40, 41, 42]
