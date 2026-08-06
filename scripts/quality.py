"""Run the repository's development checks from one small CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Command = tuple[str, ...]

COMMANDS: dict[str, tuple[Command, ...]] = {
    "check": (
        (sys.executable, "-m", "ruff", "format", "--check", "."),
        (sys.executable, "-m", "ruff", "check", "."),
        (sys.executable, "-m", "pyright"),
        (sys.executable, "-m", "pytest", "-q"),
        (sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts", "run.py"),
    ),
    "fix": (
        (sys.executable, "-m", "ruff", "format", "."),
        (sys.executable, "-m", "ruff", "check", "--fix", "."),
    ),
    "coverage": (
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--cov=src",
            "--cov=run",
            "--cov-report=term-missing",
        ),
    ),
    "mutate": (
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_tracker.py",
            "tests/test_solver.py",
            "--gremlins",
        ),
    ),
}


def _run(commands: tuple[Command, ...]) -> int:
    for command in commands:
        print(f"\n> {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=COMMANDS,
        default="check",
        nargs="?",
        help="check (default), fix, coverage, or mutate",
    )
    args = parser.parse_args(argv)
    return _run(COMMANDS[args.action])


if __name__ == "__main__":
    raise SystemExit(main())
