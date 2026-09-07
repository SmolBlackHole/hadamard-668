"""Run the repository's development checks from one small CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Command = tuple[str, ...]
DOCS_CHECK: Command = (sys.executable, "scripts/check_docs.py")

COMMANDS: dict[str, tuple[Command, ...]] = {
    "check": (
        DOCS_CHECK,
        (sys.executable, "-m", "ruff", "format", "--check", "."),
        (sys.executable, "-m", "ruff", "check", "."),
        (sys.executable, "-m", "pyright", "--pythonpath", sys.executable),
        (sys.executable, "-m", "pytest", "-q", "--basetemp", "runs/pytest-quality"),
        (sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts", "lab", "run.py"),
    ),
    "docs": (DOCS_CHECK,),
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
    """Run commands sequentially and stop at the first failing exit status."""
    for command in commands:
        print(f"\n> {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected repository quality workflow.

    Args:
        argv: Optional argument vector. Uses process arguments when omitted.

    Returns:
        The first failing command status, or zero when all checks pass.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=COMMANDS,
        default="check",
        nargs="?",
        help="check (default), docs, fix, coverage, or mutate",
    )
    args = parser.parse_args(argv)
    if args.action == "check":
        (ROOT / "runs").mkdir(exist_ok=True)
    return _run(COMMANDS[args.action])


if __name__ == "__main__":
    raise SystemExit(main())
