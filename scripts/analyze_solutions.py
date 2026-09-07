"""Canonicalize and analyze the verified GS4 solution catalog."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.solution_analysis import analyze_solution_database


def main(argv: Sequence[str] | None = None) -> None:
    """Build a canonical solution catalog from an existing database.

    Args:
        argv: Optional argument vector. Uses process arguments when omitted.

    Note:
        The command fails before analysis when the input database is missing,
        so it cannot replace a catalog with an accidental empty result.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--output", type=Path, default=Path("runs/solution-catalog.json"))
    args = parser.parse_args(argv)
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")
    result = analyze_solution_database(args.database, args.output)
    print(f"{result['valid_solutions']} valid solutions, {result['quick_orbits']} quick orbits")
    print(f"Catalog: {args.output}")


if __name__ == "__main__":
    main()
