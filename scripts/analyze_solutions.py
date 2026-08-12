"""Canonicalize and analyze the verified GS4 solution catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.solution_analysis import analyze_solution_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--output", type=Path, default=Path("data/gs4_solution_catalog_v1.json"))
    args = parser.parse_args()
    result = analyze_solution_database(args.database, args.output)
    print(f"{result['valid_solutions']} valid solutions, {result['quick_orbits']} quick orbits")
    print(f"Catalog: {args.output}")


if __name__ == "__main__":
    main()
