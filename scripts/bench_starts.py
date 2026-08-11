"""Benchmark: random vs cyclic starts (SQLite-backed, worker-safe)."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

PROJ = Path(__file__).resolve().parent.parent


class SweepEntry(TypedDict):
    seed: int
    solved: bool
    energy: int
    elapsed_s: float


def run_sweep(
    n: int,
    seeds: int,
    steps: int,
    extra_args: Sequence[str],
    label: str,
) -> list[SweepEntry]:
    out = PROJ / f"data/_bench_{n}_{label}.db"
    try:
        cmd: list[str] = [
            sys.executable,
            str(PROJ / "run.py"),
            "--sweep",
            "gs4",
            str(n),
            "--seeds",
            str(seeds),
            "--steps",
            str(steps),
            "--workers",
            "8",
            "--output",
            str(out),
            "--no-verify",
            *extra_args,
        ]
        subprocess.run(cmd, check=True, cwd=PROJ)
        con = sqlite3.connect(out)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT seed, solved, energy, elapsed_s FROM runs "
            "WHERE strategy='gs4' AND n=? ORDER BY seed",
            (n,),
        ).fetchall()
        con.close()
        entries = [
            SweepEntry(
                seed=int(row["seed"]),
                solved=bool(row["solved"]),
                energy=int(row["energy"]),
                elapsed_s=float(row["elapsed_s"]),
            )
            for row in rows
        ]
        solved = sum(entry["solved"] for entry in entries)
        best_e = min((entry["energy"] for entry in entries), default=999999)
        best_q = best_e // (64 * n)
        t_total = sum(entry["elapsed_s"] for entry in entries)
        q_vals = [entry["energy"] // (64 * n) for entry in entries]
        print(
            f"  {label}: n={n} {solved}/{len(entries)} solved "
            f"best Q={best_q} Q-dist={sorted(q_vals)[:5]}... {t_total:.1f}s"
        )
        return entries
    finally:
        Path(out).unlink(missing_ok=True)
        Path(f"{out}-wal").unlink(missing_ok=True)
        Path(f"{out}-shm").unlink(missing_ok=True)


if __name__ == "__main__":
    for n, seeds, steps in [(44, 30, 200000)]:
        print(f"\n=== n={n} ===")
        run_sweep(n, seeds, steps, ["--no-targeted-escape"], "random")
        run_sweep(n, seeds, steps, ["--start-kind", "cyclic", "--no-targeted-escape"], "cyclic")
