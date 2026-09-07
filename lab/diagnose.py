"""Exact single- and two-flip diagnosis for persisted GS4 Q=1 endpoints."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.neighborhoods import repair_witnesses
from src.output import load_runs
from src.tracker import Tracker

Int8Array = npt.NDArray[np.int8]


class PairDiagnosis(TypedDict):
    """Exact two-flip repair metrics for one Q=1 endpoint."""

    best_pair_q: int
    best_pair: list[list[int]]
    n_solving_pairs: int
    n_solving_same_sequence: int
    n_solving_same_column: int
    pair_count: int
    pair_scan_seconds: float


def _position(index: int, n: int) -> tuple[int, int]:
    return divmod(index, n)


def _support_histogram(delta: npt.NDArray[np.int8]) -> dict[str, int]:
    values, counts = np.unique(np.count_nonzero(delta, axis=1), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def pair_diagnosis(sequences: Int8Array) -> PairDiagnosis:
    """Exhaustively score unordered two-flip repairs without changing state.

    The tracker is mutated for each first flip and restored before the next
    iteration. A fresh rebuild validates the best pair before returning.
    """
    n = sequences.shape[1]
    scale = 64 * n
    work = sequences.copy()
    tracker = Tracker()
    tracker.build(work)
    initial_energy = tracker.energy()
    candidate_count = 4 * n
    best_q = 2**63 - 1
    best_pair = (-1, -1)
    solving_pairs = 0
    solving_same_sequence = 0
    solving_same_column = 0
    started = time.perf_counter()

    for first in range(candidate_count - 1):
        first_position = _position(first, n)
        tracker.accept(work, *first_position)
        pair_qs = tracker.flip_qs()[first + 1 :]
        local_offset = int(np.argmin(pair_qs))
        local_q = int(pair_qs[local_offset])
        second = first + 1 + local_offset
        if local_q < best_q:
            best_q = local_q
            best_pair = first, second

        zero_offsets = np.flatnonzero(pair_qs == 0)
        solving_pairs += int(zero_offsets.size)
        for offset in zero_offsets:
            second_position = _position(first + 1 + int(offset), n)
            solving_same_sequence += int(first_position[0] == second_position[0])
            solving_same_column += int(first_position[1] == second_position[1])
        tracker.accept(work, *first_position)

    if tracker.energy() != initial_energy or not np.array_equal(work, sequences):
        raise RuntimeError("two-flip scan did not restore the tracker state")

    positions = [_position(index, n) for index in best_pair]
    combo_q = tracker.combo_energy(positions) // scale
    flipped = sequences.copy()
    for sequence, column in positions:
        flipped[sequence, column] *= -1
    rebuilt = Tracker()
    rebuilt.build(flipped)
    rebuilt_q = rebuilt.energy() // scale
    if combo_q != best_q or rebuilt_q != best_q:
        raise RuntimeError("two-flip score disagrees with combo_energy or fresh rebuild")

    return {
        "best_pair_q": best_q,
        "best_pair": [list(position) for position in positions],
        "n_solving_pairs": solving_pairs,
        "n_solving_same_sequence": solving_same_sequence,
        "n_solving_same_column": solving_same_column,
        "pair_count": candidate_count * (candidate_count - 1) // 2,
        "pair_scan_seconds": time.perf_counter() - started,
    }


def analyze_q1_state(sequences: npt.ArrayLike) -> dict[str, object]:
    """Diagnose exact one- and two-flip repair structure at Q=1.

    Args:
        sequences: Four binary sequences with shape ``(4, n)``.

    Returns:
        JSON-serializable residual, flip-support, and repair-depth metrics.

    Raises:
        ValueError: If the input is not a binary GS4 state at Q=1.
        RuntimeError: If tracker identities or repair scores disagree.
    """
    array = np.asarray(sequences, dtype=np.int8)
    if array.ndim != 2 or array.shape[0] != 4 or np.any(np.abs(array) != 1):
        raise ValueError("expected a binary GS4 state with shape (4, n)")
    n = array.shape[1]
    tracker = Tracker()
    tracker.build(array)
    if tracker._q != 1 or tracker._u is None or tracker._delta is None:
        raise ValueError("state is not at Q=1")

    u = tracker._u.copy()
    delta = tracker._delta.copy()
    active = np.flatnonzero(u)
    if active.size != 1 or abs(int(u[active[0]])) != 1:
        raise RuntimeError("Q=1 residual is not a signed unit vector")
    active_index = int(active[0])
    sigma = int(u[active_index])
    target = -u
    exact = np.flatnonzero(np.all(delta == target, axis=1))
    single_qs = tracker.flip_qs()
    solving_singles = np.flatnonzero(single_qs == 0)
    if not np.array_equal(exact, solving_singles):
        raise RuntimeError("delta dictionary and flip_qs disagree on solving singles")

    supports = np.count_nonzero(delta, axis=1)
    pair = pair_diagnosis(array)
    repair_depth: int | str
    if exact.size:
        repair_depth = 1
    elif pair["n_solving_pairs"] > 0:
        repair_depth = 2
    else:
        repair_depth = ">2"

    beta = np.prod(array.astype(np.int16), axis=0)
    return {
        "n": n,
        "q": 1,
        "u": u.tolist(),
        "active_lag": active_index + 1,
        "sigma": sigma,
        "target_delta": target.tolist(),
        "target_delta_present": bool(exact.size),
        "min_delta_support": int(supports.min()),
        "delta_support_histogram": _support_histogram(delta),
        "n_unit_support": int(np.count_nonzero(supports == 1)),
        "n_exact_single": int(exact.size),
        "exact_single_positions": [list(_position(int(index), n)) for index in exact],
        "n_right_sign_at_active_lag": int(np.count_nonzero(delta[:, active_index] == -sigma)),
        "best_single_q": int(single_qs.min()),
        "n_neutral_singles": int(np.count_nonzero(single_qs == 1)),
        "single_flip_local_minimum": bool(np.all(single_qs >= 1)),
        "two_flip_local_minimum": pair["best_pair_q"] >= 1,
        "n_basis1": int(np.count_nonzero(beta == -1)),
        "repair_depth": repair_depth,
        **pair,
    }


def diagnose_database(
    db_path: Path,
    *,
    strategy: str = "gs4",
    selected_n: set[int] | None = None,
    extended: bool = False,
) -> dict[str, object]:
    """Diagnose unsolved Q=1 endpoints stored in a result database.

    Args:
        db_path: SQLite result database to read.
        strategy: Strategy group selected from the persisted dataset.
        selected_n: Optional sequence lengths to include.
        extended: Also exhaust triples and one-per-sequence quadruple repairs.

    Returns:
        JSON-serializable per-length summaries and individual Q=1 analyses.

    Raises:
        ValueError: If a stored solver energy violates the GS4 scale.
    """
    dataset = load_runs(db_path)
    groups = dataset.get(strategy, {})
    results: list[dict[str, object]] = []
    summary: dict[str, object] = {}

    for n_text, entries in sorted(groups.items(), key=lambda item: int(item[0])):
        n = int(n_text)
        if selected_n is not None and n not in selected_n:
            continue
        q_hist: Counter[int] = Counter()
        solved = 0
        q1_results: list[dict[str, object]] = []
        for entry in entries:
            energy = int(entry["solver_e"])
            if energy < 0 or energy % (64 * n):
                raise ValueError(f"invalid solver energy for n={n}, seed={entry['seed']}: {energy}")
            q = energy // (64 * n)
            q_hist[q] += 1
            solved += int(bool(entry["solved"]))
            if q != 1 or bool(entry["solved"]):
                continue
            raw = base64.b64decode(str(entry["seqs_b64"]))
            sequences = np.frombuffer(raw, dtype=np.int8).reshape(4, n)
            analysis = analyze_q1_state(sequences)
            if extended:
                analysis["extended_repairs"] = repair_witnesses(sequences)
            analysis.update(seed=int(entry["seed"]), energy=energy)
            q1_results.append(analysis)
            results.append(analysis)

        depth_hist = Counter(str(result["repair_depth"]) for result in q1_results)
        summary[n_text] = {
            "runs": len(entries),
            "solved": solved,
            "q_histogram": {str(q): count for q, count in sorted(q_hist.items())},
            "q1_endpoints": len(q1_results),
            "missing_target_delta": sum(
                not bool(result["target_delta_present"]) for result in q1_results
            ),
            "one_flip_local_minima": sum(
                bool(result["single_flip_local_minimum"]) for result in q1_results
            ),
            "two_flip_local_minima": sum(
                bool(result["two_flip_local_minimum"]) for result in q1_results
            ),
            "repair_depth_histogram": dict(sorted(depth_hist.items())),
            "min_delta_support_histogram": dict(
                sorted(Counter(str(result["min_delta_support"]) for result in q1_results).items())
            ),
            "best_single_q_histogram": dict(
                sorted(Counter(str(result["best_single_q"]) for result in q1_results).items())
            ),
            "active_lag_histogram": dict(
                sorted(Counter(str(result["active_lag"]) for result in q1_results).items())
            ),
            "best_pair_q_histogram": dict(
                sorted(Counter(str(result["best_pair_q"]) for result in q1_results).items())
            ),
        }

    return {
        "database": str(db_path),
        "strategy": strategy,
        "summary": summary,
        "q1_states": results,
    }


def print_summary(report: dict[str, object]) -> None:
    """Print one compact line for each sequence length in a diagnosis report.

    Args:
        report: Report returned by :func:`diagnose_database`.

    Raises:
        TypeError: If the report does not contain a mapping summary.
    """
    summary_value = report["summary"]
    if not isinstance(summary_value, dict):
        raise TypeError("invalid diagnosis summary")
    summary = cast(dict[str, dict[str, object]], summary_value)
    for n, raw in summary.items():
        print(
            f"n={n}: runs={raw['runs']} solved={raw['solved']} "
            f"Q1={raw['q1_endpoints']} missing_d={raw['missing_target_delta']} "
            f"depths={raw['repair_depth_histogram']} "
            f"best_pair_Q={raw['best_pair_q_histogram']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run database diagnosis and write the JSON report.

    Args:
        argv: Optional argument vector. Uses process arguments when omitted.

    Returns:
        Zero after successfully writing the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--output", type=Path, default=Path("runs/q1_diagnosis.json"))
    parser.add_argument("--strategy", default="gs4")
    parser.add_argument("--n", type=int, nargs="*")
    parser.add_argument(
        "--extended", action="store_true", help="Also test triples and restricted quadruples."
    )
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"database not found: {args.input}")
    selected_n = set(args.n) if args.n else None
    report = diagnose_database(
        args.input, strategy=args.strategy, selected_n=selected_n, extended=args.extended
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"Saved Q=1 diagnosis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
