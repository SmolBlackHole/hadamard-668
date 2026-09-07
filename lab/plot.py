"""Render stored sweep cohorts without modifying the database or running searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure


def load_cohorts(database: Path, first_id: int, last_id: int) -> list[dict[str, Any]]:
    """Read one snapshot, partition by provenance, and retain latest repeated seeds.

    Parity is separate. Dirty revision labels cannot identify the exact source
    tree; the report preserves IDs so callers can select a known batch explicitly.
    Only records marked valid with a consistent integer GS4 energy are accepted.
    """
    groups: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        records = connection.execute(
            "SELECT id,n,seed,energy,solved,candidate_evals,candidate_budget,"
            "code_revision,construction,solver_config_json,strategy,valid FROM runs "
            "WHERE id BETWEEN ? AND ? ORDER BY id",
            (first_id, last_id),
        ).fetchall()
    for row in records:
        if row["valid"] != 1:
            continue
        n = row["n"]
        if n < 2 or row["energy"] < 0 or row["energy"] % (64 * n):
            raise ValueError(f"run {row['id']}: invalid GS4 energy")
        if bool(row["solved"]) != (row["energy"] == 0):
            raise ValueError(f"run {row['id']}: inconsistent solved flag")
        metadata = {
            key: row[key]
            for key in ("strategy", "code_revision", "construction", "candidate_budget")
        }
        metadata["config"] = json.loads(row["solver_config_json"] or "null")
        metadata["parity"] = "odd" if n % 2 else "even"
        key = json.dumps(metadata, sort_keys=True)
        group = groups.setdefault(key, {"metadata": metadata, "records": {}, "duplicates": 0})
        identity = (n, row["seed"])
        if identity in group["records"]:
            group["duplicates"] += 1
        group["records"][identity] = dict(row)
    result: list[dict[str, Any]] = []
    for key, group in groups.items():
        result.append(
            {
                "cohort": hashlib.sha256(key.encode()).hexdigest()[:12],
                "metadata": group["metadata"],
                "duplicates": group["duplicates"],
                "records": list(group["records"].values()),
            }
        )
    return result


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate best stored Q and effort, including unsuccessful runs."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["n"]].append(row)
    result: list[dict[str, Any]] = []
    for n, rows in sorted(grouped.items()):
        q = np.array([row["energy"] // (64 * n) for row in rows])
        effort = [row["candidate_evals"] for row in rows if row["candidate_evals"] is not None]
        result.append(
            {
                "n": n,
                "runs": len(rows),
                "solved": sum(row["solved"] for row in rows),
                "q_min": int(q.min()),
                "q_p10": float(np.quantile(q, 0.1)),
                "q_median": float(np.median(q)),
                "q_p90": float(np.quantile(q, 0.9)),
                "eval_median": float(np.median(effort)) if effort else None,
                "effort_records": len(effort),
            }
        )
    return result


def render(cohort: dict[str, Any], output: Path) -> None:
    """Save a four-panel PNG and SVG plus the underlying summary CSV."""
    records = cohort["records"]
    summary = summarize(records)
    metadata = cohort["metadata"]
    ns = [row["n"] for row in summary]
    x = np.arange(len(ns))
    figure = Figure(figsize=(13, 9), layout="constrained")
    success, residual, heat, effort = figure.subplots(2, 2).flat
    success.bar(x, [100 * row["solved"] / row["runs"] for row in summary], color="#167d8d")
    for i, row in enumerate(summary):
        success.text(
            i,
            100 * row["solved"] / row["runs"] + 2,
            f"{row['solved']}/{row['runs']}",
            ha="center",
            fontsize=7,
            rotation=45,
        )
    success.set(title="Verified solve yield", ylabel="Solved runs (%)", ylim=(0, 125))
    residual.fill_between(
        x,
        [r["q_p10"] for r in summary],
        [r["q_p90"] for r in summary],
        alpha=0.2,
        color="#167d8d",
        label="10th-90th percentile",
    )
    residual.plot(x, [r["q_median"] for r in summary], "o-", color="#167d8d", label="Median")
    residual.plot(x, [r["q_min"] for r in summary], ".--", color="#c05d27", label="Minimum")
    residual.set(title="Best Q reached per run", ylabel="Q = E / (64n)")
    residual.legend(fontsize=8)
    max_q = max(r["energy"] // (64 * r["n"]) for r in records)
    distribution = np.zeros((max_q + 1, len(ns)))
    columns = {n: i for i, n in enumerate(ns)}
    for row in records:
        distribution[row["energy"] // (64 * row["n"]), columns[row["n"]]] += 1
    distribution /= np.array([r["runs"] for r in summary])
    if len(ns) == 1:
        heat.bar(np.arange(max_q + 1), distribution[:, 0], color="#167d8d")
        heat.set(title="Distribution of best Q (all runs)", xlabel="Q", ylabel="Fraction of runs")
        heat.set_xlim(min(r["q_min"] for r in summary) - 1, max_q + 1)
    else:
        picture = heat.imshow(
            distribution, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1
        )
        heat.set(title="Distribution of best Q (all runs)", ylabel="Q")
        # Matplotlib leaves these public methods' **kwargs untyped.
        figure.colorbar(picture, ax=heat, label="Fraction of runs")  # pyright: ignore[reportUnknownMemberType]
        heat.set_xticks(x, [str(n) for n in ns])
        heat.set_xlabel("Sequence length n")
    effort.bar(
        x,
        [r["eval_median"] / 1e6 if r["eval_median"] is not None else np.nan for r in summary],
        color="#c05d27",
    )
    effort.set(title="Median work, including failures", ylabel="Candidate evaluations (millions)")
    for axis in (success, residual, effort):
        axis.set_xticks(x, [str(n) for n in ns])
        axis.set_xlabel("Sequence length n")
    revision = str(metadata["code_revision"])
    figure.suptitle(  # pyright: ignore[reportUnknownMemberType]
        f"{metadata['strategy']} / {metadata['construction']} / {metadata['parity']} lengths\n"
        f"Revision {revision[:8]}{'-dirty' if revision.endswith('-dirty') else ''} | "
        f"budget {metadata['candidate_budget']} | cohort {cohort['cohort']}",
        fontsize=14,
    )
    prefix = output / cohort["cohort"]
    for extension in ("png", "svg"):
        figure.savefig(prefix.with_suffix("." + extension), dpi=160)  # pyright: ignore[reportUnknownMemberType]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main(argv: Sequence[str] | None = None) -> None:
    """Plot all provenance cohorts in a selected database ID range."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/hadamard.db"))
    parser.add_argument("--output", type=Path, default=Path("runs/plots"))
    parser.add_argument("--first-id", type=int, default=1)
    parser.add_argument("--last-id", type=int, default=2**63 - 1)
    args = parser.parse_args(argv)
    if not args.database.is_file():
        parser.error(f"database does not exist: {args.database}")
    if args.first_id < 1 or args.last_id < args.first_id:
        parser.error("require 1 <= first-id <= last-id")
    cohorts = load_cohorts(args.database, args.first_id, args.last_id)
    if not cohorts:
        parser.error("no validated runs in the selected range")
    args.output.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        render(cohort, args.output)
        print(
            f"{cohort['cohort']}: {len(cohort['records'])} runs, "
            f"{cohort['duplicates']} repeated seeds omitted"
        )
    manifest = {
        "database": str(args.database.resolve()),
        "first_id": args.first_id,
        "last_id": args.last_id,
        "cohorts": cohorts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index = [
        "# Sweep plots",
        "",
        "Database snapshot: " + str(args.database.resolve()),
        "",
        "Only records marked valid are included. Q is the best stored value, not a trace.",
        "Repeated (n, seed) entries within a cohort retain the latest ID.",
        "Dirty revision labels do not identify exact source changes; use ID filters for batches.",
        "",
    ]
    for cohort in cohorts:
        name = cohort["cohort"]
        rows = cohort["records"]
        lengths = sorted({row["n"] for row in rows})
        index.extend(
            [
                f"## {name}",
                "",
                f"n: {lengths}; {len(rows)} retained runs; "
                f"{cohort['duplicates']} repeated seeds omitted.",
                "",
                f"[SVG]({name}.svg) | [CSV]({name}.csv)",
                "",
                f"![Sweep overview]({name}.png)",
                "",
            ]
        )
    (args.output / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(f"PNG, SVG, CSV and provenance: {args.output}")


def plot_comparison(path: Path) -> None:
    """Plot a paired comparison report beside its retained raw JSON."""
    report = json.loads(path.read_text(encoding="utf-8"))
    figure = Figure(figsize=(11, 5), layout="constrained")
    yield_axis, q_axis = figure.subplots(1, 2)
    for name, groups in report["aggregates"].items():
        lengths = sorted(map(int, groups))
        yield_axis.plot(lengths, [groups[str(n)]["solve_rate"] for n in lengths], "o-", label=name)
        q_axis.plot(
            lengths, [groups[str(n)]["mean_energy"] / (64 * n) for n in lengths], "o-", label=name
        )
    yield_axis.set(title="Paired solve yield", ylabel="Solved fraction", ylim=(0, 1.05))
    q_axis.set(title="Mean best Q, including failures", ylabel="Q")
    for axis in (yield_axis, q_axis):
        axis.set_xlabel("Sequence length n")
        axis.legend()
    for suffix in ("png", "svg"):
        figure.savefig(path.with_suffix("." + suffix), dpi=160)  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    main()
