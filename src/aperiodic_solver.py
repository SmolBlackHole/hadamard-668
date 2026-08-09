"""Greedy and Tabu search for weighted aperiodic residual trackers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_tracker import BaseTracker


@dataclass(frozen=True)
class AperiodicSolverConfig:
    tabu: bool = True
    tabu_steps: int = 200
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.1


@dataclass
class AperiodicSearchStats:
    evaluations: int = 0
    greedy_accepts: int = 0
    tabu_walks: int = 0
    tabu_accepts: int = 0
    tabu_improvements: int = 0


@dataclass(frozen=True)
class AperiodicSearchResult:
    sequences: tuple[np.ndarray, ...]
    energy: int
    stats: AperiodicSearchStats


def search(
    tracker: BaseTracker,
    rng: np.random.Generator,
    *,
    steps: int,
    config: AperiodicSolverConfig | None = None,
) -> AperiodicSearchResult:
    """Minimize weighted aperiodic Q within a candidate-evaluation budget."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    cfg = config if config is not None else AperiodicSolverConfig()
    stats = AperiodicSearchStats()
    sequence_ids, columns = tracker.positions()
    candidate_count = tracker.candidate_count()

    while tracker.energy() and stats.evaluations + candidate_count <= steps:
        while tracker.energy() and stats.evaluations + candidate_count <= steps:
            scores = tracker.flip_qs()
            stats.evaluations += candidate_count
            index = int(np.argmin(scores))
            if int(scores[index]) >= tracker.energy():
                break
            tracker.accept(int(sequence_ids[index]), int(columns[index]))
            stats.greedy_accepts += 1

        if not tracker.energy() or not cfg.tabu or stats.evaluations + candidate_count > steps:
            break

        remaining_walk_steps = (steps - stats.evaluations) // candidate_count
        walk_steps = min(cfg.tabu_steps, remaining_walk_steps)
        if walk_steps <= 0:
            break
        improved, used = _tabu_walk(
            tracker,
            rng,
            sequence_ids,
            columns,
            steps=walk_steps,
            config=cfg,
        )
        stats.evaluations += used * candidate_count
        stats.tabu_walks += 1
        stats.tabu_accepts += used
        stats.tabu_improvements += int(improved)

    return AperiodicSearchResult(tracker.sequences(), tracker.energy(), stats)


def _tabu_walk(
    tracker: BaseTracker,
    rng: np.random.Generator,
    sequence_ids: np.ndarray,
    columns: np.ndarray,
    *,
    steps: int,
    config: AperiodicSolverConfig,
) -> tuple[bool, int]:
    """Walk through worse states and restore the best exact snapshot."""
    start_q = tracker.energy()
    best_q = start_q
    best = tracker.snapshot()
    tabu = np.zeros(tracker.candidate_count(), dtype=np.float64)
    used = 0

    for _ in range(steps):
        used += 1
        scores = tracker.flip_qs()
        noise = rng.random(scores.size) * config.tabu_noise
        penalized = scores.astype(np.float64) * (1.0 + tabu + noise)
        index = int(np.argmin(penalized))
        q = tracker.accept(int(sequence_ids[index]), int(columns[index]))

        tabu *= config.tabu_decay
        tabu[index] = config.tabu_tenure
        if q < best_q:
            best_q = q
            best = tracker.snapshot()
            if q == 0:
                break

    tracker.restore(best)
    return best_q < start_q, used
