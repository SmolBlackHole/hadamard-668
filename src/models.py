"""Shared data contracts for search, execution, persistence, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

Int8Array = npt.NDArray[np.int8]


@dataclass
class CandidateBudget:
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("candidate budget must be positive")

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def take(self, requested: int) -> int:
        granted = min(requested, self.remaining)
        self.used += granted
        return granted


class SearchPhase(StrEnum):
    INITIALIZE = "initialize"
    GREEDY = "greedy"
    TABU = "tabu"
    TARGETED = "targeted"
    RANDOM_KICK = "random_kick"


@dataclass(frozen=True)
class PhaseEvent:
    phase: SearchPhase
    q_before: int
    q_after: int
    lowest_q: int
    candidate_evals: int
    accepted_moves: int
    downhill_moves: int
    lateral_moves: int
    uphill_moves: int
    basis_minus_before: int
    basis_minus_after: int
    state_hash_after: str
    orbit_hash_after: str
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "q_before": self.q_before,
            "q_after": self.q_after,
            "lowest_q": self.lowest_q,
            "candidate_evals": self.candidate_evals,
            "accepted_moves": self.accepted_moves,
            "downhill_moves": self.downhill_moves,
            "lateral_moves": self.lateral_moves,
            "uphill_moves": self.uphill_moves,
            "basis_minus_before": self.basis_minus_before,
            "basis_minus_after": self.basis_minus_after,
            "state_hash_after": self.state_hash_after,
            "orbit_hash_after": self.orbit_hash_after,
            "outcome": self.outcome,
        }


@dataclass
class SearchStats:
    greedy_moves: int = 0
    tabu_moves: int = 0
    random_kicks: int = 0
    targeted_quenches: int = 0
    quench_moves: int = 0
    greedy_candidate_evals: int = 0
    tabu_candidate_evals: int = 0
    escape_candidate_evals: int = 0
    quench_candidate_evals: int = 0
    tabu_improvements: int = 0
    tabu_improvements_q1: int = 0
    tabu_improvements_q2: int = 0
    tabu_improvements_q3plus: int = 0
    tabu_solves: int = 0
    tabu_solves_q1: int = 0
    tabu_solves_q2: int = 0
    tabu_solves_q3plus: int = 0
    tabu_walks: int = 0
    tabu_walks_q1: int = 0
    tabu_walks_q2: int = 0
    tabu_walks_q3plus: int = 0
    greedy_streaks: list[int] = field(default_factory=list[int])
    greedy_energy_improvement: int = 0
    tabu_energy_improvement: int = 0
    escape_energy_improvement: int = 0
    greedy_time_s: float = 0.0
    escape_time_s: float = 0.0
    rebuild_time_s: float = 0.0
    tabu_time_s: float = 0.0
    solve_phase: str | None = None
    solve_q_before: int | None = None
    phase_events: list[PhaseEvent] = field(default_factory=list[PhaseEvent])
    _streak: int = 0

    @property
    def total_candidate_evals(self) -> int:
        return (
            self.greedy_candidate_evals
            + self.tabu_candidate_evals
            + self.escape_candidate_evals
            + self.quench_candidate_evals
        )

    @property
    def total_accepted_moves(self) -> int:
        """Accepted transitions across the top-level search and its quenches."""
        return (
            self.greedy_moves
            + self.tabu_moves
            + self.random_kicks
            + self.targeted_quenches
            + self.quench_moves
        )

    def record_solve(self, phase: str, q_before: int) -> None:
        if self.solve_phase is None:
            self.solve_phase = phase
            self.solve_q_before = q_before

    def hit_greedy(self) -> None:
        self.greedy_moves += 1
        self._streak += 1

    def hit_other(self) -> None:
        if self._streak:
            self.greedy_streaks.append(self._streak)
            self._streak = 0

    def flush(self) -> None:
        if self._streak:
            self.greedy_streaks.append(self._streak)
            self._streak = 0

    def to_dict(self) -> dict[str, object]:
        self.flush()
        return {
            "stats_schema_version": 2,
            "greedy_moves": self.greedy_moves,
            "tabu_moves": self.tabu_moves,
            "random_kicks": self.random_kicks,
            "targeted_quenches": self.targeted_quenches,
            "quench_moves": self.quench_moves,
            "total_accepted_moves": self.total_accepted_moves,
            "greedy_energy_improvement": self.greedy_energy_improvement,
            "tabu_energy_improvement": self.tabu_energy_improvement,
            "escape_energy_improvement": self.escape_energy_improvement,
            "greedy_candidate_evals": self.greedy_candidate_evals,
            "tabu_candidate_evals": self.tabu_candidate_evals,
            "escape_candidate_evals": self.escape_candidate_evals,
            "quench_candidate_evals": self.quench_candidate_evals,
            "total_candidate_evals": self.total_candidate_evals,
            "tabu_improvements": self.tabu_improvements,
            "tabu_walks": self.tabu_walks,
            "tabu_improvements_q1": self.tabu_improvements_q1,
            "tabu_improvements_q2": self.tabu_improvements_q2,
            "tabu_improvements_q3plus": self.tabu_improvements_q3plus,
            "tabu_walks_q1": self.tabu_walks_q1,
            "tabu_walks_q2": self.tabu_walks_q2,
            "tabu_walks_q3plus": self.tabu_walks_q3plus,
            "tabu_solves": self.tabu_solves,
            "tabu_solves_q1": self.tabu_solves_q1,
            "tabu_solves_q2": self.tabu_solves_q2,
            "tabu_solves_q3plus": self.tabu_solves_q3plus,
            "solve_phase": self.solve_phase,
            "solve_q_before": self.solve_q_before,
            "greedy_time_s": self.greedy_time_s,
            "escape_time_s": self.escape_time_s,
            "rebuild_time_s": self.rebuild_time_s,
            "tabu_time_s": self.tabu_time_s,
            "phase_events": [event.to_dict() for event in self.phase_events],
        }

    def display(self) -> str:
        self.flush()
        parts = [
            f"moves={self.total_accepted_moves}"
            f"({self.greedy_moves}/{self.tabu_moves}/{self.random_kicks}/"
            f"{self.targeted_quenches}/{self.quench_moves})",
            f"tabu={self.tabu_improvements}/{self.tabu_walks}",
            f"evals={self.total_candidate_evals}"
            f"({self.greedy_candidate_evals}/{self.tabu_candidate_evals}/"
            f"{self.escape_candidate_evals}/{self.quench_candidate_evals})",
            f"t={self.greedy_time_s:.1f}s/{self.tabu_time_s:.1f}s/{self.escape_time_s:.1f}s",
        ]
        if self.solve_phase is not None:
            parts.append(f"solve={self.solve_phase}@Q{self.solve_q_before}")
        if self.greedy_streaks:
            streaks = self.greedy_streaks
            parts.append(f"strk={sum(streaks) / len(streaks):.1f}/{max(streaks)}")
        else:
            parts.append("strk=-")
        return " ".join(parts)


@dataclass(frozen=True)
class SolverResult:
    sequences: Int8Array
    energy: int
    candidate_evals: int
    stats: SearchStats

    @property
    def solved(self) -> bool:
        return self.energy == 0


@dataclass(frozen=True)
class RunResult:
    strategy: str
    n: int
    seed: int
    sequences: Int8Array
    energy: int
    candidate_evals: int
    elapsed_seconds: float
    stats: SearchStats
    verified: bool
    construction: str = "unknown"
    candidate_budget: int = 0
    solver_config: dict[str, object] = field(default_factory=dict[str, object])

    @property
    def solved(self) -> bool:
        return self.energy == 0 and self.verified

    @property
    def order(self) -> int:
        return 4 * self.n

    def __str__(self) -> str:
        status = f"OK {self.order}x{self.order}" if self.solved else f"e={self.energy}"
        return f"seed={self.seed:<4} {status}  {self.elapsed_seconds:.1f}s  {self.stats.display()}"


@dataclass(frozen=True)
class AuditReport:
    checked: int
    valid: int
    quarantined: int = 0
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return (
            self.checked > 0 and self.valid + self.quarantined == self.checked and not self.failures
        )


@dataclass(frozen=True)
class MigrationReport:
    checked: int
    valid: int
    quarantined: int
