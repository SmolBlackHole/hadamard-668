"""Shared data contracts for search, execution, persistence, and verification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

Int8Array = npt.NDArray[np.int8]


def _format_energy(energy: int) -> str:
    if energy >= 1_000_000:
        return f"{energy / 1_000_000:.1f}M"
    if energy >= 1_000:
        return f"{energy / 1_000:.0f}k"
    return str(energy)


@dataclass
class SearchStats:
    singles: int = 0
    kicks: int = 0
    single_evals: int = 0
    kick_evals: int = 0
    tabu_evals: int = 0
    tabu_candidate_evals: int = 0
    quench_candidate_evals: int = 0
    target_quenches: int = 0
    tabu_hits: int = 0
    tabu_hits_q1: int = 0
    tabu_hits_q2: int = 0
    tabu_hits_q3plus: int = 0
    tabu_solves: int = 0
    tabu_solves_q1: int = 0
    tabu_solves_q2: int = 0
    tabu_solves_q3plus: int = 0
    tabu_walks: int = 0
    tabu_walks_q1: int = 0
    tabu_walks_q2: int = 0
    tabu_walks_q3plus: int = 0
    singles_streaks: list[int] = field(default_factory=list[int])
    energy_saved_singles: int = 0
    energy_saved_kicks: int = 0
    energy_saved_tabu: int = 0
    single_time_s: float = 0.0
    kick_time_s: float = 0.0
    rebuild_time_s: float = 0.0
    tabu_time_s: float = 0.0
    solve_phase: str | None = None
    solve_q_before: int | None = None
    _streak: int = 0

    @property
    def total_candidate_evals(self) -> int:
        return (
            self.single_evals
            + self.tabu_candidate_evals
            + self.kick_evals
            + self.quench_candidate_evals
        )

    def record_solve(self, phase: str, q_before: int) -> None:
        if self.solve_phase is None:
            self.solve_phase = phase
            self.solve_q_before = q_before

    def hit_single(self) -> None:
        self.singles += 1
        self._streak += 1

    def hit_other(self) -> None:
        if self._streak:
            self.singles_streaks.append(self._streak)
            self._streak = 0

    def flush(self) -> None:
        if self._streak:
            self.singles_streaks.append(self._streak)
            self._streak = 0

    def to_dict(self) -> dict[str, object]:
        self.flush()
        return {
            "singles": self.singles,
            "kicks": self.kicks,
            "e_singles": self.energy_saved_singles,
            "e_kicks": self.energy_saved_kicks,
            "e_tabu": self.energy_saved_tabu,
            "single_evals": self.single_evals,
            "kick_evals": self.kick_evals,
            "tabu_evals": self.tabu_evals,
            "tabu_candidate_evals": self.tabu_candidate_evals,
            "quench_candidate_evals": self.quench_candidate_evals,
            "total_candidate_evals": self.total_candidate_evals,
            "target_quenches": self.target_quenches,
            "tabu_hits": self.tabu_hits,
            "tabu_walks": self.tabu_walks,
            "tabu_hits_q1": self.tabu_hits_q1,
            "tabu_hits_q2": self.tabu_hits_q2,
            "tabu_hits_q3plus": self.tabu_hits_q3plus,
            "tabu_walks_q1": self.tabu_walks_q1,
            "tabu_walks_q2": self.tabu_walks_q2,
            "tabu_walks_q3plus": self.tabu_walks_q3plus,
            "tabu_solves": self.tabu_solves,
            "tabu_solves_q1": self.tabu_solves_q1,
            "tabu_solves_q2": self.tabu_solves_q2,
            "tabu_solves_q3plus": self.tabu_solves_q3plus,
            "solve_phase": self.solve_phase,
            "solve_q_before": self.solve_q_before,
            "single_time_s": self.single_time_s,
            "kick_time_s": self.kick_time_s,
            "rebuild_time_s": self.rebuild_time_s,
            "tabu_time_s": self.tabu_time_s,
        }

    def display(self) -> str:
        self.flush()
        parts = [
            f"S={self.singles}({_format_energy(self.energy_saved_singles)})",
            f"TB={self.tabu_hits}/{self.tabu_walks}({_format_energy(self.energy_saved_tabu)})",
            f"K={self.kicks}({_format_energy(self.energy_saved_kicks)})",
            f"evals={self.single_evals}/{self.tabu_candidate_evals}/{self.kick_evals}",
            f"t={self.single_time_s:.1f}s/{self.tabu_time_s:.1f}s/{self.kick_time_s:.1f}s",
        ]
        if self.singles_streaks:
            streaks = self.singles_streaks
            parts.append(f"strk={sum(streaks) / len(streaks):.1f}/{max(streaks)}")
        else:
            parts.append("strk=-")
        return " ".join(parts)


@dataclass(frozen=True)
class SolverResult:
    sequences: Int8Array
    energy: int
    steps: int
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
    steps: int
    elapsed_seconds: float
    stats: SearchStats
    verified: bool

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
    failures: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.checked > 0 and self.valid == self.checked
