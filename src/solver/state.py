"""Working-state ownership and typed phase results for GS4 search."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import Int8Array
from ..tracker import Tracker


@dataclass(frozen=True)
class TabuWalkResult:
    """Summarize a Tabu walk and its adopted best state.

    ``energy`` is ``None`` when the walk adopted no state. ``steps``
    counts executed walk transitions, not candidate evaluations; each step
    evaluates all ``4n`` single flips.

    Attributes:
        energy: Adopted best energy, possibly equal to the start, or ``None``.
        steps: Number of executed walk transitions.
        solve_q_before: ``Q`` immediately before a solving transition.
        lowest_q: Lowest ``Q`` visited by the walk.
        downhill_moves: Transitions that decreased ``Q``.
        lateral_moves: Transitions that preserved ``Q``.
        uphill_moves: Transitions that increased ``Q``.
    """

    energy: int | None
    steps: int
    solve_q_before: int | None = None
    lowest_q: int | None = None
    downhill_moves: int = 0
    lateral_moves: int = 0
    uphill_moves: int = 0


@dataclass(frozen=True)
class TargetedKickResult:
    """Summarize targeted quenches and the updated outer best state.

    Attributes:
        solved: Whether a nested quench reached zero energy.
        sequences: Solving state, or the unchanged outer current state.
        energy: Energy of ``sequences``.
        best_sequences: Best state known after all attempted quenches.
        best_energy: Energy of ``best_sequences``.
        accepted_moves: Targeted starts plus accepted nested-search moves.
    """

    solved: bool
    sequences: Int8Array
    energy: int
    best_sequences: Int8Array
    best_energy: int
    accepted_moves: int


@dataclass
class SearchState:
    """Own the working sequences, tracker and cached energy for one search."""

    sequences: Int8Array
    tracker: Tracker
    energy: int = 0
