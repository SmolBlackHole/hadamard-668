"""Validated settings for GS4 search phases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """Configure the iterated local search and its escape phases.

    Attributes:
        kick: Enable targeted and random escape phases after a stalled descent.
        tabu: Enable a Tabu walk after a stalled or low-``Q`` descent.
        tabu_steps: Maximum transitions in one Tabu walk.
        tabu_tenure: Initial soft penalty assigned to the accepted flip.
        tabu_decay: Multiplicative penalty decay applied after every step.
        tabu_noise: Scale of random multiplicative score noise.
        geo_weight: Optional weight for singleton-delta norm during greedy scans.
        targeted_escape: Allow one targeted escape per outer search.
        targeted_selection: Proposal selection: ``legacy`` first columns,
            ``random`` sampled columns, or ``residual`` sampled columns ranked
            by full post-flip ``Q``. Ranking consumes candidate evaluations.
        escape_quench_budget: Candidate limit for each nested targeted quench.
        qwindow_high: Empirical ``Q`` threshold for handing a greedy improvement
            directly to Tabu or escape handling. Zero disables the handoff.
        trace_phases: Record detailed state and cost events for every phase.
        tabu_accept_equal: Continue from a changed equal-best Tabu snapshot.
    """

    kick: bool = True
    tabu: bool = True
    tabu_steps: int = 400
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.2
    geo_weight: float = 0.0
    targeted_escape: bool = True
    escape_quench_budget: int = 10_000_000
    qwindow_high: int = 9  # empirical handoff from greedy to escape phases
    trace_phases: bool = False
    targeted_selection: str = "legacy"
    tabu_accept_equal: bool = False

    def __post_init__(self) -> None:
        if self.tabu_steps < 0:
            raise ValueError("tabu_steps cannot be negative")
        if self.tabu_tenure < 0 or not 0 <= self.tabu_decay <= 1:
            raise ValueError("tabu_tenure must be non-negative and tabu_decay must be in [0, 1]")
        if self.tabu_noise < 0 or self.geo_weight < 0:
            raise ValueError("tabu_noise and geo_weight cannot be negative")
        if self.escape_quench_budget <= 0:
            raise ValueError("escape_quench_budget must be positive")
        if self.qwindow_high < 0:
            raise ValueError("qwindow_high cannot be negative")
        if self.targeted_selection not in {"legacy", "random", "residual"}:
            raise ValueError("targeted_selection must be legacy, random, or residual")
