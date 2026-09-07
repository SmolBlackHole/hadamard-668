"""Optional phase events and basis signatures."""

from __future__ import annotations

import numpy as np

from ..canonical import orbit_hash, validation_hash
from ..models import Int8Array, PhaseEvent, SearchPhase, SearchStats
from .config import SolverConfig


def _basis_minus(sequences: Int8Array) -> int:
    return int(np.count_nonzero(np.prod(sequences, axis=0, dtype=np.int8) < 0))


def record_phase(
    stats: SearchStats,
    config: SolverConfig,
    phase: SearchPhase,
    before: Int8Array,
    after: Int8Array,
    q_before: int,
    q_after: int,
    lowest_q: int,
    candidate_evals: int,
    accepted_moves: int,
    downhill_moves: int,
    lateral_moves: int,
    uphill_moves: int,
    outcome: str,
) -> None:
    """Append a complete phase event when tracing is enabled."""
    if not config.trace_phases:
        return
    stats.phase_events.append(
        PhaseEvent(
            phase=phase,
            q_before=q_before,
            q_after=q_after,
            lowest_q=lowest_q,
            candidate_evals=candidate_evals,
            accepted_moves=accepted_moves,
            downhill_moves=downhill_moves,
            lateral_moves=lateral_moves,
            uphill_moves=uphill_moves,
            basis_minus_before=_basis_minus(before),
            basis_minus_after=_basis_minus(after),
            state_hash_after=validation_hash(after),
            orbit_hash_after=orbit_hash(after),
            outcome=outcome,
        )
    )
