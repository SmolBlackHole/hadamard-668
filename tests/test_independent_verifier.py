"""Reference checks that deliberately avoid the solver construction path."""
from __future__ import annotations

import numpy as np
import pytest
from typing import Sequence, cast

from strategies.greedy import TurynGreedySearch
from fixtures import sylvester
from verifier.verify import (
    InvalidMatrix,
    build_turyn_reference,
    independent_audit,
    verify_goethals_seidel_sequences,
    verify_turyn_candidate,
    verify_turyn_sequences,
)


TT8 = (
    (1, 1, -1, 1, -1, 1, -1, 1),
    (1, -1, -1, -1, -1, -1, -1, 1),
    (1, -1, -1, 1, 1, 1, 1, -1),
    (1, 1, 1, -1, 1, 1, -1),
)
TT36_HEX = "060989975b685d8fc80750b21c0212eceb26"


def _tt36() -> tuple[list[int], list[int], list[int], list[int]]:
    sequences = [[], [], [], []]
    for digit in TT36_HEX[:-1]:
        for index, bit in enumerate(f"{int(digit, 16):04b}"):
            sequences[index].append(1 if bit == "0" else -1)
    for index, bit in enumerate(f"{int(TT36_HEX[-1], 16):03b}"):
        sequences[index].append(1 if bit == "0" else -1)
    return cast(tuple[list[int], list[int], list[int], list[int]], tuple(sequences))


def test_known_small_hadamard_passes_pure_python_audit() -> None:
    independent_audit(sylvester(4).tolist())


@pytest.mark.parametrize("sequences,order", [(TT8, 92), (_tt36(), 428)])
def test_known_turyn_sequences_pass_every_independent_layer(
    sequences: tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]], order: int,
) -> None:
    verify_turyn_sequences(sequences)
    matrix = verify_turyn_candidate(sequences)
    assert len(matrix) == order
    assert all(len(row) == order for row in matrix)
    assert all(value in (-1, 1) for row in matrix for value in row)


def test_bit_flip_breaks_the_matrix_audit() -> None:
    matrix = verify_turyn_candidate(TT8)
    matrix[0][0] *= -1
    with pytest.raises(InvalidMatrix, match="audit failed"):
        independent_audit(matrix)


def test_random_turyn_sequences_fail_the_sequence_condition() -> None:
    random_sequences = np.random.default_rng(42).choice(
        (-1, 1), size=(4, 8)).tolist()
    with pytest.raises(InvalidMatrix, match="autocorrelation"):
        verify_turyn_sequences(
            (random_sequences[0], random_sequences[1], random_sequences[2], random_sequences[3][:-1]))


def test_solver_builder_agrees_with_the_reference_construction() -> None:
    strategy = TurynGreedySearch(n=8, sieve=False)
    sequences = strategy._seed(np.random.default_rng(3))
    compact = tuple(sequences[row, :length].tolist()
                    for row, length in enumerate(strategy.LENGTHS))
    solver_matrix, solver_metrics = strategy._build(sequences)
    reference_matrix = build_turyn_reference(compact)
    assert np.array_equal(solver_matrix, np.asarray(reference_matrix, dtype=np.int8))
    if solver_metrics["energy"] == 0:
        independent_audit(reference_matrix)
    else:
        with pytest.raises(InvalidMatrix):
            independent_audit(reference_matrix)
    verify_goethals_seidel_sequences((
        (1, 1, 1, -1), (1, -1, 1, 1),
        (1, 1, -1, 1), (1, -1, -1, -1),
    ))
