"""Property tests for metrics, gram primitives, seed energy, and output."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_constructions_helper import paley, sylvester

from energy import npaf_energy_exact
from gpu import Metrics, check_orthogonality, gram_matrix, metrics_from_gram
from output import save
from strategies.base import Pipeline, Result, SearchStrategy


def _reference_metrics(matrix: np.ndarray) -> Metrics:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return Metrics(
        energy=int(np.dot(upper, upper)),
        orthogonal_pairs=int(np.count_nonzero(upper == 0)),
        max_abs_correlation=int(np.abs(upper).max(initial=0)),
    )


# ── GPU/metrics tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "matrix",
    [
        sylvester(4),
        sylvester(8),
        paley(12),
        np.array([[1, 1], [1, -1]], dtype=np.int8),
        np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int8),
    ],
)
def test_metrics_match_independent_reference(matrix):
    assert check_orthogonality(matrix) == _reference_metrics(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        sylvester(4),
        paley(12),
        np.ones((3, 3), dtype=np.int8),
    ],
)
def test_gram_primitives_match_independent_reference(matrix):
    gram = gram_matrix(matrix)
    assert metrics_from_gram(gram) == _reference_metrics(matrix)


# ── Seed energy test ──────────────────────────────────────────────────────────


def test_seed_has_nonzero_energy():
    rng = np.random.default_rng(42)
    seqs = np.zeros((4, 8), dtype=np.int8)
    for i, L in enumerate([8, 8, 8, 7]):
        seqs[i, :L] = rng.choice((-1, 1), size=L).astype(np.int8)
    lengths = np.array((8, 8, 8, 7), dtype=np.int64)
    weights = np.array((1.0, 1.0, 2.0, 2.0), dtype=np.float64)
    energy = npaf_energy_exact(seqs, lengths, weights)
    assert energy > 0


# ── Pipeline test ─────────────────────────────────────────────────────────────


def test_pipeline_passes_matrix_and_incremented_seed():
    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "source"

        def search(self, steps, seed):
            return Result(
                matrix=np.array([[1, 1], [1, -1]], dtype=np.int8),
                metrics=Metrics(1, 0, 2),
                elapsed=0.1,
                seed=seed,
            )

    class Sink(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "sink"

        def search(self, steps, seed):
            raise AssertionError

        def refine(self, matrix, steps, seed, sequences=None):
            return Result(
                matrix=matrix,
                metrics=Metrics(1, 0, 2),
                elapsed=0.2,
                seed=seed,
            )

    r = Pipeline([(Source(), 3), (Sink(), 5)]).search(steps=0, seed=20)
    assert r.metrics.energy == 1


# ── Output tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("matrix", [sylvester(4), paley(12)])
def test_output_saves_and_audits(matrix, tmp_path):
    m = check_orthogonality(matrix)
    sha = save(
        matrix, m, tmp_path,
        strategy="test", seed=0, steps=1, wall=0.0,
        order=len(matrix), construction="test",
    )
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["is_solution"] is True
    assert info["sha256"] == sha
    assert "matrix.csv" in [p.name for p in tmp_path.iterdir()]


def test_output_rejects_an_invalid_claimed_solution(tmp_path):
    with pytest.raises(ValueError, match="other than -1 or 1"):
        save(
            np.array([[2, 0], [0, 2]], dtype=np.int8),
            Metrics(0, 1, 0),
            tmp_path / "invalid",
            strategy="test", seed=0, steps=1, wall=0.0,
            order=2, construction="test",
        )
    assert not (tmp_path / "invalid").exists()
