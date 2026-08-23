"""Tests for shared execution data contracts."""

from __future__ import annotations

import pytest

from src.models import CandidateBudget


def test_candidate_budget_rejects_negative_request_without_mutation() -> None:
    budget = CandidateBudget(10)

    with pytest.raises(ValueError, match="must not be negative"):
        budget.take(-1)

    assert budget.used == 0
    assert budget.remaining == 10
