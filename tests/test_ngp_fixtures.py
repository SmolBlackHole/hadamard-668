"""Validate known NGP pairs against our builder + energy computation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from builder import Builder
from metrics import check_orthogonality
from verify import independent_audit

NGP_JSON = (
    Path(__file__).parent.parent
    / "data"
    / "ngp_dataset"
    / "ngp_dataset_balonin_djokovic_2015"
    / "ngp_pairs.json"
)


def _load_pairs():
    data = json.loads(NGP_JSON.read_text(encoding="utf-8"))
    return [(r["length"], r["a"], r["b"], r["family"]) for r in data["records"]]


PAIRS = _load_pairs()


@pytest.mark.parametrize("n,a,b,family", PAIRS)
def test_known_ngp_energy_zero(n, a, b, family):
    """Every known NGP must have zero energy with our builder + pass pure audit."""
    bld = Builder(kind="golay_2n", n=n)
    seqs = np.array([a, b], dtype=np.int8)
    H = bld.build(seqs)
    m = check_orthogonality(H)
    assert m.energy == 0, f"family={family} n={n} energy={m.energy}"
    independent_audit(H.tolist())
