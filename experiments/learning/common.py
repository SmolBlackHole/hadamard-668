"""Shared data loading and feature computation for the learning experiments.

The solver (src/) is NOT modified; we only read its outputs and reuse its
pure functions (Tracker residual math reimplemented here standalone).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent.parent / "data"


def load_runs(name: str) -> dict[int, list[dict]]:
    """Load a benchmark file -> {n: [run, ...]}."""
    with open(DATA / name, encoding="utf-8") as fh:
        d = json.load(fh)
    return {int(n): runs for n, runs in d["gs4"].items()}


def decode_seqs(b64: str, n: int) -> np.ndarray:
    """Decode base64 int8 blob into a (4, n) +/-1 array."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int8).reshape(4, n)


def negaperiodic_residual(seqs: np.ndarray) -> np.ndarray:
    """Full NAF residual r_t (t=0..n-1) of the 4-sequence sum. Antisymmetric."""
    n = seqs.shape[1]
    j = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    v = seqs.astype(np.int32)
    return np.sum(v[:, :, None] * v[:, shifted] * sign, axis=(0, 1))


def u_of(seqs: np.ndarray) -> np.ndarray:
    """Reduced residual u_t = r_t/4 for t = 1..floor((n-1)/2)."""
    r = negaperiodic_residual(seqs)
    n = seqs.shape[1]
    m = (n - 1) // 2
    return (r[1 : m + 1] // 4).astype(np.int32)


def q_of(seqs: np.ndarray) -> int:
    u = u_of(seqs)
    return int(np.dot(u, u))


def per_seq_negacyclic_autocorr(seqs: np.ndarray) -> np.ndarray:
    """Per-sequence negaperiodic autocorrelation, shape (4, n)."""
    n = seqs.shape[1]
    t = np.arange(n)[None, :]
    j = np.arange(n)[:, None]
    shifted = (j + t) % n
    sign = np.where(j + t < n, 1, -1).astype(np.int32)
    v = seqs.astype(np.int32)
    return np.sum(v[:, :, None] * v[:, shifted] * sign, axis=1)  # (4, n)


def seq_features(seqs: np.ndarray) -> dict[str, np.ndarray | float]:
    """Structural features of a 4x n sequence block."""
    n = seqs.shape[1]
    feats: dict[str, np.ndarray | float] = {}
    feats["q"] = q_of(seqs)
    u = u_of(seqs)
    feats["u"] = u
    feats["u_max"] = float(np.abs(u).max())
    feats["u_abs_mean"] = float(np.abs(u).mean())
    feats["u_abs_std"] = float(np.abs(u).std())
    # weighted position of dominant lag (normalized to [0, 1])
    k = int(np.argmax(np.abs(u)))
    feats["k_dom"] = float(k) / max(n - 1, 1)
    feats["u_norm2_frac"] = float(np.sum(u**2)) / max(4 * n * n, 1)
    # per-sequence stats
    ac = per_seq_negacyclic_autocorr(seqs)
    feats["ac_max"] = float(np.abs(ac).max())
    feats["ac_abs_mean"] = float(np.abs(ac).mean())
    feats["ac_abs_std"] = float(np.abs(ac).std())
    # balancedness per sequence
    feats["bal"] = float(np.abs(seqs.sum(axis=1)).mean())
    # segment sums: half-1, half-2, quadrants per sequence
    halves = seqs[:, : n // 2].sum(axis=1), seqs[:, n // 2 :].sum(axis=1)
    feats["seg_h1"] = float(np.abs(halves[0]).mean())
    feats["seg_h2"] = float(np.abs(halves[1]).mean())
    # negacyclic reversal symmetry score: max |seq +- reverse|
    rev = seqs[:, ::-1]
    negrev = -rev
    sym = np.abs(seqs + rev).mean()
    nsym = np.abs(seqs + negrev).mean()
    feats["sym"] = float(sym)
    feats["nsym"] = float(nsym)
    # number of runs (sign changes) per sequence
    changes = np.abs(seqs[:, 1:] - seqs[:, :-1]).sum(axis=1) / 2
    feats["runs_mean"] = float(changes.mean())
    return feats
