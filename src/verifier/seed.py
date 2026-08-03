#!/usr/bin/env python3
"""Deterministic seed generator for search lane assignment."""

import hashlib

LANES = (
    ["gs_sds"] * 26
    + ["sat_cp_pb"] * 8
    + ["alternative_algebraic"] * 6
    + ["literature_transfer"] * 6
    + ["local_search"] * 6
    + ["evolutionary_gpu"] * 4
)


def lane_and_seed(method: str, run_index: int = 0) -> tuple[str, int]:
    """Return (lane, seed) for a method family and run index.

    When method is 'suggested', picks deterministically from LANES
    based on a hash of the empty string (no task/wallet context needed).
    """
    if method == "suggested":
        d = hashlib.sha256(b"|lane").digest()
        method = LANES[int.from_bytes(d[:8], "big") % len(LANES)]

    seed_input = f"h668|{method}|{run_index}"
    seed = int.from_bytes(hashlib.sha256(
        seed_input.encode()).digest()[-8:], "big")
    return method, seed
