"""Download all known TT(n) solutions and build ``data/tt_index.json``.

Sources:
  Uleth archive (n <= 32, all equivalence classes)
    https://www.cs.uleth.ca/~hadi/research/TurynType/
  Best et al. 2012 (TT(34), TT(38))
  Djokovic, Kotsireas 2025 (TT(40), TT(42), TT(44))

Run once: ``python data/setup.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
ULETH_BASE = "https://www.cs.uleth.ca/~hadi/research/TurynType"

# ── Paper sources (single canonical per n) ──────────────────────────────

_PAPER_HEX: dict[int, str] = {
    # Best, Djokovic, Kharaghani, Ramp (2012) — hn=1 omitted
    34: "052351540cf016cfbe5809958b32825bc",
    38: "05128f55401f041adf7f65c53567822c9cb9c",
    # Djokovic, Kotsireas (2025) — full n digits
    40: "0603E974475D3A384C3AC8B309C44C2CDCE9A726",
    42: "0080BC3834184327FD11BE854738B30B392A9D4E56",
    44: "00F32EFBCAB9D5C5C0F1F90698982729CD07C8AA4966",
    # Kharaghani, Tayfeh-Rezaie (2005) — full n digits
    36: "060989975b685d8fc80750b21c0212eceb26",
}


def _validate(hex_str: str) -> bool:
    """Check that a hex string (full n digits) encodes a valid TT sequence."""
    import numpy as np

    n = len(hex_str)
    lengths = np.array((n, n, n, n - 1), dtype=np.int64)
    weights = np.array((1, 1, 2, 2), dtype=np.int64)
    seqs = np.zeros((4, n), dtype=np.int8)
    for idx, d in enumerate(hex_str[:-1]):
        for i, b in enumerate(f"{int(d, 16):04b}"):
            seqs[i, idx] = 1 if b == "0" else -1
    for i, b in enumerate(f"{int(hex_str[-1], 16):03b}"):
        seqs[i, n - 1] = 1 if b == "0" else -1
    # Non-periodic autocorrelation
    total = np.zeros(n, dtype=np.int64)
    for seq, L, w in zip(seqs, lengths, weights, strict=False):
        s64 = seq.astype(np.int64)
        for s in range(1, int(L)):
            total[s] += int(w) * int(np.dot(s64[: L - s], s64[s:L]))
    return int(np.dot(total[1:], total[1:])) == 0


def _parse_uleth_line(line: str) -> list[str]:
    """Parse a Uleth line. hn=1 omitted, with auto-repair for non-canonical entries."""
    results: list[str] = []
    for token in line.split():
        token = token.strip()
        if not token:
            continue
        full = token + "1"
        if _validate(full):
            results.append(full)
            continue
        # hn=1 didn't work — try all 16 digits (Stephen London additions for TT(32))
        for last_digit in "023456789abcdef":
            full = token + last_digit
            if _validate(full):
                results.append(full)
                break
    return results


def download_uleth(n: int) -> list[str]:
    """Download a single Uleth archive and return all valid hex strings."""
    url = f"{ULETH_BASE}/turyn-type-{n:02d}"
    dest = DATA / f"turyn-type-{n:02d}"
    if not dest.exists():
        print(f"  downloading {url} ...")
        try:
            subprocess.run(
                [sys.executable, "-c",
                 f"import urllib.request; urllib.request.urlretrieve('{url}', r'{dest}')"],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            print(f"  WARNING: download failed for {url}, skipping")
            return []
    text = dest.read_text(encoding="utf-8")
    entries: list[str] = []
    for line in text.splitlines():
        entries.extend(_parse_uleth_line(line))
    return entries


def main() -> None:
    print("=== Hadamard-668: TT sequence data setup ===\n")

    out = DATA / "tt_index.json"
    if out.exists():
        print(f"Index already exists: {out}")
        print(f"({out.stat().st_size:,} bytes)")
        print("Delete it first to rebuild from scratch.")
        return

    index: dict[str, list[str]] = {}

    # Step 1: Download all Uleth archives (n = 2..32 even)
    print("Uleth archive (n <= 32):")
    for n in range(2, 33, 2):
        entries = download_uleth(n)
        if entries:
            print(f"  TT({n:>2}): {len(entries)} representatives")
            index[str(n)] = entries

    # Step 2: Add paper-only entries (not on Uleth)
    print("\nPaper-only entries:")
    for n, hex_str in sorted(_PAPER_HEX.items()):
        if str(n) in index:
            continue
        if len(hex_str) < n:
            hex_str += "1"
        if _validate(hex_str):
            index[str(n)] = [hex_str]
            print(f"  TT({n:>2}): 1 representative")

    # Step 3: Write index
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"\nWritten: {out} ({out.stat().st_size:,} bytes)")
    print(
        f"Total: {sum(len(v) for v in index.values())} representatives across {len(index)} n values")


if __name__ == "__main__":
    main()
