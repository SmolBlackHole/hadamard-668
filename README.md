# Hadamard-668

Finde eine reelle Hadamard-Matrix der Ordnung 668 = 4 x 167.

## Strategien

```sh
python run.py --strategy gs4       --order 668   # Goethals-Seidel, 4 negazyklische Bloecke (default)
python run.py --strategy golay_2n  --order 668   # 2N Golay-Paar
python run.py --strategy gs4_group --order 668   # GS4 group-circulant
python run.py --strategy tensor    --order 240   # Kronecker-Produkt (240 = 16*3*5)
```

## Build

```python
from builders import Builder

b = Builder(kind="gs4", n=167)
seqs = ...  # (4, 167) int8
H = b.build(seqs)  # 668x668
```

## Struktur

```txt
src/
  builders.py       # Builder(kind, n) — Hadamard-Matrix-Konstruktion
  tracker.py        # GramTracker — O(N^2) inkrementelle Energie
  gpu.py            # Metriken, Orthogonalitaetspruefung
  output.py         # CSV + JSON Persistenz
  verify.py         # Pure-Python Referenz-Verifikation
  strategies/
    kflip.py        # Iterated Local Search (KFlip)
    custom.py       # CustomSolver — Suche via KFlip + GramTracker
    registry.py     # CLI-Name -> Strategy
    base.py         # Result, SearchStrategy ABC
tests/
run.py              # CLI
```
