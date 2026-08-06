# Hadamard-668

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-51%2F51-green)]()

Suche nach einer reellen Hadamard-Matrix der Ordnung 668 über die
Goethals-Seidel-Konstruktion mit vier negazyklischen (±1)-Folgen.

## Stand

| n | Ordnung | Gelöst | 95% CI | Avg / Run |
| ---: | :---: | :---: | :---: | ---: |
| 24 | 96 | 100/100 | 0.963–1.000 | 17 ms |
| 28 | 112 | 96/100 | 0.902–0.984 | 107 ms |
| 30 | 120 | 84/100 | 0.756–0.899 | 182 ms |
| 32 | 128 | 31/100 | 0.228–0.406 | 313 ms |
| 34 | 136 | 14/100 | 0.085–0.221 | 322 ms |
| 36 | 144 | 5/100 | 0.022–0.112 | 314 ms |
| 38 | 152 | 1/100 | 0.002–0.054 | 307 ms |
| 40–167 | 160–668 | — | 0.000–0.037 | 250–554 ms |

200k Steps, 12 Worker. Details: [BENCHMARK.md](BENCHMARK.md)

## Installation

```bash
git clone <repo-url>
cd hadamard-668
pip install -e ".[dev]"
```

Dev-Werkzeuge: Ruff, Pyright, pytest, Hypothesis, pytest-cov und pytest-gremlins.

## Schnellstart

```bash
# Einzelner Run (Ordnung 128, 200k Steps)
python run.py --strategy gs4 --order 128 --steps 200000 --seed 42

# Sweep über mehrere n
python run.py --sweep gs4 32 34 36 --seeds 100 --steps 200000 --workers 12

# Ablationstest
python -m scripts.ablation

# Budget-Scaling
python -m scripts.budget_scale

# Tests
pytest tests/ -m "not slow"
pytest tests/ -m slow          # Regressionstests (n=24, ~1s)
```

## Qualität prüfen

```bash
python scripts/quality.py          # Format, Lint, Typen, Tests, compileall
python scripts/quality.py fix      # Ruff formatieren und sichere Fixes anwenden
python scripts/quality.py coverage # Testabdeckung anzeigen
python scripts/quality.py mutate   # Tracker und Solver mutieren
```

Mutationstests sind wegen ihrer Laufzeit absichtlich nicht Teil von `check`.

## Wie es funktioniert

Vier ±1-Folgen `a,b,c,d` der Länge `n` bilden über negazyklische Matrizen eine
GS4-Blockmatrix. Die Hadamard-Bedingung ist äquivalent zu:

```
NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t) = 0    für alle t = 1,…,n-1
```

Der Tracker speichert `u_t = r_t/4` (nur `⌊n/2⌋` unabhängige Residuen) und
berechnet die Orthogonalitätsenergie als `E = 64n · Σ u_t²`. Ein Single-Flip
ändert jedes `u_t` um `{-1,0,+1}` — die Energieänderung ist ein Skalarprodukt
über kleine Ganzzahlen.

Die vollständige Herleitung: [HYPOTHESIS.md](HYPOTHESIS.md)

## Architektur

```
src/
  tracker.py      # NAF-Energie-Tracker: Delta-Cache, Batch-Flip, O(1) Accept
  solver.py       # Iterated Local Search: Singles → Pairs → Kick
  generator.py    # Orchestrierung: Builder → Tracker → Solver → Metrics
  builder.py      # GS4-Blockmatrix-Konstruktion
  metrics.py      # Gram-Metriken, Orthogonalitätsprüfung
  statistics.py   # Wilson-CI, Z-Test, McNemar
  output.py       # JSON-Persistenz mit SHA-256-Dedup
  fast_hash.py    # Äquivalenzklassen-Hash (negashift + reverse)
  verify.py       # Dataset-Integritätscheck
scripts/
  ablation.py     # Systematischer Komponentenvergleich
  budget_scale.py # Solve-Rate vs. Step-Budget
tests/
data/
  benchmark.json  # Referenz-Benchmark (100 Seeds × 16 n)
  solutions.json  # Gefundene Hadamard-Matrizen
```

Kernpfad: `generator.search()` → `solver.search()` → `tracker.flip_batch()` /
`tracker.accept()` / `tracker.pair_energies()` → `builder.build()` →
`metrics.check_orthogonality()`.

## Design-Entscheidungen

- **NAF-Reduktion** statt voller Gram-Matrix: 446k → 83 Einträge bei n=167.
- **u = r/4, int8-Delta-Cache**: 8× kompakter, integer-norm2 ist gecached.
- **Batch-Flip**: `D @ u` Matmul statt 64 einzelner Dot-Produkte, Early-Exit
  beim ersten Treffer.
- **Vorkomputierte Geometrie**: `accept()` ist reine Indexed-Addition.
- **Kick statt Restart**: Kicks sind explorativ und werden immer akzeptiert.
  Restarts warfen gute Zustände weg (–20% Lösungen bei n=32).

## Was wir verworfen haben

| Feature | Grund |
| --- | --- |
| Triples | 1000-Seed-Test: keine signifikante Verbesserung (p=0.855) |
| Hard-Restarts | –20% Lösungen bei n=32, destruktiv |
| rescue_mode | kein messbarer Effekt |
| Random-Scan-Start | kein systematischer Gewinn |
| GramTracker | korrekt, aber ~5000× langsamer als NAF-Tracker |

## Nächste Schritte

Budget-Scaling zeigt: n=32 erreicht 99% bei 1.6M Steps (reines Rechenlimit),
aber n=38 bleibt bei 14% selbst mit 1.6M Steps (Suchdynamik-Limit). Ab n=36
dominiert das Plateau-Problem.

Nächster Kandidat: **Tabu-Walk** statt zufälligem Kick, um gezielt aus lokalen
Minima herauszulaufen. Referenz: `pzinn/hadamard` `improve.py`.

Siehe [RESEARCH.md](RESEARCH.md) für die priorisierte Testkaskade.

## Referenzen

- Goethals & Seidel (1970): GS4-Blockstruktur
- Djoković & Kotsireas: *Negaperiodic Golay pairs and Hadamard matrices*,
  [arXiv:1508.00640](https://arxiv.org/abs/1508.00640)
- P. Zinn et al.: *Generating Hadamard matrices with transformers*,
  [arXiv:2604.11101](https://arxiv.org/abs/2604.11101)
  ([Code](https://github.com/pzinn/hadamard))
