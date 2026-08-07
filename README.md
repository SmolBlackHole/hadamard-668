# Hadamard-668

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
![Tests](https://img.shields.io/badge/tests-passing-green)

Suche nach einer reellen Hadamard-Matrix der Ordnung 668 über die
Goethals-Seidel-Konstruktion mit vier negazyklischen (±1)-Folgen.

## Stand

Der NAF-/Tabu-Solver löst kleine und mittlere GS4-Instanzen zuverlässig und
hat mit erhöhtem Budget bereits Lösungen für `n=52` beziehungsweise Ordnung
208 erzeugt. Das eigentliche Projektziel bleibt `n=167` beziehungsweise eine
Hadamard-Matrix der Ordnung 668.

Reproduzierbare Messwerte: [docs/BENCHMARK.md](docs/BENCHMARK.md)

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

```text
NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t) = 0    für alle t = 1,…,n-1
```

Der Tracker speichert `u_t = r_t/4` (nur `⌊(n-1)/2⌋` unabhängige Residuen) und
berechnet die Orthogonalitätsenergie als `E = 64n · Σ u_t²`. Ein Single-Flip
ändert jedes `u_t` um `{-1,0,+1}` — die Energieänderung ist ein Skalarprodukt
über kleine Ganzzahlen.

Kompaktes Modell: [docs/SOLVER_MODEL.md](docs/SOLVER_MODEL.md). Ausführliche neue
Charakterisierung: [docs/TIGHT_FRAME_CHARACTERIZATION.md](docs/TIGHT_FRAME_CHARACTERIZATION.md).

## Architektur

```text
src/
  tracker.py      # NAF-Energie-Tracker: Delta-Cache, Batch-Flip, O(1) Accept
  solver.py       # Iterated Local Search: Singles → Tabu → Kick
  generator.py    # Orchestrierung: Builder → Tracker → Solver → Metrics
  builder.py      # GS4-Blockmatrix-Konstruktion
  metrics.py      # Gram-Metriken, Orthogonalitätsprüfung
  benchmark_stats.py  # Wilson-CI, Z-Test, McNemar
  output.py       # JSON-Persistenz und Integritätsprüfung
  fast_hash.py    # Äquivalenzklassen-Hash (negashift + reverse)
  verify.py       # Dataset-Integritätscheck
scripts/
  ablation.py     # Systematischer Komponentenvergleich
  budget_scale.py # Solve-Rate vs. Step-Budget
tests/
data/
  benchmark.json  # Referenz-Benchmark (100 Seeds × 16 n)
  solutions.json  # Gefundene Hadamard-Matrizen
docs/             # Modell, Benchmarks, Roadmap und Forschungsberichte
```

Kernpfad: `generator.search()` → `solver.search()` → `tracker.flip_batch()` /
`tracker.accept()` → `builder.build()` →
`metrics.check_orthogonality()`.

## Design-Entscheidungen

- **NAF-Reduktion** statt voller Gram-Matrix: 446k → 83 Einträge bei n=167.
- **u = r/4, int8-Delta-Cache**: 8× kompakter, integer-norm2 ist gecached.
- **Batch-Flip**: `D @ u` Matmul statt 64 einzelner Dot-Produkte, Early-Exit
  beim ersten Treffer.
- **Vorkomputierte Geometrie**: `accept()` ist reine Indexed-Addition.
- **Tabu-Walk**: kompiliertes, nichtmonotones Escape aus lokalen Minima.
- **Kein Pair-Rescue**: gepaarte Ablationen waren langsamer und lösten bei
  n=36–40 seltener; auch der verbleibende Top-K-Overhead wurde entfernt.
- **Kick statt Restart**: Kicks sind explorativ und werden immer akzeptiert.
  Restarts warfen gute Zustände weg (–20% Lösungen bei n=32).

## Was wir verworfen haben

| Feature | Grund |
| --- | --- |
| Triples | 1000-Seed-Test: keine signifikante Verbesserung (p=0.855) |
| Pair-Rescue | langsamer und bei n=36/38 signifikant schlechter |
| monotones Gray | viele Q-Verbesserungen, aber schlechtere Solve-Rate |
| einzelne Intervallflips | kein Low-Q-Treffer auf n=52-Zuständen |
| Hard-Restarts | –20% Lösungen bei n=32, destruktiv |
| rescue_mode | kein messbarer Effekt |
| Random-Scan-Start | kein systematischer Gewinn |
| GramTracker | korrekt, aber ~5000× langsamer als NAF-Tracker |

## Nächste Schritte

Der Solver erreicht bei `n=52` häufig die letzten ein oder zwei
Residualfehler. Der nächste Test erlaubt eine strukturierte vorübergehende
Verschlechterung und bewertet erst das lokale Minimum nach einem erneuten
Single-Abstieg. Damit wird Reachability statt nur unmittelbares Q untersucht.

Aktuelle Roadmap: [docs/TODO.md](docs/TODO.md). Offene Forschungsannahmen:
[docs/HYPOTHESES.md](docs/HYPOTHESES.md). Übersicht aller Dokumente:
[docs/README.md](docs/README.md). Eigenständige Forschungsübergabe:
[docs/RESEARCH_BRIEF.md](docs/RESEARCH_BRIEF.md).

## Referenzen

- Goethals & Seidel (1970): GS4-Blockstruktur
- Djoković & Kotsireas: *Negaperiodic Golay pairs and Hadamard matrices*,
  [arXiv:1508.00640](https://arxiv.org/abs/1508.00640)
- P. Zinn et al.: *Generating Hadamard matrices with transformers*,
  [arXiv:2604.11101](https://arxiv.org/abs/2604.11101)
  ([Code](https://github.com/pzinn/hadamard))
