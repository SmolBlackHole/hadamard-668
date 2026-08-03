# Hadamard-668

Finde eine reelle Hadamard-Matrix der Ordnung 668.

## Hintergrund

Ordnung 668 = 4 x 167 ist die **kleinste ungeloeste Ordnung** der Hadamard-Vermutung (seit 1867). Eine Hadamard-Matrix H in {-1,+1}^(668x668) mit H *H^T = 668* I zu finden, ware ein bedeutendes mathematisches Resultat.

## Projektstruktur

```txt
├── src/
│   ├── strategies/            # ABC-basierte Suchstrategien
│   └── verifier/              # Pruf-Toolchain
│       ├── verify.py          # Matrix-Prufung
│       ├── review.py          # Bundle-Review
│       ├── score.py           # Fortschritt-Metriken
│       ├── validate.py        # JSON-Manifest-Validator
│       ├── seed.py            # Deterministischer Seed-Generator
│       └── known.py           # Bekannte Matrizen (Sylvester, Paley)
├── schema/
│   ├── run.schema.json        # JSON Schema fur Suchlauf-Dokumentation
│   └── template.json          # Vorlage fur run.json
├── tests/
│   ├── test_search.py         # Engine-Smoke-Tests
│   └── fixtures/
│       └── h4/                # 4x4 Kontroll-Fixture
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Tests

```bash
.venv\Scripts\python -m pytest tests/ -v
# oder ohne pytest:
.venv\Scripts\python -m pytest
```

Die Test-Suite prueft bekannte Hadamard-Matrizen, Konstruktionen, Strategie-Vertraege, Pipelines sowie Ergebnis-Manifeste.

## Verifier (4x4 Kontrolle)

```bash
.venv\Scripts\python -m verifier.verify tests/fixtures/h4/candidate.csv --order 4 --audit --require-filename candidate.csv
.venv\Scripts\python -m verifier.review tests/fixtures/h4 --order 4
```

## Suche starten

```bash
.venv\Scripts\python run.py --strategy circulant --steps 200000 --seed 42
```

Oder importieren:

```python
from strategies.circulant import CirculantSearch

matrix, metrics, elapsed = CirculantSearch().search(steps=5000, seed=42)
```

Jeder Lauf schreibt `candidate.csv` und `run.json` in ein neues Verzeichnis unter `runs/`.

Parameter werden über `run.py` gesetzt, zum Beispiel `--strategy rowwise --steps 5000 --seed 42`.

### Pipeline

Mehrere Strategien lassen sich über eine kommaseparierte `name:steps`-Angabe
verbinden. Der empfohlene Pfad bleibt vollständig im Williamson-Raum, bevor
die Vollmatrix einmal lokal repariert wird:

```bash
.venv\Scripts\python run.py --strategy "circulant:100000,annealing_w:30000,repair:5000" --seed 42
```

Die erste Phase startet mit `search()`, jede weitere mit `refine()` auf dem
vorherigen Kandidaten. Der Seed erhöht sich pro Phase um eins. Form, Datentyp
und Vorzeichen der Matrix werden bei jeder Übergabe geprüft. Eine gemeldete
Nullenergie beendet die Pipeline erst nach einer unabhängigen vollständigen
Gram-Prüfung. Die Budgets stehen in der Strategieangabe; `--steps` wird bei
einer Pipeline nicht verwendet.

`annealing_w` akzeptiert ausschließlich Williamson-Matrizen. `repair` und
`direct` können jede passende Vorzeichenmatrix verfeinern; `cellular` und
`ca_spectral` akzeptieren nur exakt extrahierbare Goethals-Seidel-Matrizen.
Insbesondere darf `hybrid` nicht vor `annealing_w` stehen, weil `hybrid` auch
Goethals-Seidel-Kandidaten liefern kann. Unzulässige Folgestufen werden beim
Aufbau oder durch ihre Strukturprüfung abgelehnt.

`bt:<strategie>` verteilt `--steps` exakt auf begrenzte, geseedete Restarts.
Kann die innere Strategie `refine()`, wird der beste gespeicherte Checkpoint
perturbiert und tatsächlich weiterverwendet. Andernfalls sind es klar
unabhängige Restarts. Pro Seed bleiben höchstens fünf Checkpoints erhalten.

### Mehrere Seeds

Unabhängige Läufe können parallel gestartet werden:

```bash
.venv\Scripts\python run.py --strategy circulant --steps 200000 --seed 42 --runs 8 --workers 4
```

Die Seeds sind deterministisch `seed, seed+1, ...`. Jeder Lauf erhält ein
eigenes Verzeichnis mit `candidate.csv` und `run.json`. Am Ende werden alle
Energien zusammengefasst und der beste Lauf anhand einer vollständigen
Gram-Metrik markiert. Für die CUDA-Batchstrategie `montecarlo` wird die
Workerzahl automatisch auf eins begrenzt, damit nicht mehrere Prozesse um
dieselbe GPU konkurrieren.

## Benchmark

```bash
.venv\Scripts\python benchmark.py --timeout 60
.venv\Scripts\python benchmark.py --all --timeout 60
```

Der Standardbenchmark umfasst 54 geordnete Fälle: neun Kernstrategien für die
Ordnungen `4, 8, 12, 16, 20, 668`. `--all` ergänzt experimentelle Strategien
und Pipelines. Jeder Fall läuft in einem eigenen Prozess und wird nach dem
angegebenen Timeout beendet. `benchmark_results.md` enthält Wandzeit,
Rohenergie, RMS-Korrelation und orthogonale Zeilenpaare. Prozessstart,
Numba-Kompilierung und CUDA-Warm-up gehören zur gemessenen Wandzeit und müssen
bei Mikrovergleichen separat aufgewärmt werden.

## Strategien

| Gruppe | Strategien | Zustandsraum | Rechenort |
| --- | --- | --- | --- |
| Symmetrische Folgen | `circulant`, `annealing_w` | `4 × 84` unabhängige Vorzeichen | CPU/Numba und FFT |
| Allgemeine Folgen | `diffset`, `genetic`, `montecarlo`, `spectral`, `ising`, `gold`, `walsh` | `4 × 167` | CPU; große Genetic-/Monte-Carlo-Batches auf GPU |
| Zelluläre Regeln | `cellular`, `ca_spectral`, `ca_fft` | vier periodische 167er-Folgen | CPU-Faltung oder FFT |
| Teilkonstruktion | `baumert` | drei symmetrische 167er-Folgen | CPU/Numba |
| Vollmatrix | `repair`, `direct`, `rowwise` | `668 × 668` beziehungsweise Zeilenbatches | inkrementelle CPU-Deltas; finale Gram-Prüfung optional auf GPU |
| Solver/Orchestrierung | `sat`, `bt:...`, Pipelines | kompakte Bool-Folgen oder vorhandene Kandidaten | CPU/Z3 beziehungsweise Kindprozesse |

### Spectral / Douglas-Rachford

`spectral` hält vier reelle Folgen der Länge 167. Die Orthogonalprojektion
skaliert für jede Fourierfrequenz den Vierervektor so, dass die Summe seiner
Leistungen `4K` beträgt. Danach folgt die Vorzeichenprojektion im echten
Douglas-Rachford-Schritt. Dadurch entfallen SVD oder Polarzerlegung einer
668×668-Matrix vollständig; nur der finale diskrete Kandidat wird als
Goethals-Seidel-Matrix aufgebaut.

### Direct Local Search

Arbeitet direkt auf der 668×668-Matrix. Gekoppelte Eintragsflips aktualisieren
die persistente Gram-Matrix in linearer Zeit und behalten den symmetrischen
Matrixraum bei.

### Williamson-Konstruktion

Nutzt `668 = 4 × 167` und vier symmetrische zirkulante Blöcke. Dadurch sinkt
der diskrete Suchraum von 446.224 Matrixeinträgen auf `4 × 84 = 336`
unabhängige Vorzeichen.

## Lizenz

MIT
