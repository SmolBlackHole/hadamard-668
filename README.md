# Hadamard-668

Finde eine reelle Hadamard-Matrix der Ordnung 668.

## Hintergrund

Ordnung 668 = 4 x 167 ist die **kleinste ungeloeste Ordnung** der Hadamard-Vermutung (seit 1867). Eine Hadamard-Matrix H in {-1,+1}^(668x668) mit H *H^T = 668* I zu finden, ware ein bedeutendes mathematisches Resultat.

## Projektstruktur

```txt
├── src/
│   ├── strategies/            # ABC-basierte Suchstrategien
│   ├── correlations.py         # periodische Energie und inkrementelle Flips
│   ├── builders.py             # Goethals-Seidel- und Propus-Matrixbuilder
│   ├── fourier.py              # gemeinsame Fourier-Projektionen
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

Die Test-Suite prueft bekannte Hadamard-Matrizen, G-S-/Propus-Konstruktionen, Strategie-Vertraege, Pipelines sowie Ergebnis-Manifeste.

## Verifier (4x4 Kontrolle)

```bash
.venv\Scripts\python -m verifier.verify tests/fixtures/h4/candidate.csv --order 4 --audit --require-filename candidate.csv
.venv\Scripts\python -m verifier.review tests/fixtures/h4 --order 4
```

## Suche starten

```bash
.venv\Scripts\python run.py --strategy turyn_greedy --steps 200000 --seed 42
```

Oder importieren:

```python
from strategies.circulant import TurynGreedySearch

matrix, metrics, elapsed = TurynGreedySearch().search(steps=5000, seed=42)
```

Jeder Lauf schreibt `candidate.csv` und `run.json` in ein neues Verzeichnis unter `runs/`.

Parameter werden über `run.py` gesetzt, zum Beispiel `--strategy spectral --steps 5000 --seed 42`.

### Pipeline

Mehrere Strategien lassen sich über eine kommaseparierte `name:steps`-Angabe
verbinden. Der empfohlene Pfad erzeugt einen symmetrischen
Goethals-Seidel-Kandidaten und repariert die Vollmatrix anschließend lokal:

```bash
.venv\Scripts\python run.py --strategy "turyn_greedy:100000,repair:5000" --seed 42
```

Die erste Phase startet mit `search()`, jede weitere mit `refine()` auf dem
vorherigen Kandidaten. Der Seed erhöht sich pro Phase um eins. Form, Datentyp
und Vorzeichen der Matrix werden bei jeder Übergabe geprüft. Eine gemeldete
Nullenergie beendet die Pipeline erst nach einer unabhängigen vollständigen
Gram-Prüfung. Die Budgets stehen in der Strategieangabe; `--steps` wird bei
einer Pipeline nicht verwendet.

`repair` kann jede passende Vorzeichenmatrix verfeinern. Die anderen
Strategien eröffnen eine Pipeline; unzulässige Folgestufen werden bereits beim
Aufbau abgelehnt.

### Mehrere Seeds

Unabhängige Läufe können parallel gestartet werden:

```bash
.venv\Scripts\python run.py --strategy turyn_greedy --steps 200000 --seed 42 --runs 8 --workers 4
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

Der Standardbenchmark umfasst 48 geordnete Fälle: acht Kernstrategien für die
Ordnungen `4, 8, 12, 16, 20, 668`. `--all` ergänzt experimentelle Strategien
und Pipelines. Jeder Fall läuft in einem eigenen Prozess und wird nach dem
angegebenen Timeout beendet. `benchmark_results.md` enthält die lesbare
Tabelle; `benchmark_results.json` speichert jeden Fall mit Status, Backend,
Algorithmuszeit, Wandzeit, Metriken, Korrelationshistogramm und Fehlerdetails.
Die Tabelle zeigt ausschließlich die von der Strategie gemeldete
Algorithmuszeit. Die JSON-Wandzeit enthält zusätzlich Prozessstart,
Numba-Kompilierung und CUDA-Warm-up und dient nur der Timeout-Diagnose.

## Strategien

| Gruppe | Strategien | Zustandsraum | Rechenort |
| --- | --- | --- | --- |
| Symmetrische Folgen | `circulant`, `annealing` | `4 × 84` unabhängige Vorzeichen | CPU/Numba und FFT |
| Allgemeine Folgen | `diffset`, `genetic`, `montecarlo`, `spectral`, `ising` | `4 × 167` | CPU; große Genetic-/Monte-Carlo-Batches auf GPU |
| Vollmatrix | `repair` | `668 × 668` | inkrementelle CPU-Deltas; finale Gram-Prüfung optional auf GPU |
| Orchestrierung | Pipelines | vorhandene Kandidaten | CPU |

### Spectral / Douglas-Rachford

`spectral` hält vier reelle Folgen der Länge 167. Die Orthogonalprojektion
skaliert für jede Fourierfrequenz den Vierervektor so, dass die Summe seiner
Leistungen `4K` beträgt. Danach folgt die Vorzeichenprojektion im echten
Douglas-Rachford-Schritt. Dadurch entfallen SVD oder Polarzerlegung einer
668×668-Matrix vollständig; nur der finale diskrete Kandidat wird als
Goethals-Seidel-Matrix aufgebaut.

### Historische symmetrische Goethals-Seidel-Konstruktion

Nutzt `668 = 4 × 167` und vier symmetrische zirkulante Blöcke. Dadurch sinkt
der diskrete Suchraum von 446.224 Matrixeinträgen auf `4 × 84 = 336`
unabhängige Vorzeichen.

### Turyn-Typ-Konstruktion

Die vier TT(56)-Folgen haben Laengen `(56, 56, 56, 55)` und muessen die
gewichtete nichtperiodische Autokorrelationsbedingung
`N_X + N_Y + 2N_Z + 2N_W = 0` erfuellen. Die fertigen Folgen werden erst dann
ueber die feste Goethals-Seidel-Blockanordnung zu einer 668x668-Matrix erweitert.

## Lizenz

MIT
