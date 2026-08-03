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

11 Tests gegen bekannte Hadamard-Matrizen (Sylvester Ordnung 4,8,16; Paley Ordnung 8,12,20; Sylvester Ordnung 1,2), Williamson-Hilfsfunktionen, symmetrische Zirkulante und die Such-Engine.

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

Mehrere Strategien lassen sich als Pipeline über eine kommaseparierte
`name:steps`-Angabe verbinden:

```bash
.venv\Scripts\python run.py --strategy "circulant:100000,annealing:30000,repair:5000" --seed 42
```

Die erste Phase startet mit `search(steps, seed)`. Jede weitere Phase erhält
den besten Kandidaten der vorherigen Phase über `refine(matrix, steps, seed)`;
der Seed erhöht sich pro Phase um eins. Eine exakte Lösung (`energy == 0`)
beendet die Pipeline sofort. Die Schrittbudgets stehen deshalb in der
`--strategy`-Angabe; `--steps` wird bei einer Pipeline nicht verwendet.

Als Folgestufe eignen sich derzeit `annealing` (nur für Williamson-Matrizen),
`repair`, `direct` und `cellular`. Die übrigen Strategien besitzen kein
`refine()` und können nur die erste Phase bilden.

## Strategien

### Direct Local Search

Arbeitet direkt auf der 668x668-Matrix. Flippt symmetrische Eintragspaare und akzeptiert Schritte, die die Energie nicht erhohen. 222.778 Zeilenpaare mussen orthogonal werden.

### Williamson-Konstruktion

Nutzt 668 = 4 x 167. Baut die Matrix aus vier symmetrischen zirkulanten 167x167-Blocken. Reduziert die Variablen von 446.224 auf 4x84 = 336, mit 83 quadratischen Nebenbedingungen.

## Lizenz

MIT
