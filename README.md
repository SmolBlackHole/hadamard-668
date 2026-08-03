# Hadamard-668

Finde eine reelle Hadamard-Matrix der Ordnung 668.

## Hintergrund

Ordnung 668 = 4 x 167 ist die **kleinste ungeloeste Ordnung** der Hadamard-Vermutung (seit 1867). Eine Hadamard-Matrix H in {-1,+1}^(668x668) mit H *H^T = 668* I zu finden, ware ein bedeutendes mathematisches Resultat.

## Projektstruktur

```txt
├── src/
│   ├── search.py              # Such-Engine (direkt + Williamson)
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
PYTHONPATH="src" .venv\Scripts\python tests/test_search.py
```

11 Tests gegen bekannte Hadamard-Matrizen (Sylvester Ordnung 4,8,16; Paley Ordnung 8,12,20; Sylvester Ordnung 1,2), Williamson-Hilfsfunktionen, symmetrische Zirkulante und die Such-Engine.

## Verifier (4x4 Kontrolle)

```bash
.venv\Scripts\python -m verifier.verify tests/fixtures/h4/candidate.csv --order 4 --audit --require-filename candidate.csv
.venv\Scripts\python -m verifier.review tests/fixtures/h4 --order 4
```

## Suche starten

```bash
PYTHONPATH="src" .venv\Scripts\python src/search.py
```

Oder importieren:

```python
from search import run, direct_search, williamson_search

run(strategy="both", steps=5000, seed=42)
```

Ergebnis einer exakten Losung landet in `output/candidate.csv`.

Parameter (oben in `src/search.py` anderbar):

- `STEPS` — Anzahl Suchschritte
- `SEED` — Zufallsseed
- `STRATEGY` — `"direct"`, `"williamson"` oder `"both"`

## Strategien

### Direct Local Search

Arbeitet direkt auf der 668x668-Matrix. Flippt symmetrische Eintragspaare und akzeptiert Schritte, die die Energie nicht erhohen. 222.778 Zeilenpaare mussen orthogonal werden.

### Williamson-Konstruktion

Nutzt 668 = 4 x 167. Baut die Matrix aus vier symmetrischen zirkulanten 167x167-Blocken. Reduziert die Variablen von 446.224 auf 4x84 = 336, mit 83 quadratischen Nebenbedingungen.

## Lizenz

MIT
